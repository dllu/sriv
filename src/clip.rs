use std::convert::TryInto;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use anyhow::{anyhow, Context, Result};
use candle::safetensors::MmapedSafetensors;
use candle::{utils::cuda_is_available, DType, Device, Module, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, siglip};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use image::imageops::FilterType;
use sha1::Sha1;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::app::AppProxy;
use crate::color::OutputColorSpace;
use crate::source::SourceVersion;
use crate::state::ImageRequest;
use crate::thumbnails::EncodedThumbnail;

const MAX_IMAGE_BATCH: usize = 4;
const MODEL_ID: &str = "google/siglip2-base-patch16-512";
const MODEL_REVISION: &str = "a89f5c5093f902bf39d3cd4d81d2c09867f0724b";
const EMBEDDING_DIM: usize = 768;
const EMBEDDING_CACHE_EXTENSION: &str =
    "siglip2-b16-512-a89f5c5093f902bf39d3cd4d81d2c09867f0724b-prep1.embedding";

/// Load the two embedding towers without the unused classification logit parameters.
struct EmbeddingModel {
    vision: siglip::VisionModel,
    text: siglip::TextModel,
    dtype: DType,
}

impl EmbeddingModel {
    fn new(config: &siglip::Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            vision: siglip::VisionModel::new(&config.vision_config, true, vb.pp("vision_model"))?,
            text: siglip::TextModel::new(&config.text_config, vb.pp("text_model"))?,
            dtype: vb.dtype(),
        })
    }

    fn get_image_features(&self, image: &Tensor) -> Result<Tensor> {
        Ok(self
            .vision
            .forward(&image.to_dtype(self.dtype)?)?
            .to_dtype(DType::F32)?)
    }

    fn get_text_features(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.text.forward(tokens)?.to_dtype(DType::F32)?)
    }
}

struct Inference {
    model: EmbeddingModel,
    tokenizer: Tokenizer,
    device: Device,
    image_size: usize,
}

struct ModelFiles {
    config: siglip::Config,
    tokenizer: Tokenizer,
    weights: PathBuf,
}

impl ModelFiles {
    fn fetch() -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::from_env().build()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            MODEL_ID.to_string(),
            hf_hub::RepoType::Model,
            MODEL_REVISION.to_string(),
        ));
        Ok(Self {
            config: serde_json::from_slice(&fs::read(repo.get("config.json")?)?)?,
            tokenizer: load_tokenizer(repo.get("tokenizer.json")?)?,
            weights: repo.get("model.safetensors")?,
        })
    }
}

struct ModelWeights<'a> {
    tensors: MmapedSafetensors,
    stop_rx: &'a Receiver<()>,
}

impl SimpleBackend for ModelWeights<'_> {
    fn get(
        &self,
        shape: candle::Shape,
        name: &str,
        hints: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle::Result<Tensor> {
        if is_stopping(self.stop_rx) {
            candle::bail!("SigLIP 2 model loading cancelled");
        }
        SimpleBackend::get(&self.tensors, shape, name, hints, dtype, device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> candle::Result<Tensor> {
        if is_stopping(self.stop_rx) {
            candle::bail!("SigLIP 2 model loading cancelled");
        }
        SimpleBackend::get_unchecked(&self.tensors, name, dtype, device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        SimpleBackend::contains_tensor(&self.tensors, name)
    }
}

impl Inference {
    fn load(
        files: ModelFiles,
        use_cuda: bool,
        device_flag: &AtomicBool,
        stop_rx: &Receiver<()>,
    ) -> Result<Option<Self>> {
        if is_stopping(stop_rx) {
            return Ok(None);
        }
        let device = if use_cuda {
            match Device::new_cuda(0) {
                Ok(device) => device,
                Err(error) => {
                    if !is_stopping(stop_rx) {
                        eprintln!("Failed to initialize CUDA: {error}. Falling back to CPU.");
                    }
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };
        if is_stopping(stop_rx) {
            return Ok(None);
        }
        device_flag.store(device.is_cuda(), Ordering::Relaxed);
        let ModelFiles {
            config,
            tokenizer,
            weights,
        } = files;
        let dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let tensors = unsafe { MmapedSafetensors::new(weights)? };
        let vb = VarBuilder::from_backend(
            Box::new(ModelWeights { tensors, stop_rx }),
            dtype,
            device.clone(),
        );
        let model = EmbeddingModel::new(&config, vb);
        let model = if is_stopping(stop_rx) {
            drop(model);
            Ok(None)
        } else {
            model.map(Some)
        };
        if device.is_cuda() && !matches!(model, Ok(Some(_))) {
            let _ = device.synchronize();
        }
        Ok(model?.map(|model| Self {
            model,
            tokenizer,
            device,
            image_size: config.vision_config.image_size,
        }))
    }
}

fn load_tokenizer(path: PathBuf) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(anyhow::Error::msg)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 64,
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(64),
        pad_id: 0,
        pad_token: "<pad>".into(),
        ..Default::default()
    }));
    Ok(tokenizer)
}

/// Events emitted by the background SigLIP 2 worker.
#[derive(Debug)]
pub enum ClipEvent {
    SourceChanged {
        request: ImageRequest,
    },
    ImageReady {
        request: ImageRequest,
        embedding: Vec<f32>,
    },
    ImageError {
        request: ImageRequest,
        error: String,
    },
    ThumbnailError {
        request: ImageRequest,
        error: String,
    },
    TextReady {
        request_id: u64,
        embedding: Vec<f32>,
    },
    TextError {
        request_id: u64,
        error: String,
    },
}

struct ImageJob {
    request: ImageRequest,
    image_path: PathBuf,
    version: SourceVersion,
    thumbnail: EncodedThumbnail,
}

struct TextJob {
    request_id: u64,
    query: String,
}

/// Manages SigLIP 2 inference in a background thread and caching of embeddings.
pub struct ClipEngine {
    cache_base: Option<PathBuf>,
    proxy: AppProxy,
    output_color_space: OutputColorSpace,
    workers: Option<ClipWorkers>,
}

struct ClipWorkers {
    job_tx: Sender<ImageJob>,
    text_tx: Sender<TextJob>,
    result_rx: Receiver<ClipEvent>,
    using_cuda: Arc<AtomicBool>,
    threads: ClipWorkerThreads,
}

/// Inference must finish dropping its CUDA resources before the process exits.
pub(crate) struct ClipWorkerThreads {
    stop_tx: Option<Sender<()>>,
    handles: Vec<JoinHandle<()>>,
}

struct ClipWorkerContext {
    cache_base: Option<PathBuf>,
    inference: Arc<OnceLock<Option<Inference>>>,
    output_color_space: OutputColorSpace,
    result_tx: Sender<ClipEvent>,
    device_flag: Arc<AtomicBool>,
    proxy: AppProxy,
    stop_rx: Receiver<()>,
}

impl ClipEngine {
    pub(crate) fn new(
        cache_base: Option<PathBuf>,
        proxy: AppProxy,
        output_color_space: OutputColorSpace,
    ) -> Self {
        Self {
            cache_base,
            proxy,
            output_color_space,
            workers: None,
        }
    }

    /// Opening search starts loading while the user types their query.
    pub fn warm_up(&mut self) -> Result<()> {
        if self.workers.is_none() {
            self.workers = Some(ClipWorkers::start(
                self.cache_base.clone(),
                &self.proxy,
                self.output_color_space,
            )?);
        }
        Ok(())
    }

    pub fn queue_image(
        &mut self,
        request: ImageRequest,
        image_path: PathBuf,
        version: SourceVersion,
        thumbnail: EncodedThumbnail,
    ) -> Result<()> {
        self.warm_up()?;
        self.workers
            .as_ref()
            .unwrap()
            .job_tx
            .send(ImageJob {
                request,
                image_path,
                version,
                thumbnail,
            })
            .map_err(|_| anyhow!("SigLIP 2 workers stopped"))
    }

    pub fn request_text(&mut self, request_id: u64, query: String) -> Result<()> {
        self.warm_up()?;
        self.workers
            .as_ref()
            .unwrap()
            .text_tx
            .send(TextJob { request_id, query })
            .map_err(|_| anyhow!("SigLIP 2 workers stopped"))
    }

    pub fn try_recv(&self) -> Result<ClipEvent, TryRecvError> {
        self.workers
            .as_ref()
            .ok_or(TryRecvError::Empty)?
            .result_rx
            .try_recv()
    }

    pub fn device_kind(&self) -> Option<&'static str> {
        self.workers.as_ref().map(|workers| {
            if workers.using_cuda.load(Ordering::Relaxed) {
                "GPU"
            } else {
                "CPU"
            }
        })
    }

    /// Signal cancellation now. Dropping the returned handle waits for the workers.
    pub(crate) fn stop(&mut self) -> Option<ClipWorkerThreads> {
        let mut workers = self.workers.take()?;
        workers.threads.stop();
        Some(workers.threads)
    }
}

impl ClipWorkers {
    fn start(
        cache_base: Option<PathBuf>,
        proxy: &AppProxy,
        output_color_space: OutputColorSpace,
    ) -> Result<Self> {
        let (job_tx, job_rx) = unbounded::<ImageJob>();
        let (text_tx, text_rx) = unbounded::<TextJob>();
        let (result_tx, result_rx) = unbounded::<ClipEvent>();
        let (stop_tx, stop_rx) = bounded::<()>(0);
        let inference = Arc::new(OnceLock::new());
        let want_cuda = cuda_is_available();
        let using_cuda = Arc::new(AtomicBool::new(false));
        let worker_count = if want_cuda {
            1
        } else {
            num_cpus::get().clamp(1, 8)
        };
        // Drop also joins any workers started before a later thread spawn fails.
        let mut workers = Self {
            job_tx,
            text_tx,
            result_rx,
            using_cuda: using_cuda.clone(),
            threads: ClipWorkerThreads {
                stop_tx: Some(stop_tx),
                handles: Vec::with_capacity(worker_count),
            },
        };
        for worker_idx in 0..worker_count {
            let worker_ctx = ClipWorkerContext {
                cache_base: cache_base.clone(),
                inference: Arc::clone(&inference),
                output_color_space,
                result_tx: result_tx.clone(),
                device_flag: using_cuda.clone(),
                proxy: proxy.clone(),
                stop_rx: stop_rx.clone(),
            };
            let worker_rx = job_rx.clone();
            let worker_text_rx = text_rx.clone();
            let worker = thread::Builder::new()
                .name(format!("clip-worker-{worker_idx}"))
                .spawn(move || {
                    let proxy = worker_ctx.proxy.clone();
                    let stop_rx = worker_ctx.stop_rx.clone();
                    run_worker(worker_ctx, worker_rx, worker_text_rx, want_cuda);
                    // The last worker dropping its sender disconnects result_rx. Wake the UI
                    // so pending work becomes a visible failure even without further input.
                    if !is_stopping(&stop_rx) {
                        proxy.wakeup();
                    }
                })
                .context("failed to spawn clip worker thread")?;
            workers.threads.handles.push(worker);
        }
        drop(result_tx);
        Ok(workers)
    }
}

impl ClipWorkerThreads {
    fn stop(&mut self) {
        // Disconnecting a separate channel wakes every worker ahead of queued jobs.
        drop(self.stop_tx.take());
    }
}

impl Drop for ClipWorkerThreads {
    fn drop(&mut self) {
        self.stop();
        for worker in self.handles.drain(..) {
            let _ = worker.join();
        }
    }
}

fn is_stopping(stop_rx: &Receiver<()>) -> bool {
    matches!(stop_rx.try_recv(), Err(TryRecvError::Disconnected))
}

fn load_inference(
    use_cuda: bool,
    device_flag: &AtomicBool,
    stop_rx: &Receiver<()>,
) -> Result<Option<Inference>> {
    if is_stopping(stop_rx) {
        return Ok(None);
    }
    // The Hub's blocking download has no cancellation API. Keep network I/O on a
    // separate thread so closing the app never has to wait for it to acquire model files.
    let (tx, rx) = bounded(1);
    thread::Builder::new()
        .name("siglip-model-files".into())
        .spawn(move || {
            let _ = tx.send(ModelFiles::fetch());
        })?;
    let files = crossbeam_channel::select_biased! {
        recv(stop_rx) -> _ => return Ok(None),
        recv(rx) -> files => files.context("model files worker stopped")??,
    };
    Inference::load(files, use_cuda, device_flag, stop_rx)
}

fn run_worker(
    ctx: ClipWorkerContext,
    job_rx: Receiver<ImageJob>,
    text_rx: Receiver<TextJob>,
    use_cuda: bool,
) {
    let ClipWorkerContext {
        cache_base,
        inference,
        output_color_space,
        result_tx,
        device_flag,
        proxy,
        stop_rx,
    } = ctx;
    // CPU workers share the weights instead of each allocating another 1.5 GB model.
    let Some(inference) =
        inference.get_or_init(|| match load_inference(use_cuda, &device_flag, &stop_rx) {
            Ok(inference) => inference,
            Err(error) => {
                if !is_stopping(&stop_rx) {
                    eprintln!("Failed to load SigLIP 2: {error:#}");
                }
                None
            }
        })
    else {
        return;
    };
    let model = &inference.model;
    let device = &inference.device;
    let image_ctx = ImageBatchContext {
        cache_base: cache_base.as_deref(),
        model,
        device,
        clip_image_size: inference.image_size,
        output_color_space,
        result_tx: &result_tx,
        proxy: &proxy,
        stop_rx: &stop_rx,
    };
    let text_ctx = TextJobContext {
        model,
        tokenizer: &inference.tokenizer,
        device,
        result_tx: &result_tx,
        proxy: &proxy,
    };
    loop {
        // Queries can run between image batches even while thousands of images await indexing.
        crossbeam_channel::select_biased! {
            recv(stop_rx) -> _ => break,
            recv(text_rx) -> job => {
                let Ok(job) = job else { break; };
                handle_text_job(job.request_id, &job.query, &text_ctx);
            }
            recv(job_rx) -> job => {
                let Ok(job) = job else { break; };
                let mut batch = vec![job];
                let max_batch = if device.is_cuda() { MAX_IMAGE_BATCH } else { 1 };
                while batch.len() < max_batch {
                    match job_rx.try_recv() {
                        Ok(job) => batch.push(job),
                        Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                    }
                }
                process_image_batch(batch, &image_ctx);
            }
        }
    }
    // CUDA work and object destruction complete before the process can tear down the driver.
    if device.is_cuda() {
        let _ = device.synchronize();
    }
}

fn handle_text_job(request_id: u64, query: &str, ctx: &TextJobContext<'_>) {
    match process_text_job(query, ctx.model, ctx.tokenizer, ctx.device) {
        Ok(embedding) => {
            send_result(
                ctx.result_tx,
                ctx.proxy,
                ClipEvent::TextReady {
                    request_id,
                    embedding,
                },
            );
        }
        Err(err) => {
            send_result(
                ctx.result_tx,
                ctx.proxy,
                ClipEvent::TextError {
                    request_id,
                    error: format!("{}", err),
                },
            );
        }
    }
}

fn process_image_batch(batch: Vec<ImageJob>, ctx: &ImageBatchContext<'_>) {
    let mut compute_jobs = Vec::new();
    let mut compute_tensors = Vec::new();

    for job in batch {
        if is_stopping(ctx.stop_rx) {
            return;
        }
        if !ctx.source_is_current(&job) {
            continue;
        }
        let request = job.request;
        let decoded = match job.thumbnail.decode() {
            Ok(image) => image,
            Err(error) => {
                if ctx.source_is_current(&job) {
                    let error = error.to_string();
                    let event = if job.thumbnail.cached {
                        ClipEvent::ThumbnailError { request, error }
                    } else {
                        ClipEvent::ImageError { request, error }
                    };
                    send_result(ctx.result_tx, ctx.proxy, event);
                }
                continue;
            }
        };
        let tensor = tensor_from_image(&decoded, ctx.clip_image_size, ctx.output_color_space);
        if !ctx.source_is_current(&job) {
            continue;
        }
        match tensor {
            Ok(tensor) => {
                compute_jobs.push(job);
                compute_tensors.push(tensor);
            }
            Err(err) => {
                send_result(
                    ctx.result_tx,
                    ctx.proxy,
                    ClipEvent::ImageError {
                        request,
                        error: format!("{}", err),
                    },
                );
            }
        }
    }

    if compute_tensors.is_empty() || is_stopping(ctx.stop_rx) {
        return;
    }

    let embeddings = Tensor::stack(&compute_tensors, 0)
        .and_then(|stacked| stacked.to_device(ctx.device))
        .map_err(anyhow::Error::from)
        .and_then(|stacked| image_embeddings(ctx.model, stacked));
    if is_stopping(ctx.stop_rx) {
        return;
    }
    match embeddings {
        Ok(embeddings) => {
            for (job, embedding) in compute_jobs.into_iter().zip(embeddings) {
                if !ctx.source_is_current(&job) {
                    continue;
                }
                if let Err(err) =
                    cache_embedding(ctx.cache_base, &job.image_path, job.version, &embedding)
                {
                    eprintln!(
                        "Failed to cache SigLIP 2 embedding for {}: {err}",
                        job.image_path.display()
                    );
                }
                // A disk-cache failure must not discard a successfully computed embedding.
                if ctx.source_is_current(&job) {
                    send_result(
                        ctx.result_tx,
                        ctx.proxy,
                        ClipEvent::ImageReady {
                            request: job.request,
                            embedding,
                        },
                    );
                }
            }
        }
        Err(error) => {
            let error = format!("{error:#}");
            for job in &compute_jobs {
                if ctx.source_is_current(job) {
                    send_result(
                        ctx.result_tx,
                        ctx.proxy,
                        ClipEvent::ImageError {
                            request: job.request,
                            error: error.clone(),
                        },
                    );
                }
            }
        }
    }
}

fn send_result(result_tx: &Sender<ClipEvent>, proxy: &AppProxy, event: ClipEvent) {
    if result_tx.send(event).is_ok() {
        proxy.wakeup();
    }
}

struct ImageBatchContext<'a> {
    cache_base: Option<&'a Path>,
    model: &'a EmbeddingModel,
    device: &'a Device,
    clip_image_size: usize,
    output_color_space: OutputColorSpace,
    result_tx: &'a Sender<ClipEvent>,
    proxy: &'a AppProxy,
    stop_rx: &'a Receiver<()>,
}

impl ImageBatchContext<'_> {
    fn source_is_current(&self, job: &ImageJob) -> bool {
        if job.version.matches(&job.image_path) {
            return true;
        }
        send_result(
            self.result_tx,
            self.proxy,
            ClipEvent::SourceChanged {
                request: job.request,
            },
        );
        false
    }
}

struct TextJobContext<'a> {
    model: &'a EmbeddingModel,
    tokenizer: &'a Tokenizer,
    device: &'a Device,
    result_tx: &'a Sender<ClipEvent>,
    proxy: &'a AppProxy,
}

fn process_text_job(
    query: &str,
    model: &EmbeddingModel,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<Vec<f32>> {
    let ids = tokenizer
        .encode(query, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        return Err(anyhow!("tokenizer produced empty sequence for query"));
    }
    let input_ids = Tensor::new(vec![ids], device)?;
    let features = model.get_text_features(&input_ids)?;
    let features = clip::div_l2_norm(&features)?;
    let features = features.squeeze(0)?;
    let embedding = features.to_vec1::<f32>()?;
    Ok(embedding)
}

#[cfg(test)]
fn tensor_from_thumbnail(
    thumb: &EncodedThumbnail,
    image_size: usize,
    output_color_space: OutputColorSpace,
) -> Result<Tensor> {
    let decoded = thumb.decode()?;
    tensor_from_image(&decoded, image_size, output_color_space)
}

fn tensor_from_image(
    decoded: &image::DynamicImage,
    image_size: usize,
    output_color_space: OutputColorSpace,
) -> Result<Tensor> {
    let srgb = crate::image_io::thumbnail_for_clip(decoded, output_color_space)?;
    // Fixed-resolution SigLIP stretches the whole image, rather than center-cropping it.
    let resized = image::imageops::resize(
        &srgb,
        image_size as u32,
        image_size as u32,
        FilterType::Triangle,
    );
    let (height, width) = (resized.height() as usize, resized.width() as usize);
    let data = resized.into_raw();
    let tensor = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2.0 / 255.0, -1.0)?;
    Ok(tensor)
}

fn image_embeddings(model: &EmbeddingModel, tensor: Tensor) -> Result<Vec<Vec<f32>>> {
    let features = model.get_image_features(&tensor)?;
    let features = clip::div_l2_norm(&features)?;
    Ok(features.to_vec2::<f32>()?)
}

fn embedding_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + 4 * embedding.len());
    bytes.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
    bytes.extend(embedding.iter().flat_map(|value| value.to_le_bytes()));
    bytes
}

fn cache_embedding(
    cache_base: Option<&Path>,
    image_path: &Path,
    version: SourceVersion,
    embedding: &[f32],
) -> Result<()> {
    if let Some(base) = cache_base {
        version.write_cache(
            image_path,
            &cache_file_path(base, image_path, EMBEDDING_CACHE_EXTENSION),
            &embedding_bytes(embedding),
        )?;
    }
    Ok(())
}

pub fn load_cached_embedding(
    cache_base: &Path,
    image_path: &Path,
    version: SourceVersion,
) -> Result<Option<Vec<f32>>> {
    let embed_path = cache_file_path(cache_base, image_path, EMBEDDING_CACHE_EXTENSION);
    let Some(mut file) = version.open_cache(&embed_path) else {
        return Ok(None);
    };
    let mut header = [0u8; 4];
    file.read_exact(&mut header)?;
    let len = u32::from_le_bytes(header) as usize;
    if len != EMBEDDING_DIM {
        return Ok(None);
    }
    let mut buffer = vec![0u8; len * 4];
    file.read_exact(&mut buffer)?;
    let mut embedding = Vec::with_capacity(len);
    for chunk in buffer.chunks_exact(4) {
        embedding.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    if embedding.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    Ok(Some(embedding))
}

/// Computes the path in the cache directory for the given image and extension.
pub fn cache_file_path(cache_base: &Path, image_path: &Path, extension: &str) -> PathBuf {
    let path_str = image_path.to_string_lossy();
    let mut hasher = Sha1::new();
    hasher.update(path_str.as_bytes());
    let hex = hasher.digest().to_string();
    let shard = &hex[..3];
    let name = &hex[3..];
    cache_base.join(shard).join(format!("{name}.{extension}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn siglip_preprocessing_preserves_both_edges_and_normalizes_rgb() {
        let image = image::RgbImage::from_raw(2, 1, vec![255, 0, 0, 0, 0, 255]).unwrap();
        let thumbnail = EncodedThumbnail::encode(&image::DynamicImage::ImageRgb8(image)).unwrap();
        let tensor = tensor_from_thumbnail(&thumbnail, 2, OutputColorSpace::Srgb).unwrap();
        assert_eq!(
            tensor.to_vec3::<f32>().unwrap(),
            vec![
                vec![vec![1.0, -1.0], vec![1.0, -1.0]],
                vec![vec![-1.0, -1.0], vec![-1.0, -1.0]],
                vec![vec![-1.0, 1.0], vec![-1.0, 1.0]],
            ]
        );
    }

    /// Explicit integration check: downloads the pinned model and uses a photograph of cats.
    #[test]
    #[ignore = "downloads SigLIP 2 weights; set SRIV_SEARCH_TEST_IMAGE to a cat photograph"]
    fn siglip2_inference_smoke() {
        let started = std::time::Instant::now();
        let (_stop_tx, stop_rx) = bounded(0);
        let inference = Inference::load(
            ModelFiles::fetch().unwrap(),
            cuda_is_available(),
            &AtomicBool::new(false),
            &stop_rx,
        )
        .unwrap()
        .unwrap();
        eprintln!(
            "model loaded on {:?} in {:?}",
            inference.device,
            started.elapsed()
        );
        let path = std::env::var("SRIV_SEARCH_TEST_IMAGE").expect("set SRIV_SEARCH_TEST_IMAGE");
        let image = image::load_from_memory(&std::fs::read(path).unwrap())
            .unwrap()
            .thumbnail(512, 512);
        let thumbnail = EncodedThumbnail::encode(&image).unwrap();
        let pixels = tensor_from_thumbnail(&thumbnail, 512, OutputColorSpace::Srgb)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .to_device(&inference.device)
            .unwrap();
        if let Ok(directory) = std::env::var("SRIV_SEARCH_REFERENCE_DIR") {
            fs::create_dir_all(&directory).unwrap();
            fs::write(
                Path::new(&directory).join("thumbnail.png"),
                &thumbnail.bytes,
            )
            .unwrap();
            pixels
                .to_device(&Device::Cpu)
                .unwrap()
                .write_npy(Path::new(&directory).join("pixels.npy"))
                .unwrap();
        }
        let started = std::time::Instant::now();
        let images = image_embeddings(&inference.model, pixels).unwrap();
        eprintln!("512 px image encoded in {:?}", started.elapsed());
        let pixels = tensor_from_thumbnail(&thumbnail, 512, OutputColorSpace::Srgb)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .to_device(&inference.device)
            .unwrap();
        for batch_size in [1, MAX_IMAGE_BATCH] {
            let batch = pixels.repeat((batch_size, 1, 1, 1)).unwrap();
            let started = std::time::Instant::now();
            let result = image_embeddings(&inference.model, batch).unwrap();
            eprintln!(
                "warm batch of {batch_size}: {:?} ({:.1} images/sec)",
                started.elapsed(),
                batch_size as f64 / started.elapsed().as_secs_f64()
            );
            for embedding in result {
                assert!(
                    embedding
                        .iter()
                        .zip(&images[0])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        > 0.999
                );
            }
        }
        assert_eq!(images[0].len(), EMBEDDING_DIM);
        assert!((images[0].iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-4);
        let mut scores = Vec::new();
        for (index, query) in [
            "two cats resting on a couch",
            "a satellite photograph of a desert",
            "an airplane flying in the sky",
        ]
        .into_iter()
        .enumerate()
        {
            let text = process_text_job(
                query,
                &inference.model,
                &inference.tokenizer,
                &inference.device,
            )
            .unwrap();
            assert_eq!(text.len(), EMBEDDING_DIM);
            assert!(text.iter().all(|v| v.is_finite()));
            if let Ok(directory) = std::env::var("SRIV_SEARCH_REFERENCE_DIR") {
                Tensor::new(text.as_slice(), &Device::Cpu)
                    .unwrap()
                    .write_npy(Path::new(&directory).join(format!("text-{index}.npy")))
                    .unwrap();
                let ids = inference.tokenizer.encode(query, true).unwrap();
                fs::write(
                    Path::new(&directory).join(format!("tokens-{index}.json")),
                    serde_json::to_vec(ids.get_ids()).unwrap(),
                )
                .unwrap();
            }
            scores.push(images[0].iter().zip(text).map(|(a, b)| a * b).sum::<f32>());
        }
        let long = inference
            .tokenizer
            .encode("cat ".repeat(1000), true)
            .unwrap();
        assert_eq!(long.len(), 64);
        assert_eq!(inference.tokenizer.encode("cat", true).unwrap().len(), 64);
        eprintln!("cat / desert / airplane cosine similarities: {scores:?}");
        assert!(scores[0] > scores[1] && scores[0] > scores[2]);
        if let Ok(directory) = std::env::var("SRIV_SEARCH_REFERENCE_DIR") {
            Tensor::new(images[0].as_slice(), &Device::Cpu)
                .unwrap()
                .write_npy(Path::new(&directory).join("image.npy"))
                .unwrap();
        }
    }

    #[test]
    fn stopping_worker_skips_model_fetch() {
        let (stop_tx, stop_rx) = bounded(0);
        drop(stop_tx);
        assert!(load_inference(false, &AtomicBool::new(false), &stop_rx)
            .unwrap()
            .is_none());
    }

    #[test]
    fn stopping_workers_signals_cancellation_before_waiting_for_them() {
        let (stop_tx, stop_rx) = bounded(0);
        let (resume_tx, resume_rx) = bounded(0);
        let (finished_tx, finished_rx) = bounded(1);
        let timeout = Duration::from_secs(2);
        let worker = thread::spawn(move || {
            assert!(stop_rx.recv().is_err());
            resume_rx.recv_timeout(timeout).unwrap();
            finished_tx.send(()).unwrap();
        });
        let mut threads = ClipWorkerThreads {
            stop_tx: Some(stop_tx),
            handles: vec![worker],
        };
        threads.stop();
        resume_tx.send_timeout((), timeout).unwrap();
        drop(threads);
        assert_eq!(finished_rx.try_recv(), Ok(()));
    }

    #[test]
    fn model_weight_loading_stops_between_tensors() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sriv-test-weights-{}-{unique}.safetensors",
            std::process::id()
        ));
        Tensor::new(&[1.0f32, 2.0], &Device::Cpu)
            .unwrap()
            .save_safetensors("weight", &path)
            .unwrap();
        let (stop_tx, stop_rx) = bounded(0);
        let tensors = unsafe { MmapedSafetensors::new(&path) }.unwrap();
        let vb = VarBuilder::from_backend(
            Box::new(ModelWeights {
                tensors,
                stop_rx: &stop_rx,
            }),
            DType::F32,
            Device::Cpu,
        );
        assert_eq!(
            vb.get(2, "weight").unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
        drop(stop_tx);
        assert!(vb.get(2, "weight").is_err());
        assert!(vb.get_unchecked("weight").is_err());
        drop(vb);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn private_embedding_skips_disk_cache() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("sriv-private-clip-{}-{unique}", std::process::id()));
        let image_path = dir.join("image.png");
        let embedding = [0.1; EMBEDDING_DIM];
        fs::create_dir(&dir).unwrap();
        fs::write(&image_path, b"source image").unwrap();
        let version = SourceVersion::read(&image_path).unwrap();

        let cache_base = dir.join("cache");
        cache_embedding(None, &image_path, version, &embedding).unwrap();
        assert!(!cache_base.exists());
        cache_embedding(Some(&cache_base), &image_path, version, &embedding).unwrap();
        assert_eq!(
            load_cached_embedding(&cache_base, &image_path, version).unwrap(),
            Some(embedding.to_vec())
        );
        // The original CLIP cache is never used, even if it is newer than the source.
        let legacy = dir.join("legacy.png");
        fs::write(&legacy, b"another image").unwrap();
        let legacy_version = SourceVersion::read(&legacy).unwrap();
        legacy_version
            .write_cache(
                &legacy,
                &cache_file_path(&cache_base, &legacy, "clip"),
                &embedding_bytes(&embedding),
            )
            .unwrap();
        assert!(load_cached_embedding(&cache_base, &legacy, legacy_version)
            .unwrap()
            .is_none());
        cache_embedding(Some(&cache_base), &image_path, version, &[0.0; 512]).unwrap();
        assert!(load_cached_embedding(&cache_base, &image_path, version)
            .unwrap()
            .is_none());
        cache_embedding(
            Some(&cache_base),
            &image_path,
            version,
            &[f32::NAN; EMBEDDING_DIM],
        )
        .unwrap();
        assert!(load_cached_embedding(&cache_base, &image_path, version)
            .unwrap()
            .is_none());
        fs::remove_dir_all(dir).unwrap();
    }
}
