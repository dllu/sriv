use std::convert::TryInto;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{anyhow, Context, Result};
use candle::{utils::cuda_is_available, DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{clip, siglip};
use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use image::imageops::FilterType;
use sha1::Sha1;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::app::AppProxy;
use crate::color::OutputColorSpace;
use crate::state::ImageRequest;
use crate::thumbnails::{decode_thumbnail, DecodeOutcome, EncodedThumbnail};

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

impl Inference {
    fn load(use_cuda: bool, device_flag: &AtomicBool) -> Result<Self> {
        let device = if use_cuda {
            match Device::new_cuda(0) {
                Ok(device) => device,
                Err(error) => {
                    eprintln!("Failed to initialize CUDA: {error}. Falling back to CPU.");
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };
        device_flag.store(device.is_cuda(), Ordering::Relaxed);
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            MODEL_ID.to_string(),
            hf_hub::RepoType::Model,
            MODEL_REVISION.to_string(),
        ));
        let config: siglip::Config = serde_json::from_slice(&fs::read(repo.get("config.json")?)?)?;
        let tokenizer = load_tokenizer(repo.get("tokenizer.json")?)?;
        let weights = repo.get("model.safetensors")?;
        let dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
        let model = EmbeddingModel::new(&config, vb)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            image_size: config.vision_config.image_size,
        })
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
    thumbnail: EncodedThumbnail,
}

struct TextJob {
    request_id: u64,
    query: String,
}

/// Manages SigLIP 2 inference in a background thread and caching of embeddings.
#[derive(Debug)]
pub struct ClipEngine {
    job_tx: Sender<ImageJob>,
    text_tx: Sender<TextJob>,
    result_rx: Receiver<ClipEvent>,
    using_cuda: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct ClipRequestSender {
    job_tx: Sender<ImageJob>,
}

impl ClipRequestSender {
    pub fn queue_image(
        &self,
        request: ImageRequest,
        image_path: PathBuf,
        thumbnail: EncodedThumbnail,
    ) -> Result<()> {
        self.job_tx
            .send(ImageJob {
                request,
                image_path,
                thumbnail,
            })
            .map_err(|err| anyhow!("failed to queue clip image job: {err}"))
    }
}

#[derive(Clone)]
struct ClipWorkerContext {
    cache_base: Option<PathBuf>,
    inference: Arc<OnceLock<std::result::Result<Inference, String>>>,
    output_color_space: OutputColorSpace,
    result_tx: Sender<ClipEvent>,
    device_flag: Arc<AtomicBool>,
    proxy: AppProxy,
}

impl ClipEngine {
    pub(crate) fn new(
        cache_base: Option<PathBuf>,
        proxy: AppProxy,
        output_color_space: OutputColorSpace,
    ) -> Result<Self> {
        let (job_tx, job_rx) = unbounded::<ImageJob>();
        let (text_tx, text_rx) = unbounded::<TextJob>();
        let (result_tx, result_rx) = unbounded::<ClipEvent>();
        let inference = Arc::new(OnceLock::new());
        let want_cuda = cuda_is_available();
        let using_cuda = Arc::new(AtomicBool::new(false));
        let worker_count = if want_cuda {
            1
        } else {
            num_cpus::get().clamp(1, 8)
        };
        for worker_idx in 0..worker_count {
            let worker_ctx = ClipWorkerContext {
                cache_base: cache_base.clone(),
                inference: Arc::clone(&inference),
                output_color_space,
                result_tx: result_tx.clone(),
                device_flag: using_cuda.clone(),
                proxy: proxy.clone(),
            };
            let worker_rx = job_rx.clone();
            let worker_text_rx = text_rx.clone();
            thread::Builder::new()
                .name(format!("clip-worker-{worker_idx}"))
                .spawn(move || {
                    let proxy = worker_ctx.proxy.clone();
                    if let Err(err) = run_worker(worker_ctx, worker_rx, worker_text_rx, want_cuda) {
                        eprintln!("clip worker terminated: {err:#}");
                    }
                    // The last worker dropping its sender disconnects result_rx. Wake the UI
                    // so pending work becomes a visible failure even without further input.
                    proxy.wakeup();
                })
                .context("failed to spawn clip worker thread")?;
        }
        drop(job_rx);
        drop(text_rx);
        drop(result_tx);
        Ok(Self {
            job_tx,
            text_tx,
            result_rx,
            using_cuda,
        })
    }

    pub fn request_sender(&self) -> ClipRequestSender {
        ClipRequestSender {
            job_tx: self.job_tx.clone(),
        }
    }

    /// Requests a text embedding for the given query string.
    pub fn request_text(&self, request_id: u64, query: String) -> Result<()> {
        self.text_tx
            .send(TextJob { request_id, query })
            .map_err(|err| anyhow!("failed to queue clip text job: {err}"))
    }

    /// Attempts to retrieve the next event from the worker without blocking.
    pub fn try_recv(&self) -> Result<ClipEvent, TryRecvError> {
        self.result_rx.try_recv()
    }

    pub fn device_kind(&self) -> &'static str {
        if self.using_cuda.load(Ordering::Relaxed) {
            "GPU"
        } else {
            "CPU"
        }
    }
}

fn run_worker(
    ctx: ClipWorkerContext,
    job_rx: Receiver<ImageJob>,
    text_rx: Receiver<TextJob>,
    use_cuda: bool,
) -> Result<()> {
    let ClipWorkerContext {
        cache_base,
        inference,
        output_color_space,
        result_tx,
        device_flag,
        proxy,
    } = ctx;
    // CPU workers share the weights instead of each allocating another 1.5 GB model.
    let inference = inference
        .get_or_init(|| {
            Inference::load(use_cuda, &device_flag).map_err(|error| format!("{error:#}"))
        })
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))?;
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
    Ok(())
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
    let mut compute_indices = Vec::new();
    let mut compute_paths = Vec::new();
    let mut compute_tensors = Vec::new();

    for ImageJob {
        request,
        image_path,
        thumbnail: thumb,
    } in batch
    {
        let decoded = match decode_thumbnail(&thumb) {
            DecodeOutcome::Ready(image) => image,
            DecodeOutcome::Regenerate(error) => {
                send_result(
                    ctx.result_tx,
                    ctx.proxy,
                    ClipEvent::ThumbnailError { request, error },
                );
                continue;
            }
        };
        match tensor_from_image(&decoded, ctx.clip_image_size, ctx.output_color_space) {
            Ok(tensor) => {
                compute_indices.push(request);
                compute_paths.push(image_path);
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

    if compute_tensors.is_empty() {
        return;
    }

    let tensor_refs: Vec<&Tensor> = compute_tensors.iter().collect();
    let stacked = match Tensor::stack(&tensor_refs, 0) {
        Ok(t) => t,
        Err(err) => {
            report_batch_error(&compute_indices, ctx.result_tx, ctx.proxy, err);
            return;
        }
    };

    let stacked = match stacked.to_device(ctx.device) {
        Ok(t) => t,
        Err(err) => {
            report_batch_error(&compute_indices, ctx.result_tx, ctx.proxy, err);
            return;
        }
    };

    match image_embeddings(ctx.model, stacked) {
        Ok(embeddings) => {
            for ((request, path), embedding) in compute_indices
                .into_iter()
                .zip(compute_paths)
                .zip(embeddings)
            {
                if let Err(err) = cache_embedding(ctx.cache_base, &path, &embedding) {
                    eprintln!(
                        "Failed to cache SigLIP 2 embedding for {}: {err}",
                        path.display()
                    );
                }
                // A disk-cache failure must not discard a successfully computed embedding.
                send_result(
                    ctx.result_tx,
                    ctx.proxy,
                    ClipEvent::ImageReady { request, embedding },
                );
            }
        }
        Err(err) => {
            report_batch_error(&compute_indices, ctx.result_tx, ctx.proxy, err);
        }
    }
}

fn send_result(result_tx: &Sender<ClipEvent>, proxy: &AppProxy, event: ClipEvent) {
    if result_tx.send(event).is_ok() {
        proxy.wakeup();
    }
}

fn report_batch_error(
    indices: &[ImageRequest],
    result_tx: &Sender<ClipEvent>,
    proxy: &AppProxy,
    err: impl fmt::Display,
) {
    let msg = format!("{}", err);
    for &request in indices {
        send_result(
            result_tx,
            proxy,
            ClipEvent::ImageError {
                request,
                error: msg.clone(),
            },
        );
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

fn write_embedding(path: &Path, embedding: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("clip");
    let tmp_path = path.with_extension(format!("{ext}.tmp"));
    {
        let mut file = File::create(&tmp_path)?;
        let len = embedding.len() as u32;
        file.write_all(&len.to_le_bytes())?;
        for value in embedding {
            file.write_all(&value.to_le_bytes())?;
        }
        file.sync_all()?;
    }
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn cache_embedding(cache_base: Option<&Path>, image_path: &Path, embedding: &[f32]) -> Result<()> {
    if let Some(base) = cache_base {
        write_embedding(
            &cache_file_path(base, image_path, EMBEDDING_CACHE_EXTENSION),
            embedding,
        )?;
    }
    Ok(())
}

pub fn load_cached_embedding(cache_base: &Path, image_path: &Path) -> Result<Option<Vec<f32>>> {
    let embed_path = cache_file_path(cache_base, image_path, EMBEDDING_CACHE_EXTENSION);
    let embed_meta = match fs::metadata(&embed_path) {
        Ok(meta) => meta,
        Err(_) => return Ok(None),
    };
    let image_meta = fs::metadata(image_path).ok();
    let image_mtime = image_meta.and_then(|m| m.modified().ok());
    let embed_mtime = embed_meta.modified().ok();
    if let (Some(img_time), Some(emb_time)) = (image_mtime, embed_mtime) {
        if emb_time < img_time {
            return Ok(None);
        }
    }
    let mut file = File::open(&embed_path)?;
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
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let inference = Inference::load(cuda_is_available(), &AtomicBool::new(false)).unwrap();
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
    fn private_embedding_skips_disk_cache() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("sriv-private-clip-{}-{unique}", std::process::id()));
        let image_path = dir.join("image.png");
        let embedding = [0.1; EMBEDDING_DIM];
        cache_embedding(None, &image_path, &embedding).unwrap();
        assert!(!dir.exists());

        let cache_base = dir.join("cache");
        cache_embedding(Some(&cache_base), &image_path, &embedding).unwrap();
        assert_eq!(
            load_cached_embedding(&cache_base, &image_path).unwrap(),
            Some(embedding.to_vec())
        );
        // The original CLIP cache is never used, even if it is newer than the source.
        let legacy = dir.join("legacy.png");
        write_embedding(&cache_file_path(&cache_base, &legacy, "clip"), &embedding).unwrap();
        assert!(load_cached_embedding(&cache_base, &legacy)
            .unwrap()
            .is_none());
        let current = cache_file_path(&cache_base, &image_path, EMBEDDING_CACHE_EXTENSION);
        write_embedding(&current, &[0.0; 512]).unwrap();
        assert!(load_cached_embedding(&cache_base, &image_path)
            .unwrap()
            .is_none());
        write_embedding(&current, &[f32::NAN; EMBEDDING_DIM]).unwrap();
        assert!(load_cached_embedding(&cache_base, &image_path)
            .unwrap()
            .is_none());
        fs::remove_dir_all(dir).unwrap();
    }
}
