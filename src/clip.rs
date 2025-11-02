use std::convert::TryInto;
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{anyhow, Context, Result};
use candle::{utils::cuda_is_available, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{self, ClipModel};
use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use nannou::image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use sha1::Sha1;
use tokenizers::Tokenizer;

const MAX_IMAGE_BATCH: usize = 16;

/// Events emitted by the background CLIP worker.
#[derive(Debug)]
pub enum ClipEvent {
    ImageReady {
        index: usize,
        embedding: Vec<f32>,
    },
    ImageError {
        index: usize,
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

enum ClipJob {
    Image {
        index: usize,
        image_path: PathBuf,
        thumbnail: RgbImage,
    },
    Text {
        request_id: u64,
        query: String,
    },
}

/// Manages CLIP inference in a background thread and caching of embeddings.
#[derive(Debug)]
pub struct ClipEngine {
    job_tx: Sender<ClipJob>,
    result_rx: Receiver<ClipEvent>,
    using_cuda: Arc<AtomicBool>,
}

#[derive(Clone)]
pub struct ClipRequestSender {
    job_tx: Sender<ClipJob>,
}

impl ClipRequestSender {
    pub fn queue_image(
        &self,
        index: usize,
        image_path: PathBuf,
        thumbnail: RgbImage,
    ) -> Result<()> {
        self.job_tx
            .send(ClipJob::Image {
                index,
                image_path,
                thumbnail,
            })
            .map_err(|err| anyhow!("failed to queue clip image job: {err}"))
    }
}

#[derive(Clone)]
struct ClipWorkerContext {
    cache_base: PathBuf,
    config: clip::ClipConfig,
    result_tx: Sender<ClipEvent>,
    device_flag: Arc<AtomicBool>,
}

impl ClipEngine {
    pub fn new(cache_base: PathBuf) -> Result<Self> {
        let (job_tx, job_rx) = unbounded::<ClipJob>();
        let (result_tx, result_rx) = unbounded::<ClipEvent>();
        let config = clip::ClipConfig::vit_base_patch32();
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
                config: config.clone(),
                result_tx: result_tx.clone(),
                device_flag: using_cuda.clone(),
            };
            let worker_rx = job_rx.clone();
            thread::Builder::new()
                .name(format!("clip-worker-{worker_idx}"))
                .spawn(move || {
                    if let Err(err) = run_worker(worker_ctx, worker_rx, want_cuda) {
                        eprintln!("clip worker terminated: {err:#}");
                    }
                })
                .context("failed to spawn clip worker thread")?;
        }
        drop(job_rx);
        drop(result_tx);
        Ok(Self {
            job_tx,
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
        self.job_tx
            .send(ClipJob::Text { request_id, query })
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

fn run_worker(ctx: ClipWorkerContext, job_rx: Receiver<ClipJob>, use_cuda: bool) -> Result<()> {
    let ClipWorkerContext {
        cache_base,
        config,
        result_tx,
        device_flag,
    } = ctx;
    let device = if use_cuda {
        match Device::new_cuda(0) {
            Ok(device) => {
                device_flag.store(true, Ordering::Relaxed);
                device
            }
            Err(err) => {
                eprintln!("Failed to initialize CUDA device: {err}. Falling back to CPU.");
                device_flag.store(false, Ordering::Relaxed);
                Device::Cpu
            }
        }
    } else {
        device_flag.store(false, Ordering::Relaxed);
        Device::Cpu
    };
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        "openai/clip-vit-base-patch32".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/15".to_string(),
    ));
    let model_file = repo.get("model.safetensors")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(anyhow::Error::msg)?;
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or_else(|| anyhow!("tokenizer does not provide <|endoftext|> token"))?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&model_file), DType::F32, &device)?
    };
    let model = ClipModel::new(vb, &config)?;
    let mut channel_closed = false;
    while let Ok(job) = job_rx.recv() {
        match job {
            ClipJob::Image {
                index,
                image_path,
                thumbnail,
            } => {
                let mut batch = vec![(index, image_path, thumbnail)];
                let mut deferred_job: Option<ClipJob> = None;
                while batch.len() < MAX_IMAGE_BATCH {
                    match job_rx.try_recv() {
                        Ok(ClipJob::Image {
                            index,
                            image_path,
                            thumbnail,
                        }) => batch.push((index, image_path, thumbnail)),
                        Ok(other) => {
                            deferred_job = Some(other);
                            break;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            channel_closed = true;
                            break;
                        }
                    }
                }
                let batch_ctx = ImageBatchContext {
                    cache_base: &cache_base,
                    model: &model,
                    device: &device,
                    clip_image_size: config.image_size,
                    result_tx: &result_tx,
                };
                process_image_batch(batch, &batch_ctx);
                if let Some(job) = deferred_job {
                    match job {
                        ClipJob::Text { request_id, query } => handle_text_job(
                            request_id, &query, &model, &tokenizer, pad_id, &device, &result_tx,
                        ),
                        ClipJob::Image {
                            index,
                            image_path,
                            thumbnail,
                        } => process_image_batch(vec![(index, image_path, thumbnail)], &batch_ctx),
                    }
                }
                if channel_closed {
                    break;
                }
            }
            ClipJob::Text { request_id, query } => {
                handle_text_job(
                    request_id, &query, &model, &tokenizer, pad_id, &device, &result_tx,
                );
            }
        }
    }
    Ok(())
}

fn handle_text_job(
    request_id: u64,
    query: &str,
    model: &ClipModel,
    tokenizer: &Tokenizer,
    pad_id: u32,
    device: &Device,
    result_tx: &Sender<ClipEvent>,
) {
    match process_text_job(query, model, tokenizer, pad_id, device) {
        Ok(embedding) => {
            let _ = result_tx.send(ClipEvent::TextReady {
                request_id,
                embedding,
            });
        }
        Err(err) => {
            let _ = result_tx.send(ClipEvent::TextError {
                request_id,
                error: format!("{}", err),
            });
        }
    }
}

fn process_image_batch(batch: Vec<(usize, PathBuf, RgbImage)>, ctx: &ImageBatchContext<'_>) {
    let mut compute_indices = Vec::new();
    let mut compute_paths = Vec::new();
    let mut compute_tensors = Vec::new();

    for (index, image_path, thumb) in batch {
        match tensor_from_thumbnail(thumb, ctx.clip_image_size) {
            Ok(tensor) => {
                compute_indices.push(index);
                compute_paths.push(image_path);
                compute_tensors.push(tensor);
            }
            Err(err) => {
                let _ = ctx.result_tx.send(ClipEvent::ImageError {
                    index,
                    error: format!("{}", err),
                });
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
            report_batch_error(&compute_indices, ctx.result_tx, err);
            return;
        }
    };

    let stacked = match stacked.to_device(ctx.device) {
        Ok(t) => t,
        Err(err) => {
            report_batch_error(&compute_indices, ctx.result_tx, err);
            return;
        }
    };

    match image_embeddings(ctx.model, stacked) {
        Ok(embeddings) => {
            for ((index, path), embedding) in compute_indices
                .into_iter()
                .zip(compute_paths.into_iter())
                .zip(embeddings.into_iter())
            {
                let embed_path = cache_file_path(ctx.cache_base, &path, "clip");
                match write_embedding(&embed_path, &embedding) {
                    Ok(()) => {
                        let _ = ctx
                            .result_tx
                            .send(ClipEvent::ImageReady { index, embedding });
                    }
                    Err(err) => {
                        let _ = ctx.result_tx.send(ClipEvent::ImageError {
                            index,
                            error: format!("{}", err),
                        });
                    }
                }
            }
        }
        Err(err) => {
            report_batch_error(&compute_indices, ctx.result_tx, err);
        }
    }
}

fn report_batch_error(indices: &[usize], result_tx: &Sender<ClipEvent>, err: impl fmt::Display) {
    let msg = format!("{}", err);
    for &idx in indices {
        let _ = result_tx.send(ClipEvent::ImageError {
            index: idx,
            error: msg.clone(),
        });
    }
}

struct ImageBatchContext<'a> {
    cache_base: &'a Path,
    model: &'a ClipModel,
    device: &'a Device,
    clip_image_size: usize,
    result_tx: &'a Sender<ClipEvent>,
}

fn process_text_job(
    query: &str,
    model: &ClipModel,
    tokenizer: &Tokenizer,
    pad_id: u32,
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
    let max_len = ids.len();
    let mut tokens = vec![ids];
    for token_vec in tokens.iter_mut() {
        if token_vec.len() < max_len {
            token_vec.extend(std::iter::repeat_n(pad_id, max_len - token_vec.len()));
        }
    }
    let input_ids = Tensor::new(tokens, device)?;
    let features = model.get_text_features(&input_ids)?;
    let features = clip::div_l2_norm(&features)?;
    let features = features.squeeze(0)?;
    let embedding = features.to_vec1::<f32>()?;
    Ok(embedding)
}

fn tensor_from_thumbnail(thumb: RgbImage, clip_image_size: usize) -> Result<Tensor> {
    let mut dyn_thumb = DynamicImage::ImageRgb8(thumb);
    if dyn_thumb.width() != clip_image_size as u32 || dyn_thumb.height() != clip_image_size as u32 {
        dyn_thumb = dyn_thumb.resize_to_fill(
            clip_image_size as u32,
            clip_image_size as u32,
            FilterType::Triangle,
        );
    }
    let resized = dyn_thumb.to_rgb8();
    let (height, width) = (resized.height() as usize, resized.width() as usize);
    let data = resized.into_raw();
    let tensor = Tensor::from_vec(data, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2.0 / 255.0, -1.0)?;
    Ok(tensor)
}

fn image_embeddings(model: &ClipModel, tensor: Tensor) -> Result<Vec<Vec<f32>>> {
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

pub fn load_cached_embedding(cache_base: &Path, image_path: &Path) -> Result<Option<Vec<f32>>> {
    let embed_path = cache_file_path(cache_base, image_path, "clip");
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
    let mut buffer = vec![0u8; len * 4];
    file.read_exact(&mut buffer)?;
    let mut embedding = Vec::with_capacity(len);
    for chunk in buffer.chunks_exact(4) {
        embedding.push(f32::from_le_bytes(chunk.try_into().unwrap()));
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
