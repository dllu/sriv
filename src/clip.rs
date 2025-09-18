use std::convert::TryInto;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use candle::{utils::cuda_is_available, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{self, ClipModel};
use crossbeam_channel::{unbounded, Receiver, Sender, TryRecvError};
use nannou::image::{self, imageops::FilterType, DynamicImage, GenericImageView, RgbImage};
use sha1::Sha1;
use tokenizers::Tokenizer;

use crate::adjust_orientation;

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
        thumb_size: u32,
    },
    Text {
        request_id: u64,
        query: String,
    },
}

/// Manages CLIP inference in a background thread and caching of embeddings.
#[derive(Debug)]
pub struct ClipEngine {
    cache_base: PathBuf,
    job_tx: Sender<ClipJob>,
    result_rx: Receiver<ClipEvent>,
}

impl ClipEngine {
    pub fn new(cache_base: PathBuf) -> Result<Self> {
        let (job_tx, job_rx) = unbounded::<ClipJob>();
        let (result_tx, result_rx) = unbounded::<ClipEvent>();
        let config = clip::ClipConfig::vit_base_patch32();
        let use_cuda = cuda_is_available();
        let worker_count = if use_cuda {
            1
        } else {
            num_cpus::get().clamp(1, 8)
        };
        for worker_idx in 0..worker_count {
            let worker_cache = cache_base.clone();
            let worker_rx = job_rx.clone();
            let worker_tx = result_tx.clone();
            let worker_config = config.clone();
            thread::Builder::new()
                .name(format!("clip-worker-{worker_idx}"))
                .spawn(move || {
                    if let Err(err) =
                        run_worker(worker_cache, worker_config, worker_rx, worker_tx, use_cuda)
                    {
                        eprintln!("clip worker terminated: {err:#}");
                    }
                })
                .context("failed to spawn clip worker thread")?;
        }
        Ok(Self {
            cache_base,
            job_tx,
            result_rx,
        })
    }

    /// Attempts to load a cached embedding if it is fresher than the original image.
    pub fn load_cached_embedding(&self, image_path: &Path) -> Result<Option<Vec<f32>>> {
        load_cached_embedding(&self.cache_base, image_path)
    }

    /// Ensures the embedding for `image_path` is computed, queuing work if required.
    pub fn ensure_embedding(&self, index: usize, image_path: &Path, thumb_size: u32) -> Result<()> {
        self.job_tx
            .send(ClipJob::Image {
                index,
                image_path: image_path.to_path_buf(),
                thumb_size,
            })
            .map_err(|err| anyhow!("failed to queue clip image job: {err}"))
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
}

fn run_worker(
    cache_base: PathBuf,
    config: clip::ClipConfig,
    job_rx: Receiver<ClipJob>,
    result_tx: Sender<ClipEvent>,
    use_cuda: bool,
) -> Result<()> {
    let device = if use_cuda {
        match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("Failed to initialize CUDA device: {err}. Falling back to CPU.");
                Device::Cpu
            }
        }
    } else {
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
    for job in job_rx.iter() {
        match job {
            ClipJob::Image {
                index,
                image_path,
                thumb_size,
            } => {
                match process_image_job(
                    &image_path,
                    thumb_size,
                    &cache_base,
                    &model,
                    &device,
                    config.image_size,
                ) {
                    Ok(embedding) => {
                        let _ = result_tx.send(ClipEvent::ImageReady { index, embedding });
                    }
                    Err(err) => {
                        let _ = result_tx.send(ClipEvent::ImageError {
                            index,
                            error: format!("{}", err),
                        });
                    }
                }
            }
            ClipJob::Text { request_id, query } => {
                match process_text_job(&query, &model, &tokenizer, pad_id, &device) {
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
        }
    }
    Ok(())
}

fn process_image_job(
    image_path: &Path,
    thumb_size: u32,
    cache_base: &Path,
    model: &ClipModel,
    device: &Device,
    clip_image_size: usize,
) -> Result<Vec<f32>> {
    if let Some(embedding) = load_cached_embedding(cache_base, image_path)? {
        return Ok(embedding);
    }
    let thumb = generate_thumbnail(image_path, thumb_size)?;
    let tensor = tensor_from_thumbnail(thumb, clip_image_size, device)?;
    let embedding = image_embedding(model, tensor)?;
    let embed_path = cache_file_path(cache_base, image_path, "clip");
    write_embedding(&embed_path, &embedding)?;
    Ok(embedding)
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
            token_vec.extend(std::iter::repeat(pad_id).take(max_len - token_vec.len()));
        }
    }
    let input_ids = Tensor::new(tokens, device)?;
    let features = model.get_text_features(&input_ids)?;
    let features = clip::div_l2_norm(&features)?;
    let features = features.squeeze(0)?;
    let embedding = features.to_vec1::<f32>()?;
    Ok(embedding)
}

fn tensor_from_thumbnail(
    thumb: RgbImage,
    clip_image_size: usize,
    device: &Device,
) -> Result<Tensor> {
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
    Ok(tensor.unsqueeze(0)?.to_device(device)?)
}

fn image_embedding(model: &ClipModel, tensor: Tensor) -> Result<Vec<f32>> {
    let features = model.get_image_features(&tensor)?;
    let features = clip::div_l2_norm(&features)?;
    let features = features.squeeze(0)?;
    Ok(features.to_vec1::<f32>()?)
}

fn generate_thumbnail(image_path: &Path, thumb_size: u32) -> Result<RgbImage> {
    let img_orig = image::open(image_path).with_context(|| {
        format!(
            "failed to open image for clip embedding: {}",
            image_path.display()
        )
    })?;
    let oriented = adjust_orientation(img_orig, image_path);
    let mut thumb = oriented.thumbnail(thumb_size, thumb_size);
    let (w0, h0) = thumb.dimensions();
    if w0 == 0 || h0 == 0 {
        bail!("thumbnail has zero dimensions for {}", image_path.display());
    }
    let w = w0.max(2);
    let h = h0.max(2);
    if w != w0 || h != h0 {
        thumb = thumb.resize_exact(w, h, FilterType::Nearest);
    }
    Ok(thumb.to_rgb8())
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

fn load_cached_embedding(cache_base: &Path, image_path: &Path) -> Result<Option<Vec<f32>>> {
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
