//! Compressed thumbnails stay in RAM; only nearby images are decoded by workers.

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, TryRecvError};
use image::{DynamicImage, ImageDecoder, ImageEncoder};
use std::collections::{HashMap, VecDeque};
use std::io::Cursor;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::app::AppProxy;
use crate::state::ImageRequest;

pub const THUMBNAIL_SIZE: u32 = 512;
const DECODED_CACHE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct EncodedThumbnail {
    pub bytes: Arc<[u8]>,
    pub size: [u32; 2],
    pub cached: bool,
}

impl EncodedThumbnail {
    pub fn from_png(bytes: Vec<u8>) -> Result<Self> {
        // Reading the header does not inflate the pixels.
        let decoder = image::codecs::png::PngDecoder::new(Cursor::new(&bytes))?;
        let (width, height) = decoder.dimensions();
        drop(decoder);
        anyhow::ensure!(
            width > 0 && height > 0 && width <= THUMBNAIL_SIZE && height <= THUMBNAIL_SIZE,
            "invalid thumbnail dimensions {width}x{height}"
        );
        Ok(Self {
            bytes: bytes.into(),
            size: [width, height],
            cached: false,
        })
    }

    pub fn encode(image: &DynamicImage) -> Result<Self> {
        let mut bytes = Vec::new();
        // Fast DEFLATE keeps viewport decompression inexpensive.
        image::codecs::png::PngEncoder::new_with_quality(
            &mut bytes,
            image::codecs::png::CompressionType::Fast,
            image::codecs::png::FilterType::Adaptive,
        )
        .write_image(
            image.as_bytes(),
            image.width(),
            image.height(),
            image.color().into(),
        )?;
        Self::from_png(bytes)
    }

    pub fn decode(&self) -> Result<DynamicImage> {
        Ok(image::load_from_memory_with_format(
            &self.bytes,
            image::ImageFormat::Png,
        )?)
    }
}

struct ThumbnailEntry {
    request: ImageRequest,
    image: EncodedThumbnail,
}

pub struct ThumbnailStore {
    encoded: HashMap<usize, ThumbnailEntry>,
    decoded: HashMap<usize, DynamicImage>,
    lru: VecDeque<usize>,
    decoded_bytes: usize,
    budget: usize,
}

impl ThumbnailStore {
    pub fn new() -> Self {
        Self::with_budget(DECODED_CACHE_BYTES)
    }

    fn with_budget(budget: usize) -> Self {
        Self {
            encoded: HashMap::new(),
            decoded: HashMap::new(),
            lru: VecDeque::new(),
            decoded_bytes: 0,
            budget,
        }
    }

    pub fn insert(&mut self, request: ImageRequest, image: EncodedThumbnail) {
        self.remove(request.index);
        self.encoded
            .insert(request.index, ThumbnailEntry { request, image });
    }

    pub fn remove(&mut self, index: usize) {
        self.encoded.remove(&index);
        self.evict(index);
    }

    fn evict(&mut self, index: usize) {
        if let Some(image) = self.decoded.remove(&index) {
            self.decoded_bytes -= image.as_bytes().len();
        }
        self.lru.retain(|&entry| entry != index);
    }

    pub fn decoded(&mut self, index: usize) -> Option<&DynamicImage> {
        if self.decoded.contains_key(&index) {
            self.lru.retain(|&entry| entry != index);
            self.lru.push_back(index);
        }
        self.decoded.get(&index)
    }

    pub fn size(&self, index: usize) -> Option<[u32; 2]> {
        self.encoded.get(&index).map(|entry| entry.image.size)
    }

    pub fn complete(&mut self, request: ImageRequest, image: DynamicImage) -> bool {
        if !self
            .encoded
            .get(&request.index)
            .is_some_and(|entry| entry.request == request)
        {
            return false;
        }
        self.evict(request.index);
        self.decoded_bytes += image.as_bytes().len();
        self.decoded.insert(request.index, image);
        self.lru.push_back(request.index);
        // A single oversized entry can still be uploaded; subsequent entries evict it.
        while self.decoded_bytes > self.budget && self.decoded.len() > 1 {
            self.evict(*self.lru.front().unwrap());
        }
        true
    }

    pub fn decode_jobs(&self, indices: impl IntoIterator<Item = usize>) -> Vec<DecodeJob> {
        indices
            .into_iter()
            .filter_map(|index| {
                let entry = self.encoded.get(&index)?;
                (!self.decoded.contains_key(&index)).then(|| DecodeJob {
                    request: entry.request,
                    image: entry.image.clone(),
                })
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct DecodeJob {
    request: ImageRequest,
    image: EncodedThumbnail,
}

#[derive(Default)]
struct DecodeQueue {
    jobs: VecDeque<DecodeJob>,
    active: Vec<ImageRequest>,
    closed: bool,
}

impl DecodeQueue {
    fn replace(&mut self, jobs: Vec<DecodeJob>) {
        if self.closed {
            return;
        }
        self.jobs = jobs
            .into_iter()
            .filter(|job| !self.active.contains(&job.request))
            .collect();
    }

    fn pop(&mut self) -> Option<DecodeJob> {
        let job = self.jobs.pop_front()?;
        self.active.push(job.request);
        Some(job)
    }

    fn finish(&mut self, request: ImageRequest) {
        self.active.retain(|&active| active != request);
    }
}

pub enum DecodeOutcome {
    Ready(DynamicImage),
    Regenerate(String),
}

/// Decode failures in disk caches recover through the original-image worker.
pub(crate) fn decode_thumbnail(image: &EncodedThumbnail) -> DecodeOutcome {
    match image.decode() {
        Ok(image) => DecodeOutcome::Ready(image),
        Err(error) if image.cached => DecodeOutcome::Regenerate(error.to_string()),
        Err(error) => {
            eprintln!("Failed to decode in-memory thumbnail: {error}");
            DecodeOutcome::Ready(crate::image_io::fallback_thumbnail())
        }
    }
}

pub struct DecodeUpdate {
    pub request: ImageRequest,
    pub outcome: DecodeOutcome,
}

pub struct ThumbnailDecoder {
    queue: Arc<(Mutex<DecodeQueue>, Condvar)>,
    results: Receiver<DecodeUpdate>,
}

impl ThumbnailDecoder {
    pub fn new(proxy: AppProxy) -> Result<Self> {
        let queue = Arc::new((Mutex::new(DecodeQueue::default()), Condvar::new()));
        let (tx, results) = unbounded();
        // If a later spawn fails, Drop closes the queue and releases earlier workers.
        let decoder = Self {
            queue: Arc::clone(&queue),
            results,
        };
        for index in 0..num_cpus::get().clamp(1, 4) {
            let queue = Arc::clone(&queue);
            let tx = tx.clone();
            let proxy = proxy.clone();
            thread::Builder::new()
                .name(format!("thumbnail-decode-{index}"))
                .spawn(move || loop {
                    let job = {
                        let (lock, wake) = &*queue;
                        let mut state = lock.lock().unwrap();
                        loop {
                            if state.closed {
                                return;
                            }
                            if let Some(job) = state.pop() {
                                break job;
                            }
                            state = wake.wait(state).unwrap();
                        }
                    };
                    let outcome = decode_thumbnail(&job.image);
                    if tx
                        .send(DecodeUpdate {
                            request: job.request,
                            outcome,
                        })
                        .is_err()
                    {
                        return;
                    }
                    proxy.wakeup();
                })?;
        }
        Ok(decoder)
    }

    /// Replace work on every viewport change, so rapid scrolling cannot build a backlog.
    pub fn request(&self, jobs: Vec<DecodeJob>) {
        let (lock, wake) = &*self.queue;
        lock.lock().unwrap().replace(jobs);
        wake.notify_all();
    }

    pub fn try_recv(&self) -> std::result::Result<DecodeUpdate, TryRecvError> {
        let update = self.results.try_recv()?;
        self.queue.0.lock().unwrap().finish(update.request);
        Ok(update)
    }
}

impl Drop for ThumbnailDecoder {
    fn drop(&mut self) {
        let (lock, wake) = &*self.queue;
        let mut state = lock.lock().unwrap();
        state.closed = true;
        state.jobs.clear();
        wake.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thumbnail() -> EncodedThumbnail {
        EncodedThumbnail::encode(&DynamicImage::new_rgba8(2, 2)).unwrap()
    }

    fn job(index: usize, generation: u64) -> DecodeJob {
        DecodeJob {
            request: ImageRequest { index, generation },
            image: thumbnail(),
        }
    }

    #[test]
    fn scrolling_replaces_pending_work_and_keeps_inflight_generations_separate() {
        let mut queue = DecodeQueue::default();
        queue.replace(vec![job(0, 0), job(1, 0)]);
        let first = queue.pop().unwrap();
        queue.replace(vec![job(0, 0), job(2, 0)]);
        assert_eq!(queue.pop().unwrap().request.index, 2);
        assert!(queue.pop().is_none());
        queue.replace(vec![job(0, 1)]);
        assert_eq!(queue.pop().unwrap().request.generation, 1);
        queue.finish(first.request);
        assert!(queue.active.contains(&ImageRequest {
            index: 0,
            generation: 1
        }));
    }

    #[test]
    fn decoded_lru_obeys_byte_budget_and_rejects_results_after_invalidation() {
        let mut store = ThumbnailStore::with_budget(32);
        for index in 0..3 {
            let request = ImageRequest {
                index,
                generation: 0,
            };
            store.insert(request, thumbnail());
            store.complete(request, DynamicImage::new_rgba8(2, 2));
            if index == 1 {
                assert!(store.decoded(0).is_some());
            }
        }
        assert!(store.decoded(1).is_none());
        assert!(store.decoded(0).is_some());
        assert_eq!(store.decoded_bytes, 32);
        assert_eq!(store.encoded.len(), 3);
        let stale = ImageRequest {
            index: 0,
            generation: 0,
        };
        store.remove(0);
        assert!(!store.complete(stale, DynamicImage::new_rgba8(2, 2)));
        store.insert(
            ImageRequest {
                index: 0,
                generation: 1,
            },
            thumbnail(),
        );
        assert!(!store.complete(stale, DynamicImage::new_rgba8(2, 2)));
        assert_eq!(store.decode_jobs([0, 1, 2]).len(), 2);
    }

    #[test]
    fn png_roundtrip_preserves_alpha_and_jobs_share_the_compressed_allocation() {
        let image = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            17,
            9,
            image::Rgba([10, 20, 30, 40]),
        ));
        let png = EncodedThumbnail::encode(&image).unwrap();
        assert_eq!(png.size, [17, 9]);
        assert_eq!(png.decode().unwrap().to_rgba8(), image.to_rgba8());
        let mut store = ThumbnailStore::new();
        store.insert(
            ImageRequest {
                index: 0,
                generation: 0,
            },
            png.clone(),
        );
        let jobs = store.decode_jobs([0]);
        assert!(Arc::ptr_eq(&png.bytes, &jobs[0].image.bytes));
        assert_eq!(store.decoded_bytes, 0);
        let mut queue = DecodeQueue {
            closed: true,
            ..Default::default()
        };
        queue.replace(jobs);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn corrupt_cached_pixels_request_regeneration() {
        let mut bytes = thumbnail().bytes.to_vec();
        let idat = bytes.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
        bytes[idat + 4] ^= 0xff;
        // Header validation deliberately leaves decompression to a viewport/indexing worker.
        let mut thumbnail = EncodedThumbnail::from_png(bytes).unwrap();
        thumbnail.cached = true;
        assert!(matches!(
            decode_thumbnail(&thumbnail),
            DecodeOutcome::Regenerate(_)
        ));
    }

    #[test]
    #[ignore = "set SRIV_SEARCH_TEST_IMAGE to a photo for a thumbnail decode measurement"]
    fn thumbnail_decode_benchmark() {
        let path = std::env::var("SRIV_SEARCH_TEST_IMAGE").expect("set SRIV_SEARCH_TEST_IMAGE");
        let image = image::load_from_memory(&std::fs::read(path).unwrap())
            .unwrap()
            .thumbnail(512, 512);
        let image = DynamicImage::ImageRgba8(image.to_rgba8());
        let thumbnail = EncodedThumbnail::encode(&image).unwrap();
        let mut timings = Vec::new();
        for _ in 0..200 {
            let started = std::time::Instant::now();
            std::hint::black_box(thumbnail.decode().unwrap());
            timings.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        timings.sort_by(f64::total_cmp);
        eprintln!(
            "{:?}: PNG {} bytes, decoded {} bytes; decode median {:.3} ms, p95 {:.3} ms",
            thumbnail.size,
            thumbnail.bytes.len(),
            image.as_bytes().len(),
            timings[100],
            timings[190]
        );
    }
}
