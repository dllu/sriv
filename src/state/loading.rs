use super::TiledTexture;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Every pipeline carries the same file generation from request through completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageRequest {
    pub index: usize,
    pub generation: u64,
}

#[derive(Debug)]
pub enum EmbeddingState {
    WaitingForThumbnail,
    Pending,
    Ready(Vec<f32>),
    Failed(String),
}

impl EmbeddingState {
    pub fn embedding(&self) -> Option<&[f32]> {
        match self {
            Self::Ready(embedding) => Some(embedding),
            _ => None,
        }
    }

    pub fn is_pending(&self) -> bool {
        matches!(self, Self::WaitingForThumbnail | Self::Pending)
    }

    /// A worker can finish CLIP before the thumbnail reaches the UI. Never regress it.
    pub fn thumbnail_ready(&mut self, mut state: Self, worker_error: Option<&str>) {
        if matches!(self, Self::WaitingForThumbnail) {
            if matches!(state, Self::Pending) {
                if let Some(error) = worker_error {
                    state = Self::Failed(error.to_string());
                }
            }
            *self = state;
        }
    }

    pub fn worker_stopped(&mut self, error: &str) {
        // Thumbnails can still deliver cached embeddings after the workers stop.
        if matches!(self, Self::Pending) {
            *self = Self::Failed(error.to_string());
        }
    }
}

struct ImageState {
    generation: u64,
    embedding: EmbeddingState,
}

/// File generations and embedding progress have one owner. UI progress reads are O(1).
pub struct ImageStates {
    entries: Vec<ImageState>,
    pending: usize,
    failed: usize,
}

impl ImageStates {
    pub fn new(count: usize) -> Self {
        Self {
            entries: (0..count)
                .map(|_| ImageState {
                    generation: 0,
                    embedding: EmbeddingState::WaitingForThumbnail,
                })
                .collect(),
            pending: count,
            failed: 0,
        }
    }

    pub fn request(&self, index: usize) -> ImageRequest {
        ImageRequest {
            index,
            generation: self.entries[index].generation,
        }
    }

    pub fn add_images(&mut self, count: usize) {
        self.entries.extend((0..count).map(|_| ImageState {
            generation: 0,
            embedding: EmbeddingState::WaitingForThumbnail,
        }));
        self.pending += count;
    }

    pub fn is_current(&self, request: ImageRequest) -> bool {
        self.entries
            .get(request.index)
            .is_some_and(|entry| entry.generation == request.generation)
    }

    pub fn invalidate(&mut self, index: usize) -> ImageRequest {
        self.entries[index].generation += 1;
        self.transition(index, |state| *state = EmbeddingState::WaitingForThumbnail);
        self.request(index)
    }

    pub fn thumbnail_ready(
        &mut self,
        request: ImageRequest,
        state: EmbeddingState,
        worker_error: Option<&str>,
    ) -> bool {
        if !self.is_current(request) {
            return false;
        }
        self.transition(request.index, |embedding| {
            embedding.thumbnail_ready(state, worker_error)
        });
        true
    }

    pub fn image_ready(&mut self, request: ImageRequest, embedding: Vec<f32>) -> bool {
        if !self.is_current(request) {
            return false;
        }
        self.transition(request.index, |state| {
            *state = EmbeddingState::Ready(embedding)
        });
        true
    }

    pub fn image_failed(&mut self, request: ImageRequest, error: String) -> bool {
        if !self.is_current(request) {
            return false;
        }
        self.transition(request.index, |state| {
            *state = EmbeddingState::Failed(error)
        });
        true
    }

    pub fn worker_stopped(&mut self, error: &str) {
        for index in 0..self.entries.len() {
            self.transition(index, |state| state.worker_stopped(error));
        }
    }

    pub fn embedding(&self, index: usize) -> Option<&[f32]> {
        self.entries[index].embedding.embedding()
    }

    pub fn ready_embeddings(&self) -> impl Iterator<Item = (usize, &[f32])> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                entry
                    .embedding
                    .embedding()
                    .map(|embedding| (index, embedding))
            })
    }

    pub fn error(&self, index: usize) -> Option<&str> {
        match &self.entries[index].embedding {
            EmbeddingState::Failed(error) => Some(error),
            _ => None,
        }
    }

    pub fn pending(&self) -> usize {
        self.pending
    }
    pub fn failed(&self) -> usize {
        self.failed
    }

    fn transition(&mut self, index: usize, apply: impl FnOnce(&mut EmbeddingState)) {
        let state = &mut self.entries[index].embedding;
        self.pending -= usize::from(state.is_pending());
        self.failed -= usize::from(matches!(state, EmbeddingState::Failed(_)));
        apply(state);
        self.pending += usize::from(state.is_pending());
        self.failed += usize::from(matches!(state, EmbeddingState::Failed(_)));
    }
}

#[derive(Debug)]
enum FullImageState {
    InFlight(ImageRequest),
    Failed { retry_at: Instant, error: String },
    Ready(TiledTexture),
}

/// Own the load lifecycle and the LRU together: an image cannot be both loaded and failed.
pub struct FullImageCache {
    entries: HashMap<usize, FullImageState>,
    usage: VecDeque<usize>,
    capacity: usize,
    retry_delay: Duration,
}

impl FullImageCache {
    pub fn new(capacity: usize, retry_delay: Duration) -> Self {
        assert!(capacity > 0);
        Self {
            entries: HashMap::new(),
            usage: VecDeque::new(),
            capacity,
            retry_delay,
        }
    }

    pub fn get(&self, index: &usize) -> Option<&TiledTexture> {
        match self.entries.get(index) {
            Some(FullImageState::Ready(texture)) => Some(texture),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, index: &usize) -> Option<&mut TiledTexture> {
        match self.entries.get_mut(index) {
            Some(FullImageState::Ready(texture)) => Some(texture),
            _ => None,
        }
    }

    pub fn contains_key(&self, index: &usize) -> bool {
        self.get(index).is_some()
    }

    pub fn retry_at(&self, index: usize) -> Option<Instant> {
        match self.entries.get(&index) {
            Some(FullImageState::Failed { retry_at, .. }) => Some(*retry_at),
            _ => None,
        }
    }

    pub fn error(&self, index: usize) -> Option<&str> {
        match self.entries.get(&index) {
            Some(FullImageState::Failed { error, .. }) => Some(error),
            _ => None,
        }
    }

    pub fn request(&mut self, request: ImageRequest, now: Instant) -> bool {
        match self.entries.get(&request.index) {
            Some(FullImageState::Ready(_)) => {
                self.touch(request.index);
                return false;
            }
            Some(FullImageState::InFlight(_)) => return false,
            Some(FullImageState::Failed { retry_at, .. }) if now < *retry_at => return false,
            _ => {}
        }
        self.entries
            .insert(request.index, FullImageState::InFlight(request));
        true
    }

    pub fn is_pending(&self, request: ImageRequest) -> bool {
        matches!(self.entries.get(&request.index), Some(FullImageState::InFlight(pending)) if *pending == request)
    }

    pub fn loaded(&mut self, request: ImageRequest, texture: TiledTexture, selected: usize) {
        if !self.is_pending(request) {
            return;
        }
        self.entries
            .insert(request.index, FullImageState::Ready(texture));
        self.touch(request.index);
        if self.usage.len() > self.capacity {
            // Late neighbor loads must not evict the image the user is currently viewing.
            if let Some(pos) = self.usage.iter().rposition(|&index| index != selected) {
                let index = self.usage.remove(pos).unwrap();
                self.entries.remove(&index);
            }
        }
    }

    pub fn failed(&mut self, request: ImageRequest, error: String, now: Instant) {
        if self.is_pending(request) {
            self.entries.insert(
                request.index,
                FullImageState::Failed {
                    retry_at: now + self.retry_delay,
                    error,
                },
            );
        }
    }

    pub fn invalidate(&mut self, index: usize) {
        self.entries.remove(&index);
        self.usage.retain(|&cached| cached != index);
    }

    fn touch(&mut self, index: usize) {
        self.usage.retain(|&cached| cached != index);
        self.usage.push_front(index);
    }
}

#[derive(Debug)]
pub enum SelectionPreload {
    Waiting { deadline: Instant },
    Requested,
}

impl SelectionPreload {
    pub fn after(now: Instant, delay: Duration) -> Self {
        Self::Waiting {
            deadline: now + delay,
        }
    }

    pub fn deadline(&self) -> Option<Instant> {
        match self {
            Self::Waiting { deadline } => Some(*deadline),
            Self::Requested => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_can_append_images_without_resetting_loaded_generations_or_progress() {
        let mut images = ImageStates::new(0);
        images.add_images(1);
        let first = images.invalidate(0);
        images.image_ready(first, vec![1.0]);
        images.add_images(2);
        assert_eq!(images.request(0), first);
        assert_eq!(images.embedding(0), Some([1.0].as_slice()));
        assert_eq!((images.pending(), images.failed()), (2, 0));
        images.image_failed(images.request(1), "failed".into());
        images.thumbnail_ready(images.request(2), EmbeddingState::Ready(vec![2.0]), None);
        assert_eq!((images.pending(), images.failed()), (0, 1));
    }

    #[test]
    fn old_image_results_are_rejected_after_invalidation() {
        let mut images = ImageStates::new(1);
        let old = images.request(0);
        let new = images.invalidate(0);
        assert!(images.image_ready(new, vec![2.0]));
        assert!(!images.image_ready(old, vec![1.0]));
        assert!(!images.image_failed(old, "stale error".into()));
        assert!(!images.thumbnail_ready(old, EmbeddingState::Pending, None));
        assert_eq!(images.embedding(0), Some([2.0].as_slice()));
        assert_eq!((images.pending(), images.failed()), (0, 0));
        assert!(!images.is_current(ImageRequest {
            index: 1,
            generation: 0
        }));
    }

    #[test]
    fn embedding_progress_survives_event_reordering_failure_and_reload() {
        let mut images = ImageStates::new(2);
        let first = images.request(0);
        let second = images.request(1);
        assert_eq!(images.pending(), 2);
        images.image_failed(first, "inference failed".into());
        images.thumbnail_ready(first, EmbeddingState::Pending, None);
        assert_eq!((images.pending(), images.failed()), (1, 1));
        images.image_ready(second, vec![1.0]);
        images.thumbnail_ready(second, EmbeddingState::Pending, None);
        assert_eq!((images.pending(), images.failed()), (0, 1));
        let reloaded = images.invalidate(0);
        assert_eq!((images.pending(), images.failed()), (1, 0));
        images.thumbnail_ready(reloaded, EmbeddingState::Pending, None);
        images.worker_stopped("offline");
        assert_eq!((images.pending(), images.failed()), (0, 1));
        assert_eq!(
            images.ready_embeddings().collect::<Vec<_>>(),
            vec![(1, [1.0].as_slice())]
        );
    }

    #[test]
    fn thumbnail_delivery_cannot_undo_an_early_embedding_result() {
        for mut state in [
            EmbeddingState::Ready(vec![1.0]),
            EmbeddingState::Failed("error".into()),
        ] {
            state.thumbnail_ready(EmbeddingState::Pending, None);
            assert!(!state.is_pending());
        }
        let mut state = EmbeddingState::WaitingForThumbnail;
        state.thumbnail_ready(EmbeddingState::Failed("queue closed".into()), None);
        assert!(!state.is_pending());
    }

    #[test]
    fn worker_shutdown_handles_both_thumbnail_delivery_orders_and_cached_results() {
        let mut before = EmbeddingState::WaitingForThumbnail;
        before.thumbnail_ready(EmbeddingState::Pending, None);
        before.worker_stopped("offline");
        assert!(matches!(before, EmbeddingState::Failed(_)));

        let mut after = EmbeddingState::WaitingForThumbnail;
        after.worker_stopped("offline");
        after.thumbnail_ready(EmbeddingState::Pending, Some("offline"));
        assert!(matches!(after, EmbeddingState::Failed(_)));

        let mut cached = EmbeddingState::WaitingForThumbnail;
        cached.worker_stopped("offline");
        cached.thumbnail_ready(EmbeddingState::Ready(vec![1.0]), Some("offline"));
        assert_eq!(cached.embedding(), Some([1.0].as_slice()));
    }

    #[test]
    fn full_image_failure_retries_exactly_at_deadline_and_recovers() {
        let now = Instant::now();
        let delay = Duration::from_secs(5);
        let mut cache = FullImageCache::new(2, delay);
        let request = ImageRequest {
            index: 0,
            generation: 0,
        };
        assert!(cache.request(request, now));
        assert!(!cache.request(request, now));
        cache.failed(request, "decode failed".into(), now);
        assert_eq!(cache.error(0), Some("decode failed"));
        assert_eq!(cache.retry_at(0), Some(now + delay));
        assert!(!cache.request(request, now + delay - Duration::from_nanos(1)));
        assert!(cache.request(request, now + delay));
        cache.loaded(request, texture(1), 0);
        cache.failed(request, "late error".into(), now);
        assert_eq!(cache.get(&0).unwrap().full_w, 1);
        assert_eq!(cache.error(0), None);
    }

    #[test]
    fn full_image_invalidation_rejects_late_success_and_failure() {
        let now = Instant::now();
        let mut cache = FullImageCache::new(2, Duration::from_secs(5));
        let old = ImageRequest {
            index: 0,
            generation: 0,
        };
        let new = ImageRequest {
            index: 0,
            generation: 1,
        };
        cache.request(old, now);
        cache.invalidate(0);
        cache.request(new, now);
        cache.loaded(old, texture(100), 0);
        cache.failed(old, "stale".into(), now);
        assert!(cache.is_pending(new));
        cache.loaded(new, texture(200), 0);
        cache.loaded(old, texture(100), 0);
        assert_eq!(cache.get(&0).unwrap().full_w, 200);
    }

    #[test]
    fn full_image_lru_pins_selection_and_touches_cached_requests() {
        let now = Instant::now();
        let mut cache = FullImageCache::new(2, Duration::from_secs(5));
        let request = |index| ImageRequest {
            index,
            generation: 0,
        };
        for index in 0..3 {
            assert!(cache.request(request(index), now));
            cache.loaded(request(index), texture(1), 0);
        }
        assert!(cache.get(&0).is_some());
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_some());
        assert!(!cache.request(request(0), now));
        cache.request(request(3), now);
        cache.loaded(request(3), texture(1), 3);
        assert!(cache.get(&0).is_some());
        assert!(cache.get(&2).is_none());
        assert!(cache.get(&3).is_some());
    }

    fn texture(width: u32) -> TiledTexture {
        TiledTexture::new(
            width,
            1,
            vec![super::super::TiledFrame {
                delay: Duration::ZERO,
                tiles: Vec::new(),
            }],
        )
    }
}
