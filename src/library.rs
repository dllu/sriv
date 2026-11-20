use crate::image_io;
use crate::source::SourceVersion;
use crossbeam_channel::{bounded, Receiver, TryRecvError};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

pub type SharedImagePaths = Arc<RwLock<Vec<PathBuf>>>;

pub struct DiscoveredImage {
    pub path: PathBuf,
    pub version: Option<SourceVersion>,
}

pub enum LibraryUpdate {
    Images(Vec<DiscoveredImage>),
    Sidecars(Vec<bool>),
    Finished,
}

pub enum LibraryState {
    Discovering {
        updates: Receiver<LibraryUpdate>,
        paths: SharedImagePaths,
    },
    Ready,
}

impl LibraryState {
    pub fn start(
        inputs: Vec<PathBuf>,
        clear_cache: Option<PathBuf>,
        wake: impl Fn() + Send + 'static,
    ) -> (Self, SharedImagePaths) {
        // Bound pending batches so discovering a large library cannot outrun the UI.
        let (tx, updates) = bounded(4);
        let paths = Arc::new(RwLock::new(Vec::new()));
        thread::spawn(move || {
            if let Some(cache) = clear_cache {
                if let Err(error) = fs::remove_dir_all(&cache) {
                    if error.kind() != std::io::ErrorKind::NotFound {
                        eprintln!(
                            "Failed to clear thumbnail cache {}: {error}",
                            cache.display()
                        );
                    }
                }
            }
            discover_images(&inputs, |update| {
                if tx.send(update).is_err() {
                    return false;
                }
                wake();
                true
            });
        });
        (
            Self::Discovering {
                updates,
                paths: Arc::clone(&paths),
            },
            paths,
        )
    }

    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Discovering { .. })
    }

    pub fn try_recv(&mut self) -> Option<LibraryUpdate> {
        let Self::Discovering { updates, paths } = self else {
            return None;
        };
        match updates.try_recv() {
            Ok(LibraryUpdate::Images(images)) => {
                paths
                    .write()
                    .unwrap()
                    .extend(images.iter().map(|image| image.path.clone()));
                Some(LibraryUpdate::Images(images))
            }
            Ok(LibraryUpdate::Finished) | Err(TryRecvError::Disconnected) => {
                *self = Self::Ready;
                Some(LibraryUpdate::Finished)
            }
            Ok(update) => Some(update),
            Err(TryRecvError::Empty) => None,
        }
    }
}

fn collect_candidates(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for input in inputs {
        let path = match input.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                eprintln!("Failed to access {}: {error}", input.display());
                continue;
            }
        };
        if path.is_dir() {
            let entries = match fs::read_dir(&path) {
                Ok(entries) => entries,
                Err(error) => {
                    eprintln!("Failed to list {}: {error}", path.display());
                    continue;
                }
            };
            for entry in entries.flatten() {
                // A directory entry normally supplies its type without a metadata lookup.
                // Resolve individual paths only for symlinks; the parent is already canonical.
                match entry.file_type() {
                    Ok(kind) if kind.is_file() => paths.push(entry.path()),
                    Ok(kind) if kind.is_symlink() => {
                        if let Ok(target) = entry.path().canonicalize() {
                            if target.is_file() {
                                paths.push(target);
                            }
                        }
                    }
                    _ => {}
                }
            }
        } else if path.is_file() {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

fn discover_images(inputs: &[PathBuf], mut emit: impl FnMut(LibraryUpdate) -> bool) {
    let mut batch = Vec::new();
    let mut image_paths = Vec::new();
    let mut last_batch = Instant::now();
    for path in collect_candidates(inputs) {
        // Format detection reads the source file. Keep it off the event loop, even
        // for images whose thumbnails already exist in the disk cache.
        if !image_io::is_supported_image_path(&path) {
            continue;
        }
        let version = SourceVersion::read(&path);
        image_paths.push(path.clone());
        batch.push(DiscoveredImage { path, version });
        if image_paths.len() == 1
            || batch.len() >= 64
            || last_batch.elapsed() >= Duration::from_millis(16)
        {
            if !emit(LibraryUpdate::Images(std::mem::take(&mut batch))) {
                return;
            }
            last_batch = Instant::now();
        }
    }
    if !batch.is_empty() && !emit(LibraryUpdate::Images(batch)) {
        return;
    }
    // Sidecars are optional decorations; they must not delay the first thumbnails.
    if !emit(LibraryUpdate::Sidecars(image_io::detect_thumb_sidecars(
        &image_paths,
    ))) {
        return;
    }
    emit(LibraryUpdate::Finished);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgb, RgbImage};
    use std::path::Path;
    use std::sync::mpsc::channel;
    use std::time::SystemTime;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("sriv-discovery-{}-{unique}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn image(path: &Path) {
        RgbImage::from_pixel(2, 2, Rgb([32, 64, 128]))
            .save_with_format(path, ImageFormat::Png)
            .unwrap();
    }

    #[test]
    fn discovery_yields_before_inspecting_the_rest_of_the_library() {
        let directory = TestDirectory::new();
        let first = directory.0.join("a.png");
        let later = directory.0.join("b.png");
        image(&first);
        image(&later);
        let mut discovered = Vec::new();
        let mut finished = false;
        discover_images(&[directory.0.clone()], |update| {
            match update {
                LibraryUpdate::Images(images) => {
                    assert_eq!(images.len(), 1);
                    assert_eq!(images[0].path, first);
                    assert!(images[0].version.is_some());
                    discovered.extend(images.into_iter().map(|image| image.path));
                    // This would be too late if discovery inspected every image before
                    // making the first batch available to the thumbnail workers.
                    fs::write(&later, b"not an image").unwrap();
                }
                LibraryUpdate::Sidecars(flags) => assert_eq!(flags, vec![false]),
                LibraryUpdate::Finished => finished = true,
            }
            true
        });
        assert_eq!(discovered, vec![first]);
        assert!(finished);
    }

    #[test]
    fn discovery_preserves_content_detection_sorting_and_sidecars() {
        let directory = TestDirectory::new();
        let first = directory.0.join("a.unknown");
        let second = directory.0.join("b.jpg");
        image(&first);
        image(&second);
        fs::write(directory.0.join("fake.jpg"), b"not an image").unwrap();
        fs::write(directory.0.join("a.RAF"), b"raw").unwrap();
        fs::write(directory.0.join("a.RAF.xmp"), b"sidecar").unwrap();
        let child = directory.0.join("child");
        fs::create_dir(&child).unwrap();
        image(&child.join("nested.png"));

        let mut discovered = Vec::new();
        let mut flags = Vec::new();
        discover_images(&[directory.0.clone()], |update| {
            match update {
                LibraryUpdate::Images(images) => {
                    discovered.extend(images.into_iter().map(|image| image.path));
                }
                LibraryUpdate::Sidecars(sidecars) => flags = sidecars,
                LibraryUpdate::Finished => {}
            }
            true
        });
        assert_eq!(discovered, vec![first, second]);
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn discovery_publishes_worker_paths_and_clears_cache_before_loading() {
        let directory = TestDirectory::new();
        let first = directory.0.join("a.png");
        image(&first);
        let cache = directory.0.join("cache");
        fs::create_dir(&cache).unwrap();
        fs::write(cache.join("stale"), b"cached").unwrap();
        let (wake_tx, wake_rx) = channel();
        let (mut state, worker_paths) =
            LibraryState::start(vec![directory.0.clone()], Some(cache.clone()), move || {
                let _ = wake_tx.send(());
            });
        let mut discovered = Vec::new();
        while state.is_loading() {
            wake_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            while let Some(update) = state.try_recv() {
                if let LibraryUpdate::Images(images) = update {
                    assert!(!cache.exists());
                    discovered.extend(images.into_iter().map(|image| image.path));
                    assert_eq!(*worker_paths.read().unwrap(), discovered);
                }
            }
        }
        assert_eq!(discovered, vec![first]);
        assert!(state.try_recv().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn discovery_keeps_symlink_cache_keys_canonical() {
        let directory = TestDirectory::new();
        let target = directory.0.join("a.png");
        image(&target);
        std::os::unix::fs::symlink(&target, directory.0.join("b.alias")).unwrap();
        assert_eq!(
            collect_candidates(&[directory.0.clone()]),
            vec![target.clone(), target]
        );
    }
}
