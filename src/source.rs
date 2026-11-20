use std::fs::{self, File, FileTimes, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

#[cfg(unix)]
use std::{os::unix::fs::MetadataExt, time::Duration};

use crate::state::ImageRequest;

/// Identify the source read by a worker, including replacements with a preserved mtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceVersion {
    modified: SystemTime,
    len: u64,
    #[cfg(unix)]
    identity: (u64, u64),
    #[cfg(unix)]
    changed: (i64, i64),
}

impl SourceVersion {
    pub fn read(path: &Path) -> Option<Self> {
        let metadata = fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        Some(Self {
            modified: metadata.modified().ok()?,
            len: metadata.len(),
            #[cfg(unix)]
            identity: (metadata.dev(), metadata.ino()),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    pub fn matches(self, path: &Path) -> bool {
        Self::read(path) == Some(self)
    }

    fn cache_time(self) -> SystemTime {
        #[cfg(unix)]
        if let Ok(seconds) = u64::try_from(self.changed.0) {
            if let Some(changed) =
                SystemTime::UNIX_EPOCH.checked_add(Duration::new(seconds, self.changed.1 as u32))
            {
                return self.modified.max(changed);
            }
        }
        self.modified
    }

    pub fn open_cache(self, path: &Path) -> Option<File> {
        let file = File::open(path).ok()?;
        let modified = file.metadata().and_then(|meta| meta.modified()).ok()?;
        (modified >= self.cache_time()).then_some(file)
    }

    /// Publish complete cache files and date them to the source that produced them.
    pub fn write_cache(self, source: &Path, path: &Path, bytes: &[u8]) -> io::Result<()> {
        static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let temp = path.with_extension(format!("tmp-{}-{id}", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let result = file
            .write_all(bytes)
            .and_then(|()| file.set_times(FileTimes::new().set_modified(self.cache_time())));
        drop(file);
        if let Err(error) = result {
            let _ = fs::remove_file(temp);
            return Err(error);
        }
        // The source can change even after this check. Its newer timestamp then makes
        // this cache entry ineligible on the next read.
        if self.matches(source) {
            if let Err(error) = fs::rename(&temp, path) {
                let _ = fs::remove_file(temp);
                return Err(error);
            }
        } else {
            let _ = fs::remove_file(temp);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceRequest {
    pub request: ImageRequest,
    pub version: SourceVersion,
    pub read_cache: bool,
}

#[derive(Debug)]
pub enum SourceOutcome<T> {
    Ready(T),
    Changed,
    Failed(String),
}

impl SourceRequest {
    pub fn load<T>(
        self,
        path: &Path,
        load: impl FnOnce() -> anyhow::Result<T>,
    ) -> SourceOutcome<T> {
        if !self.version.matches(path) {
            return SourceOutcome::Changed;
        }
        let result = load();
        if !self.version.matches(path) {
            return SourceOutcome::Changed;
        }
        match result {
            Ok(value) => SourceOutcome::Ready(value),
            Err(error) => SourceOutcome::Failed(format!("{error:#}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_changed_during_load_cannot_publish_a_result_or_replace_the_new_cache() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("sriv-source-{}-{unique}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let image = dir.join("image.png");
        let cached = dir.join("thumbnail.png");
        fs::write(&image, b"old source").unwrap();
        let old = SourceVersion::read(&image).unwrap();
        let job = SourceRequest {
            request: ImageRequest {
                index: 0,
                generation: 0,
            },
            version: old,
            read_cache: true,
        };
        let outcome = job.load(&image, || {
            fs::write(&image, b"new source with a different size")?;
            let new = SourceVersion::read(&image).unwrap();
            new.write_cache(&image, &cached, b"new thumbnail")?;
            old.write_cache(&image, &cached, b"stale thumbnail")?;
            Ok(())
        });
        assert!(matches!(outcome, SourceOutcome::Changed));
        assert_eq!(fs::read(&cached).unwrap(), b"new thumbnail");
        fs::remove_dir_all(dir).unwrap();
    }
}
