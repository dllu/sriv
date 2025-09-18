use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver as CbReceiver, Sender as CbSender};
use image::imageops::FilterType;
use nannou::event::{ModifiersState, MouseButton, MouseScrollDelta, TouchPhase, Update};
use nannou::image::imageops::crop_imm;
use nannou::image::{self, DynamicImage, GenericImageView, RgbaImage};
use nannou::prelude::*;
use nannou::wgpu;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use toml::Value as TomlValue;

mod clip;

use clip::{ClipEngine, ClipEvent};
/// Maximum number of full-resolution images to cache in memory.
const FULL_CACHE_CAPACITY: usize = 4;
/// How long to wait before retrying a full-resolution load request.
const FULL_PENDING_RETRY: Duration = Duration::from_secs(5);
/// Number of extra rows of thumbnails to keep warm beyond the viewport.
const THUMB_PREFETCH_ROWS: usize = 1;
/// Additional thumbnail textures to keep available beyond the visible count.
const THUMB_POOL_MARGIN: usize = 12;

/// List of recognized raw file extensions for detecting XMP sidecars.
const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "bay", "cap", "cr2", "cr3", "crw", "cs1", "dcr", "dng", "erf", "fff",
    "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "ptx", "pxn",
    "raf", "raw", "rwl", "rw2", "rwz", "sr2", "srf", "srw", "x3f",
];

/// The display mode of the viewer.
#[derive(Debug)]
enum Mode {
    Thumbnails,
    Single,
}

/// A user-defined key binding specification.
#[derive(Debug)]
struct KeyBinding {
    key: Key,
    ctrl: bool,
    shift: bool,
    alt: bool,
    super_key: bool,
    command: String,
}

/// Parse a single binding spec (e.g. "ctrl+u") and command into a KeyBinding.
fn parse_binding_spec(spec: &str, command: &str) -> Option<KeyBinding> {
    let mut ctrl = false;
    let mut shift = false;
    let mut alt = false;
    let mut super_key = false;
    let mut key_opt: Option<Key> = None;
    for part in spec.split('+').map(|s| s.trim()) {
        match part.to_lowercase().as_str() {
            "ctrl" | "control" => ctrl = true,
            "shift" => shift = true,
            "alt" => alt = true,
            "super" | "cmd" | "meta" => super_key = true,
            tok => {
                // Parse as main key
                if key_opt.is_some() {
                    return None;
                }
                let key = match tok.to_ascii_uppercase().as_str() {
                    // Letters
                    c if c.len() == 1 && c.chars().all(|ch| ch.is_ascii_alphabetic()) => {
                        let ch = c.chars().next().unwrap();
                        // Match variant by character
                        match ch {
                            'A' => Key::A,
                            'B' => Key::B,
                            'C' => Key::C,
                            'D' => Key::D,
                            'E' => Key::E,
                            'F' => Key::F,
                            'G' => Key::G,
                            'H' => Key::H,
                            'I' => Key::I,
                            'J' => Key::J,
                            'K' => Key::K,
                            'L' => Key::L,
                            'M' => Key::M,
                            'N' => Key::N,
                            'O' => Key::O,
                            'P' => Key::P,
                            'Q' => Key::Q,
                            'R' => Key::R,
                            'S' => Key::S,
                            'T' => Key::T,
                            'U' => Key::U,
                            'V' => Key::V,
                            'W' => Key::W,
                            'X' => Key::X,
                            'Y' => Key::Y,
                            'Z' => Key::Z,
                            _ => return None,
                        }
                    }
                    // Digits
                    d if d.len() == 1 && d.chars().all(|ch| ch.is_ascii_digit()) => match d {
                        "0" => Key::Key0,
                        "1" => Key::Key1,
                        "2" => Key::Key2,
                        "3" => Key::Key3,
                        "4" => Key::Key4,
                        "5" => Key::Key5,
                        "6" => Key::Key6,
                        "7" => Key::Key7,
                        "8" => Key::Key8,
                        "9" => Key::Key9,
                        _ => return None,
                    },
                    _ => return None,
                };
                key_opt = Some(key);
            }
        }
    }
    key_opt.map(|key| KeyBinding {
        key,
        ctrl,
        shift,
        alt,
        super_key,
        command: command.to_string(),
    })
}

/// Parse TOML-formatted bindings into a list of KeyBindings.
fn parse_bindings(s: &str) -> Vec<KeyBinding> {
    let mut bindings = Vec::new();
    if let Ok(TomlValue::Table(table)) = toml::from_str::<TomlValue>(s) {
        for (spec, val) in table {
            if let TomlValue::String(cmd) = val {
                if let Some(binding) = parse_binding_spec(&spec, &cmd) {
                    bindings.push(binding);
                }
            }
        }
    }
    bindings
}

/// Mouse click handler: select thumbnail on left-click in thumbnail mode.
fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if let Mode::Thumbnails = model.mode {
        if button == MouseButton::Left {
            let pos = app.mouse.position();
            let rect = app.window_rect();
            let cell = model.thumb_size as f32 + model.gap;
            let cols = ((rect.w() + model.gap) / cell).floor() as usize;
            let cols = cols.max(1);
            let total = model.image_paths.len();
            let rows = total.div_ceil(cols);
            let half_gap = model.gap / 2.0;
            let scroll = model.scroll_offset;
            let row_min_f = (scroll - model.thumb_size as f32 - half_gap) / cell;
            let row_min = row_min_f.ceil().max(0.0) as usize;
            let row_max_f = (rect.h() + scroll - half_gap) / cell;
            let row_max = row_max_f.floor().min(rows.saturating_sub(1) as f32) as usize;
            for row in row_min..=row_max {
                let base_y = rect.h() / 2.0
                    - (model.thumb_size as f32) / 2.0
                    - half_gap
                    - (row as f32) * cell;
                for col in 0..cols {
                    let i = row * cols + col;
                    if i >= total {
                        break;
                    }
                    let x = -rect.w() / 2.0
                        + (model.thumb_size as f32) / 2.0
                        + half_gap
                        + (col as f32) * cell;
                    let y = base_y + scroll;
                    let (width, height) = if let Some(tex) = model.thumb_textures.get(&i) {
                        let [tw, th] = tex.size();
                        (tw as f32, th as f32)
                    } else {
                        let size = model.thumb_size as f32;
                        (size, size)
                    };
                    let x_min = x - width / 2.0;
                    let x_max = x + width / 2.0;
                    let y_min = y - height / 2.0;
                    let y_max = y + height / 2.0;
                    if pos.x >= x_min && pos.x <= x_max && pos.y >= y_min && pos.y <= y_max {
                        model.current = i;
                        model.selection_changed_at = Instant::now();
                        model.selection_pending = false;
                        return;
                    }
                }
            }
        }
    }
}

/// Mouse wheel scroll handler to scroll thumbnails in thumbnail view.
fn mouse_wheel(app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    match model.mode {
        Mode::Thumbnails => {
            // Determine scroll amount: line vs pixel delta
            let scroll_amount = match delta {
                MouseScrollDelta::LineDelta(_x, y) => y * -100.0,
                MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
            };
            // Update scroll offset and clamp to content bounds
            model.scroll_offset += scroll_amount;
            let rect = app.window_rect();
            let cell = model.thumb_size as f32 + model.gap;
            // Number of columns and rows in the thumbnail grid
            let cols = ((rect.w() + model.gap) / cell).floor() as usize;
            let cols = cols.max(1);
            let total = model.image_paths.len();
            let rows = ((total + cols - 1) / cols) as f32;
            // Maximum scroll to show last row at bottom
            let max_scroll = (rows * cell - rect.h()).max(0.0);
            model.scroll_offset = model.scroll_offset.clamp(0.0, max_scroll);
        }
        Mode::Single => {
            // Zoom in/out around mouse cursor
            let mouse_pos = app.mouse.position();
            let old_zoom = model.zoom;
            // Determine zoom factor from scroll delta
            let zoom_factor = match delta {
                MouseScrollDelta::LineDelta(_x, y) => 1.0 + y * 0.2,
                MouseScrollDelta::PixelDelta(pos) => 1.0 + pos.y as f32 * 0.002,
            };
            let new_zoom = (old_zoom * zoom_factor).clamp(0.01, 10.0);
            // Adjust pan so the point under cursor stays fixed
            model.pan = mouse_pos + (model.pan - mouse_pos) * (new_zoom / old_zoom);
            model.zoom = new_zoom;
        }
    }
}

/// A single tile of a full-resolution image used to bypass GPU texture size limits.
#[derive(Debug)]
struct Tile {
    /// Pixel offset from the left edge of the full image.
    x_offset: u32,
    /// Pixel offset from the top edge of the full image.
    y_offset: u32,
    /// Width of this tile in pixels.
    width: u32,
    /// Height of this tile in pixels.
    height: u32,
    /// Raw pixel data stored on the CPU.
    pixel_data: Vec<u8>,
    /// Lazily-created GPU texture.
    texture: RefCell<Option<wgpu::Texture>>,
}
/// A full-resolution image represented as a set of tiles.
#[derive(Debug)]
struct TiledTexture {
    /// Full image width in pixels.
    full_w: u32,
    /// Full image height in pixels.
    full_h: u32,
    /// Tiles composing the full image.
    tiles: Vec<Tile>,
}
// Implement creation and sizing of tiled full-resolution textures.
impl TiledTexture {
    /// Return the full image dimensions [width, height].
    fn size(&self) -> [u32; 2] {
        [self.full_w, self.full_h]
    }
}

/// The application state.
#[derive(Debug)]
struct SearchState {
    input: String,
    focused: bool,
    results: Vec<(usize, f32)>,
    current: usize,
    pending_request: Option<u64>,
    error: Option<String>,
    last_embedding: Option<Vec<f32>>,
}

#[derive(Debug)]
struct Model {
    image_paths: Vec<PathBuf>,
    thumb_textures: HashMap<usize, wgpu::Texture>,
    thumb_has_xmp: Vec<bool>,
    thumb_rx: Receiver<(usize, DynamicImage)>,
    thumb_req_tx: CbSender<usize>,
    thumb_pending: HashSet<usize>,
    thumb_usage: VecDeque<usize>,
    thumb_capacity: usize,
    // Channels for full-resolution image loading
    full_req_tx: CbSender<usize>,
    full_resp_rx: CbReceiver<(usize, u32, u32, Vec<(u32, u32, u32, u32, Vec<u8>)>)>,
    // Track indices currently requested but not yet loaded
    full_pending: HashMap<usize, Instant>,
    // LRU cache of full-resolution tiled textures
    full_textures: HashMap<usize, TiledTexture>,
    // Usage order for LRU eviction: front = most recently used
    full_usage: VecDeque<usize>,
    mode: Mode,
    current: usize,
    thumb_size: u32,
    gap: f32,
    // Vertical scroll offset for thumbnail view
    scroll_offset: f32,
    // Zoom scale for single image view (1.0 = 100%)
    zoom: f32,
    // Pan offset for single image view (in window coords)
    pan: Vec2,
    // Window rect at last update, for resize detection
    prev_window_rect: Rect,
    // Whether to auto-fit the current image when loaded or resized
    fit_mode: bool,
    // Timestamp when thumbnail selection last changed
    selection_changed_at: Instant,
    // Whether preload has been requested after selection
    selection_pending: bool,
    // User-defined key bindings
    key_bindings: Vec<KeyBinding>,
    // Channel for sending command output messages
    command_tx: std::sync::mpsc::Sender<String>,
    // Channel for receiving command output messages
    command_rx: Receiver<String>,
    // Captured command output for display
    command_output: Option<String>,
    // CLIP embedding management
    clip_engine: ClipEngine,
    clip_embeddings: Vec<Option<Vec<f32>>>,
    clip_missing: HashSet<usize>,
    clip_inflight: HashSet<usize>,
    next_search_request_id: u64,
    search: Option<SearchState>,
}
/// Compute the cache path for an image based on a SHA1 of its path.
/// The cache layout is: cache_base/<first 3 hex chars>/<remaining hex chars>.png
fn thumbnail_cache_path(cache_base: &Path, image_path: &Path) -> PathBuf {
    clip::cache_file_path(cache_base, image_path, "png")
}

/// Adjust image orientation based on EXIF orientation tag.
pub(crate) fn adjust_orientation(img: DynamicImage, path: &Path) -> DynamicImage {
    let mut oriented = img;
    if let Ok(exif) = rexif::parse_file(path) {
        for entry in exif.entries {
            if entry.tag.to_string() == "Orientation" {
                if let rexif::TagValue::U16(vals) = entry.value {
                    if let Some(&code) = vals.get(0) {
                        oriented = match code {
                            2 => oriented.fliph(),
                            3 => oriented.rotate180(),
                            4 => oriented.flipv(),
                            5 => oriented.rotate90().fliph(),
                            6 => oriented.rotate90(),
                            7 => oriented.rotate270().fliph(),
                            8 => oriented.rotate270(),
                            _ => oriented,
                        }
                    }
                }
                break;
            }
        }
    }
    oriented
}

/// Scan a directory for raw files that have matching XMP sidecars.
fn scan_raw_sidecars(dir: &Path) -> HashMap<String, bool> {
    let mut map = HashMap::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext_raw = match path.extension().and_then(|s| s.to_str()) {
                Some(ext) => ext,
                None => continue,
            };
            let ext_lower = ext_raw.to_ascii_lowercase();
            if !RAW_EXTENSIONS.contains(&ext_lower.as_str()) {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(stem) => stem,
                None => continue,
            };
            let mut has_xmp = false;
            // Typical variants: foo.RAF.xmp, foo.raf.xmp, foo.xmp
            let base_xmp = path.with_extension("xmp");
            let candidates = [
                path.with_extension(format!("{}.xmp", ext_raw)),
                path.with_extension(format!("{}.xmp", ext_lower)),
                base_xmp.clone(),
                path.parent()
                    .map(|parent| parent.join(format!("{}.xmp", stem)))
                    .unwrap_or_else(|| base_xmp.clone()),
            ];
            for candidate in candidates.iter() {
                if candidate.exists() {
                    has_xmp = true;
                    break;
                }
            }
            let key = stem.to_string();
            map.entry(key)
                .and_modify(|flag| *flag |= has_xmp)
                .or_insert(has_xmp);
        }
    }
    map
}

/// Determine which images have corresponding raw files with XMP sidecars.
fn detect_thumb_sidecars(image_paths: &[PathBuf]) -> Vec<bool> {
    let mut dir_cache: HashMap<PathBuf, HashMap<String, bool>> = HashMap::new();
    let mut flags = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem.to_string(),
            None => {
                flags.push(false);
                continue;
            }
        };
        let parent = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let entry = dir_cache
            .entry(parent.clone())
            .or_insert_with(|| scan_raw_sidecars(&parent));
        let flag = entry.get(&stem).copied().unwrap_or(false);
        flags.push(flag);
    }
    flags
}

/// The model function for initializing the application state.
fn model(app: &App) -> Model {
    // Parse command-line arguments: files or directories.
    let mut regen_cache = false;
    let mut args: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--clear-cache" || arg == "--regen-cache" {
            regen_cache = true;
        } else {
            args.push(arg);
        }
    }
    if args.is_empty() {
        eprintln!("Usage: sriv-rs [--clear-cache] <image files or directories>...");
        std::process::exit(1);
    }
    // Collect image file paths.
    let mut image_paths: Vec<PathBuf> = Vec::new();
    for arg in args {
        let pb = PathBuf::from(&arg);
        if pb.is_dir() {
            for entry in fs::read_dir(&pb).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        match ext.to_lowercase().as_str() {
                            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "gif" | "webp" | "tif" => {
                                image_paths.push(path.canonicalize().unwrap());
                            }
                            _ => {}
                        }
                    }
                }
            }
        } else if pb.is_file() {
            image_paths.push(pb.canonicalize().unwrap());
        }
    }
    if image_paths.is_empty() {
        eprintln!("No image files found in arguments.");
        std::process::exit(1);
    }
    image_paths.sort();
    let thumb_has_xmp = detect_thumb_sidecars(&image_paths);
    // Prepare thumbnail size, gap, and cache base directory.
    let thumb_size: u32 = 256;
    let cache_home = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| {
                let mut pb = PathBuf::from(h);
                pb.push(".cache");
                pb
            })
        })
        .unwrap_or_else(|| PathBuf::from("."));
    let cache_base = cache_home.join("sriv");
    if regen_cache {
        if let Err(e) = fs::remove_dir_all(&cache_base) {
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!(
                    "Failed to clear thumbnail cache {}: {}",
                    cache_base.display(),
                    e
                );
            }
        }
    }
    let clip_engine = ClipEngine::new(cache_base.clone()).unwrap_or_else(|err| {
        eprintln!("Failed to initialize CLIP: {err}");
        std::process::exit(1);
    });
    let mut clip_embeddings = Vec::with_capacity(image_paths.len());
    let mut clip_missing: HashSet<usize> = HashSet::new();
    let mut clip_inflight: HashSet<usize> = HashSet::new();
    for (idx, path) in image_paths.iter().enumerate() {
        match clip_engine.load_cached_embedding(path) {
            Ok(Some(vec)) => clip_embeddings.push(Some(vec)),
            Ok(None) => {
                clip_embeddings.push(None);
                clip_missing.insert(idx);
            }
            Err(err) => {
                eprintln!(
                    "Failed to load cached CLIP embedding for {}: {}",
                    path.display(),
                    err
                );
                clip_embeddings.push(None);
                clip_missing.insert(idx);
            }
        }
    }
    for &idx in clip_missing.iter() {
        if let Err(err) = clip_engine.ensure_embedding(idx, &image_paths[idx], thumb_size) {
            eprintln!(
                "Failed to queue CLIP embedding for {}: {}",
                image_paths[idx].display(),
                err
            );
        } else {
            clip_inflight.insert(idx);
        }
    }
    // Create the window first, so textures can reference a focused window.
    let _window = app
        .new_window()
        .size(800, 600)
        .title("sriv")
        .view(view)
        .key_pressed(key_pressed)
        .received_character(received_character)
        .mouse_wheel(mouse_wheel)
        .mouse_pressed(mouse_pressed)
        .build()
        .unwrap();
    // Channel for receiving thumbnails from background threads.
    let (tx, rx) = channel::<(usize, DynamicImage)>();
    // Channel for requesting thumbnail loads on demand.
    let (thumb_req_tx, thumb_req_rx) = unbounded::<usize>();
    let num_workers = rayon::current_num_threads().clamp(1, 8);
    let shared_paths = Arc::new(image_paths.clone());
    for _ in 0..num_workers {
        let paths = Arc::clone(&shared_paths);
        let cache_base = cache_base.clone();
        let thumb_req_rx = thumb_req_rx.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            while let Ok(i) = thumb_req_rx.recv() {
                if let Some(p) = paths.get(i) {
                    let cache_path = thumbnail_cache_path(&cache_base, p);
                    let mut result: Option<DynamicImage> = None;
                    if let (Ok(meta_orig), Ok(meta_cache)) =
                        (fs::metadata(p), fs::metadata(&cache_path))
                    {
                        if let (Ok(orig_mtime), Ok(cache_mtime)) =
                            (meta_orig.modified(), meta_cache.modified())
                        {
                            if cache_mtime >= orig_mtime {
                                if let Ok(img) = image::open(&cache_path) {
                                    result = Some(DynamicImage::ImageRgba8(img.to_rgba8()));
                                }
                            }
                        }
                    }
                    if result.is_none() {
                        if let Ok(img_orig) = image::open(p) {
                            let img = adjust_orientation(img_orig, p);
                            let mut thumb = img.thumbnail(thumb_size, thumb_size);
                            let (w0, h0) = thumb.dimensions();
                            if w0 != 0 && h0 != 0 {
                                let w = w0.max(2);
                                let h = h0.max(2);
                                if w != w0 || h != h0 {
                                    thumb = thumb.resize_exact(w, h, FilterType::Nearest);
                                }
                                if let Some(parent) = cache_path.parent() {
                                    let _ = fs::create_dir_all(parent);
                                }
                                let dyn_thumb = DynamicImage::ImageRgba8(thumb.to_rgba8());
                                let _ = dyn_thumb.save(&cache_path);
                                result = Some(dyn_thumb);
                            }
                        }
                    }
                    if let Some(img) = result {
                        let _ = tx.send((i, img));
                    } else {
                        let placeholder = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
                            2,
                            2,
                            image::Rgba([128, 128, 128, 255]),
                        ));
                        let _ = tx.send((i, placeholder));
                    }
                }
            }
        });
    }
    drop(thumb_req_rx);
    // Initialize channels and state for full-resolution LRU cache.
    // Channel for requesting full-resolution images (by index)
    let (full_req_tx, full_req_rx) = unbounded::<usize>();
    // Channel for receiving loaded full-resolution image tile data
    let (full_resp_tx, full_resp_rx) =
        unbounded::<(usize, u32, u32, Vec<(u32, u32, u32, u32, Vec<u8>)>)>();
    // Spawn a pool of loader threads for full images: load, crop, and convert to raw tile data off the main thread
    {
        // Shared image paths for all workers
        let paths = Arc::new(image_paths.clone());
        // Spawn worker threads matching thumbnail thread count
        for _ in 0..num_workers {
            let req_rx = full_req_rx.clone();
            let resp_tx = full_resp_tx.clone();
            let paths = Arc::clone(&paths);
            thread::spawn(move || {
                while let Ok(idx) = req_rx.recv() {
                    if let Some(path) = paths.get(idx) {
                        if let Ok(img_orig) = image::open(path) {
                            let img = adjust_orientation(img_orig, path);
                            let rgba = img.to_rgba8();
                            let full_w = rgba.width();
                            let full_h = rgba.height();
                            const MAX_TILE_SIZE: u32 = 8192;
                            let mut tiles_data = Vec::new();
                            for y in (0..full_h).step_by(MAX_TILE_SIZE as usize) {
                                for x in (0..full_w).step_by(MAX_TILE_SIZE as usize) {
                                    let tile_w = (full_w - x).min(MAX_TILE_SIZE);
                                    let tile_h = (full_h - y).min(MAX_TILE_SIZE);
                                    let sub_image: RgbaImage =
                                        crop_imm(&rgba, x, y, tile_w, tile_h).to_image();
                                    let raw_pixels = sub_image.into_raw();
                                    tiles_data.push((x, y, tile_w, tile_h, raw_pixels));
                                }
                            }
                            let _ = resp_tx.send((idx, full_w, full_h, tiles_data));
                        }
                    }
                }
            });
        }
    }
    let full_pending: HashMap<usize, Instant> = HashMap::new();
    let full_textures: HashMap<usize, TiledTexture> = HashMap::new();
    let full_usage: VecDeque<usize> = VecDeque::new();
    // Get initial window rect for resize tracking
    let initial_rect = app.window_rect();
    // Load user key bindings from config file
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    let config_path = config_home.join("sriv").join("bindings.toml");
    let key_bindings = if let Ok(contents) = fs::read_to_string(&config_path) {
        parse_bindings(&contents)
    } else {
        Vec::new()
    };
    // Channel for receiving command output from custom commands
    let (command_tx, command_rx) = channel::<String>();
    let mut model = Model {
        image_paths,
        thumb_textures: HashMap::new(),
        thumb_has_xmp,
        thumb_rx: rx,
        thumb_req_tx: thumb_req_tx.clone(),
        thumb_pending: HashSet::new(),
        thumb_usage: VecDeque::new(),
        thumb_capacity: 0,
        full_req_tx,
        full_resp_rx,
        full_pending,
        full_textures,
        full_usage,
        mode: Mode::Thumbnails,
        current: 0,
        thumb_size,
        gap: 10.0,
        scroll_offset: 0.0,
        zoom: 1.0,
        pan: vec2(0.0, 0.0),
        prev_window_rect: initial_rect,
        fit_mode: false,
        selection_changed_at: Instant::now(),
        selection_pending: false,
        // Custom key bindings
        key_bindings,
        // Command output handling
        command_tx,
        command_rx,
        command_output: None,
        clip_engine,
        clip_embeddings,
        clip_missing,
        clip_inflight,
        next_search_request_id: 0,
        search: None,
    };
    update_thumbnail_requests(app, &mut model);
    model
}

fn main() -> Result<()> {
    // Launch the nannou application with our model initializer and update callback.
    nannou::app(model).update(update).run();
    Ok(())
}

/// Navigate to a given index in single-image mode: update current, preload neighbors, and fit if loaded.
fn navigate_to(app: &App, model: &mut Model, new_idx: usize) {
    let len = model.image_paths.len();
    model.current = new_idx;
    // Preload the target and its neighbors
    request_full_texture(model, new_idx);
    if new_idx > 0 {
        request_full_texture(model, new_idx - 1);
    }
    if new_idx + 1 < len {
        request_full_texture(model, new_idx + 1);
    }
    // Apply fit if already loaded
    if model.full_textures.contains_key(&new_idx) {
        apply_fit(app, model);
    }
}

fn focus_image(app: &App, model: &mut Model, idx: usize) {
    let len = model.image_paths.len();
    if len == 0 {
        return;
    }
    let idx = idx.min(len - 1);
    match model.mode {
        Mode::Single => navigate_to(app, model, idx),
        Mode::Thumbnails => {
            model.current = idx;
            model.selection_changed_at = Instant::now();
            model.selection_pending = false;
            ensure_thumbnail_visible(app, model, idx);
        }
    }
}

fn ensure_thumbnail_visible(app: &App, model: &mut Model, idx: usize) {
    if !matches!(model.mode, Mode::Thumbnails) {
        return;
    }
    let rect = app.window_rect();
    let cell = model.thumb_size as f32 + model.gap;
    let mut cols = ((rect.w() + model.gap) / cell).floor() as usize;
    cols = cols.max(1);
    let row = idx / cols;
    let top = row as f32 * cell;
    let bottom = top + cell;
    let view_height = rect.h();
    let mut scroll = model.scroll_offset;
    if top < scroll {
        scroll = top;
    } else if bottom > scroll + view_height {
        scroll = bottom - view_height;
    }
    let rows = model.image_paths.len().div_ceil(cols);
    let max_scroll = (rows as f32 * cell - view_height).max(0.0);
    model.scroll_offset = scroll.clamp(0.0, max_scroll);
}

fn advance_search(app: &App, model: &mut Model, delta: isize) {
    let mut target = None;
    if let Some(search) = model.search.as_mut() {
        if search.results.is_empty() {
            return;
        }
        let len = search.results.len() as isize;
        let mut idx = search.current as isize + delta;
        if len == 0 {
            return;
        }
        idx = ((idx % len) + len) % len;
        search.current = idx as usize;
        target = search
            .results
            .get(search.current)
            .map(|(image_idx, _)| *image_idx);
    }
    if let Some(idx) = target {
        focus_image(app, model, idx);
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

fn handle_text_result(app: &App, model: &mut Model, request_id: u64, embedding: Vec<f32>) {
    let mut focus_target = None;
    if let Some(search) = model.search.as_mut() {
        if search.pending_request != Some(request_id) {
            return;
        }
        search.pending_request = None;
        search.error = None;
        search.last_embedding = Some(embedding);
        if let Some(text_embed) = search.last_embedding.as_ref() {
            let mut scored = Vec::new();
            for (idx, maybe_embed) in model.clip_embeddings.iter().enumerate() {
                if let Some(img_embed) = maybe_embed {
                    scored.push((idx, cosine_similarity(text_embed, img_embed)));
                }
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            search.results = scored;
            search.current = 0;
            focus_target = search.results.first().map(|(idx, _)| *idx);
            if focus_target.is_none()
                && model.clip_missing.is_empty()
                && model.clip_inflight.is_empty()
            {
                search.error = Some("No matches found".to_string());
            }
        }
    }
    if let Some(idx) = focus_target {
        focus_image(app, model, idx);
    }
}

fn update_search_with_image_embedding(app: &App, model: &mut Model, index: usize) {
    let mut focus_target = None;
    if let Some(search) = model.search.as_mut() {
        if let (Some(text_embed), Some(img_embed)) = (
            search.last_embedding.as_ref(),
            model
                .clip_embeddings
                .get(index)
                .and_then(|item| item.as_ref()),
        ) {
            let had_results = !search.results.is_empty();
            let score = cosine_similarity(text_embed, img_embed);
            if let Some(entry) = search.results.iter_mut().find(|(idx, _)| *idx == index) {
                entry.1 = score;
            } else {
                search.results.push((index, score));
            }
            search
                .results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            if !search.results.is_empty() {
                search.error = None;
                if search.current >= search.results.len() {
                    search.current = search.results.len() - 1;
                }
            }
            if !had_results {
                search.current = 0;
                focus_target = Some(index);
            } else if let Some(pos) = search
                .results
                .iter()
                .position(|(idx, _)| *idx == model.current)
            {
                search.current = pos;
            }
        }
    }
    if let Some(idx) = focus_target {
        focus_image(app, model, idx);
    }
}

fn handle_search_key(app: &App, model: &mut Model, key: Key) -> bool {
    let mods = app.keys.mods;
    if mods.ctrl() || mods.alt() || mods.logo() {
        return false;
    }

    if key == Key::Slash {
        if let Some(search) = model.search.as_mut() {
            search.focused = true;
        } else {
            model.search = Some(SearchState {
                input: String::new(),
                focused: true,
                results: Vec::new(),
                current: 0,
                pending_request: None,
                error: None,
                last_embedding: None,
            });
        }
        return true;
    }

    if let Some(true) = model.search.as_ref().map(|s| s.focused) {
        match key {
            Key::Escape => {
                model.search = None;
                return true;
            }
            Key::Return => {
                let query_opt = model.search.as_ref().and_then(|s| {
                    let trimmed = s.input.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                });
                if let Some(query) = query_opt {
                    if let Some(search) = model.search.as_mut() {
                        search.pending_request = None;
                        search.error = None;
                        search.last_embedding = None;
                        search.results.clear();
                        search.current = 0;
                    }
                    let request_id = model.next_search_request_id;
                    model.next_search_request_id = model.next_search_request_id.wrapping_add(1);
                    match model.clip_engine.request_text(request_id, query) {
                        Ok(()) => {
                            if let Some(search) = model.search.as_mut() {
                                search.pending_request = Some(request_id);
                                search.focused = false;
                            }
                        }
                        Err(err) => {
                            if let Some(search) = model.search.as_mut() {
                                search.error = Some(format!("Failed to queue search: {err}"));
                                search.focused = false;
                            }
                        }
                    }
                } else if let Some(search) = model.search.as_mut() {
                    search.error = Some("Enter a search phrase".to_string());
                }
                return true;
            }
            Key::Back => {
                let mut remove_search = false;
                if let Some(search) = model.search.as_mut() {
                    if search.input.is_empty() {
                        remove_search = true;
                    } else {
                        search.input.pop();
                        search.pending_request = None;
                        search.error = None;
                        search.last_embedding = None;
                        search.results.clear();
                        search.current = 0;
                    }
                }
                if remove_search {
                    model.search = None;
                }
                return true;
            }
            _ => {
                return true;
            }
        }
    }

    if let Some(false) = model.search.as_ref().map(|s| s.focused) {
        match key {
            Key::Escape => {
                model.search = None;
                return true;
            }
            Key::N => {
                if matches!(model.mode, Mode::Thumbnails)
                    && model
                        .search
                        .as_ref()
                        .map(|s| !s.results.is_empty())
                        .unwrap_or(false)
                {
                    let delta = if mods.shift() { -1 } else { 1 };
                    advance_search(app, model, delta);
                    return true;
                }
            }
            Key::P => {
                if matches!(model.mode, Mode::Thumbnails)
                    && model
                        .search
                        .as_ref()
                        .map(|s| !s.results.is_empty())
                        .unwrap_or(false)
                {
                    let delta = if mods.shift() { 1 } else { -1 };
                    advance_search(app, model, delta);
                    return true;
                }
            }
            _ => {}
        }
    }

    false
}
/// Directions for arrow key navigation.
enum ArrowDirection {
    Left,
    Right,
    Up,
    Down,
}

/// Handle arrow navigation in both thumbnail and single modes.
/// Returns true if event was fully consumed (e.g., panned in single mode).
fn handle_arrow(app: &App, model: &mut Model, dir: ArrowDirection) -> bool {
    let len = model.image_paths.len();
    let rect = app.window_rect();
    // Thumbnail grid parameters for thumbnail navigation
    let cell = model.thumb_size as f32 + model.gap;
    let cols = ((rect.w() + model.gap) / cell).floor() as usize;
    let cols = cols.max(1);
    match model.mode {
        Mode::Thumbnails => {
            // Compute target row and column
            let row = match dir {
                ArrowDirection::Up => {
                    if model.current < cols {
                        (len - 1) / cols
                    } else {
                        model.current / cols - 1
                    }
                }
                ArrowDirection::Down => model.current / cols + 1,
                _ => model.current / cols,
            };
            let col = match dir {
                ArrowDirection::Left => {
                    let c = model.current % cols;
                    if c == 0 {
                        cols - 1
                    } else {
                        c - 1
                    }
                }
                ArrowDirection::Right => {
                    let c = model.current % cols;
                    if c + 1 >= cols {
                        0
                    } else {
                        c + 1
                    }
                }
                _ => model.current % cols,
            };
            let idx = row * cols + col;
            if idx < len {
                model.current = idx;
            }
            false
        }
        Mode::Single => {
            let pan_step = 200.0;
            match dir {
                ArrowDirection::Left | ArrowDirection::Right => {
                    if let Some(tex) = model.full_textures.get(&model.current) {
                        let [tw, _] = tex.size();
                        let disp_w = tw as f32 * model.zoom;
                        if disp_w > rect.w() {
                            if let ArrowDirection::Left = dir {
                                model.pan.x += pan_step;
                            } else {
                                model.pan.x -= pan_step;
                            }
                            let max_pan = (disp_w - rect.w()) / 2.0;
                            model.pan.x = model.pan.x.min(max_pan).max(-max_pan);
                            return true;
                        }
                    }
                }
                ArrowDirection::Up | ArrowDirection::Down => {
                    if let Some(tex) = model.full_textures.get(&model.current) {
                        let [_, th] = tex.size();
                        let disp_h = th as f32 * model.zoom;
                        if disp_h > rect.h() {
                            if let ArrowDirection::Up = dir {
                                model.pan.y -= pan_step;
                            } else {
                                model.pan.y += pan_step;
                            }
                            let max_pan = (disp_h - rect.h()) / 2.0;
                            model.pan.y = model.pan.y.min(max_pan).max(-max_pan);
                            return true;
                        }
                    }
                }
            }
            false
        }
    }
}

fn received_character(_app: &App, model: &mut Model, ch: char) {
    if ch.is_control() {
        return;
    }
    if let Some(search) = model.search.as_mut() {
        if search.focused {
            search.input.push(ch);
            search.pending_request = None;
            search.error = None;
            search.last_embedding = None;
            search.results.clear();
            search.current = 0;
        }
    }
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    if handle_search_key(app, model, key) {
        return;
    }
    let len = model.image_paths.len();
    let rect = app.window_rect();
    let cell = model.thumb_size as f32 + model.gap;
    let cols = ((rect.w() + model.gap) / cell).floor() as usize;
    let cols = cols.max(1);
    if app.keys.mods == ModifiersState::empty() {
        match key {
            // Quit on 'q'
            Key::Q => {
                // Exit the application
                app.quit();
            }
            // g/G: jump to first/last in thumbnail mode
            Key::G => {
                if let Mode::Thumbnails = model.mode {
                    let len = model.image_paths.len();
                    // if Shift+G, go to last thumbnail; otherwise go to first
                    if app.keys.mods.shift() {
                        if len > 0 {
                            model.current = len - 1;
                        }
                    } else {
                        model.current = 0;
                    }
                }
            }
            Key::N => {
                // Next image in single-image mode
                if let Mode::Single = model.mode {
                    if model.current + 1 < len {
                        navigate_to(app, model, model.current + 1);
                    }
                }
            }
            Key::P => {
                // Previous image in single-image mode
                if let Mode::Single = model.mode {
                    if model.current > 0 {
                        navigate_to(app, model, model.current - 1);
                    }
                }
            }
            // Skip 10 images forward
            Key::RBracket => {
                if let Mode::Single = model.mode {
                    let new_idx = (model.current + 10).min(len.saturating_sub(1));
                    navigate_to(app, model, new_idx);
                }
            }
            // Skip 10 images backward
            Key::LBracket => {
                if let Mode::Single = model.mode {
                    let new_idx = model.current.saturating_sub(10);
                    navigate_to(app, model, new_idx);
                }
            }
            Key::H | Key::Left => {
                if handle_arrow(app, model, ArrowDirection::Left) {
                    return;
                }
            }
            Key::L | Key::Right => {
                if handle_arrow(app, model, ArrowDirection::Right) {
                    return;
                }
            }
            Key::K | Key::Up => {
                if handle_arrow(app, model, ArrowDirection::Up) {
                    return;
                }
            }
            Key::J | Key::Down => {
                if handle_arrow(app, model, ArrowDirection::Down) {
                    return;
                }
            }
            Key::Return => {
                // Toggle between thumbnail and single-image modes.
                match model.mode {
                    Mode::Thumbnails => {
                        // Pre-load current and adjacent images, then fit
                        let len = model.image_paths.len();
                        let idx = model.current;
                        request_full_texture(model, idx);
                        if idx > 0 {
                            request_full_texture(model, idx - 1);
                        }
                        if idx + 1 < len {
                            request_full_texture(model, idx + 1);
                        }
                        // Enter single mode and fit image to window
                        model.mode = Mode::Single;
                        apply_fit(app, model);
                    }
                    Mode::Single => {
                        model.mode = Mode::Thumbnails;
                    }
                }
            }
            // Fit single image to window
            Key::W => {
                if let Mode::Single = model.mode {
                    let rect = app.window_rect();
                    if let Some(tex) = model.full_textures.get(&model.current) {
                        let [w, h] = tex.size();
                        let fit = (rect.w() / w as f32).min(rect.h() / h as f32);
                        model.zoom = fit;
                    } else {
                        model.zoom = 1.0;
                    }
                    model.pan = vec2(0.0, 0.0);
                }
            }
            // Toggle full screen
            Key::F => {
                let window = app.main_window();
                // Query current state, then toggle
                let is_fs = window.is_fullscreen();
                window.set_fullscreen(!is_fs);
            }
            // Show at 100% scale
            Key::Equals => {
                if let Mode::Single = model.mode {
                    model.zoom = 1.0;
                    model.pan = vec2(0.0, 0.0);
                }
            }
            // Clear command output display
            Key::X => {
                model.command_output = None;
            }
            _ => {}
        }
    } else if app.keys.mods == ModifiersState::SHIFT {
        match key {
            Key::G => {
                if let Mode::Thumbnails = model.mode {
                    let len = model.image_paths.len();
                    if len > 0 {
                        model.current = len - 1;
                    }
                }
            }
            _ => {}
        }
    }
    // Custom key bindings execution
    let current_file = model.image_paths[model.current]
        .to_string_lossy()
        .to_string();
    for binding in &model.key_bindings {
        if key == binding.key
            && app.keys.mods.ctrl() == binding.ctrl
            && app.keys.mods.shift() == binding.shift
            && app.keys.mods.alt() == binding.alt
            && app.keys.mods.logo() == binding.super_key
        {
            let cmd = binding.command.replace("{file}", &current_file);
            let tx = model.command_tx.clone();
            thread::spawn(move || {
                match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let mut s = stdout.to_string();
                        if !stderr.is_empty() {
                            if !s.is_empty() {
                                s.push('\n');
                            }
                            s.push_str(&stderr);
                        }
                        let _ = tx.send(s);
                    }
                    Err(e) => {
                        let _ = tx.send(format!("Failed to execute command: {}", e));
                    }
                }
            });
        }
    }

    // Auto-scroll to keep current thumbnail in view
    if let Mode::Thumbnails = model.mode {
        let row = model.current / cols;
        let y0 = rect.h() / 2.0 - (model.thumb_size as f32) / 2.0 - model.gap / 2.0;
        let target_y = y0 - (row as f32) * cell + model.scroll_offset;
        let top_bound = rect.h() / 2.0 - (model.thumb_size as f32) / 2.0 - model.gap / 2.0;
        let bottom_bound = -rect.h() / 2.0 + (model.thumb_size as f32) / 2.0 + model.gap / 2.0;
        if target_y > top_bound {
            model.scroll_offset -= target_y - top_bound;
        } else if target_y < bottom_bound {
            model.scroll_offset += bottom_bound - target_y;
        }
    }
    // On thumbnail mode selection (via keys), reset preload timer
    if let Mode::Thumbnails = model.mode {
        model.selection_changed_at = Instant::now();
        model.selection_pending = false;
    }
}

/// Update function to process incoming thumbnail images.
fn update(app: &App, model: &mut Model, _update: Update) {
    loop {
        match model.clip_engine.try_recv() {
            Ok(event) => match event {
                ClipEvent::ImageReady { index, embedding } => {
                    if let Some(slot) = model.clip_embeddings.get_mut(index) {
                        *slot = Some(embedding);
                    }
                    model.clip_missing.remove(&index);
                    model.clip_inflight.remove(&index);
                    update_search_with_image_embedding(app, model, index);
                }
                ClipEvent::ImageError { index, error } => {
                    model.clip_inflight.remove(&index);
                    if let Some(path) = model.image_paths.get(index) {
                        eprintln!(
                            "Failed to compute CLIP embedding for {}: {}",
                            path.display(),
                            error
                        );
                    } else {
                        eprintln!("Failed to compute CLIP embedding: {}", error);
                    }
                }
                ClipEvent::TextReady {
                    request_id,
                    embedding,
                } => {
                    handle_text_result(app, model, request_id, embedding);
                }
                ClipEvent::TextError { request_id, error } => {
                    if let Some(search) = model.search.as_mut() {
                        if search.pending_request == Some(request_id) {
                            search.pending_request = None;
                            search.error = Some(error);
                        }
                    }
                }
            },
            Err(crossbeam_channel::TryRecvError::Empty) => break,
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }
    }
    // Receive command output messages for display
    while let Ok(msg) = model.command_rx.try_recv() {
        model.command_output = Some(msg);
    }
    // Receive thumbnails from background thread and create textures
    // Process incoming thumbnails
    while let Ok((i, img)) = model.thumb_rx.try_recv() {
        model.thumb_pending.remove(&i);
        let tex = wgpu::Texture::from_image(app, &img);
        insert_thumbnail_texture(model, i, tex);
    }
    // Process loaded full-resolution tile data
    while let Ok((idx, full_w, full_h, tiles_data)) = model.full_resp_rx.try_recv() {
        // Store raw pixel data for lazy texture creation
        let mut tiles = Vec::new();

        for (x_offset, y_offset, width, height, pixel_data) in tiles_data {
            tiles.push(Tile {
                x_offset,
                y_offset,
                width,
                height,
                pixel_data,
                texture: RefCell::new(None),
            });
        }
        let tiled = TiledTexture {
            full_w,
            full_h,
            tiles,
        };
        // Insert into cache and update LRU
        model.full_textures.insert(idx, tiled);
        // Mark use and remove pending
        if let Some(pos) = model.full_usage.iter().position(|&i| i == idx) {
            model.full_usage.remove(pos);
        }
        model.full_usage.push_front(idx);
        model.full_pending.remove(&idx);
        // Evict least recently used if over capacity
        if model.full_usage.len() > FULL_CACHE_CAPACITY {
            if let Some(old_idx) = model.full_usage.pop_back() {
                model.full_textures.remove(&old_idx);
            }
        }
        // If this is the current image and in fit mode, resize to fit
        if idx == model.current && model.fit_mode {
            apply_fit(app, model);
        }
    }
    // Handle window resize: update view parameters and re-apply fit if in fit mode
    let rect = app.window_rect();
    if rect != model.prev_window_rect {
        model.prev_window_rect = rect;
        if let Mode::Single = model.mode {
            if model.fit_mode {
                apply_fit(app, model);
            }
        }
    }
    // Schedule preload of selected thumbnail if stable for >200ms
    if let Mode::Thumbnails = model.mode {
        if !model.selection_pending
            && model.selection_changed_at.elapsed() >= Duration::from_millis(200)
        {
            request_full_texture(model, model.current);
            model.selection_pending = true;
        }
    }
    // Clamp thumbnail scrolling to content bounds
    if let Mode::Thumbnails = model.mode {
        let cell = model.thumb_size as f32 + model.gap;
        let cols = ((rect.w() + model.gap) / cell).floor() as usize;
        let cols = cols.max(1);
        let total = model.image_paths.len();
        let rows = ((total + cols - 1) / cols) as f32;
        let max_scroll = (rows * cell - rect.h()).max(0.0);
        model.scroll_offset = model.scroll_offset.clamp(0.0, max_scroll);
    }
    if matches!(model.mode, Mode::Single) {
        if !model.full_textures.contains_key(&model.current) {
            request_full_texture(model, model.current);
        }
    }
    update_thumbnail_requests(app, model);
}
/// Ensure the full-resolution texture for `idx` is loaded and update LRU cache.
/// Request loading of full-resolution image at `idx` in background.  Adds to pending set.
fn request_full_texture(model: &mut Model, idx: usize) {
    if model.full_textures.contains_key(&idx) {
        return;
    }
    let now = Instant::now();
    let needs_request = match model.full_pending.get(&idx) {
        None => true,
        Some(&sent_at) => now.duration_since(sent_at) > FULL_PENDING_RETRY,
    };
    if needs_request {
        model.full_pending.insert(idx, now);
        if let Err(err) = model.full_req_tx.send(idx) {
            model.full_pending.remove(&idx);
            eprintln!("failed to request full image load for index {idx}: {err}");
        }
    }
}

fn insert_thumbnail_texture(model: &mut Model, idx: usize, tex: wgpu::Texture) {
    model.thumb_textures.insert(idx, tex);
    touch_thumbnail(model, idx);
    enforce_thumbnail_capacity(model);
}

fn touch_thumbnail(model: &mut Model, idx: usize) {
    if let Some(pos) = model.thumb_usage.iter().position(|&v| v == idx) {
        model.thumb_usage.remove(pos);
    }
    model.thumb_usage.push_front(idx);
}

fn enforce_thumbnail_capacity(model: &mut Model) {
    while model.thumb_textures.len() > model.thumb_capacity {
        if let Some(old_idx) = model.thumb_usage.pop_back() {
            model.thumb_textures.remove(&old_idx);
        } else {
            break;
        }
    }
}

fn update_thumbnail_requests(app: &App, model: &mut Model) {
    if !matches!(model.mode, Mode::Thumbnails) {
        return;
    }
    let total = model.image_paths.len();
    if total == 0 {
        model.thumb_capacity = 0;
        model.thumb_usage.clear();
        model.thumb_textures.clear();
        model.thumb_pending.clear();
        return;
    }
    let rect = app.window_rect();
    let visible = visible_thumbnail_indices(model, rect);
    let required = visible.len();
    let target_capacity = required
        .saturating_add(THUMB_POOL_MARGIN)
        .min(total)
        .max(required);
    if model.thumb_capacity != target_capacity {
        model.thumb_capacity = target_capacity;
        enforce_thumbnail_capacity(model);
    }
    for &idx in &visible {
        if model.thumb_textures.contains_key(&idx) {
            touch_thumbnail(model, idx);
        } else if !model.thumb_pending.contains(&idx) {
            model.thumb_pending.insert(idx);
            let _ = model.thumb_req_tx.send(idx);
        }
    }
    enforce_thumbnail_capacity(model);
}

fn visible_thumbnail_indices(model: &Model, rect: Rect) -> Vec<usize> {
    let total = model.image_paths.len();
    if total == 0 {
        return Vec::new();
    }
    let cell = model.thumb_size as f32 + model.gap;
    let mut cols = ((rect.w() + model.gap) / cell).floor() as isize;
    if cols < 1 {
        cols = 1;
    }
    let cols = cols as usize;
    let rows = (total + cols - 1) / cols;
    if rows == 0 {
        return Vec::new();
    }
    let half_gap = model.gap / 2.0;
    let scroll = model.scroll_offset;
    let row_min_f = (scroll - model.thumb_size as f32 - half_gap) / cell;
    let row_max_f = (rect.h() + scroll - half_gap) / cell;
    let mut row_min = row_min_f.ceil() as isize - THUMB_PREFETCH_ROWS as isize;
    let mut row_max = row_max_f.floor() as isize + THUMB_PREFETCH_ROWS as isize;
    let max_row = rows as isize - 1;
    if row_min < 0 {
        row_min = 0;
    }
    if row_max > max_row {
        row_max = max_row;
    }
    if row_max < row_min {
        row_max = row_min;
    }
    let mut indices = Vec::new();
    for row in row_min..=row_max {
        let base = row as usize * cols;
        for col in 0..cols {
            let idx = base + col;
            if idx >= total {
                break;
            }
            indices.push(idx);
        }
    }
    indices
}
/// Apply fit-to-window for current single-image view
fn apply_fit(app: &App, model: &mut Model) {
    model.fit_mode = true;
    let rect = app.window_rect();
    if let Some(tex) = model.full_textures.get(&model.current) {
        let [w, h] = tex.size();
        model.zoom = (rect.w() / w as f32).min(rect.h() / h as f32);
    } else {
        model.zoom = 1.0;
    }
    model.pan = vec2(0.0, 0.0);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    let rect = app.window_rect();
    match model.mode {
        Mode::Thumbnails => {
            let cell = model.thumb_size as f32 + model.gap;
            let cols = ((rect.w() + model.gap) / cell).floor() as usize;
            let cols = cols.max(1);
            // Only draw thumbnails within the visible viewport rows
            let total = model.image_paths.len();
            let rows = total.div_ceil(cols);
            let half_gap = model.gap / 2.0;
            let scroll = model.scroll_offset;
            // Compute visible row range
            let row_min_f = (scroll - model.thumb_size as f32 - half_gap) / cell;
            let row_min = row_min_f.ceil().max(0.0) as usize;
            let row_max_f = (rect.h() + scroll - half_gap) / cell;
            let row_max = row_max_f.floor().min(rows.saturating_sub(1) as f32) as usize;
            for row in row_min..=row_max {
                let base_y = rect.h() / 2.0
                    - (model.thumb_size as f32) / 2.0
                    - half_gap
                    - (row as f32) * cell;
                for col in 0..cols {
                    let i = row * cols + col;
                    if i >= total {
                        break;
                    }
                    let x = -rect.w() / 2.0
                        + (model.thumb_size as f32) / 2.0
                        + half_gap
                        + (col as f32) * cell;
                    let y = base_y + scroll;
                    if let Some(tex) = model.thumb_textures.get(&i) {
                        let [tw, th] = tex.size();
                        let w = tw as f32;
                        let h = th as f32;
                        draw.texture(tex).x_y(x, y).w_h(w, h);
                        if model.thumb_has_xmp.get(i).copied().unwrap_or(false) {
                            let icon_w = 38.0;
                            let icon_h = 18.0;
                            let margin = 6.0;
                            let icon_center_x = x + w / 2.0 - icon_w / 2.0 - margin;
                            let icon_center_y = y + h / 2.0 - icon_h / 2.0 - margin;
                            draw.rect()
                                .x_y(icon_center_x, icon_center_y)
                                .w_h(icon_w, icon_h)
                                .color(srgba(0.0, 0.0, 0.0, 0.65));
                            draw.text("XMP")
                                .font_size(12)
                                .w_h(icon_w, icon_h)
                                .x_y(icon_center_x, icon_center_y - 1.0)
                                .color(WHITE);
                        }
                        if i == model.current {
                            draw.rect()
                                .x_y(x, y)
                                .w_h(w + 4.0, h + 4.0)
                                .no_fill()
                                .stroke(WHITE)
                                .stroke_weight(2.0);
                        }
                    } else {
                        let thumb_w = model.thumb_size as f32;
                        let thumb_h = model.thumb_size as f32;
                        draw.rect()
                            .x_y(x, y)
                            .w_h(thumb_w, thumb_h)
                            .color(srgba(0.5, 0.5, 0.5, 1.0));
                        if i == model.current {
                            draw.rect()
                                .x_y(x, y)
                                .w_h(thumb_w + 4.0, thumb_h + 4.0)
                                .no_fill()
                                .stroke(WHITE)
                                .stroke_weight(2.0);
                        }
                    }
                }
            }
            // Bottom info bar in thumbnail mode: filename and index/total
            let bar_h = 20.0;
            let bar_y = -rect.h() / 2.0 + bar_h / 2.0;
            // Background
            draw.rect()
                .x_y(0.0, bar_y)
                .w_h(rect.w(), bar_h)
                .color(srgba(0.0, 0.0, 0.0, 0.5));
            let full_path = model.image_paths[model.current].to_string_lossy();
            draw.text(&full_path)
                .font_size(14)
                .w_h(rect.w(), bar_h)
                .x_y(0.0, bar_y)
                .left_justify()
                .color(WHITE);
            // Index of selected image
            let count = format!("{}/{}", model.current + 1, model.image_paths.len());
            draw.text(&count)
                .font_size(14)
                .w_h(rect.w(), bar_h)
                .x_y(0.0, bar_y)
                .right_justify()
                .color(WHITE);
        }
        Mode::Single => {
            // Attempt to draw the full-resolution tiled texture if loaded;
            // otherwise display a loading message.
            if let Some(tex) = model.full_textures.get(&model.current) {
                // Draw each tile at the correct position, applying zoom and pan
                let [full_w, full_h] = tex.size();
                for tile in &tex.tiles {
                    // Compute tile center relative to full image center
                    let x_center =
                        tile.x_offset as f32 - full_w as f32 / 2.0 + tile.width as f32 / 2.0;
                    let y_center =
                        full_h as f32 / 2.0 - tile.y_offset as f32 - tile.height as f32 / 2.0;
                    // Lazy-create GPU texture if needed
                    if tile.texture.borrow().is_none() {
                        let size = wgpu::Extent3d {
                            width: tile.width,
                            height: tile.height,
                            depth_or_array_layers: 1,
                        };
                        let descriptor = wgpu::TextureDescriptor {
                            label: None,
                            size,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_DST,
                            view_formats: &[],
                        };
                        let handle = app.main_window().device().create_texture(&descriptor);
                        app.main_window().queue().write_texture(
                            wgpu::ImageCopyTexture {
                                texture: &handle,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            &tile.pixel_data,
                            wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * tile.width),
                                rows_per_image: Some(tile.height),
                            },
                            size,
                        );
                        let n_texture =
                            wgpu::Texture::from_handle_and_descriptor(Arc::new(handle), descriptor);
                        *tile.texture.borrow_mut() = Some(n_texture);
                    }
                    let n_texture = tile.texture.borrow().as_ref().unwrap().clone();
                    draw.texture(&n_texture)
                        .x_y(
                            model.pan.x + x_center * model.zoom,
                            model.pan.y + y_center * model.zoom,
                        )
                        .w_h(
                            tile.width as f32 * model.zoom,
                            tile.height as f32 * model.zoom,
                        );
                }
                // Draw bottom info bar with full path, dimensions, and zoom
                let bar_h = 20.0;
                let bar_y = -rect.h() / 2.0 + bar_h / 2.0;
                // Background
                draw.rect()
                    .x_y(0.0, bar_y)
                    .w_h(rect.w(), bar_h)
                    .color(srgba(0.0, 0.0, 0.0, 0.5));
                // Full path, left-aligned
                let full_path = model.image_paths[model.current].to_string_lossy();
                draw.text(&full_path)
                    .font_size(14)
                    .color(WHITE)
                    .w_h(rect.w(), bar_h)
                    .x_y(0.0, bar_y)
                    .left_justify();
                // Dimensions and zoom, right-aligned
                let info = format!("{}×{}  {:.2}×", full_w, full_h, model.zoom);
                draw.text(&info)
                    .font_size(14)
                    .color(WHITE)
                    .w_h(rect.w(), bar_h)
                    .x_y(0.0, bar_y)
                    .right_justify();
            } else {
                draw.text("Loading...")
                    .font_size(24)
                    .color(WHITE)
                    .x_y(0.0, 0.0);
                // Draw bottom info bar with full path, dimensions, and zoom
                let bar_h = 20.0;
                let bar_y = -rect.h() / 2.0 + bar_h / 2.0;
                // Background
                draw.rect()
                    .x_y(0.0, bar_y)
                    .w_h(rect.w(), bar_h)
                    .color(srgba(0.0, 0.0, 0.0, 0.5));
                // Full path, left-aligned
                let full_path = model.image_paths[model.current].to_string_lossy();
                draw.text(&full_path)
                    .font_size(14)
                    .color(WHITE)
                    .w_h(rect.w(), bar_h)
                    .x_y(0.0, bar_y)
                    .left_justify();
            }
        }
    }

    if let Some(search) = &model.search {
        let prompt = if search.focused {
            format!("/{}_", search.input)
        } else {
            format!("/{}", search.input)
        };
        let mut status_parts = Vec::new();
        if let Some(err) = &search.error {
            status_parts.push(err.clone());
        } else if search.pending_request.is_some() {
            status_parts.push("searching…".to_string());
        } else if !search.results.is_empty() {
            status_parts.push(format!(
                "match {}/{}",
                search.current + 1,
                search.results.len()
            ));
        }
        let pending = model.clip_missing.len() + model.clip_inflight.len();
        if pending > 0 {
            status_parts.push(format!(
                "pending embeddings: {} ({})",
                pending,
                model.clip_engine.device_kind()
            ));
        }
        let status = status_parts.join(" | ");
        let bar_h = 28.0;
        let bar_y = rect.top() - bar_h / 2.0 - 10.0;
        let bg = if search.focused {
            srgba(0.2549, 0.2039, 0.3490, 0.9)
        } else {
            srgba(0.0471, 0.0471, 0.0471, 0.9)
        };
        draw.rect().x_y(0.0, bar_y).w_h(rect.w(), bar_h).color(bg);
        draw.text(&prompt)
            .font_size(16)
            .color(WHITE)
            .w_h(rect.w(), bar_h)
            .x_y(0.0, bar_y)
            .left_justify();
        draw.text(&status)
            .font_size(14)
            .color(WHITE)
            .w_h(rect.w(), bar_h)
            .x_y(0.0, bar_y)
            .right_justify();
    }

    // Draw command output overlay if present
    if let Some(ref out) = model.command_output {
        let box_height = rect.h() / 2.0;
        let box_center_y = rect.h() / 4.0;
        // Semi-transparent background
        draw.rect()
            .x_y(0.0, box_center_y)
            .w_h(rect.w(), box_height)
            .color(srgba(0.0, 0.0, 0.0, 0.8));
        let lines: Vec<&str> = out.lines().collect();
        let font_size = 16;
        let margin = 10.0;
        let line_spacing = 2.0;
        let mut y = rect.h() / 2.0 - margin - (font_size as f32) / 2.0;
        let text_width = rect.w() - 2.0 * margin;
        for line in lines {
            if y < 0.0 {
                break;
            }
            draw.text(line)
                .font_size(font_size)
                .w_h(text_width, font_size as f32)
                .x_y(0.0, y)
                .left_justify()
                .color(WHITE);
            y -= font_size as f32 + line_spacing;
        }
    }
    draw.to_frame(app, &frame).unwrap();
}
