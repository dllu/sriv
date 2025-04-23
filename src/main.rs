use anyhow::Result;
use image::imageops::FilterType;
use nannou::event::{MouseButton, MouseScrollDelta, TouchPhase, Update};
use nannou::image::imageops::crop_imm;
use nannou::image::{self, DynamicImage, GenericImageView, RgbaImage};
use nannou::prelude::*;
use nannou::wgpu;
use sha1::Sha1;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use crossbeam_channel::{unbounded, Sender as CbSender, Receiver as CbReceiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use toml::Value as TomlValue;
/// Maximum number of full-resolution images to cache in memory.
const FULL_CACHE_CAPACITY: usize = 64;

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
            for (i, tex_opt) in model.thumb_textures.iter().enumerate() {
                let row = i / cols;
                let col = i % cols;
                let x = -rect.w() / 2.0
                    + (model.thumb_size as f32) / 2.0
                    + model.gap / 2.0
                    + (col as f32) * cell;
                let y = rect.h() / 2.0
                    - (model.thumb_size as f32) / 2.0
                    - model.gap / 2.0
                    - (row as f32) * cell
                    + model.scroll_offset;
                let width = tex_opt
                    .as_ref()
                    .map(|t| t.size()[0] as f32)
                    .unwrap_or(model.thumb_size as f32);
                let height = tex_opt
                    .as_ref()
                    .map(|t| t.size()[1] as f32)
                    .unwrap_or(model.thumb_size as f32);
                let x_min = x - width / 2.0;
                let x_max = x + width / 2.0;
                let y_min = y - height / 2.0;
                let y_max = y + height / 2.0;
                if pos.x >= x_min && pos.x <= x_max && pos.y >= y_min && pos.y <= y_max {
                    model.current = i;
                    // Reset preload timer on click selection
                    model.selection_changed_at = Instant::now();
                    model.selection_pending = false;
                    break;
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
            let total = model.thumb_textures.len();
            let rows = ((total + cols - 1) / cols) as f32;
            // Maximum scroll to show last row at bottom
            let max_scroll = (rows * cell - rect.h()).max(0.0);
            model.scroll_offset = model.scroll_offset.clamp(0.0, max_scroll);
            // Update view parameters for thumbnail prioritization
            if let Ok(mut vp) = model.view_params.lock() {
                vp.2 = model.scroll_offset;
            }
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
            let new_zoom = (old_zoom * zoom_factor).clamp(0.1, 10.0);
            // Adjust pan so the point under cursor stays fixed
            model.pan = mouse_pos + (model.pan - mouse_pos) * (new_zoom / old_zoom);
            model.zoom = new_zoom;
        }
    }
}

/// A single tile of a full-resolution image used to bypass GPU texture size limits.
#[derive(Debug)]
struct Tile {
    /// The GPU texture for this tile.
    texture: wgpu::Texture,
    /// Pixel offset from the left edge of the full image.
    x_offset: u32,
    /// Pixel offset from the top edge of the full image.
    y_offset: u32,
    /// Width of this tile in pixels.
    width: u32,
    /// Height of this tile in pixels.
    height: u32,
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
struct Model {
    image_paths: Vec<PathBuf>,
    thumb_textures: Vec<Option<wgpu::Texture>>,
    thumb_rx: Receiver<(usize, DynamicImage)>,
    // View parameters (window width, height, scroll offset) for prioritization
    view_params: Arc<Mutex<(f32, f32, f32)>>,
    // Channels for full-resolution image loading
    full_req_tx: CbSender<usize>,
    full_resp_rx: CbReceiver<(usize, u32, u32, Vec<(u32, u32, u32, u32, Vec<u8>)>)>,
    // Track indices currently requested but not yet loaded
    full_pending: HashSet<usize>,
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
}
/// Compute the cache path for an image based on a SHA1 of its path.
/// The cache layout is: cache_base/<first 3 hex chars>/<remaining hex chars>.png
fn thumbnail_cache_path(cache_base: &Path, image_path: &Path) -> PathBuf {
    let hex = {
        let path_str = image_path.to_string_lossy();
        let mut hasher = Sha1::new();
        hasher.update(path_str.as_bytes());
        hasher.digest().to_string()
    };
    let shard = &hex[..3];
    let name = &hex[3..];
    cache_base.join(shard).join(format!("{}.png", name))
}

/// The model function for initializing the application state.
fn model(app: &App) -> Model {
    // Parse command-line arguments: files or directories.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: sriv-rs <image files or directories>...");
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
    // Prepare thumbnail size, gap, and cache base directory.
    let thumb_size: u32 = 256;
    // Gap between thumbnails
    let gap: f32 = 10.0;
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
    // Create the window first, so textures can reference a focused window.
    let _window = app
        .new_window()
        .size(800, 600)
        .title("sriv")
        .view(view)
        .key_pressed(key_pressed)
        .mouse_wheel(mouse_wheel)
        .mouse_pressed(mouse_pressed)
        .build()
        .unwrap();
    // Channel for receiving thumbnails from background thread.
    let (tx, rx) = channel::<(usize, DynamicImage)>();
    // Prioritized thumbnail generation: shared queue and view parameters
    let rect0 = app.window_rect();
    let thumb_queue = Arc::new(Mutex::new(VecDeque::new()));
    let view_params = Arc::new(Mutex::new((rect0.w(), rect0.h(), 0.0_f32)));
    // Flag to signal completion of initial thumbnail enqueue scan
    let scanning_done: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    // Kick off first-pass scanning on a background thread: load cached thumbnails & queue misses
    {
        let paths = image_paths.clone();
        let cache_base = cache_base.clone();
        let thumb_queue = Arc::clone(&thumb_queue);
        let tx = tx.clone();
        let scanning_done = Arc::clone(&scanning_done);
        thread::spawn(move || {
            for (i, p) in paths.iter().enumerate() {
                let cache_path = thumbnail_cache_path(&cache_base, p);
                let mut loaded = false;
                if let (Ok(meta_orig), Ok(meta_cache)) =
                    (fs::metadata(p), fs::metadata(&cache_path))
                {
                    if let (Ok(orig_mtime), Ok(cache_mtime)) =
                        (meta_orig.modified(), meta_cache.modified())
                    {
                        if cache_mtime >= orig_mtime {
                            if let Ok(img) = image::open(&cache_path) {
                                let _ = tx.send((i, img));
                                loaded = true;
                            }
                        }
                    }
                }
                if !loaded {
                    thumb_queue.lock().unwrap().push_back(i);
                }
            }
            // Signal that initial scanning and queuing is complete
            scanning_done.store(true, Ordering::SeqCst);
        });
    }
    // Spawn worker threads to generate missing thumbnails by priority
    let num_workers = rayon::current_num_threads().clamp(1, 8);
    for _ in 0..num_workers {
        let paths = image_paths.clone();
        let cache_base = cache_base.clone();
        let thumb_queue = Arc::clone(&thumb_queue);
        let tx = tx.clone();
        let scanning_done = Arc::clone(&scanning_done);
        // Clone view parameters and gap for prioritization
        let view_params = Arc::clone(&view_params);
        thread::spawn(move || {
            loop {
                let idx_opt = {
                    let mut q = thumb_queue.lock().unwrap();
                    if q.is_empty() {
                        // If scanning is complete and no more tasks, exit
                        if scanning_done.load(Ordering::SeqCst) {
                            break;
                        }
                        // Wait briefly for tasks
                        drop(q);
                        thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    // Prioritize thumbnails near viewport center
                    if q.len() > 1 {
                        let (rect_w, rect_h, scroll_offset) = *view_params.lock().unwrap();
                        let cell = thumb_size as f32 + gap;
                        let cols = ((rect_w + gap) / cell).floor() as usize;
                        let y0 = rect_h / 2.0 - thumb_size as f32 / 2.0 - gap / 2.0;
                        let slice = q.make_contiguous();
                        slice.sort_by(|&i1, &i2| {
                            let row1 = (i1 / cols) as f32;
                            let y1 = y0 - row1 * cell + scroll_offset;
                            let row2 = (i2 / cols) as f32;
                            let y2 = y0 - row2 * cell + scroll_offset;
                            y1.abs()
                                .partial_cmp(&y2.abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    q.pop_front()
                };
                if let Some(i) = idx_opt {
                    let p = &paths[i];
                    let cache_path = thumbnail_cache_path(&cache_base, p);
                    if let Ok(img) = image::open(p) {
                        let mut thumb = img.thumbnail(thumb_size, thumb_size);
                        let (w0, h0) = thumb.dimensions();
                        if w0 != 0 && h0 != 0 {
                            let w = if w0 < 2 { 2 } else { w0 };
                            let h = if h0 < 2 { 2 } else { h0 };
                            if w != w0 || h != h0 {
                                thumb = thumb.resize_exact(w, h, FilterType::Nearest);
                            }
                            if let Some(parent) = cache_path.parent() {
                                let _ = fs::create_dir_all(parent);
                            }
                            let _ = thumb.save(&cache_path);
                            let _ = tx.send((i, thumb));
                        }
                    }
                }
            }
        });
    }
    // Initialize thumbnail texture placeholders.
    let thumb_textures: Vec<Option<wgpu::Texture>> = (0..image_paths.len()).map(|_| None).collect();
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
                        if let Ok(img) = image::open(path) {
                            let rgba = img.to_rgba8();
                            let full_w = rgba.width();
                            let full_h = rgba.height();
                            const MAX_TILE_SIZE: u32 = 8192;
                            let mut tiles_data = Vec::new();
                            for y in (0..full_h).step_by(MAX_TILE_SIZE as usize) {
                                for x in (0..full_w).step_by(MAX_TILE_SIZE as usize) {
                                    let tile_w = (full_w - x).min(MAX_TILE_SIZE);
                                    let tile_h = (full_h - y).min(MAX_TILE_SIZE);
                                    let sub_image: RgbaImage = crop_imm(&rgba, x, y, tile_w, tile_h).to_image();
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
    let full_pending: HashSet<usize> = HashSet::new();
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
    Model {
        image_paths,
        thumb_textures,
        thumb_rx: rx,
        view_params,
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
    }
}

fn main() -> Result<()> {
    // Launch the nannou application with our model initializer and update callback.
    nannou::app(model).update(update).run();
    Ok(())
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    let len = model.image_paths.len();
    let rect = app.window_rect();
    let cell = model.thumb_size as f32 + model.gap;
    let cols = ((rect.w() + model.gap) / cell).floor() as usize;
    let cols = cols.max(1);
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
            // navigate to next image in single-image mode
            if let Mode::Single = model.mode {
                if model.current < len - 1 {
                    model.current += 1;
                    // Pre-load current and adjacent images
                    let idx = model.current;
                    request_full_texture(model, idx);
                    if idx > 0 {
                        request_full_texture(model, idx - 1);
                    }
                    if idx + 1 < len {
                        request_full_texture(model, idx + 1);
                    }
                    // If already loaded, fit the image to window
                    if model.full_textures.contains_key(&idx) {
                        apply_fit(app, model);
                    }
                }
            }
        }
        Key::P => {
            // navigate to previous image in single-image mode
            if let Mode::Single = model.mode {
                if model.current > 0 {
                    model.current -= 1;
                    // Load current and pre-load neighbors
                    request_full_texture(model, model.current);
                    let idx = model.current;
                    if idx > 0 {
                        request_full_texture(model, idx - 1);
                    }
                    if idx + 1 < len {
                        request_full_texture(model, idx + 1);
                    }
                    // If already loaded, fit the image to window
                    if model.full_textures.contains_key(&idx) {
                        apply_fit(app, model);
                    }
                }
            }
        }
        // Skip 10 images forward
        Key::RBracket => {
            if let Mode::Single = model.mode {
                let len = model.image_paths.len();
                let new_idx = (model.current + 10).min(len.saturating_sub(1));
                model.current = new_idx;
                // Request current and neighbors
                request_full_texture(model, new_idx);
                if new_idx > 0 {
                    request_full_texture(model, new_idx - 1);
                }
                if new_idx + 1 < len {
                    request_full_texture(model, new_idx + 1);
                }
                // If already loaded, fit the image to window
                if model.full_textures.contains_key(&new_idx) {
                    apply_fit(app, model);
                }
            }
        }
        // Skip 10 images backward
        Key::LBracket => {
            if let Mode::Single = model.mode {
                let new_idx = model.current.saturating_sub(10);
                model.current = new_idx;
                // Request current and neighbors
                request_full_texture(model, new_idx);
                if new_idx > 0 {
                    request_full_texture(model, new_idx - 1);
                }
                if new_idx + 1 < model.image_paths.len() {
                    request_full_texture(model, new_idx + 1);
                }
                // If already loaded, fit the image to window
                if model.full_textures.contains_key(&new_idx) {
                    apply_fit(app, model);
                }
            }
        }
        Key::H | Key::Left => match model.mode {
            Mode::Thumbnails => {
                let row = model.current / cols;
                let col = model.current % cols;
                let new_col = if col == 0 { cols - 1 } else { col - 1 };
                let idx = row * cols + new_col;
                if idx < len {
                    model.current = idx;
                }
            }
            Mode::Single => {
                // Pan left if zoomed wider than view
                if let Some(tex) = model.full_textures.get(&model.current) {
                    let [tw, _] = tex.size();
                    let disp_w = tw as f32 * model.zoom;
                    if disp_w > rect.w() {
                        let pan_step = 20.0;
                        model.pan.x += pan_step;
                        let max_pan = (disp_w - rect.w()) / 2.0;
                        model.pan.x = model.pan.x.min(max_pan).max(-max_pan);
                        return;
                    }
                }
            }
        },
        Key::L | Key::Right => match model.mode {
            Mode::Thumbnails => {
                let row = model.current / cols;
                let col = model.current % cols;
                let new_col = if col + 1 >= cols { 0 } else { col + 1 };
                let idx = row * cols + new_col;
                if idx < len {
                    model.current = idx;
                }
            }
            Mode::Single => {
                // Pan right if zoomed wider than view
                if let Some(tex) = model.full_textures.get(&model.current) {
                    let [tw, _] = tex.size();
                    let disp_w = tw as f32 * model.zoom;
                    if disp_w > rect.w() {
                        let pan_step = 20.0;
                        model.pan.x -= pan_step;
                        let max_pan = (disp_w - rect.w()) / 2.0;
                        model.pan.x = model.pan.x.min(max_pan).max(-max_pan);
                        return;
                    }
                }
            }
        },
        Key::K | Key::Up => match model.mode {
            Mode::Thumbnails => {
                let row = if model.current < cols {
                    (len - 1) / cols
                } else {
                    model.current / cols - 1
                };
                let col = model.current % cols;
                let idx = row * cols + col;
                if idx < len {
                    model.current = idx;
                }
            }
            Mode::Single => {
                // Pan up if zoomed taller than view
                if let Some(tex) = model.full_textures.get(&model.current) {
                    let [_, th] = tex.size();
                    let disp_h = th as f32 * model.zoom;
                    if disp_h > rect.h() {
                        let pan_step = 20.0;
                        model.pan.y -= pan_step;
                        let max_pan = (disp_h - rect.h()) / 2.0;
                        model.pan.y = model.pan.y.min(max_pan).max(-max_pan);
                        return;
                    }
                }
            }
        },
        Key::J | Key::Down => match model.mode {
            Mode::Thumbnails => {
                let row = model.current / cols + 1;
                let idx = row * cols + (model.current % cols);
                if idx < len {
                    model.current = idx;
                }
            }
            Mode::Single => {
                // Pan down if zoomed taller than view
                if let Some(tex) = model.full_textures.get(&model.current) {
                    let [_, th] = tex.size();
                    let disp_h = th as f32 * model.zoom;
                    if disp_h > rect.h() {
                        let pan_step = 20.0;
                        model.pan.y += pan_step;
                        let max_pan = (disp_h - rect.h()) / 2.0;
                        model.pan.y = model.pan.y.min(max_pan).max(-max_pan);
                        return;
                    }
                }
            }
        },
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
    // Update view parameters after auto-scroll
    if let Ok(mut vp) = model.view_params.lock() {
        vp.2 = model.scroll_offset;
    }
    // On thumbnail mode selection (via keys), reset preload timer
    if let Mode::Thumbnails = model.mode {
        model.selection_changed_at = Instant::now();
        model.selection_pending = false;
    }
}

/// Update function to process incoming thumbnail images.
fn update(app: &App, model: &mut Model, _update: Update) {
    // Receive command output messages for display
    while let Ok(msg) = model.command_rx.try_recv() {
        model.command_output = Some(msg);
    }
    // Receive thumbnails from background thread and create textures
    // Process incoming thumbnails
    while let Ok((i, img)) = model.thumb_rx.try_recv() {
        let tex = wgpu::Texture::from_image(app, &img);
        model.thumb_textures[i] = Some(tex);
    }
    // Process loaded full-resolution tile data
    while let Ok((idx, full_w, full_h, tiles_data)) = model.full_resp_rx.try_recv() {
        // Build GPU textures manually from raw tile pixel data
        let mut tiles = Vec::new();

        let mut write_texture = Duration::from_millis(0);
        for (x_offset, y_offset, width, height, pixel_data) in tiles_data {
            // Create raw GPU texture handle and upload pixel data
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let descriptor = wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            };
            // Create the raw texture handle
            let handle = app.main_window().device().create_texture(&descriptor);
            // Upload pixel data into the texture
            let tic = Instant::now();
            app.main_window().queue().write_texture(
                wgpu::ImageCopyTexture {
                    texture: &handle,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &pixel_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                size,
            );
            write_texture += tic.elapsed();
            // Wrap the raw handle and descriptor into a nannou Texture
            let texture = wgpu::Texture::from_handle_and_descriptor(Arc::new(handle), descriptor);
            tiles.push(Tile {
                texture,
                x_offset,
                y_offset,
                width,
                height,
            });
        }
        println!("Timing: {:?}", write_texture);
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
        // Update view parameters for thumbnail prioritization
        if let Ok(mut vp) = model.view_params.lock() {
            vp.0 = rect.w();
            vp.1 = rect.h();
            vp.2 = model.scroll_offset;
        }
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
        let total = model.thumb_textures.len();
        let rows = ((total + cols - 1) / cols) as f32;
        let max_scroll = (rows * cell - rect.h()).max(0.0);
        model.scroll_offset = model.scroll_offset.clamp(0.0, max_scroll);
    }
}
/// Ensure the full-resolution texture for `idx` is loaded and update LRU cache.
/// Request loading of full-resolution image at `idx` in background.  Adds to pending set.
fn request_full_texture(model: &mut Model, idx: usize) {
    if !model.full_textures.contains_key(&idx) && !model.full_pending.contains(&idx) {
        model.full_pending.insert(idx);
        let _ = model.full_req_tx.send(idx);
    }
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
            let total = model.thumb_textures.len();
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
                    let tex_opt = &model.thumb_textures[i];
                    match tex_opt {
                        Some(tex) => {
                            let [tw, th] = tex.size();
                            let w = tw as f32;
                            let h = th as f32;
                            draw.texture(tex).x_y(x, y).w_h(w, h);
                            if i == model.current {
                                draw.rect()
                                    .x_y(x, y)
                                    .w_h(w + 4.0, h + 4.0)
                                    .no_fill()
                                    .stroke(WHITE)
                                    .stroke_weight(2.0);
                            }
                        }
                        None => {
                            draw.rect()
                                .x_y(x, y)
                                .w_h(model.thumb_size as f32, model.thumb_size as f32)
                                .color(srgba(0.5, 0.5, 0.5, 1.0));
                            if i == model.current {
                                draw.rect()
                                    .x_y(x, y)
                                    .w_h(
                                        model.thumb_size as f32 + 4.0,
                                        model.thumb_size as f32 + 4.0,
                                    )
                                    .no_fill()
                                    .stroke(WHITE)
                                    .stroke_weight(2.0);
                            }
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
                    draw.texture(&tile.texture)
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
