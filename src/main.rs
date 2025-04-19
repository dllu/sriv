use anyhow::Result;
use nannou::image::{self, DynamicImage, GenericImageView, RgbaImage};
use image::imageops::FilterType;
use rayon::prelude::*;
use nannou::prelude::*;
use nannou::event::{MouseScrollDelta, TouchPhase, Update};
use nannou::image::imageops::crop_imm;
use std::fs;
use std::path::{PathBuf, Component};
use std::sync::mpsc::{channel, Receiver};
use std::thread;

/// The display mode of the viewer.
#[derive(Debug)]
enum Mode {
    Thumbnails,
    Single,
}

/// Mouse wheel scroll handler to scroll thumbnails in thumbnail view.
fn mouse_wheel(app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    match model.mode {
        Mode::Thumbnails => {
            // Determine scroll amount: line vs pixel delta
            let scroll_amount = match delta {
                MouseScrollDelta::LineDelta(_x, y) => y * 20.0,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
            };
            model.scroll_offset += scroll_amount;
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
            let new_zoom = (old_zoom * zoom_factor).max(0.1).min(10.0);
            // Adjust pan so the point under cursor stays fixed
            model.pan = mouse_pos + (model.pan - mouse_pos) * (new_zoom / old_zoom);
            model.zoom = new_zoom;
        }
        _ => {}
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
    /// Create a tiled texture from a full-resolution image.
    fn new(app: &App, img: &DynamicImage) -> Self {
        // Convert image to RGBA8 for consistent pixel format.
        let rgba = img.to_rgba8();
        let full_w = rgba.width();
        let full_h = rgba.height();
        // Maximum texture dimension per tile (adjust as needed for GPU limits).
        const MAX_TILE_SIZE: u32 = 8192;
        let mut tiles = Vec::new();
        // Split image into tiles of size at most MAX_TILE_SIZE.
        for y in (0..full_h).step_by(MAX_TILE_SIZE as usize) {
            for x in (0..full_w).step_by(MAX_TILE_SIZE as usize) {
                let tile_w = (full_w - x).min(MAX_TILE_SIZE);
                let tile_h = (full_h - y).min(MAX_TILE_SIZE);
                // Crop the sub-image.
                let sub_image: RgbaImage = crop_imm(&rgba, x, y, tile_w, tile_h).to_image();
                // Wrap in DynamicImage for texture creation.
                let dyn_img = DynamicImage::ImageRgba8(sub_image);
                // Create GPU texture from sub-image.
                let texture = wgpu::Texture::from_image(app, &dyn_img);
                tiles.push(Tile {
                    texture,
                    x_offset: x,
                    y_offset: y,
                    width: tile_w,
                    height: tile_h,
                });
            }
        }
        TiledTexture { full_w, full_h, tiles }
    }

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
    // Full-resolution images as tiled textures, loaded lazily on demand.
    full_textures: Vec<Option<TiledTexture>>,
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
    // Prepare thumbnail size and cache base directory.
    let thumb_size = 256;
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
    let _window = app.new_window()
        .size(800, 600)
        .view(view)
        .key_pressed(key_pressed)
        .mouse_wheel(mouse_wheel)
        .build()
        .unwrap();
    // Channel for receiving thumbnails from background thread.
    let (tx, rx) = channel::<(usize, DynamicImage)>();
    // Spawn background thread to generate thumbnails in parallel.
    {
        let paths = image_paths.clone();
        let thumb_size = thumb_size;
        let cache_base = cache_base.clone();
        thread::spawn(move || {
            paths.par_iter().enumerate().for_each(|(i, p)| {
                // Compute cache thumbnail path (mirroring full path under cache_base, with .png extension).
                let rel = p.components()
                    .filter_map(|c| match c {
                        Component::Normal(os) => Some(os),
                        _ => None,
                    })
                    .collect::<PathBuf>();
                let mut cache_path = cache_base.join(&rel);
                cache_path.set_extension("png");
                // Attempt to load a valid cached thumbnail.
                let thumb_opt = if let (Ok(meta_orig), Ok(meta_cache)) = (fs::metadata(p), fs::metadata(&cache_path)) {
                    if let (Ok(orig_mtime), Ok(cache_mtime)) = (meta_orig.modified(), meta_cache.modified()) {
                        if cache_mtime >= orig_mtime {
                            if let Ok(img) = image::open(&cache_path) {
                                Some(img)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }.or_else(|| {
                    // Generate a new thumbnail.
                    if let Ok(img) = image::open(p) {
                        let mut thumb = img.thumbnail(thumb_size, thumb_size);
                        let (w0, h0) = thumb.dimensions();
                        if w0 == 0 || h0 == 0 {
                            None
                        } else {
                            let w = if w0 < 2 { 2 } else { w0 };
                            let h = if h0 < 2 { 2 } else { h0 };
                            if w != w0 || h != h0 {
                                thumb = thumb.resize_exact(w, h, FilterType::Nearest);
                            }
                            if let Some(parent) = cache_path.parent() {
                                let _ = fs::create_dir_all(parent);
                            }
                            let _ = thumb.save(&cache_path);
                            Some(thumb)
                        }
                    } else {
                        None
                    }
                });
                if let Some(thumb) = thumb_opt {
                    let _ = tx.send((i, thumb));
                }
            });
        });
    }
    // Initialize thumbnail texture placeholders.
    let thumb_textures: Vec<Option<wgpu::Texture>> = (0..image_paths.len())
        .map(|_| None)
        .collect();
    // Initialize placeholders for full-resolution tiled textures (lazy loading).
    let full_textures: Vec<Option<TiledTexture>> = (0..image_paths.len())
        .map(|_| None)
        .collect();
    Model {
        image_paths,
        thumb_textures,
        thumb_rx: rx,
        full_textures,
        mode: Mode::Thumbnails,
        current: 0,
        thumb_size,
        gap: 10.0,
        scroll_offset: 0.0,
        zoom: 1.0,
        pan: vec2(0.0, 0.0),
    }
}

fn main() -> Result<()> {
    // Launch the nannou application with our model initializer and update callback.
    nannou::app(model)
        .update(update)
        .run();
    Ok(())
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    let len = model.image_paths.len();
    let rect = app.window_rect();
    let cell = model.thumb_size as f32 + model.gap;
    let cols = ((rect.w() + model.gap) / cell).floor() as usize;
    let cols = cols.max(1);
    match key {
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
                if let Some(tex) = &model.full_textures[model.current] {
                    let [tw, th] = tex.size();
                    let disp_w = tw as f32 * model.zoom;
                    if disp_w > rect.w() {
                        let pan_step = 20.0;
                        model.pan.x += pan_step;
                        let max_pan = (disp_w - rect.w()) / 2.0;
                        model.pan.x = model.pan.x.min(max_pan).max(-max_pan);
                        return;
                    }
                }
                // Navigate to previous image
                model.current = if model.current == 0 { len - 1 } else { model.current - 1 };
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
                if let Some(tex) = &model.full_textures[model.current] {
                    let [tw, th] = tex.size();
                    let disp_w = tw as f32 * model.zoom;
                    if disp_w > rect.w() {
                        let pan_step = 20.0;
                        model.pan.x -= pan_step;
                        let max_pan = (disp_w - rect.w()) / 2.0;
                        model.pan.x = model.pan.x.min(max_pan).max(-max_pan);
                        return;
                    }
                }
                // Navigate to next image
                model.current = (model.current + 1) % len;
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
                if let Some(tex) = &model.full_textures[model.current] {
                    let [tw, th] = tex.size();
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
            _ => {}
        }
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
                if let Some(tex) = &model.full_textures[model.current] {
                    let [tw, th] = tex.size();
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
            _ => {}
        }
        Key::Return => {
            // Toggle between thumbnail view and single-image view.
            match model.mode {
                Mode::Thumbnails => {
                    // On entering single mode, load and pad the full-resolution image to ensure 2D texture.
                    if model.full_textures[model.current].is_none() {
                        let path = &model.image_paths[model.current];
                        match image::open(path) {
                            Ok(mut img) => {
                                // Ensure minimum dimensions to avoid 1D textures
                                let (w0, h0) = img.dimensions();
                                let w = w0.max(2);
                                let h = h0.max(2);
                                if w != w0 || h != h0 {
                                    img = img.resize_exact(w, h, FilterType::Nearest);
                                }
                                // Create a tiled texture to handle large images
                                let tiled = TiledTexture::new(app, &img);
                                model.full_textures[model.current] = Some(tiled);
                            }
                            Err(e) => eprintln!("Warning: failed to open full image {:?}: {}", path, e),
                        }
                    }
                    model.mode = Mode::Single;
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
                if let Some(tex) = &model.full_textures[model.current] {
                    let [w, h] = tex.size();
                    let fit = (rect.w() / w as f32).min(rect.h() / h as f32);
                    model.zoom = fit;
                } else {
                    model.zoom = 1.0;
                }
                model.pan = vec2(0.0, 0.0);
            }
        }
        // Show at 100% scale
        Key::Equals => {
            if let Mode::Single = model.mode {
                model.zoom = 1.0;
                model.pan = vec2(0.0, 0.0);
            }
        }
        _ => {}
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
}
/// Update function to process incoming thumbnail images.
fn update(app: &App, model: &mut Model, _update: Update) {
    // Receive thumbnails from background thread and create textures
    while let Ok((i, img)) = model.thumb_rx.try_recv() {
        let tex = wgpu::Texture::from_image(app, &img);
        model.thumb_textures[i] = Some(tex);
    }
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
            for (i, tex_opt) in model.thumb_textures.iter().enumerate() {
                let row = i / cols;
                let col = i % cols;
                let x = -rect.w() / 2.0 + (model.thumb_size as f32) / 2.0 + model.gap / 2.0
                    + col as f32 * cell;
                let y = rect.h() / 2.0 - (model.thumb_size as f32) / 2.0 - model.gap / 2.0
                    - row as f32 * cell;
                // Apply vertical scroll offset
                let y = y + model.scroll_offset;
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
                        // Placeholder grey square while loading
                        draw.rect()
                            .x_y(x, y)
                            .w_h(model.thumb_size as f32, model.thumb_size as f32)
                            .color(srgba(0.5, 0.5, 0.5, 1.0));
                        if i == model.current {
                            draw.rect()
                                .x_y(x, y)
                                .w_h(model.thumb_size as f32 + 4.0, model.thumb_size as f32 + 4.0)
                                .no_fill()
                                .stroke(WHITE)
                                .stroke_weight(2.0);
                        }
                    }
                }
            }
        }
        Mode::Single => {
            // Attempt to draw the full-resolution tiled texture if loaded;
            // otherwise display a loading message.
            if let Some(tex) = &model.full_textures[model.current] {
                // Draw each tile at the correct position, applying zoom and pan
                let [full_w, full_h] = tex.size();
                for tile in &tex.tiles {
                    // Compute tile center relative to full image center
                    let x_center = tile.x_offset as f32 - full_w as f32 / 2.0 + tile.width as f32 / 2.0;
                    let y_center = full_h as f32 / 2.0 - tile.y_offset as f32 - tile.height as f32 / 2.0;
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
                // Draw filename below image
                let filename = model.image_paths[model.current]
                    .file_name()
                    .unwrap()
                    .to_string_lossy();
                draw.text(&filename)
                    .font_size(24)
                    .color(WHITE)
                    .x_y(0.0, -rect.h() / 2.0 + 16.0);
            } else {
                draw.text("Loading...")
                    .font_size(24)
                    .color(WHITE)
                    .x_y(0.0, 0.0);
            }
        }
    }

    draw.to_frame(app, &frame).unwrap();
}
