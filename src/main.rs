use anyhow::Result;
use nannou::image::{self, DynamicImage, GenericImageView};
use rayon::prelude::*;
use nannou::prelude::*;
use std::fs;
use std::path::PathBuf;

/// The display mode of the viewer.
#[derive(Debug)]
enum Mode {
    Thumbnails,
    Single,
}

/// The application state.
#[derive(Debug)]
struct Model {
    image_paths: Vec<PathBuf>,
    thumb_textures: Vec<wgpu::Texture>,
    // Full-resolution textures, loaded lazily on demand.
    full_textures: Vec<Option<wgpu::Texture>>,
    mode: Mode,
    current: usize,
    thumb_size: u32,
    gap: f32,
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
                            "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "gif" => {
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
    // Generate thumbnails in parallel, filtering out any images that fail to decode.
    let thumb_size = 256;
    let thumb_data: Vec<(PathBuf, DynamicImage)> = image_paths
        .par_iter()
        .filter_map(|p| match image::open(p) {
            Ok(img) => {
                // Generate thumbnail and skip any with zero dimensions.
                let thumb = img.thumbnail(thumb_size, thumb_size);
                let (w, h) = thumb.dimensions();
                if w == 0 || h == 0 {
                    eprintln!("Warning: thumbnail for {:?} has zero dimension {}x{}, skipping", p, w, h);
                    None
                } else {
                    Some((p.clone(), thumb))
                }
            }
            Err(e) => {
                eprintln!("Warning: failed to open image {:?}: {}", p, e);
                None
            }
        })
        .collect();
    // Separate filtered paths and thumbnail images.
    let (image_paths, thumb_images): (Vec<PathBuf>, Vec<DynamicImage>) = thumb_data.into_iter().unzip();
    // Exit if no valid images could be opened.
    if image_paths.is_empty() {
        eprintln!("No valid image files found after decoding.");
        std::process::exit(1);
    }
    // Create the window first, so textures can reference a focused window.
    let _window = app.new_window()
        .size(800, 600)
        .view(view)
        .key_pressed(key_pressed)
        .build()
        .unwrap();
    // Create textures for thumbnails.
    let thumb_textures: Vec<wgpu::Texture> = thumb_images
        .iter()
        .map(|img| wgpu::Texture::from_image(app, img))
        .collect();
    // Initialize placeholders for full-resolution textures (lazy loading).
    let full_textures: Vec<Option<wgpu::Texture>> = (0..image_paths.len())
        .map(|_| None)
        .collect();
    Model {
        image_paths,
        thumb_textures,
        full_textures,
        mode: Mode::Thumbnails,
        current: 0,
        thumb_size,
        gap: 10.0,
    }
}

fn main() -> Result<()> {
    // Launch the nannou application with our model initializer.
    nannou::app(model)
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
                model.current = (model.current + 1) % len;
            }
        },
        Key::K | Key::Up => {
            if let Mode::Thumbnails = model.mode {
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
        }
        Key::J | Key::Down => {
            if let Mode::Thumbnails = model.mode {
                let row = model.current / cols + 1;
                let idx = row * cols + (model.current % cols);
                if idx < len {
                    model.current = idx;
                }
            }
        }
        Key::Return => {
            // Toggle between thumbnail view and single-image view.
            match model.mode {
                Mode::Thumbnails => {
                    // On entering single mode, lazily load the full-resolution texture if needed.
                    if model.full_textures[model.current].is_none() {
                        let path = &model.image_paths[model.current];
                        match wgpu::Texture::from_path(app, path) {
                            Ok(tex) => model.full_textures[model.current] = Some(tex),
                            Err(e) => eprintln!("Warning: failed to load full image {:?}: {}", path, e),
                        }
                    }
                    model.mode = Mode::Single;
                }
                Mode::Single => {
                    model.mode = Mode::Thumbnails;
                }
            }
        }
        _ => {}
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
            for (i, tex) in model.thumb_textures.iter().enumerate() {
                let row = i / cols;
                let col = i % cols;
                let x = -rect.w() / 2.0 + (model.thumb_size as f32) / 2.0 + model.gap / 2.0
                    + col as f32 * cell;
                let y = rect.h() / 2.0 - (model.thumb_size as f32) / 2.0 - model.gap / 2.0
                    - row as f32 * cell;
                draw.texture(tex).x_y(x, y).w_h(model.thumb_size as f32, model.thumb_size as f32);
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
        Mode::Single => {
            // Attempt to draw the full-resolution texture if loaded;
            // otherwise display a loading message.
            if let Some(tex) = &model.full_textures[model.current] {
                let [w, h] = tex.size();
                let img_w = w as f32;
                let img_h = h as f32;
                let scale = (rect.w() / img_w).max(rect.h() / img_h);
                draw.texture(tex).w_h(img_w * scale, img_h * scale);

                let filename = model.image_paths[model.current]
                    .file_name()
                    .unwrap()
                    .to_string_lossy();
                draw.text(&filename)
                    .font_size(24)
                    .color(WHITE)
                    .x_y(0.0, -rect.h() / 2.0 + 16.0);
            } else {
                draw.text("Loading..." )
                    .font_size(24)
                    .color(WHITE)
                    .x_y(0.0, 0.0);
            }
        }
    }

    draw.to_frame(app, &frame).unwrap();
}
