use ::image as image_rs;
use ::image::imageops::FilterType as ImageRsFilterType;
use ::image::{self, DynamicImage, RgbaImage};
use anyhow::{anyhow, Result};
use crossbeam_channel::unbounded;
use libheif_rs::{ColorSpace as HeifColorSpace, HeifContext, LibHeif, Plane, RgbChroma};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::channel;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::ModifiersState;
use winit::window::{Fullscreen, Window};
mod clip;
mod geometry;
mod grid;
mod input;
mod renderer;
mod state;

use clip::{ClipEngine, ClipEvent};
use geometry::{srgba, vec2, Rect, Rgba, Vec2, WHITE};
use grid::ThumbnailGrid;
use input::Key;
use renderer::{Renderer, Scene};
use state::{
    parse_bindings, parse_ui_font_path, CommandEvent, FullPendingState, Mode, Model, SearchState,
    TerminalSession, TerminalState, ThumbRequestQueue, ThumbnailEntry, ThumbnailTexture,
    ThumbnailUpdate, Tile, TilePixelFormat, TiledTexture,
};

type FullImageTile = (u32, u32, u32, u32, TilePixelFormat, Vec<u8>);

#[derive(Debug)]
enum FullImageMessage {
    Loaded {
        index: usize,
        full_w: u32,
        full_h: u32,
        tiles: Vec<FullImageTile>,
    },
    Failed {
        index: usize,
        error: String,
    },
}

/// Maximum number of full-resolution images to cache in memory.
const FULL_CACHE_CAPACITY: usize = 4;
/// How long to wait before retrying a full-resolution load request.
const FULL_PENDING_RETRY: Duration = Duration::from_secs(5);
/// Number of extra rows of thumbnails to keep warm beyond the viewport.
pub(crate) const THUMB_PREFETCH_ROWS: usize = 1;
/// Number of files to poll for modifications each update tick.
const FILE_WATCH_BATCH: usize = 32;
const TERMINAL_PANEL_FRACTION: f32 = 0.42;
const TERMINAL_PANEL_MIN_HEIGHT: f32 = 180.0;
const TERMINAL_TAB_HEIGHT: f32 = 28.0;
const TERMINAL_STATUS_HEIGHT: f32 = 22.0;
const TERMINAL_MARGIN: f32 = 12.0;
const TERMINAL_FONT_SIZE: u32 = 14;
const TERMINAL_CELL_WIDTH: f32 = 8.4;
const TERMINAL_CELL_HEIGHT: f32 = 18.0;
const TERMINAL_SCROLLBACK: usize = 4_000;
const TERMINAL_NOMINAL_ROWS: u16 = 240;

#[derive(Clone, Copy, Debug)]
enum UserEvent {
    Wake,
}

#[derive(Clone)]
struct AppProxy(EventLoopProxy<UserEvent>);

impl AppProxy {
    fn wakeup(&self) -> std::result::Result<(), winit::event_loop::EventLoopClosed<UserEvent>> {
        self.0.send_event(UserEvent::Wake)
    }
}

#[derive(Default)]
struct Keys {
    mods: ModifiersState,
}

#[derive(Default)]
struct Mouse {
    position: Vec2,
}

impl Mouse {
    fn position(&self) -> Vec2 {
        self.position
    }
}

struct App {
    window: Arc<Window>,
    proxy: AppProxy,
    keys: Keys,
    mouse: Mouse,
    rect: Rect,
    exit_requested: Cell<bool>,
}

impl App {
    fn create_proxy(&self) -> AppProxy {
        self.proxy.clone()
    }

    fn quit(&self) {
        self.exit_requested.set(true);
    }

    fn update_rect(&mut self) {
        let logical = self
            .window
            .inner_size()
            .to_logical::<f32>(self.window.scale_factor());
        self.rect = Rect::from_w_h(logical.width, logical.height);
    }
}

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "bmp", "tiff", "gif", "webp", "tif", "heif", "heic",
];

/// List of recognized raw file extensions for detecting XMP sidecars.
const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "bay", "cap", "cr2", "cr3", "crw", "cs1", "dcr", "dng", "erf", "fff",
    "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "ptx", "pxn",
    "raf", "raw", "rwl", "rw2", "rwz", "sr2", "srf", "srw", "x3f",
];

fn extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn is_heif_path(path: &Path) -> bool {
    matches!(extension_lower(path).as_deref(), Some("heif" | "heic"))
}

fn is_supported_image_path(path: &Path) -> bool {
    extension_lower(path)
        .as_deref()
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext))
}

fn terminal_panel_height(rect: Rect) -> f32 {
    (rect.h() * TERMINAL_PANEL_FRACTION)
        .max(TERMINAL_PANEL_MIN_HEIGHT)
        .min(rect.h() - 40.0)
}

fn terminal_panel_rect(rect: Rect) -> Rect {
    let height = terminal_panel_height(rect);
    Rect::from_x_y_w_h(0.0, rect.bottom() + height / 2.0, rect.w(), height)
}

fn terminal_body_rect(panel_rect: Rect) -> Rect {
    let height = (panel_rect.h() - TERMINAL_TAB_HEIGHT - TERMINAL_STATUS_HEIGHT).max(40.0);
    let center_y = panel_rect.bottom() + TERMINAL_STATUS_HEIGHT + height / 2.0;
    Rect::from_x_y_w_h(0.0, center_y, panel_rect.w(), height)
}

fn terminal_grid_size(rect: Rect) -> (u16, u16) {
    let panel_rect = terminal_panel_rect(rect);
    let body_rect = terminal_body_rect(panel_rect);
    let width = (body_rect.w() - TERMINAL_MARGIN * 2.0).max(TERMINAL_CELL_WIDTH);
    let height = (body_rect.h() - TERMINAL_MARGIN * 2.0).max(TERMINAL_CELL_HEIGHT);
    let cols = (width / TERMINAL_CELL_WIDTH).floor().max(1.0) as u16;
    let rows = (height / TERMINAL_CELL_HEIGHT).floor().max(1.0) as u16;
    (rows, cols)
}

fn terminal_pty_size(rows: u16, cols: u16) -> PtySize {
    PtySize {
        rows,
        cols,
        pixel_width: 0,
        pixel_height: 0,
    }
}

fn terminal_title(command: &str) -> String {
    let trimmed = command.trim();
    let mut title = trimmed
        .lines()
        .next()
        .unwrap_or("command")
        .trim()
        .to_string();
    if title.len() > 28 {
        title.truncate(28);
        title.push('…');
    }
    if title.is_empty() {
        "command".to_string()
    } else {
        title
    }
}

fn active_terminal_session(model: &Model) -> Option<&TerminalSession> {
    model.terminal.sessions.get(model.terminal.active)
}

fn active_terminal_session_mut(model: &mut Model) -> Option<&mut TerminalSession> {
    model.terminal.sessions.get_mut(model.terminal.active)
}

fn sync_terminal_viewport(app: &App, model: &mut Model) {
    let Some(rect) = current_window_rect(app, model) else {
        return;
    };
    let (rows, cols) = terminal_grid_size(rect);
    model.terminal.rows = rows;
    model.terminal.cols = cols;
}

fn set_active_terminal(model: &mut Model, active: usize) {
    if model.terminal.sessions.is_empty() {
        model.terminal.active = 0;
        return;
    }
    model.terminal.active = active.min(model.terminal.sessions.len() - 1);
    if let Some(session) = active_terminal_session_mut(model) {
        session
            .parser
            .screen_mut()
            .set_scrollback(session.scrollback_offset);
        session.scrollback_offset = session.parser.screen().scrollback();
    }
}

fn cycle_terminal_tab(model: &mut Model, delta: isize) {
    let len = model.terminal.sessions.len();
    if len == 0 {
        return;
    }
    let len = len as isize;
    let next = ((model.terminal.active as isize + delta).rem_euclid(len)) as usize;
    set_active_terminal(model, next);
}

fn scroll_active_terminal(model: &mut Model, delta: isize) {
    let Some(session) = active_terminal_session_mut(model) else {
        return;
    };
    if delta < 0 {
        session.scrollback_offset = session
            .scrollback_offset
            .saturating_sub(delta.unsigned_abs());
    } else {
        session.scrollback_offset = session.scrollback_offset.saturating_add(delta as usize);
    }
    session
        .parser
        .screen_mut()
        .set_scrollback(session.scrollback_offset);
    session.scrollback_offset = session.parser.screen().scrollback();
}

fn close_active_terminal(model: &mut Model) {
    if model.terminal.sessions.is_empty() {
        return;
    }
    let active = model.terminal.active.min(model.terminal.sessions.len() - 1);
    model.terminal.sessions.remove(active);
    if model.terminal.sessions.is_empty() {
        model.terminal.active = 0;
        model.terminal.visible = false;
        return;
    }
    set_active_terminal(model, active.min(model.terminal.sessions.len() - 1));
}

fn launch_terminal_command(app: &App, model: &mut Model, command: String) {
    sync_terminal_viewport(app, model);
    let rows = TERMINAL_NOMINAL_ROWS.max(model.terminal.rows.max(1));
    let cols = model.terminal.cols.max(1);
    let session_id = model.terminal.next_id;
    model.terminal.next_id += 1;
    model.terminal.visible = true;

    let mut parser = vt100::Parser::new(rows, cols, TERMINAL_SCROLLBACK);
    parser.screen_mut().set_scrollback(0);

    let title = terminal_title(&command);
    let pty_system = native_pty_system();
    let proxy = app.create_proxy();
    let tx = model.command_tx.clone();

    let session = match pty_system.openpty(terminal_pty_size(rows, cols)) {
        Ok(pair) => match pair.master.try_clone_reader() {
            Ok(mut reader) => {
                let mut builder = CommandBuilder::new("sh");
                builder.arg("-c");
                builder.arg(&command);

                match pair.slave.spawn_command(builder) {
                    Ok(mut child) => {
                        let reader_tx = tx.clone();
                        let reader_proxy = proxy.clone();
                        thread::spawn(move || {
                            let mut buf = vec![0_u8; 8192];
                            loop {
                                match reader.read(&mut buf) {
                                    Ok(0) => break,
                                    Ok(n) => {
                                        let _ = reader_tx.send(CommandEvent::Output {
                                            session_id,
                                            bytes: buf[..n].to_vec(),
                                        });
                                        let _ = reader_proxy.wakeup();
                                    }
                                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                                    Err(err) => {
                                        let _ = reader_tx.send(CommandEvent::Failed {
                                            session_id,
                                            error: format!("Terminal read failed: {err}"),
                                        });
                                        let _ = reader_proxy.wakeup();
                                        break;
                                    }
                                }
                            }
                        });
                        thread::spawn(move || {
                            let event = match child.wait() {
                                Ok(status) => CommandEvent::Finished {
                                    session_id,
                                    exit_code: status.exit_code(),
                                    signal: status.signal().map(str::to_string),
                                },
                                Err(err) => CommandEvent::Failed {
                                    session_id,
                                    error: format!("Failed to wait for command: {err}"),
                                },
                            };
                            let _ = tx.send(event);
                            let _ = proxy.wakeup();
                        });

                        TerminalSession {
                            id: session_id,
                            title,
                            command,
                            parser,
                            master: Some(pair.master),
                            scrollback_offset: 0,
                            running: true,
                            exit_code: None,
                            signal: None,
                            error: None,
                        }
                    }
                    Err(error) => TerminalSession {
                        id: session_id,
                        title,
                        command,
                        parser,
                        master: Some(pair.master),
                        scrollback_offset: 0,
                        running: false,
                        exit_code: None,
                        signal: None,
                        error: Some(format!("Failed to spawn command: {error}")),
                    },
                }
            }
            Err(error) => TerminalSession {
                id: session_id,
                title,
                command,
                parser,
                master: Some(pair.master),
                scrollback_offset: 0,
                running: false,
                exit_code: None,
                signal: None,
                error: Some(format!("Failed to open PTY reader: {error}")),
            },
        },
        Err(error) => {
            parser.process(format!("Failed to allocate PTY: {error}\r\n").as_bytes());
            TerminalSession {
                id: session_id,
                title,
                command,
                parser,
                master: None,
                scrollback_offset: 0,
                running: false,
                exit_code: None,
                signal: None,
                error: Some(format!("Failed to allocate PTY: {error}")),
            }
        }
    };

    model.terminal.sessions.push(session);
    set_active_terminal(model, model.terminal.sessions.len() - 1);
}

fn launch_detached_command(command: String) {
    thread::spawn(move || {
        let result = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Err(err) = result {
            eprintln!("Failed to launch detached command: {err}");
        }
    });
}

fn vt_color_to_rgba(color: vt100::Color, bold: bool, default: Rgba) -> Rgba {
    match color {
        vt100::Color::Default => default,
        vt100::Color::Rgb(r, g, b) => {
            srgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
        }
        vt100::Color::Idx(idx) => ansi_color(idx, bold),
    }
}

fn ansi_color(idx: u8, bold: bool) -> Rgba {
    let idx = if bold && idx < 8 { idx + 8 } else { idx };
    let [r, g, b] = match idx {
        0 => [12, 12, 12],
        1 => [197, 15, 31],
        2 => [19, 161, 14],
        3 => [193, 156, 0],
        4 => [0, 55, 218],
        5 => [136, 23, 152],
        6 => [58, 150, 221],
        7 => [204, 204, 204],
        8 => [118, 118, 118],
        9 => [231, 72, 86],
        10 => [22, 198, 12],
        11 => [249, 241, 165],
        12 => [59, 120, 255],
        13 => [180, 0, 158],
        14 => [97, 214, 214],
        15 => [242, 242, 242],
        16..=231 => {
            let cube = idx - 16;
            let r = cube / 36;
            let g = (cube % 36) / 6;
            let b = cube % 6;
            let convert = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
            [convert(r), convert(g), convert(b)]
        }
        232..=255 => {
            let level = 8 + (idx - 232) * 10;
            [level, level, level]
        }
    };
    srgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}

/// Mouse click handler: select thumbnail on left-click in thumbnail mode.
fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if let Mode::Thumbnails = model.mode {
        if button == MouseButton::Left {
            let pos = app.mouse.position();
            let Some(rect) = current_window_rect(app, model) else {
                return;
            };
            let grid = ThumbnailGrid::new(model, rect);
            if let Some((row_min, row_max)) = grid.visible_rows() {
                for row in row_min..=row_max {
                    for col in 0..grid.cols() {
                        let i = row * grid.cols() + col;
                        if i >= grid.total() {
                            break;
                        }
                        let center = grid.index_center(i).unwrap();
                        let x = center.x;
                        let y = center.y;
                        let (width, height) = if let Some(slot) = model.thumb_visible.get(&i) {
                            let [tw, th] = slot.size;
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
}

/// Mouse wheel scroll handler to scroll thumbnails in thumbnail view.
fn mouse_wheel(app: &App, model: &mut Model, delta: MouseScrollDelta, _phase: TouchPhase) {
    if model.terminal.visible && !model.terminal.sessions.is_empty() {
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => (-y * 3.0).round() as isize,
            MouseScrollDelta::PixelDelta(pos) => (-pos.y as f32 / 24.0).round() as isize,
        };
        if scroll_amount != 0 {
            scroll_active_terminal(model, scroll_amount);
        }
        return;
    }
    match model.mode {
        Mode::Thumbnails => {
            // Determine scroll amount: line vs pixel delta
            let scroll_amount = match delta {
                MouseScrollDelta::LineDelta(_x, y) => y * -100.0,
                MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
            };
            // Update scroll offset and clamp to content bounds
            model.scroll_offset += scroll_amount;
            let Some(rect) = current_window_rect(app, model) else {
                return;
            };
            let grid = ThumbnailGrid::new(model, rect);
            model.scroll_offset = model.scroll_offset.clamp(0.0, grid.max_scroll());
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

/// Compute the cache path for an image based on a SHA1 of its path.
/// The cache layout is: cache_base/<first 3 hex chars>/<remaining hex chars>.png
fn thumbnail_cache_path(cache_base: &Path, image_path: &Path) -> PathBuf {
    clip::cache_file_path(cache_base, image_path, "png")
}

fn orientation_from_tag_value(value: &rexif::TagValue) -> Option<u16> {
    let raw = match value {
        rexif::TagValue::U16(vals) => vals.first().copied(),
        rexif::TagValue::I16(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::U8(vals) => vals.first().map(|&v| v as u16),
        rexif::TagValue::I8(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::U32(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::I32(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::URational(vals) => vals.first().and_then(|r| {
            let num = r.numerator;
            let den = r.denominator;
            if den == 0 || num % den != 0 {
                return None;
            }
            u16::try_from(num / den).ok()
        }),
        rexif::TagValue::IRational(vals) => vals.first().and_then(|r| {
            let num = r.numerator;
            let den = r.denominator;
            if den == 0 || num % den != 0 {
                return None;
            }
            u16::try_from(num / den).ok()
        }),
        _ => None,
    }?;

    (1..=8).contains(&raw).then_some(raw)
}

fn parse_exif_quiet(path: &Path) -> Option<rexif::ExifData> {
    let data = fs::read(path).ok()?;
    rexif::parse_buffer_quiet(&data).0.ok()
}

fn adjust_orientation_full(img: image_rs::DynamicImage, path: &Path) -> image_rs::DynamicImage {
    let mut oriented = img;
    if let Some(exif) = parse_exif_quiet(path) {
        for entry in exif.entries {
            if entry.tag == rexif::ExifTag::Orientation {
                if let Some(code) = orientation_from_tag_value(&entry.value) {
                    oriented = match code {
                        2 => oriented.fliph(),
                        3 => oriented.rotate180(),
                        4 => oriented.flipv(),
                        5 => oriented.rotate90().fliph(),
                        6 => oriented.rotate90(),
                        7 => oriented.rotate270().fliph(),
                        8 => oriented.rotate270(),
                        _ => oriented,
                    };
                }
                break;
            }
        }
    }
    oriented
}

fn push_u16_rgba(pixel_data: &mut Vec<u8>, rgba: [u16; 4]) {
    for channel in rgba {
        pixel_data.extend_from_slice(&channel.to_ne_bytes());
    }
}

fn rgba16_bytes_to_rgba8(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let value = u16::from_ne_bytes([chunk[0], chunk[1]]);
        out.push(((value as u32 + 128) / 257) as u8);
    }
    out
}

fn libheif() -> &'static LibHeif {
    static LIBHEIF: OnceLock<LibHeif> = OnceLock::new();
    LIBHEIF.get_or_init(LibHeif::new)
}

fn heif_target_color_space(has_alpha: bool, high_bit_depth: bool) -> HeifColorSpace {
    match (has_alpha, high_bit_depth, cfg!(target_endian = "little")) {
        (false, false, _) => HeifColorSpace::Rgb(RgbChroma::Rgb),
        (true, false, _) => HeifColorSpace::Rgb(RgbChroma::Rgba),
        (false, true, true) => HeifColorSpace::Rgb(RgbChroma::HdrRgbLe),
        (false, true, false) => HeifColorSpace::Rgb(RgbChroma::HdrRgbBe),
        (true, true, true) => HeifColorSpace::Rgb(RgbChroma::HdrRgbaLe),
        (true, true, false) => HeifColorSpace::Rgb(RgbChroma::HdrRgbaBe),
    }
}

fn heif_bytes_per_pixel(has_alpha: bool, high_bit_depth: bool) -> usize {
    let channels = if has_alpha { 4 } else { 3 };
    let bytes_per_channel = if high_bit_depth { 2 } else { 1 };
    channels * bytes_per_channel
}

fn copy_heif_interleaved_plane(plane: &Plane<&[u8]>, bytes_per_pixel: usize) -> Result<Vec<u8>> {
    if plane.stride == 0 {
        return Err(anyhow!("HEIF row stride is zero"));
    }
    let width = plane.width as usize;
    let height = plane.height as usize;
    let row_size = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| anyhow!("HEIF image row size overflows usize"))?;
    if row_size > plane.stride {
        return Err(anyhow!(
            "HEIF row stride {} is smaller than row size {}",
            plane.stride,
            row_size
        ));
    }
    let expected_len = row_size
        .checked_mul(height)
        .ok_or_else(|| anyhow!("HEIF image size overflows usize"))?;
    let mut pixels = Vec::with_capacity(expected_len);
    for row in plane.data.chunks_exact(plane.stride).take(height) {
        pixels.extend_from_slice(&row[..row_size]);
    }
    if pixels.len() != expected_len {
        return Err(anyhow!("HEIF pixel data ended before all rows were copied"));
    }
    Ok(pixels)
}

fn native_u16_samples(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn image_buffer_from_raw<P, Container>(
    width: u32,
    height: u32,
    buf: Container,
    format_name: &str,
) -> Result<image_rs::ImageBuffer<P, Container>>
where
    P: image_rs::Pixel,
    Container: std::ops::Deref<Target = [P::Subpixel]>,
{
    image_rs::ImageBuffer::from_raw(width, height, buf)
        .ok_or_else(|| anyhow!("decoded HEIF {format_name} data has the wrong length"))
}

fn decode_heif_image(path: &Path) -> Result<image_rs::DynamicImage> {
    let lib_heif = libheif();
    let bytes = fs::read(path)?;
    let ctx = HeifContext::read_from_bytes(&bytes)?;
    let handle = ctx.primary_image_handle()?;
    let has_alpha = handle.has_alpha_channel();
    let high_bit_depth = handle.luma_bits_per_pixel() > 8 || handle.chroma_bits_per_pixel() > 8;
    let color_space = heif_target_color_space(has_alpha, high_bit_depth);
    let decoded = lib_heif.decode(&handle, color_space, None)?;
    let planes = decoded.planes();
    let plane = planes
        .interleaved
        .ok_or_else(|| anyhow!("decoded HEIF did not provide interleaved pixel data"))?;
    let width = plane.width;
    let height = plane.height;
    let bytes_per_pixel = heif_bytes_per_pixel(has_alpha, high_bit_depth);
    let pixels = copy_heif_interleaved_plane(&plane, bytes_per_pixel)?;

    match (has_alpha, high_bit_depth) {
        (false, false) => Ok(image_rs::DynamicImage::ImageRgb8(image_buffer_from_raw(
            width, height, pixels, "RGB8",
        )?)),
        (true, false) => Ok(image_rs::DynamicImage::ImageRgba8(image_buffer_from_raw(
            width, height, pixels, "RGBA8",
        )?)),
        (false, true) => Ok(image_rs::DynamicImage::ImageRgb16(image_buffer_from_raw(
            width,
            height,
            native_u16_samples(&pixels),
            "RGB16",
        )?)),
        (true, true) => Ok(image_rs::DynamicImage::ImageRgba16(image_buffer_from_raw(
            width,
            height,
            native_u16_samples(&pixels),
            "RGBA16",
        )?)),
    }
}

fn decode_image_no_limits(path: &Path) -> Result<image_rs::DynamicImage> {
    if is_heif_path(path) {
        return decode_heif_image(path);
    }
    let mut reader = image_rs::io::Reader::open(path)?;
    reader.limits(image_rs::io::Limits::no_limits());
    Ok(adjust_orientation_full(reader.decode()?, path))
}

fn collect_tiled_pixels<F>(
    full_w: u32,
    full_h: u32,
    format: TilePixelFormat,
    bytes_per_pixel: usize,
    mut fill_pixel: F,
) -> Vec<FullImageTile>
where
    F: FnMut(u32, u32, &mut Vec<u8>),
{
    const MAX_TILE_SIZE: u32 = 8192;
    let mut tiles = Vec::new();
    for y in (0..full_h).step_by(MAX_TILE_SIZE as usize) {
        for x in (0..full_w).step_by(MAX_TILE_SIZE as usize) {
            let tile_w = (full_w - x).min(MAX_TILE_SIZE);
            let tile_h = (full_h - y).min(MAX_TILE_SIZE);
            let mut pixel_data =
                Vec::with_capacity(tile_w as usize * tile_h as usize * bytes_per_pixel);
            for py in y..(y + tile_h) {
                for px in x..(x + tile_w) {
                    fill_pixel(px, py, &mut pixel_data);
                }
            }
            tiles.push((x, y, tile_w, tile_h, format, pixel_data));
        }
    }
    tiles
}

fn load_full_image_tiles(path: &Path) -> Result<(u32, u32, Vec<FullImageTile>)> {
    let img = decode_image_no_limits(path)?;
    let (full_w, full_h) = (img.width(), img.height());
    let tiles = match img {
        image_rs::DynamicImage::ImageLuma8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [l] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[l, l, l, u8::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageLumaA8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [l, a] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[l, l, l, a]);
            },
        ),
        image_rs::DynamicImage::ImageRgb8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [r, g, b] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[r, g, b, u8::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageRgba8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                pixel_data.extend_from_slice(&buf.get_pixel(px, py).0);
            },
        ),
        image_rs::DynamicImage::ImageLuma16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [l] = buf.get_pixel(px, py).0;
                push_u16_rgba(pixel_data, [l, l, l, u16::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageLumaA16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [l, a] = buf.get_pixel(px, py).0;
                push_u16_rgba(pixel_data, [l, l, l, a]);
            },
        ),
        image_rs::DynamicImage::ImageRgb16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [r, g, b] = buf.get_pixel(px, py).0;
                push_u16_rgba(pixel_data, [r, g, b, u16::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageRgba16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                push_u16_rgba(pixel_data, buf.get_pixel(px, py).0);
            },
        ),
        other => {
            let rgba = other.to_rgba8();
            collect_tiled_pixels(
                full_w,
                full_h,
                TilePixelFormat::Rgba8,
                4,
                |px, py, pixel_data| {
                    pixel_data.extend_from_slice(&rgba.get_pixel(px, py).0);
                },
            )
        }
    };
    Ok((full_w, full_h, tiles))
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use image_rs::{ImageBuffer, Rgb};

    #[test]
    fn orientation_from_urational_rounds_down() {
        let value = rexif::TagValue::URational(vec![rexif::URational {
            numerator: 6,
            denominator: 2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), Some(3));
    }

    #[test]
    fn orientation_from_irational_with_negative_denominator() {
        let value = rexif::TagValue::IRational(vec![rexif::IRational {
            numerator: -12,
            denominator: -2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), Some(6));
    }

    #[test]
    fn orientation_from_irational_non_integer() {
        let value = rexif::TagValue::IRational(vec![rexif::IRational {
            numerator: 3,
            denominator: 2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), None);
    }

    #[test]
    fn tiles_rgb16_images_as_rgba16_without_whole_image_rgba_conversion() {
        let img = image_rs::DynamicImage::ImageRgb16(ImageBuffer::from_fn(3, 2, |x, y| {
            Rgb([
                (x as u16) * 1000 + 1,
                (y as u16) * 1000 + 2,
                (x as u16 + y as u16) * 1000 + 3,
            ])
        }));
        let (full_w, full_h) = (img.width(), img.height());
        let tiles = match img {
            image_rs::DynamicImage::ImageRgb16(buf) => collect_tiled_pixels(
                full_w,
                full_h,
                TilePixelFormat::Rgba16,
                8,
                |px, py, pixel_data| {
                    let [r, g, b] = buf.get_pixel(px, py).0;
                    push_u16_rgba(pixel_data, [r, g, b, u16::MAX]);
                },
            ),
            _ => unreachable!(),
        };

        assert_eq!(tiles.len(), 1);
        let (_, _, width, height, format, pixel_data) = &tiles[0];
        assert_eq!((*width, *height), (3, 2));
        assert!(matches!(format, TilePixelFormat::Rgba16));
        assert_eq!(pixel_data.len(), 3 * 2 * 8);
        assert_eq!(&pixel_data[0..8], &[1, 0, 2, 0, 3, 0, 255, 255]);
    }

    #[test]
    fn rgba16_fallback_to_rgba8_uses_full_u16_range() {
        let bytes = [
            0_u16.to_ne_bytes(),
            257_u16.to_ne_bytes(),
            32_768_u16.to_ne_bytes(),
            u16::MAX.to_ne_bytes(),
        ]
        .concat();
        assert_eq!(rgba16_bytes_to_rgba8(&bytes), vec![0, 1, 128, 255]);
    }

    #[test]
    fn supported_image_extensions_include_heif_and_heic() {
        assert!(is_supported_image_path(Path::new("photo.heif")));
        assert!(is_supported_image_path(Path::new("photo.HEIC")));
        assert!(is_heif_path(Path::new("photo.HEIF")));
        assert!(!is_supported_image_path(Path::new("notes.txt")));
    }

    #[test]
    fn copy_heif_interleaved_plane_removes_row_padding() {
        let data = [1_u8, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 88, 88];
        let plane = Plane {
            data: &data[..],
            width: 2,
            height: 2,
            stride: 8,
            bits_per_pixel: 24,
            storage_bits_per_pixel: 24,
        };
        let pixels = copy_heif_interleaved_plane(&plane, 3).unwrap();
        assert_eq!(pixels, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
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
                if path.is_file() && is_supported_image_path(&path) {
                    image_paths.push(path.canonicalize().unwrap());
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
    let mut file_mod_times = Vec::with_capacity(image_paths.len());
    for path in &image_paths {
        file_mod_times.push(current_mod_time(path));
    }
    // Channel for receiving thumbnails from background threads.
    let (thumb_tx, thumb_rx) = channel::<ThumbnailUpdate>();
    let thumb_queue = ThumbRequestQueue::new();
    thumb_queue.enqueue_batch(0..image_paths.len());
    let num_workers = rayon::current_num_threads().clamp(1, 8);
    let shared_paths = Arc::new(image_paths.clone());
    let clip_engine = ClipEngine::new(cache_base.clone()).unwrap_or_else(|err| {
        eprintln!("Failed to initialize CLIP: {err}");
        std::process::exit(1);
    });
    let clip_sender = clip_engine.request_sender();
    for _ in 0..num_workers {
        let paths = Arc::clone(&shared_paths);
        let cache_base = cache_base.clone();
        let tx = thumb_tx.clone();
        let thumb_queue = thumb_queue.clone();
        let clip_sender = clip_sender.clone();
        thread::spawn(move || {
            while let Some(i) = thumb_queue.pop() {
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
                        if let Ok(img) = decode_image_no_limits(p) {
                            let mut thumb = img.thumbnail(thumb_size, thumb_size);
                            let (w0, h0) = (thumb.width(), thumb.height());
                            if w0 != 0 && h0 != 0 {
                                let w = w0.max(2);
                                let h = h0.max(2);
                                if w != w0 || h != h0 {
                                    thumb = thumb.resize_exact(w, h, ImageRsFilterType::Nearest);
                                }
                                if let Some(parent) = cache_path.parent() {
                                    let _ = fs::create_dir_all(parent);
                                }
                                if thumb.save(&cache_path).is_ok() {
                                    if let Ok(img) = image::open(&cache_path) {
                                        result = Some(DynamicImage::ImageRgba8(img.to_rgba8()));
                                    }
                                }
                            }
                        }
                    }
                    let image = result.unwrap_or_else(|| {
                        DynamicImage::ImageRgba8(RgbaImage::from_pixel(
                            2,
                            2,
                            image::Rgba([128, 128, 128, 255]),
                        ))
                    });
                    let clip_embedding = match clip::load_cached_embedding(&cache_base, p) {
                        Ok(value) => value,
                        Err(err) => {
                            eprintln!(
                                "Failed to load cached CLIP embedding for {}: {}",
                                p.display(),
                                err
                            );
                            None
                        }
                    };
                    if clip_embedding.is_none() {
                        let clip_thumb = image.to_rgb8();
                        if let Err(err) = clip_sender.queue_image(i, p.clone(), clip_thumb) {
                            eprintln!(
                                "Failed to queue CLIP embedding for {}: {}",
                                p.display(),
                                err
                            );
                        }
                    }
                    let update = ThumbnailUpdate {
                        index: i,
                        image,
                        clip_embedding,
                    };
                    if tx.send(update).is_err() {
                        break;
                    }
                }
            }
        });
    }
    let clip_missing: HashSet<usize> = (0..image_paths.len()).collect();
    let clip_inflight: HashSet<usize> = HashSet::new();
    // Initialize channels and state for full-resolution LRU cache.
    // Channel for requesting full-resolution images (by index)
    let (full_req_tx, full_req_rx) = unbounded::<usize>();
    // Channel for receiving loaded full-resolution image tile data
    let (full_resp_tx, full_resp_rx) = unbounded::<FullImageMessage>();
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
                        match load_full_image_tiles(path) {
                            Ok((full_w, full_h, tiles_data)) => {
                                let _ = resp_tx.send(FullImageMessage::Loaded {
                                    index: idx,
                                    full_w,
                                    full_h,
                                    tiles: tiles_data,
                                });
                            }
                            Err(err) => {
                                let _ = resp_tx.send(FullImageMessage::Failed {
                                    index: idx,
                                    error: format!("failed to open {}: {}", path.display(), err),
                                });
                            }
                        }
                    } else {
                        let _ = resp_tx.send(FullImageMessage::Failed {
                            index: idx,
                            error: "image index out of range".to_string(),
                        });
                    }
                }
            });
        }
    }
    let full_pending: HashMap<usize, FullPendingState> = HashMap::new();
    let full_textures: HashMap<usize, TiledTexture> = HashMap::new();
    let full_usage: VecDeque<usize> = VecDeque::new();
    // Get initial window rect for resize tracking
    let initial_rect = app.rect;
    // Load user configuration files.
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    let config_dir = config_home.join("sriv");
    let bindings_path = config_dir.join("bindings.toml");
    let key_bindings = if let Ok(contents) = fs::read_to_string(&bindings_path) {
        parse_bindings(&contents)
    } else {
        Vec::new()
    };
    let app_config_path = config_dir.join("config.toml");
    let ui_font_path = fs::read_to_string(&app_config_path)
        .ok()
        .and_then(|contents| parse_ui_font_path(&contents));
    // Channel for receiving command terminal updates from custom commands
    let (command_tx, command_rx) = channel::<CommandEvent>();
    Model {
        image_paths,
        ui_font_path,
        thumb_visible: HashMap::new(),
        thumb_data: HashMap::new(),
        thumb_has_xmp,
        thumb_rx,
        thumb_queue: thumb_queue.clone(),
        file_mod_times,
        file_watch_cursor: 0,
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
        prev_scroll: 0.0,
        fit_mode: false,
        selection_changed_at: Instant::now(),
        selection_pending: false,
        // Custom key bindings
        key_bindings,
        // Command terminal handling
        command_tx,
        command_rx,
        terminal: TerminalState {
            sessions: Vec::new(),
            visible: false,
            active: 0,
            next_id: 1,
            rows: 24,
            cols: 80,
        },
        clip_engine,
        clip_missing,
        clip_inflight,
        pending_clip_embeddings: HashMap::new(),
        next_search_request_id: 0,
        search: None,
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = AppProxy(event_loop.create_proxy());
    event_loop.run_app(&mut SrivApplication { state: None, proxy })?;
    Ok(())
}

struct ApplicationState {
    app: App,
    renderer: Renderer,
    model: Model,
    next_update: Instant,
}

impl ApplicationState {
    fn update(&mut self) {
        update(&self.app, &mut self.model, &self.renderer);
        self.app.window.request_redraw();
        self.next_update = Instant::now() + Duration::from_millis(16);
    }

    fn handle_exit_request(&self, event_loop: &ActiveEventLoop) {
        if self.app.exit_requested.replace(false) {
            event_loop.exit();
        }
    }
}

struct SrivApplication {
    state: Option<ApplicationState>,
    proxy: AppProxy,
}

impl ApplicationHandler<UserEvent> for SrivApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("sriv")
            .with_inner_size(LogicalSize::new(800.0, 600.0));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("Failed to create the sriv window: {error}");
                event_loop.exit();
                return;
            }
        };
        let mut app = App {
            window: Arc::clone(&window),
            proxy: self.proxy.clone(),
            keys: Keys::default(),
            mouse: Mouse::default(),
            rect: Rect::default(),
            exit_requested: Cell::new(false),
        };
        app.update_rect();
        let mut renderer = match pollster::block_on(Renderer::new(window, event_loop)) {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("Failed to initialize graphics: {error:#}");
                event_loop.exit();
                return;
            }
        };
        let model = model(&app);
        renderer.load_ui_font(model.ui_font_path.as_deref());
        self.state = Some(ApplicationState {
            app,
            renderer,
            model,
            next_update: Instant::now(),
        });
        if let Some(state) = &mut self.state {
            state.update();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: UserEvent) {
        if let Some(state) = &mut self.state {
            state.update();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &mut self.state else {
            return;
        };
        let now = Instant::now();
        if now >= state.next_update {
            state.update();
        }
        state.handle_exit_request(event_loop);
        event_loop.set_control_flow(ControlFlow::WaitUntil(state.next_update));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else {
            return;
        };
        if window_id != state.app.window.id() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.app.update_rect();
                state.renderer.resize(size.width, size.height);
                state.update();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                state.app.update_rect();
                let size = state.app.window.inner_size();
                state.renderer.resize(size.width, size.height);
                state.update();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let logical = position.to_logical::<f32>(state.app.window.scale_factor());
                state.app.mouse.position = vec2(
                    logical.x - state.app.rect.w() / 2.0,
                    state.app.rect.h() / 2.0 - logical.y,
                );
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                state.app.keys.mods = modifiers.state();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                mouse_pressed(&state.app, &mut state.model, button);
                state.app.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                let delta = match delta {
                    MouseScrollDelta::LineDelta(x, y) => MouseScrollDelta::LineDelta(x, y),
                    MouseScrollDelta::PixelDelta(position) => {
                        let logical = position.to_logical::<f64>(state.app.window.scale_factor());
                        MouseScrollDelta::PixelDelta(winit::dpi::PhysicalPosition::new(
                            logical.x, logical.y,
                        ))
                    }
                };
                mouse_wheel(&state.app, &mut state.model, delta, phase);
                state.app.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if let Some(key) = Key::from_physical_key(event.physical_key) {
                    key_pressed(&state.app, &mut state.model, key);
                }
                if !state.app.keys.mods.control_key()
                    && !state.app.keys.mods.alt_key()
                    && !state.app.keys.mods.super_key()
                {
                    if let Some(text) = event.text {
                        for ch in text.chars() {
                            received_character(&state.app, &mut state.model, ch);
                        }
                    }
                }
                state.handle_exit_request(event_loop);
                state.app.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = view(&state.app, &mut state.model, &mut state.renderer) {
                    eprintln!("Failed to render frame: {error:#}");
                }
            }
            _ => {}
        }
    }
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
    let Some(rect) = current_window_rect(app, model) else {
        return;
    };
    let grid = ThumbnailGrid::new(model, rect);
    if let Some(row) = grid.row_for_index(idx) {
        let view_height = grid.rect().h();
        let mut scroll = model.scroll_offset;
        let top = grid.row_top(row);
        let bottom = grid.row_bottom(row);
        if top < scroll {
            scroll = top;
        } else if bottom > scroll + view_height {
            scroll = bottom - view_height;
        }
        model.scroll_offset = scroll.clamp(0.0, grid.max_scroll());
    }
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
            for idx in 0..model.image_paths.len() {
                if let Some(entry) = model.thumb_data.get(&idx) {
                    if let Some(img_embed) = entry.clip_embedding.as_ref() {
                        scored.push((idx, cosine_similarity(text_embed, img_embed)));
                        continue;
                    }
                }
                if let Some(img_embed) = model.pending_clip_embeddings.get(&idx) {
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
                .thumb_data
                .get(&index)
                .and_then(|entry| entry.clip_embedding.as_ref())
                .or_else(|| model.pending_clip_embeddings.get(&index)),
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
    if mods.control_key() || mods.alt_key() || mods.super_key() {
        return false;
    }

    if key == Key::Slash {
        if let Some(search) = model.search.as_mut() {
            if search.focused {
                return false;
            }
            search.focused = true;
            search.skip_next_char = true;
        } else {
            model.search = Some(SearchState {
                input: String::new(),
                focused: true,
                skip_next_char: true,
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
                        search.skip_next_char = false;
                    }
                    let request_id = model.next_search_request_id;
                    model.next_search_request_id = model.next_search_request_id.wrapping_add(1);
                    match model.clip_engine.request_text(request_id, query) {
                        Ok(()) => {
                            if let Some(search) = model.search.as_mut() {
                                search.pending_request = Some(request_id);
                                search.focused = false;
                                search.skip_next_char = false;
                            }
                        }
                        Err(err) => {
                            if let Some(search) = model.search.as_mut() {
                                search.error = Some(format!("Failed to queue search: {err}"));
                                search.focused = false;
                                search.skip_next_char = false;
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
                        search.skip_next_char = false;
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
            Key::N
                if matches!(model.mode, Mode::Thumbnails)
                    && model
                        .search
                        .as_ref()
                        .map(|s| !s.results.is_empty())
                        .unwrap_or(false) =>
            {
                let delta = if mods.shift_key() { -1 } else { 1 };
                advance_search(app, model, delta);
                return true;
            }
            Key::P
                if matches!(model.mode, Mode::Thumbnails)
                    && model
                        .search
                        .as_ref()
                        .map(|s| !s.results.is_empty())
                        .unwrap_or(false) =>
            {
                let delta = if mods.shift_key() { 1 } else { -1 };
                advance_search(app, model, delta);
                return true;
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
    let Some(rect) = current_window_rect(app, model) else {
        return matches!(model.mode, Mode::Single);
    };
    match model.mode {
        Mode::Thumbnails => {
            if len == 0 {
                return false;
            }
            let grid = ThumbnailGrid::new(model, rect);
            let cols = grid.cols();
            if cols == 0 {
                return false;
            }
            let current = model.current.min(len - 1);
            let mut row = current / cols;
            let mut col = current % cols;
            let total_rows = grid.rows();
            let mut changed = false;
            match dir {
                ArrowDirection::Up => {
                    if row > 0 {
                        row -= 1;
                        let row_len = grid.row_length(row).max(1);
                        col = col.min(row_len - 1);
                        changed = true;
                    }
                }
                ArrowDirection::Down => {
                    if row + 1 < total_rows {
                        row += 1;
                        let row_len = grid.row_length(row).max(1);
                        col = col.min(row_len - 1);
                        changed = true;
                    }
                }
                ArrowDirection::Left => {
                    if col > 0 {
                        col -= 1;
                        changed = true;
                    } else if row > 0 {
                        row -= 1;
                        let row_len = grid.row_length(row).max(1);
                        col = row_len - 1;
                        changed = true;
                    }
                }
                ArrowDirection::Right => {
                    let row_len = grid.row_length(row);
                    if col + 1 < row_len {
                        col += 1;
                        changed = true;
                    } else if row + 1 < total_rows {
                        row += 1;
                        col = 0;
                        changed = true;
                    }
                }
            }
            if changed {
                let mut idx = row * cols + col;
                if idx >= len {
                    idx = len - 1;
                }
                model.current = idx;
            }
            // Compute target row and column
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
            if search.skip_next_char {
                search.skip_next_char = false;
                return;
            }
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

    if app.keys.mods == ModifiersState::empty() {
        match key {
            Key::X => {
                model.terminal.visible = !model.terminal.visible;
                return;
            }
            Key::Left | Key::H if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                cycle_terminal_tab(model, -1);
                return;
            }
            Key::Right | Key::L
                if model.terminal.visible && !model.terminal.sessions.is_empty() =>
            {
                cycle_terminal_tab(model, 1);
                return;
            }
            Key::Up | Key::K if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                scroll_active_terminal(model, 1);
                return;
            }
            Key::Down | Key::J if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                scroll_active_terminal(model, -1);
                return;
            }
            Key::PageUp if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                let page = model.terminal.rows.max(1) as isize;
                scroll_active_terminal(model, page);
                return;
            }
            Key::PageDown if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                let page = model.terminal.rows.max(1) as isize;
                scroll_active_terminal(model, -page);
                return;
            }
            Key::Back if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                close_active_terminal(model);
                return;
            }
            _ => {}
        }
    }

    let len = model.image_paths.len();
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
                    if app.keys.mods.shift_key() {
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
            Key::H | Key::Left if handle_arrow(app, model, ArrowDirection::Left) => {
                return;
            }
            Key::L | Key::Right if handle_arrow(app, model, ArrowDirection::Right) => {
                return;
            }
            Key::K | Key::Up if handle_arrow(app, model, ArrowDirection::Up) => {
                return;
            }
            Key::J | Key::Down if handle_arrow(app, model, ArrowDirection::Down) => {
                return;
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
                    if let Some(rect) = current_window_rect(app, model) {
                        if let Some(tex) = model.full_textures.get(&model.current) {
                            let [w, h] = tex.size();
                            let fit = (rect.w() / w as f32).min(rect.h() / h as f32);
                            model.zoom = fit;
                        } else {
                            model.zoom = 1.0;
                        }
                    } else {
                        model.zoom = 1.0;
                    }
                    model.pan = vec2(0.0, 0.0);
                }
            }
            // Toggle full screen
            Key::F => {
                let fullscreen = if app.window.fullscreen().is_some() {
                    None
                } else {
                    Some(Fullscreen::Borderless(None))
                };
                app.window.set_fullscreen(fullscreen);
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
    } else if app.keys.mods == ModifiersState::SHIFT && key == Key::G {
        if let Mode::Thumbnails = model.mode {
            let len = model.image_paths.len();
            if len > 0 {
                model.current = len - 1;
            }
        }
    }
    // Custom key bindings execution
    let current_file = model.image_paths[model.current]
        .to_string_lossy()
        .to_string();
    let mut commands_to_launch = Vec::new();
    for binding in &model.key_bindings {
        if key == binding.key
            && app.keys.mods.control_key() == binding.ctrl
            && app.keys.mods.shift_key() == binding.shift
            && app.keys.mods.alt_key() == binding.alt
            && app.keys.mods.super_key() == binding.super_key
        {
            commands_to_launch.push((
                binding.command.replace("{file}", &current_file),
                binding.use_terminal,
            ));
        }
    }
    for (cmd, use_terminal) in commands_to_launch {
        if use_terminal {
            launch_terminal_command(app, model, cmd);
        } else {
            launch_detached_command(cmd);
        }
    }

    // Auto-scroll to keep current thumbnail in view
    if let Mode::Thumbnails = model.mode {
        ensure_thumbnail_visible(app, model, model.current);
    }
    // On thumbnail mode selection (via keys), reset preload timer
    if let Mode::Thumbnails = model.mode {
        model.selection_changed_at = Instant::now();
        model.selection_pending = false;
    }
}

/// Update function to process incoming thumbnail images.
fn update(app: &App, model: &mut Model, renderer: &Renderer) {
    sync_terminal_viewport(app, model);

    while let Ok(update) = model.thumb_rx.try_recv() {
        handle_thumbnail_update(app, model, update);
    }
    loop {
        match model.clip_engine.try_recv() {
            Ok(event) => match event {
                ClipEvent::ImageReady { index, embedding } => {
                    if let Some(entry) = model.thumb_data.get_mut(&index) {
                        entry.clip_embedding = Some(embedding);
                        model.pending_clip_embeddings.remove(&index);
                        model.clip_missing.remove(&index);
                        model.clip_inflight.remove(&index);
                        update_search_with_image_embedding(app, model, index);
                    } else {
                        model.pending_clip_embeddings.insert(index, embedding);
                        model.clip_inflight.remove(&index);
                        model.clip_missing.remove(&index);
                    }
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
    // Receive command terminal events.
    while let Ok(event) = model.command_rx.try_recv() {
        match event {
            CommandEvent::Output { session_id, bytes } => {
                if let Some(session) = model
                    .terminal
                    .sessions
                    .iter_mut()
                    .find(|session| session.id == session_id)
                {
                    session.parser.process(&bytes);
                    session
                        .parser
                        .screen_mut()
                        .set_scrollback(session.scrollback_offset);
                    session.scrollback_offset = session.parser.screen().scrollback();
                }
            }
            CommandEvent::Finished {
                session_id,
                exit_code,
                signal,
            } => {
                if let Some(session) = model
                    .terminal
                    .sessions
                    .iter_mut()
                    .find(|session| session.id == session_id)
                {
                    session.running = false;
                    session.exit_code = Some(exit_code);
                    session.signal = signal;
                    session.master = None;
                }
            }
            CommandEvent::Failed { session_id, error } => {
                if let Some(session) = model
                    .terminal
                    .sessions
                    .iter_mut()
                    .find(|session| session.id == session_id)
                {
                    if session.error.is_none() {
                        session.parser.process(format!("{error}\r\n").as_bytes());
                    }
                    session.error = Some(error);
                    session.running = false;
                    session.master = None;
                }
            }
        }
    }
    detect_file_changes(app, model);

    // Process loaded full-resolution tile data
    while let Ok(message) = model.full_resp_rx.try_recv() {
        match message {
            FullImageMessage::Loaded {
                index: idx,
                full_w,
                full_h,
                tiles,
            } => {
                // Store raw pixel data for lazy texture creation
                let mut prepared_tiles = Vec::new();

                for (x_offset, y_offset, width, height, format, pixel_data) in tiles {
                    prepared_tiles.push(Tile {
                        x_offset,
                        y_offset,
                        width,
                        height,
                        format,
                        pixel_data,
                        texture: None,
                    });
                }
                let tiled = TiledTexture {
                    full_w,
                    full_h,
                    tiles: prepared_tiles,
                };
                // Insert into cache and update LRU
                model.full_textures.insert(idx, tiled);
                touch_full_texture(model, idx);
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
            FullImageMessage::Failed { index: idx, error } => {
                model.full_pending.insert(
                    idx,
                    FullPendingState::Failed {
                        last_error_at: Instant::now(),
                    },
                );
                let path_info = model
                    .image_paths
                    .get(idx)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| format!("image index {idx}"));
                eprintln!("failed to load full image {path_info}: {error}");
                model.full_textures.remove(&idx);
                if let Some(pos) = model.full_usage.iter().position(|&i| i == idx) {
                    model.full_usage.remove(pos);
                }
            }
        }
    }
    // Handle window resize: update view parameters and re-apply fit if in fit mode
    let window_rect = current_window_rect(app, model);
    if let Some(rect) = window_rect {
        if rect != model.prev_window_rect {
            model.prev_window_rect = rect;
            if let Mode::Single = model.mode {
                if model.fit_mode {
                    apply_fit(app, model);
                }
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
        if let Some(rect) = window_rect {
            let grid = ThumbnailGrid::new(model, rect);
            model.scroll_offset = model.scroll_offset.clamp(0.0, grid.max_scroll());
        }
    }
    if matches!(model.mode, Mode::Single) && !model.full_textures.contains_key(&model.current) {
        request_full_texture(model, model.current);
    }
    update_thumbnail_requests(app, model, renderer);
}

fn terminal_status_text(session: &TerminalSession) -> String {
    if let Some(error) = &session.error {
        return error.clone();
    }
    if session.running {
        return "running".to_string();
    }
    if let Some(signal) = &session.signal {
        return format!("terminated by {signal}");
    }
    match session.exit_code {
        Some(0) => "completed".to_string(),
        Some(code) => format!("exit {code}"),
        None => "finished".to_string(),
    }
}

fn terminal_tab_label(session: &TerminalSession) -> String {
    if let Some(error) = &session.error {
        return format!("{}  {}", session.title, error);
    }
    if session.running {
        return format!("{}  running", session.title);
    }
    if let Some(signal) = &session.signal {
        return format!("{}  {}", session.title, signal);
    }
    if matches!(session.exit_code, Some(0) | None) {
        return session.title.clone();
    }
    format!("{}  exit {}", session.title, session.exit_code.unwrap())
}

fn terminal_row_start(screen: &vt100::Screen, visible_rows: u16, cols: u16) -> u16 {
    let (rows, _) = screen.size();
    let (cursor_row, _) = screen.cursor_position();
    let mut last_content_row = 0_u16;
    for row in 0..rows {
        let mut has_content = false;
        for col in 0..cols {
            if screen
                .cell(row, col)
                .is_some_and(|cell| cell.has_contents())
            {
                has_content = true;
                break;
            }
        }
        if has_content {
            last_content_row = row;
        }
    }
    let anchor_row = last_content_row.max(cursor_row);
    let max_start = rows.saturating_sub(visible_rows);
    anchor_row
        .saturating_add(1)
        .saturating_sub(visible_rows)
        .min(max_start)
}

fn terminal_view_row_start(session: &TerminalSession, visible_rows: u16, visible_cols: u16) -> u16 {
    if session.scrollback_offset > 0 {
        return 0;
    }
    terminal_row_start(session.parser.screen(), visible_rows, visible_cols)
}

fn draw_terminal_panel(draw: &mut Scene<'_>, model: &Model, rect: Rect) {
    if !model.terminal.visible {
        return;
    }

    let panel_rect = terminal_panel_rect(rect);
    let body_rect = terminal_body_rect(panel_rect);
    let tabs_y = panel_rect.top() - TERMINAL_TAB_HEIGHT / 2.0;
    let status_y = panel_rect.bottom() + TERMINAL_STATUS_HEIGHT / 2.0;
    let panel_bg = srgba(0.03, 0.04, 0.05, 0.72);
    let body_bg = srgba(0.05, 0.06, 0.08, 0.82);
    let default_fg = srgba(0.88, 0.9, 0.92, 1.0);
    let default_bg = body_bg;

    draw.rect()
        .x_y(panel_rect.x(), panel_rect.y())
        .w_h(panel_rect.w(), panel_rect.h())
        .color(panel_bg);
    draw.rect()
        .x_y(body_rect.x(), body_rect.y())
        .w_h(body_rect.w(), body_rect.h())
        .color(body_bg);

    if model.terminal.sessions.is_empty() {
        draw.text("No terminal sessions yet")
            .font_size(16)
            .color(default_fg)
            .x_y(body_rect.x(), body_rect.y());
        return;
    }

    let tab_count = model.terminal.sessions.len().max(1) as f32;
    let tab_width = panel_rect.w() / tab_count;
    let tabs_start = panel_rect.left() + tab_width / 2.0;
    for (idx, session) in model.terminal.sessions.iter().enumerate() {
        let x = tabs_start + idx as f32 * tab_width;
        let is_active = idx == model.terminal.active;
        let tab_bg = if is_active && session.running {
            srgba(0.15, 0.38, 0.2, 0.82)
        } else if is_active {
            srgba(0.2, 0.24, 0.28, 0.82)
        } else if session.running {
            srgba(0.1, 0.2, 0.14, 0.7)
        } else {
            srgba(0.11, 0.12, 0.15, 0.7)
        };
        let label = terminal_tab_label(session);
        draw.rect()
            .x_y(x, tabs_y)
            .w_h((tab_width - 2.0).max(1.0), TERMINAL_TAB_HEIGHT - 4.0)
            .color(tab_bg);
        draw.text(&label)
            .font_size(13)
            .color(default_fg)
            .w_h((tab_width - 14.0).max(1.0), TERMINAL_TAB_HEIGHT - 4.0)
            .x_y(x, tabs_y - 1.0)
            .left_justify();
    }

    let session = active_terminal_session(model).unwrap();
    let screen = session.parser.screen();
    let (rows, cols) = screen.size();
    let visible_rows = rows.min(model.terminal.rows.max(1));
    let visible_cols = cols.min(model.terminal.cols.max(1));
    let row_start = terminal_view_row_start(session, visible_rows, visible_cols);
    let origin_x = body_rect.left() + TERMINAL_MARGIN + TERMINAL_CELL_WIDTH / 2.0;
    let origin_y = body_rect.top() - TERMINAL_MARGIN - TERMINAL_CELL_HEIGHT / 2.0;

    for visible_row in 0..visible_rows {
        let row = row_start + visible_row;
        for col in 0..visible_cols {
            let Some(cell) = screen.cell(row, col) else {
                continue;
            };
            if cell.is_wide_continuation() {
                continue;
            }

            let mut fg = vt_color_to_rgba(cell.fgcolor(), cell.bold(), default_fg);
            let mut bg = vt_color_to_rgba(cell.bgcolor(), false, default_bg);
            if cell.inverse() {
                std::mem::swap(&mut fg, &mut bg);
            }

            let x = origin_x + col as f32 * TERMINAL_CELL_WIDTH;
            let y = origin_y - visible_row as f32 * TERMINAL_CELL_HEIGHT;

            if bg != default_bg {
                draw.rect()
                    .x_y(x, y)
                    .w_h(TERMINAL_CELL_WIDTH, TERMINAL_CELL_HEIGHT)
                    .color(bg);
            }

            if cell.has_contents() {
                draw.text(cell.contents())
                    .font_size(TERMINAL_FONT_SIZE)
                    .color(fg)
                    .w_h(TERMINAL_CELL_WIDTH * 2.0, TERMINAL_CELL_HEIGHT)
                    .x_y(x, y - 1.0)
                    .left_justify();
            }

            if cell.underline() {
                draw.line()
                    .start(vec2(
                        x - TERMINAL_CELL_WIDTH / 2.0,
                        y - TERMINAL_CELL_HEIGHT / 2.6,
                    ))
                    .end(vec2(
                        x + TERMINAL_CELL_WIDTH / 2.0,
                        y - TERMINAL_CELL_HEIGHT / 2.6,
                    ))
                    .weight(1.0)
                    .color(fg);
            }
        }
    }

    let status = format!(
        "{} | {} | x toggle | arrows tabs/scroll | backspace close tab",
        session.command,
        terminal_status_text(session)
    );
    draw.rect()
        .x_y(0.0, status_y)
        .w_h(panel_rect.w(), TERMINAL_STATUS_HEIGHT)
        .color(srgba(0.08, 0.09, 0.11, 0.78));
    draw.text(&status)
        .font_size(12)
        .color(default_fg)
        .w_h(panel_rect.w() - 20.0, TERMINAL_STATUS_HEIGHT)
        .x_y(0.0, status_y - 1.0)
        .left_justify();
}

fn touch_full_texture(model: &mut Model, idx: usize) {
    if !model.full_textures.contains_key(&idx) {
        return;
    }
    if let Some(pos) = model.full_usage.iter().position(|&i| i == idx) {
        model.full_usage.remove(pos);
    }
    model.full_usage.push_front(idx);
}

/// Ensure the full-resolution texture for `idx` is loaded and update LRU cache.
/// Request loading of full-resolution image at `idx` in background.  Adds to pending set.
fn request_full_texture(model: &mut Model, idx: usize) {
    if model.full_textures.contains_key(&idx) {
        touch_full_texture(model, idx);
        return;
    }
    let now = Instant::now();
    let should_request = match model.full_pending.get(&idx) {
        None => true,
        Some(FullPendingState::InFlight { .. }) => false,
        Some(FullPendingState::Failed { last_error_at }) => {
            now.duration_since(*last_error_at) > FULL_PENDING_RETRY
        }
    };
    if should_request {
        model
            .full_pending
            .insert(idx, FullPendingState::InFlight { _requested_at: now });
        if let Err(err) = model.full_req_tx.send(idx) {
            model
                .full_pending
                .insert(idx, FullPendingState::Failed { last_error_at: now });
            eprintln!("failed to request full image load for index {idx}: {err}");
        }
    }
}

fn update_thumbnail_requests(app: &App, model: &mut Model, renderer: &Renderer) {
    if !matches!(model.mode, Mode::Thumbnails) {
        return;
    }
    let total = model.image_paths.len();
    if total == 0 {
        model.thumb_visible.clear();
        return;
    }
    let Some(rect) = current_window_rect(app, model) else {
        return;
    };
    let grid = ThumbnailGrid::new(model, rect);
    let visible = grid.visible_indices();

    let window_changed = if rect != model.prev_window_rect {
        model.prev_window_rect = rect;
        true
    } else {
        false
    };
    let scroll_changed = if (model.scroll_offset - model.prev_scroll).abs() > f32::EPSILON {
        model.prev_scroll = model.scroll_offset;
        true
    } else {
        false
    };
    if window_changed || scroll_changed {
        model
            .thumb_queue
            .reprioritize(|idx| grid.viewport_priority(idx));
    }
    let visible_set: HashSet<usize> = visible.iter().copied().collect();
    let mut to_remove = Vec::new();
    for idx in model.thumb_visible.keys() {
        if !visible_set.contains(idx) {
            to_remove.push(*idx);
        }
    }
    for idx in to_remove {
        model.thumb_visible.remove(&idx);
    }

    for idx in visible {
        let center = grid.index_center(idx).unwrap_or(vec2(0.0, 0.0));
        if let Some(slot) = model.thumb_visible.get_mut(&idx) {
            slot.center = center;
            continue;
        }
        if let Some(entry) = model.thumb_data.get(&idx) {
            let texture = renderer.texture_from_image(&entry.image);
            let size = texture.size();
            model.thumb_visible.insert(
                idx,
                ThumbnailTexture {
                    texture,
                    center,
                    size,
                },
            );
        }
    }
}

fn handle_thumbnail_update(app: &App, model: &mut Model, update: ThumbnailUpdate) {
    let ThumbnailUpdate {
        index,
        image,
        clip_embedding,
    } = update;
    let mut final_embedding = clip_embedding;
    if final_embedding.is_none() {
        if let Some(pending) = model.pending_clip_embeddings.remove(&index) {
            final_embedding = Some(pending);
        }
    } else {
        model.pending_clip_embeddings.remove(&index);
    }
    let has_embedding = final_embedding.is_some();
    model.thumb_data.insert(
        index,
        ThumbnailEntry {
            image,
            clip_embedding: final_embedding,
        },
    );
    if has_embedding {
        model.clip_missing.remove(&index);
        model.clip_inflight.remove(&index);
        update_search_with_image_embedding(app, model, index);
    } else {
        model.clip_missing.remove(&index);
        model.clip_inflight.insert(index);
    }
}

fn detect_file_changes(app: &App, model: &mut Model) {
    let total = model.image_paths.len();
    if total == 0 {
        return;
    }
    let mut candidates: HashSet<usize> = HashSet::new();
    candidates.insert(model.current);
    if matches!(model.mode, Mode::Thumbnails) {
        if let Some(rect) = current_window_rect(app, model) {
            for idx in ThumbnailGrid::new(model, rect).visible_indices() {
                candidates.insert(idx);
            }
        }
    }
    let batch = FILE_WATCH_BATCH.min(total);
    for _ in 0..batch {
        let idx = model.file_watch_cursor;
        model.file_watch_cursor = (model.file_watch_cursor + 1) % total;
        candidates.insert(idx);
    }
    for idx in candidates {
        check_image_modification(model, idx);
    }
}

fn check_image_modification(model: &mut Model, idx: usize) {
    if idx >= model.image_paths.len() {
        return;
    }
    let path = &model.image_paths[idx];
    let old_mod = model.file_mod_times[idx];
    let new_mod = current_mod_time(path);
    let changed = match (old_mod, new_mod) {
        (Some(old), Some(new)) => match new.duration_since(old) {
            Ok(diff) => diff > Duration::ZERO,
            Err(_) => true,
        },
        (None, None) => false,
        _ => true,
    };
    model.file_mod_times[idx] = new_mod;
    if changed {
        handle_image_modified(model, idx);
    }
}

fn handle_image_modified(model: &mut Model, idx: usize) {
    model.thumb_data.remove(&idx);
    model.pending_clip_embeddings.remove(&idx);
    model.thumb_visible.remove(&idx);
    model.thumb_queue.enqueue(idx);

    model.full_textures.remove(&idx);
    if let Some(pos) = model.full_usage.iter().position(|&i| i == idx) {
        model.full_usage.remove(pos);
    }
    model.full_pending.remove(&idx);
    if matches!(model.mode, Mode::Single) && idx == model.current {
        request_full_texture(model, idx);
    }

    model.clip_missing.insert(idx);
    model.clip_inflight.remove(&idx);
}

fn current_mod_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|meta| meta.modified()).ok()
}

fn current_window_rect(app: &App, model: &Model) -> Option<Rect> {
    let _ = model;
    Some(app.rect)
}

/// Apply fit-to-window for current single-image view
fn apply_fit(app: &App, model: &mut Model) {
    model.fit_mode = true;
    if let Some(rect) = current_window_rect(app, model) {
        if let Some(tex) = model.full_textures.get(&model.current) {
            let [w, h] = tex.size();
            model.zoom = (rect.w() / w as f32).min(rect.h() / h as f32);
        } else {
            model.zoom = 1.0;
        }
    } else {
        model.zoom = 1.0;
    }
    model.pan = vec2(0.0, 0.0);
}

fn prepare_current_full_textures(renderer: &Renderer, model: &mut Model) {
    let Some(tiled) = model.full_textures.get_mut(&model.current) else {
        return;
    };
    for tile in &mut tiled.tiles {
        if tile.texture.is_some() {
            continue;
        }
        let (format, bytes_per_row, pixels): (
            wgpu::TextureFormat,
            u32,
            std::borrow::Cow<'_, [u8]>,
        ) = match tile.format {
            TilePixelFormat::Rgba8 => (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                4 * tile.width,
                std::borrow::Cow::Borrowed(&tile.pixel_data),
            ),
            TilePixelFormat::Rgba16 if renderer.supports_rgba16() => (
                wgpu::TextureFormat::Rgba16Unorm,
                8 * tile.width,
                std::borrow::Cow::Borrowed(&tile.pixel_data),
            ),
            TilePixelFormat::Rgba16 => (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                4 * tile.width,
                std::borrow::Cow::Owned(rgba16_bytes_to_rgba8(&tile.pixel_data)),
            ),
        };
        tile.texture = Some(renderer.create_texture(
            "sriv full-resolution tile",
            tile.width,
            tile.height,
            format,
            bytes_per_row,
            pixels.as_ref(),
        ));
    }
}

fn view(app: &App, model: &mut Model, renderer: &mut Renderer) -> Result<()> {
    prepare_current_full_textures(renderer, model);
    let Some(rect) = current_window_rect(app, model) else {
        return Ok(());
    };
    let mut draw = Scene::new(rect);
    match model.mode {
        Mode::Thumbnails => {
            let grid = ThumbnailGrid::new(model, rect);
            if let Some((row_min, row_max)) = grid.visible_rows() {
                for row in row_min..=row_max {
                    for col in 0..grid.cols() {
                        let i = row * grid.cols() + col;
                        if i >= grid.total() {
                            break;
                        }
                        let center = match grid.index_center(i) {
                            Some(c) => c,
                            None => continue,
                        };

                        if let Some(slot) = model.thumb_visible.get(&i) {
                            let [tw, th] = slot.size;
                            let w = tw as f32;
                            let h = th as f32;
                            draw.texture(&slot.texture)
                                .x_y(center.x, center.y)
                                .w_h(w, h);

                            if model.thumb_has_xmp.get(i).copied().unwrap_or(false) {
                                let icon_w = 40.0;
                                let icon_h = 20.0;
                                let margin = 6.0;
                                let icon_center_x = center.x + w / 2.0 - icon_w / 2.0 - margin;
                                let icon_center_y = center.y + h / 2.0 - icon_h / 2.0 - margin;
                                draw.rect()
                                    .x_y(icon_center_x, icon_center_y)
                                    .w_h(icon_w, icon_h)
                                    .color(srgba(1.0, 0.0, 0.0, 0.85));
                                draw.text("XMP")
                                    .font_size(12)
                                    .w_h(icon_w, icon_h)
                                    .x_y(icon_center_x, icon_center_y - 1.0)
                                    .color(WHITE);
                            }
                            if i == model.current {
                                draw.rect()
                                    .x_y(center.x, center.y)
                                    .w_h(w + 4.0, h + 4.0)
                                    .no_fill()
                                    .stroke(WHITE)
                                    .stroke_weight(2.0);
                            }
                        } else {
                            let thumb_w = model.thumb_size as f32;
                            let thumb_h = model.thumb_size as f32;
                            draw.rect()
                                .x_y(center.x, center.y)
                                .w_h(thumb_w, thumb_h)
                                .color(srgba(0.5, 0.5, 0.5, 1.0));
                            if i == model.current {
                                draw.rect()
                                    .x_y(center.x, center.y)
                                    .w_h(thumb_w + 4.0, thumb_h + 4.0)
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
                    draw.texture(tile.texture.as_ref().unwrap())
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
        let bar_y = rect.top() - bar_h / 2.0;
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

    draw_terminal_panel(&mut draw, model, rect);
    renderer.render(&draw, app.window.scale_factor())
}
