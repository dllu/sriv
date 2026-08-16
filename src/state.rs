use crate::clip::ClipEngine;
use crate::geometry::{Rect, Vec2};
use crate::input::Key;
use crate::renderer::GpuTexture;
use crossbeam_channel::{Receiver as CbReceiver, Sender as CbSender};
use image::DynamicImage;
use portable_pty::MasterPty;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Instant, SystemTime};
use toml::Value as TomlValue;
use vt100::Parser;

#[derive(Debug)]
pub enum Mode {
    Thumbnails,
    Single,
}

#[derive(Debug)]
pub struct KeyBinding {
    pub key: Key,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub super_key: bool,
    pub command: String,
    pub use_terminal: bool,
}

#[derive(Clone, Debug)]
pub struct ThumbRequestQueue {
    inner: Arc<ThumbQueueInner>,
}

#[derive(Debug)]
struct ThumbQueueInner {
    state: Mutex<ThumbQueueState>,
    condvar: Condvar,
}

#[derive(Debug, Default)]
struct ThumbQueueState {
    order: VecDeque<usize>,
    members: HashSet<usize>,
    closed: bool,
}

impl ThumbRequestQueue {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ThumbQueueInner {
                state: Mutex::new(ThumbQueueState::default()),
                condvar: Condvar::new(),
            }),
        }
    }

    pub fn enqueue(&self, index: usize) {
        self.enqueue_batch(std::iter::once(index));
    }

    pub fn enqueue_batch<I>(&self, indices: I)
    where
        I: IntoIterator<Item = usize>,
    {
        let mut state = self.inner.state.lock().unwrap();
        if state.closed {
            return;
        }
        let mut inserted = false;
        for index in indices {
            if state.members.insert(index) {
                state.order.push_back(index);
                inserted = true;
            }
        }
        if inserted {
            self.inner.condvar.notify_all();
        }
    }

    pub fn pop(&self) -> Option<usize> {
        let mut state = self.inner.state.lock().unwrap();
        loop {
            if let Some(idx) = state.order.pop_front() {
                state.members.remove(&idx);
                return Some(idx);
            }
            if state.closed {
                return None;
            }
            state = self.inner.condvar.wait(state).unwrap();
        }
    }

    pub fn reprioritize<F>(&self, mut priority: F)
    where
        F: FnMut(usize) -> f32,
    {
        let mut state = self.inner.state.lock().unwrap();
        if state.order.len() <= 1 {
            return;
        }
        let mut scored: Vec<(usize, f32)> = state
            .order
            .iter()
            .copied()
            .map(|idx| (idx, priority(idx)))
            .collect();
        scored.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        state.order.clear();
        for (idx, _) in scored {
            state.order.push_back(idx);
        }
    }

    pub fn close(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.closed = true;
        self.inner.condvar.notify_all();
    }
}

fn parse_binding_spec(spec: &str, command: &str, use_terminal: bool) -> Option<KeyBinding> {
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
                if key_opt.is_some() {
                    return None;
                }
                let key = match tok.to_ascii_uppercase().as_str() {
                    c if c.len() == 1 && c.chars().all(|ch| ch.is_ascii_alphabetic()) => {
                        let ch = c.chars().next().unwrap();
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
        use_terminal,
    })
}

pub fn parse_bindings(s: &str) -> Vec<KeyBinding> {
    let mut bindings = Vec::new();
    if let Ok(TomlValue::Table(table)) = toml::from_str::<TomlValue>(s) {
        for (spec, val) in table {
            match val {
                TomlValue::String(cmd) => {
                    if let Some(binding) = parse_binding_spec(&spec, &cmd, true) {
                        bindings.push(binding);
                    }
                }
                TomlValue::Table(binding_table) => {
                    let command = binding_table.get("command").and_then(TomlValue::as_str);
                    let use_terminal = binding_table
                        .get("terminal")
                        .and_then(TomlValue::as_bool)
                        .unwrap_or(true);
                    if let Some(command) = command {
                        if let Some(binding) = parse_binding_spec(&spec, command, use_terminal) {
                            bindings.push(binding);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    bindings
}

pub fn parse_ui_font_path(s: &str) -> Option<PathBuf> {
    let value = toml::from_str::<TomlValue>(s).ok()?;
    let table = value.as_table()?;
    table
        .get("ui_font_path")
        .and_then(TomlValue::as_str)
        .map(PathBuf::from)
}

#[derive(Debug)]
pub struct SearchState {
    pub input: String,
    pub focused: bool,
    pub skip_next_char: bool,
    pub results: Vec<(usize, f32)>,
    pub current: usize,
    pub pending_request: Option<u64>,
    pub error: Option<String>,
    pub last_embedding: Option<Vec<f32>>,
}

#[derive(Debug)]
pub enum FullPendingState {
    InFlight { _requested_at: Instant },
    Failed { last_error_at: Instant },
}

pub type FullImageTile = (u32, u32, u32, u32, TilePixelFormat, Vec<u8>);

#[derive(Debug)]
pub enum FullImageMessage {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TilePixelFormat {
    /// Encoded with the output space's sRGB transfer function; sampled through an sRGB texture.
    Rgba8,
    /// Linear-light output-space values; sampled through a linear 16-bit UNORM texture.
    Rgba16,
}

#[derive(Debug)]
pub struct Tile {
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    pub format: TilePixelFormat,
    pub pixel_data: Vec<u8>,
    pub texture: Option<GpuTexture>,
}

#[derive(Debug)]
pub struct TiledTexture {
    pub full_w: u32,
    pub full_h: u32,
    pub tiles: Vec<Tile>,
}

impl TiledTexture {
    pub fn size(&self) -> [u32; 2] {
        [self.full_w, self.full_h]
    }
}

#[derive(Debug)]
pub struct ThumbnailEntry {
    pub image: DynamicImage,
    pub clip_embedding: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct ThumbnailUpdate {
    pub index: usize,
    pub image: DynamicImage,
    pub clip_embedding: Option<Vec<f32>>,
}

#[derive(Debug)]
pub enum CommandEvent {
    Output {
        session_id: u64,
        bytes: Vec<u8>,
    },
    Finished {
        session_id: u64,
        exit_code: u32,
        signal: Option<String>,
    },
    Failed {
        session_id: u64,
        error: String,
    },
}

pub struct TerminalSession {
    pub id: u64,
    pub title: String,
    pub command: String,
    pub parser: Parser,
    pub master: Option<Box<dyn MasterPty + Send>>,
    pub scrollback_offset: usize,
    pub running: bool,
    pub exit_code: Option<u32>,
    pub signal: Option<String>,
    pub error: Option<String>,
}

pub struct TerminalState {
    pub sessions: Vec<TerminalSession>,
    pub visible: bool,
    pub active: usize,
    pub next_id: u64,
    pub rows: u16,
    pub cols: u16,
}

pub struct Model {
    pub image_paths: Vec<PathBuf>,
    pub ui_font_path: Option<PathBuf>,
    pub thumb_visible: HashMap<usize, ThumbnailTexture>,
    pub thumb_data: HashMap<usize, ThumbnailEntry>,
    pub thumb_has_xmp: Vec<bool>,
    pub thumb_rx: Receiver<ThumbnailUpdate>,
    pub thumb_queue: ThumbRequestQueue,
    pub file_mod_times: Vec<Option<SystemTime>>,
    pub file_watch_cursor: usize,
    pub full_req_tx: CbSender<usize>,
    pub full_resp_rx: CbReceiver<FullImageMessage>,
    pub full_pending: HashMap<usize, FullPendingState>,
    pub full_textures: HashMap<usize, TiledTexture>,
    pub full_usage: VecDeque<usize>,
    pub mode: Mode,
    pub current: usize,
    pub thumb_size: u32,
    pub gap: f32,
    pub scroll_offset: f32,
    pub zoom: f32,
    pub pan: Vec2,
    pub prev_window_rect: Rect,
    pub prev_scroll: f32,
    pub fit_mode: bool,
    pub selection_changed_at: Instant,
    pub selection_pending: bool,
    pub key_bindings: Vec<KeyBinding>,
    pub command_tx: Sender<CommandEvent>,
    pub command_rx: Receiver<CommandEvent>,
    pub terminal: TerminalState,
    pub clip_engine: ClipEngine,
    pub clip_missing: HashSet<usize>,
    pub clip_inflight: HashSet<usize>,
    pub pending_clip_embeddings: HashMap<usize, Vec<f32>>,
    pub next_search_request_id: u64,
    pub search: Option<SearchState>,
}

impl Drop for Model {
    fn drop(&mut self) {
        self.thumb_queue.close();
    }
}

#[derive(Debug)]
pub struct ThumbnailTexture {
    pub texture: GpuTexture,
    pub center: Vec2,
    pub size: [u32; 2],
}
