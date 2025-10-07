use crate::clip::{ClipEngine, ThumbnailCache};
use crate::FullImageMessage;
use crossbeam_channel::{Receiver as CbReceiver, Sender as CbSender};
use nannou::image::DynamicImage;
use nannou::prelude::{Key, Rect, Vec2};
use nannou::wgpu;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Instant, SystemTime};
use toml::Value as TomlValue;

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
}

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
    })
}

pub fn parse_bindings(s: &str) -> Vec<KeyBinding> {
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

#[derive(Debug)]
pub struct Tile {
    pub x_offset: u32,
    pub y_offset: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_data: Vec<u8>,
    pub texture: RefCell<Option<wgpu::Texture>>,
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
pub struct Model {
    pub image_paths: Vec<PathBuf>,
    pub thumb_textures: HashMap<usize, wgpu::Texture>,
    pub thumb_cache: ThumbnailCache,
    pub thumb_has_xmp: Vec<bool>,
    pub thumb_rx: Receiver<(usize, DynamicImage)>,
    pub thumb_req_tx: CbSender<usize>,
    pub thumb_pending: HashSet<usize>,
    pub thumb_usage: VecDeque<usize>,
    pub thumb_capacity: usize,
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
    pub fit_mode: bool,
    pub selection_changed_at: Instant,
    pub selection_pending: bool,
    pub key_bindings: Vec<KeyBinding>,
    pub command_tx: Sender<String>,
    pub command_rx: Receiver<String>,
    pub command_output: Option<String>,
    pub clip_engine: ClipEngine,
    pub clip_embeddings: Vec<Option<Vec<f32>>>,
    pub clip_missing: HashSet<usize>,
    pub clip_inflight: HashSet<usize>,
    pub next_search_request_id: u64,
    pub search: Option<SearchState>,
}
