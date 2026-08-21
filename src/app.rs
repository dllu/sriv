use anyhow::Result;
use crossbeam_channel::unbounded;
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{ModifiersState, PhysicalKey};
#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::window::{Fullscreen, Window};

use crate::clip::{self, ClipEngine, ClipEvent};
use crate::color::OutputColorSpace;
use crate::geometry::{vec2, Rect, Vec2};
use crate::grid::ThumbnailGrid;
use crate::image_io;
use crate::input::Key;
use crate::renderer::Renderer;
use crate::state::{
    parse_bindings, parse_ui_font_path, CommandEvent, FullImageMessage, FullPendingState, Mode,
    Model, SearchState, TerminalState, ThumbRequestQueue, ThumbnailEntry, ThumbnailTexture,
    ThumbnailUpdate, Tile, TiledFrame, TiledTexture,
};
use crate::ui;

/// Maximum number of full-resolution images to cache in memory.
const FULL_CACHE_CAPACITY: usize = 4;
/// How long to wait before retrying a full-resolution load request.
const FULL_PENDING_RETRY: Duration = Duration::from_secs(5);
/// Delay before preloading the currently selected thumbnail.
const SELECTION_PRELOAD_DELAY: Duration = Duration::from_millis(200);
/// Number of files to poll for modifications each update tick.
const FILE_WATCH_BATCH: usize = 32;
/// File polling is deliberately much slower than interactive updates.
const FILE_WATCH_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug)]
enum UserEvent {
    Wake,
}

#[derive(Clone)]
pub(crate) struct AppProxy {
    event_loop: EventLoopProxy<UserEvent>,
    wake_pending: Arc<AtomicBool>,
}

impl AppProxy {
    fn new(event_loop: EventLoopProxy<UserEvent>) -> Self {
        Self {
            event_loop,
            wake_pending: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn wakeup(&self) {
        if self.wake_pending.swap(true, AtomicOrdering::AcqRel) {
            return;
        }
        if self.event_loop.send_event(UserEvent::Wake).is_err() {
            self.wake_pending.store(false, AtomicOrdering::Release);
        }
    }

    fn clear_wakeup(&self) {
        self.wake_pending.store(false, AtomicOrdering::Release);
    }
}

#[derive(Default)]
struct Keys {
    mods: ModifiersState,
    pressed_while_focused: HashSet<PhysicalKey>,
    ignored_until_release: HashSet<PhysicalKey>,
}

impl Keys {
    fn focus_changed(&mut self, focused: bool) {
        if !focused {
            self.mods = ModifiersState::empty();
            self.pressed_while_focused.clear();
            self.ignored_until_release.clear();
        }
    }

    fn should_handle(
        &mut self,
        physical_key: PhysicalKey,
        state: ElementState,
        repeat: bool,
        is_synthetic: bool,
    ) -> bool {
        if state == ElementState::Released {
            self.pressed_while_focused.remove(&physical_key);
            self.ignored_until_release.remove(&physical_key);
            return false;
        }

        // Synthetic presses describe keys that were already down when focus arrived. Keep
        // ignoring that physical key until its release: on X11, its first subsequent repeat can
        // be reported as a non-repeat.
        if is_synthetic || self.ignored_until_release.contains(&physical_key) {
            self.pressed_while_focused.remove(&physical_key);
            self.ignored_until_release.insert(physical_key);
            return false;
        }

        // Platforms without synthetic focus presses can still expose the held key as a repeat.
        if repeat && !self.pressed_while_focused.contains(&physical_key) {
            self.ignored_until_release.insert(physical_key);
            return false;
        }

        self.pressed_while_focused.insert(physical_key);
        true
    }
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

pub(crate) struct App {
    window: Arc<Window>,
    proxy: AppProxy,
    keys: Keys,
    mouse: Mouse,
    rect: Rect,
    exit_requested: Cell<bool>,
}

impl App {
    pub(crate) fn create_proxy(&self) -> AppProxy {
        self.proxy.clone()
    }

    pub(crate) fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    pub(crate) fn rect(&self) -> Rect {
        self.rect
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

/// Mouse click handler: select thumbnail on left-click in thumbnail mode.
fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    if let Mode::Thumbnails = model.mode {
        if button == MouseButton::Left {
            let pos = app.mouse.position();
            let rect = app.rect();
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
            ui::scroll_active_terminal(model, scroll_amount);
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
            let rect = app.rect();
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

/// The model function for initializing the application state.
fn model(app: &App, output_color_space: OutputColorSpace) -> Model {
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
        eprintln!("Usage: sriv [--clear-cache] <image files or directories>...");
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
                if path.is_file() && image_io::is_supported_image_path(&path) {
                    image_paths.push(path.canonicalize().unwrap());
                }
            }
        } else if pb.is_file() && image_io::is_supported_image_path(&pb) {
            image_paths.push(pb.canonicalize().unwrap());
        }
    }
    if image_paths.is_empty() {
        eprintln!("No image files found in arguments.");
        std::process::exit(1);
    }
    image_paths.sort();
    let thumb_has_xmp = image_io::detect_thumb_sidecars(&image_paths);
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
    let worker_proxy = app.create_proxy();
    let clip_engine =
        ClipEngine::new(cache_base.clone(), worker_proxy.clone()).unwrap_or_else(|err| {
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
        let proxy = worker_proxy.clone();
        thread::spawn(move || {
            while let Some(i) = thumb_queue.pop() {
                if let Some(p) = paths.get(i) {
                    let image =
                        image_io::load_thumbnail(&cache_base, p, thumb_size, output_color_space);
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
                        let clip_thumb =
                            match image_io::thumbnail_for_clip(&image, output_color_space) {
                                Ok(image) => image,
                                Err(error) => {
                                    eprintln!(
                                    "Failed to prepare an sRGB CLIP thumbnail for {}: {error:#}",
                                    p.display()
                                );
                                    image.to_rgb8()
                                }
                            };
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
                    match tx.send(update) {
                        Ok(()) => {
                            proxy.wakeup();
                        }
                        Err(_) => break,
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
        // Spawn worker threads matching thumbnail thread count
        for _ in 0..num_workers {
            let req_rx = full_req_rx.clone();
            let resp_tx = full_resp_tx.clone();
            let paths = Arc::clone(&shared_paths);
            let proxy = worker_proxy.clone();
            thread::spawn(move || {
                while let Ok(idx) = req_rx.recv() {
                    if let Some(path) = paths.get(idx) {
                        match image_io::load_full_image_tiles(path, output_color_space) {
                            Ok((full_w, full_h, frames)) => {
                                if resp_tx
                                    .send(FullImageMessage::Loaded {
                                        index: idx,
                                        full_w,
                                        full_h,
                                        frames,
                                    })
                                    .is_ok()
                                {
                                    proxy.wakeup();
                                }
                            }
                            Err(err) => {
                                if resp_tx
                                    .send(FullImageMessage::Failed {
                                        index: idx,
                                        error: format!(
                                            "failed to open {}: {}",
                                            path.display(),
                                            err
                                        ),
                                    })
                                    .is_ok()
                                {
                                    proxy.wakeup();
                                }
                            }
                        }
                    } else {
                        if resp_tx
                            .send(FullImageMessage::Failed {
                                index: idx,
                                error: "image index out of range".to_string(),
                            })
                            .is_ok()
                        {
                            proxy.wakeup();
                        }
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

pub(crate) fn run() -> Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = AppProxy::new(event_loop.create_proxy());
    event_loop.run_app(&mut SrivApplication { state: None, proxy })?;
    Ok(())
}

struct ApplicationState {
    app: App,
    renderer: Renderer,
    model: Model,
    next_file_watch: Instant,
}

impl ApplicationState {
    fn update(&mut self, check_files: bool) -> bool {
        update(&self.app, &mut self.model, &self.renderer, check_files)
    }

    fn update_and_redraw_if_changed(&mut self, check_files: bool) {
        if self.update(check_files) {
            self.app.window.request_redraw();
        }
    }

    fn refresh_and_redraw(&mut self) {
        self.update(false);
        self.app.window.request_redraw();
    }

    fn next_scheduled_update(&self) -> Instant {
        let mut next = self.next_file_watch;
        if matches!(self.model.mode, Mode::Thumbnails) && !self.model.selection_pending {
            next = next.min(self.model.selection_changed_at + SELECTION_PRELOAD_DELAY);
        }
        if matches!(self.model.mode, Mode::Single)
            && !self.model.full_textures.contains_key(&self.model.current)
        {
            if let Some(FullPendingState::Failed { last_error_at }) =
                self.model.full_pending.get(&self.model.current)
            {
                next = next.min(*last_error_at + FULL_PENDING_RETRY);
            }
        }
        if matches!(self.model.mode, Mode::Single) {
            if let Some(next_frame_at) = self
                .model
                .full_textures
                .get(&self.model.current)
                .and_then(TiledTexture::next_animation_frame_at)
            {
                next = next.min(next_frame_at);
            }
        }
        next
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
        #[cfg(target_os = "linux")]
        let attributes = attributes.with_name("sriv", "sriv");
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
        let model = model(&app, renderer.output_color_space());
        renderer.load_ui_font(model.ui_font_path.as_deref());
        self.state = Some(ApplicationState {
            app,
            renderer,
            model,
            next_file_watch: Instant::now() + FILE_WATCH_INTERVAL,
        });
        if let Some(state) = &mut self.state {
            state.refresh_and_redraw();
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: UserEvent) {
        self.proxy.clear_wakeup();
        if let Some(state) = &mut self.state {
            state.update_and_redraw_if_changed(false);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &mut self.state else {
            return;
        };
        let now = Instant::now();
        let check_files = now >= state.next_file_watch;
        let scheduled_update_due = now >= state.next_scheduled_update();
        if check_files {
            state.next_file_watch = now + FILE_WATCH_INTERVAL;
        }
        if scheduled_update_due {
            state.update_and_redraw_if_changed(check_files);
        }
        state.handle_exit_request(event_loop);
        event_loop.set_control_flow(ControlFlow::WaitUntil(state.next_scheduled_update()));
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
            WindowEvent::Focused(focused) => {
                state.app.keys.focus_changed(focused);
            }
            WindowEvent::Resized(size) => {
                state.app.update_rect();
                state.renderer.resize(size.width, size.height);
                state.refresh_and_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                state.app.update_rect();
                let size = state.app.window.inner_size();
                state.renderer.resize(size.width, size.height);
                state.refresh_and_redraw();
            }
            WindowEvent::Occluded(false) => {
                state.app.window.request_redraw();
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
                state.refresh_and_redraw();
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
                state.refresh_and_redraw();
            }
            WindowEvent::KeyboardInput {
                event,
                is_synthetic,
                ..
            } => {
                if !state.app.keys.should_handle(
                    event.physical_key,
                    event.state,
                    event.repeat,
                    is_synthetic,
                ) {
                    return;
                }
                let key = Key::from_physical_key(event.physical_key);
                let search_is_focused = state
                    .model
                    .search
                    .as_ref()
                    .map(|search| search.focused)
                    .unwrap_or(false);
                if !search_is_focused
                    && is_unmodified_q_key_down(event.state, key, state.app.keys.mods)
                {
                    state.app.quit();
                }
                if let Some(key) = key {
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
                state.refresh_and_redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = ui::render(&state.app, &mut state.model, &mut state.renderer) {
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
    if let Some(texture) = model.full_textures.get_mut(&new_idx) {
        texture.restart_animation(Instant::now());
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
    let rect = app.rect();
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
    let rect = app.rect();
    match model.mode {
        Mode::Thumbnails => {
            if len == 0 {
                return false;
            }
            let grid = ThumbnailGrid::new(model, rect);
            let cols = grid.cols();
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
                ui::cycle_terminal_tab(model, -1);
                return;
            }
            Key::Right | Key::L
                if model.terminal.visible && !model.terminal.sessions.is_empty() =>
            {
                ui::cycle_terminal_tab(model, 1);
                return;
            }
            Key::Up | Key::K if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                ui::scroll_active_terminal(model, 1);
                return;
            }
            Key::Down | Key::J if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                ui::scroll_active_terminal(model, -1);
                return;
            }
            Key::PageUp if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                let page = model.terminal.rows.max(1) as isize;
                ui::scroll_active_terminal(model, page);
                return;
            }
            Key::PageDown if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                let page = model.terminal.rows.max(1) as isize;
                ui::scroll_active_terminal(model, -page);
                return;
            }
            Key::Back if model.terminal.visible && !model.terminal.sessions.is_empty() => {
                ui::close_active_terminal(model);
                return;
            }
            _ => {}
        }
    }

    let len = model.image_paths.len();
    if app.keys.mods == ModifiersState::empty() {
        match key {
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
                        if let Some(texture) = model.full_textures.get_mut(&idx) {
                            texture.restart_animation(Instant::now());
                        }
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
                    let rect = app.rect();
                    if let Some(tex) = model.full_textures.get(&model.current) {
                        let [w, h] = tex.size();
                        model.zoom = (rect.w() / w as f32).min(rect.h() / h as f32);
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
            ui::launch_terminal_command(app, model, cmd);
        } else {
            ui::launch_detached_command(cmd);
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

fn is_unmodified_q_key_down(
    state: ElementState,
    key: Option<Key>,
    modifiers: ModifiersState,
) -> bool {
    state == ElementState::Pressed && key == Some(Key::Q) && modifiers.is_empty()
}

/// Process queued work and scheduled maintenance, returning whether the scene changed.
fn update(app: &App, model: &mut Model, renderer: &Renderer, check_files: bool) -> bool {
    let mut redraw_needed = false;
    ui::sync_terminal_viewport(app, model);

    while let Ok(update) = model.thumb_rx.try_recv() {
        handle_thumbnail_update(app, model, update);
        redraw_needed = true;
    }
    loop {
        match model.clip_engine.try_recv() {
            Ok(event) => {
                redraw_needed = true;
                match event {
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
                }
            }
            Err(crossbeam_channel::TryRecvError::Empty) => break,
            Err(crossbeam_channel::TryRecvError::Disconnected) => break,
        }
    }
    // Receive command terminal events.
    while let Ok(event) = model.command_rx.try_recv() {
        redraw_needed = true;
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
    if check_files && detect_file_changes(app, model) {
        redraw_needed = true;
    }

    // Process loaded full-resolution tile data
    while let Ok(message) = model.full_resp_rx.try_recv() {
        redraw_needed = true;
        match message {
            FullImageMessage::Loaded {
                index: idx,
                full_w,
                full_h,
                frames,
            } => {
                // Store raw pixel data for lazy texture creation
                let prepared_frames = frames
                    .into_iter()
                    .map(|frame| TiledFrame {
                        delay: frame.delay,
                        tiles: frame
                            .tiles
                            .into_iter()
                            .map(
                                |(x_offset, y_offset, width, height, format, pixel_data)| Tile {
                                    x_offset,
                                    y_offset,
                                    width,
                                    height,
                                    format,
                                    pixel_data,
                                    texture: None,
                                },
                            )
                            .collect(),
                    })
                    .collect();
                let mut tiled = TiledTexture::new(full_w, full_h, prepared_frames);
                if idx == model.current && matches!(model.mode, Mode::Single) {
                    tiled.restart_animation(Instant::now());
                }
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
    if matches!(model.mode, Mode::Single) {
        if let Some(texture) = model.full_textures.get_mut(&model.current) {
            redraw_needed |= texture.advance_animation(Instant::now());
        }
    }
    // Handle window resize: update view parameters and re-apply fit if in fit mode
    let rect = app.rect();
    if rect != model.prev_window_rect {
        model.prev_window_rect = rect;
        redraw_needed = true;
        if let Mode::Single = model.mode {
            if model.fit_mode {
                apply_fit(app, model);
            }
        }
    }
    // Schedule preload of selected thumbnail if stable for >200ms
    if let Mode::Thumbnails = model.mode {
        if !model.selection_pending
            && Instant::now() >= model.selection_changed_at + SELECTION_PRELOAD_DELAY
        {
            request_full_texture(model, model.current);
            model.selection_pending = true;
        }
    }
    // Clamp thumbnail scrolling to content bounds
    if let Mode::Thumbnails = model.mode {
        let grid = ThumbnailGrid::new(model, rect);
        let clamped = model.scroll_offset.clamp(0.0, grid.max_scroll());
        if clamped != model.scroll_offset {
            model.scroll_offset = clamped;
            redraw_needed = true;
        }
    }
    if matches!(model.mode, Mode::Single) && !model.full_textures.contains_key(&model.current) {
        request_full_texture(model, model.current);
    }
    update_thumbnail_requests(app, model, renderer);
    redraw_needed
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

/// Touch an already loaded texture in the LRU, or queue it for background loading.
fn request_full_texture(model: &mut Model, idx: usize) {
    if model.full_textures.contains_key(&idx) {
        touch_full_texture(model, idx);
        return;
    }
    let now = Instant::now();
    let should_request = match model.full_pending.get(&idx) {
        None => true,
        Some(FullPendingState::InFlight) => false,
        Some(FullPendingState::Failed { last_error_at }) => {
            now.duration_since(*last_error_at) > FULL_PENDING_RETRY
        }
    };
    if should_request {
        model.full_pending.insert(idx, FullPendingState::InFlight);
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
    let rect = app.rect();
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

fn detect_file_changes(app: &App, model: &mut Model) -> bool {
    let total = model.image_paths.len();
    if total == 0 {
        return false;
    }
    let mut candidates: HashSet<usize> = HashSet::new();
    candidates.insert(model.current);
    if matches!(model.mode, Mode::Thumbnails) {
        for idx in ThumbnailGrid::new(model, app.rect()).visible_indices() {
            candidates.insert(idx);
        }
    }
    let batch = FILE_WATCH_BATCH.min(total);
    for _ in 0..batch {
        let idx = model.file_watch_cursor;
        model.file_watch_cursor = (model.file_watch_cursor + 1) % total;
        candidates.insert(idx);
    }
    let mut changed = false;
    for idx in candidates {
        changed |= check_image_modification(model, idx);
    }
    changed
}

fn check_image_modification(model: &mut Model, idx: usize) -> bool {
    if idx >= model.image_paths.len() {
        return false;
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
    changed
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

/// Apply fit-to-window for current single-image view
fn apply_fit(app: &App, model: &mut Model) {
    model.fit_mode = true;
    let rect = app.rect();
    if let Some(tex) = model.full_textures.get(&model.current) {
        let [w, h] = tex.size();
        model.zoom = (rect.w() / w as f32).min(rect.h() / h as f32);
    } else {
        model.zoom = 1.0;
    }
    model.pan = vec2(0.0, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    const Q_KEY: PhysicalKey = PhysicalKey::Code(KeyCode::KeyQ);

    #[test]
    fn key_held_before_focus_is_ignored_until_released() {
        let mut keys = Keys::default();

        assert!(!keys.should_handle(Q_KEY, ElementState::Pressed, false, true));
        keys.focus_changed(true);

        // X11 can report the first repeat after focus as a non-repeat.
        assert!(!keys.should_handle(Q_KEY, ElementState::Pressed, false, false));
        assert!(!keys.should_handle(Q_KEY, ElementState::Pressed, true, false));
        assert!(!keys.should_handle(Q_KEY, ElementState::Released, false, false));
        assert!(keys.should_handle(Q_KEY, ElementState::Pressed, false, false));
    }

    #[test]
    fn repeats_are_handled_only_after_a_focused_press() {
        let mut keys = Keys::default();

        assert!(keys.should_handle(Q_KEY, ElementState::Pressed, false, false));
        assert!(keys.should_handle(Q_KEY, ElementState::Pressed, true, false));

        keys.focus_changed(false);
        keys.focus_changed(true);
        assert!(!keys.should_handle(Q_KEY, ElementState::Pressed, true, false));
        assert!(!keys.should_handle(Q_KEY, ElementState::Released, false, false));
        assert!(keys.should_handle(Q_KEY, ElementState::Pressed, false, false));
    }

    #[test]
    fn quit_shortcut_requires_unmodified_q_key_down() {
        assert!(is_unmodified_q_key_down(
            ElementState::Pressed,
            Some(Key::Q),
            ModifiersState::empty(),
        ));
        assert!(!is_unmodified_q_key_down(
            ElementState::Released,
            Some(Key::Q),
            ModifiersState::empty(),
        ));
        assert!(!is_unmodified_q_key_down(
            ElementState::Pressed,
            Some(Key::W),
            ModifiersState::empty(),
        ));
        for modifiers in [
            ModifiersState::SHIFT,
            ModifiersState::CONTROL,
            ModifiersState::ALT,
            ModifiersState::SUPER,
        ] {
            assert!(!is_unmodified_q_key_down(
                ElementState::Pressed,
                Some(Key::Q),
                modifiers,
            ));
        }
    }
}
