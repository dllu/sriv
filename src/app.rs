use anyhow::Result;
use crossbeam_channel::unbounded;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
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
use crate::geometry::{fit_image_zoom, image_rect, vec2, Rect, Vec2};
use crate::grid::ThumbnailGrid;
use crate::image_io;
use crate::input::Key;
use crate::library::{LibraryState, LibraryUpdate};
use crate::renderer::Renderer;
use crate::state::{
    parse_bindings, parse_ui_font_path, CommandEvent, EmbeddingState, FullImageCache,
    FullImageMessage, ImageRequest, ImageStates, Mode, Model, SearchPhase, SearchState,
    SelectionPreload, TerminalState, ThumbRequestQueue, ThumbnailEntry, ThumbnailTexture,
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
            let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
            if let Some((row_min, row_max)) = grid.visible_rows() {
                for row in row_min..=row_max {
                    for col in 0..grid.cols() {
                        let i = row * grid.cols() + col;
                        if i >= grid.total() {
                            break;
                        }
                        let center = grid.index_center(i).unwrap();
                        let size = if let Some(slot) = model.thumb_visible.get(&i) {
                            slot.size
                        } else {
                            [model.thumb_size; 2]
                        };
                        let bounds = image_rect(rect, center, size, 1.0, app.scale_factor());
                        if pos.x >= bounds.left()
                            && pos.x <= bounds.right()
                            && pos.y >= bounds.bottom()
                            && pos.y <= bounds.top()
                        {
                            model.current = i;
                            model.selection_preload =
                                SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY);
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
            let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
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
            model.fit_mode = false;
        }
    }
}

/// The model function for initializing the application state.
fn model(app: &App, output_color_space: OutputColorSpace) -> Model {
    // Parse command-line arguments: files or directories.
    let mut regen_cache = false;
    let mut private = false;
    let mut args: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--clear-cache" || arg == "--regen-cache" {
            regen_cache = true;
        } else if arg == "--private" {
            private = true;
        } else {
            args.push(arg);
        }
    }
    if args.is_empty() {
        eprintln!("Usage: sriv [--clear-cache] [--private] <image files or directories>...");
        std::process::exit(1);
    }
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
    let worker_proxy = app.create_proxy();
    let discovery_proxy = worker_proxy.clone();
    let (library, shared_paths) = LibraryState::start(
        args.into_iter().map(PathBuf::from).collect(),
        regen_cache.then(|| cache_base.clone()),
        move || discovery_proxy.wakeup(),
    );
    let cache_dir = (!private).then_some(cache_base);
    // Channel for receiving thumbnails from background threads.
    let (thumb_tx, thumb_rx) = channel::<ThumbnailUpdate>();
    let thumb_queue = ThumbRequestQueue::new();
    let num_workers = rayon::current_num_threads().clamp(1, 8);
    let clip_engine =
        ClipEngine::new(cache_dir.clone(), worker_proxy.clone()).unwrap_or_else(|err| {
            eprintln!("Failed to initialize CLIP: {err}");
            std::process::exit(1);
        });
    let clip_sender = clip_engine.request_sender();
    for _ in 0..num_workers {
        let paths = Arc::clone(&shared_paths);
        let cache_dir = cache_dir.clone();
        let tx = thumb_tx.clone();
        let thumb_queue = thumb_queue.clone();
        let clip_sender = clip_sender.clone();
        let proxy = worker_proxy.clone();
        thread::spawn(move || {
            while let Some(request) = thumb_queue.pop() {
                let i = request.index;
                let path = paths.read().unwrap().get(i).cloned();
                if let Some(p) = path {
                    let image = image_io::load_thumbnail(
                        cache_dir.as_deref(),
                        &p,
                        thumb_size,
                        output_color_space,
                    );
                    let clip_embedding = match cache_dir
                        .as_deref()
                        .map(|dir| clip::load_cached_embedding(dir, &p))
                        .transpose()
                    {
                        Ok(value) => value.flatten(),
                        Err(err) => {
                            eprintln!(
                                "Failed to load cached CLIP embedding for {}: {}",
                                p.display(),
                                err
                            );
                            None
                        }
                    };
                    let embedding = if let Some(embedding) = clip_embedding {
                        EmbeddingState::Ready(embedding)
                    } else {
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
                        match clip_sender.queue_image(request, p.clone(), clip_thumb) {
                            Ok(()) => EmbeddingState::Pending,
                            Err(err) => EmbeddingState::Failed(format!(
                                "Failed to queue CLIP embedding: {err}"
                            )),
                        }
                    };
                    let update = ThumbnailUpdate {
                        request,
                        image,
                        embedding,
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
    // Initialize channels and state for full-resolution LRU cache.
    // Channel for requesting full-resolution images (by index)
    let (full_req_tx, full_req_rx) = unbounded::<ImageRequest>();
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
                while let Ok(request) = req_rx.recv() {
                    let idx = request.index;
                    let path = paths.read().unwrap().get(idx).cloned();
                    if let Some(path) = path {
                        match image_io::load_full_image_tiles(&path, output_color_space) {
                            Ok((full_w, full_h, frames)) => {
                                if resp_tx
                                    .send(FullImageMessage::Loaded {
                                        request,
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
                                        request,
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
                                request,
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
    let full_images = FullImageCache::new(FULL_CACHE_CAPACITY, FULL_PENDING_RETRY);
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
        image_paths: Vec::new(),
        library,
        ui_font_path,
        thumb_visible: HashMap::new(),
        thumb_data: HashMap::new(),
        thumb_has_xmp: Vec::new(),
        thumb_rx,
        thumb_queue: thumb_queue.clone(),
        file_mod_times: Vec::new(),
        image_states: ImageStates::new(0),
        file_watch_cursor: 0,
        full_req_tx,
        full_resp_rx,
        full_images,
        mode: Mode::Thumbnails,
        current: 0,
        thumb_size,
        gap: 10.0,
        scroll_offset: 0.0,
        zoom: 1.0,
        pan: vec2(0.0, 0.0),
        prev_window_rect: initial_rect,
        prev_scale_factor: app.scale_factor(),
        prev_scroll: 0.0,
        fit_mode: false,
        selection_preload: SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY),
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
        clip_error: None,
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
        if matches!(self.model.mode, Mode::Thumbnails) && !self.model.image_paths.is_empty() {
            if let Some(deadline) = self.model.selection_preload.deadline() {
                next = next.min(deadline);
            }
        }
        if matches!(self.model.mode, Mode::Single)
            && !self.model.full_images.contains_key(&self.model.current)
        {
            if let Some(retry_at) = self.model.full_images.retry_at(self.model.current) {
                next = next.min(retry_at);
            }
        }
        if matches!(self.model.mode, Mode::Single) {
            if let Some(next_frame_at) = self
                .model
                .full_images
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
                // Only a search that was already focused receives this event's text.
                // Opening it with '/' consumes that whole event, not the next character.
                if search_is_focused
                    && !state.app.keys.mods.control_key()
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
                // Keep input handling cheap. `update` can drain background queues and create GPU
                // textures, so running it for every key press makes a burst of input wait behind
                // unrelated image work. Redraw requests are coalesced by winit; background work
                // wakes the event loop separately, and thumbnail selection schedules its own
                // deferred preload.
                state.app.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Input can move the viewport without running `update`. Prepare cached
                // thumbnails for this frame instead of waiting for a worker or timer wakeup.
                update_thumbnail_requests(&state.app, &mut state.model, &state.renderer, false);
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
    // Navigation always starts fitted, whether this image is cached or still loading.
    if let Some(texture) = model.full_images.get_mut(&new_idx) {
        texture.restart_animation(Instant::now());
    }
    apply_fit(app, model);
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
            model.selection_preload =
                SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY);
            ensure_thumbnail_visible(app, model, idx);
        }
    }
}

fn ensure_thumbnail_visible(app: &App, model: &mut Model, idx: usize) {
    if !matches!(model.mode, Mode::Thumbnails) {
        return;
    }
    let rect = app.rect();
    let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
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
    if let Some(idx) = model
        .search
        .as_mut()
        .and_then(|search| search.advance(delta))
    {
        focus_image(app, model, idx);
    }
}

fn handle_text_result(app: &App, model: &mut Model, request_id: u64, embedding: Vec<f32>) {
    let images = model.image_states.ready_embeddings();
    let target = model
        .search
        .as_mut()
        .and_then(|search| search.complete(request_id, embedding, images));
    if let Some(idx) = target {
        focus_image(app, model, idx);
    }
}

fn update_search_with_image_embedding(app: &App, model: &mut Model, index: usize) {
    let target = model.image_states.embedding(index).and_then(|embedding| {
        model
            .search
            .as_mut()
            .and_then(|search| search.image_ready(index, embedding, model.current))
    });
    if let Some(idx) = target {
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
        } else {
            model.search = Some(SearchState::new());
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
                    let request_id = model.next_search_request_id;
                    model.next_search_request_id = model.next_search_request_id.wrapping_add(1);
                    let search = model.search.as_mut().unwrap();
                    search.submit(request_id);
                    if let Err(err) = model.clip_engine.request_text(request_id, query) {
                        search.fail(request_id, format!("Failed to queue search: {err}"));
                    }
                } else if let Some(search) = model.search.as_mut() {
                    search.phase = SearchPhase::Failed("Enter a search phrase".to_string());
                }
                return true;
            }
            Key::Back => {
                let mut remove_search = false;
                if let Some(search) = model.search.as_mut() {
                    if search.input.is_empty() {
                        remove_search = true;
                    } else {
                        search.edit(|input| {
                            input.pop();
                        });
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
                        .map(|s| !s.results().is_empty())
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
                        .map(|s| !s.results().is_empty())
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
            let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
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
                    if let Some(tex) = model.full_images.get(&model.current) {
                        let [tw, _] = tex.size();
                        let disp_w = tw as f32 * model.zoom / app.scale_factor() as f32;
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
                    if let Some(tex) = model.full_images.get(&model.current) {
                        let [_, th] = tex.size();
                        let disp_h = th as f32 * model.zoom / app.scale_factor() as f32;
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
            search.edit(|input| input.push(ch));
        }
    }
}

fn key_pressed(app: &App, model: &mut Model, key: Key) {
    if handle_search_key(app, model, key) {
        return;
    }
    if model.image_paths.is_empty() {
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
                        if let Some(texture) = model.full_images.get_mut(&idx) {
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
                    apply_fit(app, model);
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
            // Show one image pixel per physical screen pixel.
            Key::Equals => {
                if let Mode::Single = model.mode {
                    model.fit_mode = false;
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
                expand_binding_command(&binding.command, &current_file),
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
        model.selection_preload = SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY);
    }
}

fn expand_binding_command(command: &str, file: &str) -> String {
    // {file} is a shell argument: quote it, including any literal apostrophes.
    let quoted_file = format!("'{}'", file.replace('\'', "'\\''"));
    command.replace("{file}", &quoted_file)
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
    let mut redraw_needed = receive_library_updates(app, model);
    ui::sync_terminal_viewport(app, model);
    if check_files && detect_file_changes(app, model) {
        redraw_needed = true;
    }

    while let Ok(update) = model.thumb_rx.try_recv() {
        handle_thumbnail_update(app, model, update);
        redraw_needed = true;
    }
    loop {
        match model.clip_engine.try_recv() {
            Ok(event) => {
                redraw_needed = true;
                match event {
                    ClipEvent::ImageReady { request, embedding } => {
                        if model.image_states.image_ready(request, embedding) {
                            update_search_with_image_embedding(app, model, request.index);
                        }
                    }
                    ClipEvent::ImageError { request, error } => {
                        if model.image_states.image_failed(request, error.clone()) {
                            eprintln!(
                                "Failed to compute CLIP embedding for {}: {error}",
                                model.image_paths[request.index].display()
                            );
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
                            search.fail(request_id, error);
                        }
                    }
                }
            }
            Err(crossbeam_channel::TryRecvError::Empty) => break,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                if model.clip_error.is_none() {
                    let error =
                        "CLIP unavailable: all workers stopped; see stderr for details".to_string();
                    model.image_states.worker_stopped(&error);
                    if let Some(search) = &mut model.search {
                        if let SearchPhase::Pending { request_id } = search.phase {
                            search.fail(request_id, error.clone());
                        }
                    }
                    model.clip_error = Some(error);
                    redraw_needed = true;
                }
                break;
            }
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
                    session.status.finish(exit_code, signal);
                }
            }
            CommandEvent::OutputClosed { session_id } => {
                if let Some(session) = model
                    .terminal
                    .sessions
                    .iter_mut()
                    .find(|session| session.id == session_id)
                {
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
                    if session.status.fail(error.clone()) {
                        session.parser.process(format!("{error}\r\n").as_bytes());
                    }
                    session.master = None;
                }
            }
        }
    }
    // Process loaded full-resolution tile data
    while let Ok(message) = model.full_resp_rx.try_recv() {
        redraw_needed = true;
        match message {
            FullImageMessage::Loaded {
                request,
                full_w,
                full_h,
                frames,
            } => {
                if !accept_full_result(model, request) {
                    continue;
                }
                let idx = request.index;
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
                model.full_images.loaded(request, tiled, model.current);
                // If this is the current image and in fit mode, resize to fit
                if idx == model.current && model.fit_mode {
                    apply_fit(app, model);
                }
            }
            FullImageMessage::Failed { request, error } => {
                if !accept_full_result(model, request) {
                    continue;
                }
                let idx = request.index;
                model
                    .full_images
                    .failed(request, error.clone(), Instant::now());
                let path_info = model
                    .image_paths
                    .get(idx)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| format!("image index {idx}"));
                eprintln!("failed to load full image {path_info}: {error}");
            }
        }
    }
    if matches!(model.mode, Mode::Single) {
        if let Some(texture) = model.full_images.get_mut(&model.current) {
            redraw_needed |= texture.advance_animation(Instant::now());
        }
    }
    // Handle window resize: update view parameters and re-apply fit if in fit mode
    let rect = app.rect();
    let viewport_changed =
        rect != model.prev_window_rect || app.scale_factor() != model.prev_scale_factor;
    if viewport_changed {
        model.prev_window_rect = rect;
        model.prev_scale_factor = app.scale_factor();
        redraw_needed = true;
        if let Mode::Single = model.mode {
            if model.fit_mode {
                apply_fit(app, model);
            }
        }
    }
    // Schedule preload of selected thumbnail if stable for >200ms
    if let Mode::Thumbnails = model.mode {
        if model
            .selection_preload
            .deadline()
            .is_some_and(|deadline| !model.image_paths.is_empty() && Instant::now() >= deadline)
        {
            request_full_texture(model, model.current);
            model.selection_preload = SelectionPreload::Requested;
        }
    }
    // Clamp thumbnail scrolling to content bounds
    if let Mode::Thumbnails = model.mode {
        let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
        let clamped = model.scroll_offset.clamp(0.0, grid.max_scroll());
        if clamped != model.scroll_offset {
            model.scroll_offset = clamped;
            redraw_needed = true;
        }
    }
    if matches!(model.mode, Mode::Single) && !model.full_images.contains_key(&model.current) {
        request_full_texture(model, model.current);
    }
    update_thumbnail_requests(app, model, renderer, viewport_changed);
    redraw_needed
}

fn receive_library_updates(app: &App, model: &mut Model) -> bool {
    let mut changed = false;
    // Yield to drawing and input even when discovery can produce batches continuously.
    for _ in 0..8 {
        let Some(update) = model.library.try_recv() else {
            return changed;
        };
        changed = true;
        match update {
            LibraryUpdate::Images(images) => {
                let start = model.image_paths.len();
                model.image_states.add_images(images.len());
                for image in images {
                    model.image_paths.push(image.path);
                    model.file_mod_times.push(image.modified);
                    model.thumb_has_xmp.push(false);
                }
                model.thumb_queue.enqueue_batch(
                    (start..model.image_paths.len()).map(|index| model.image_states.request(index)),
                );
                if start == 0 {
                    model.selection_preload =
                        SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY);
                }
            }
            LibraryUpdate::Sidecars(flags) => model.thumb_has_xmp = flags,
            LibraryUpdate::Finished => {
                if model.image_paths.is_empty() {
                    eprintln!("No image files found in arguments.");
                    std::process::exit(1);
                }
                if model.image_paths.len() == 1 && matches!(model.mode, Mode::Thumbnails) {
                    model.mode = Mode::Single;
                    navigate_to(app, model, 0);
                }
            }
        }
    }
    app.proxy.wakeup();
    changed
}

fn accept_full_result(model: &Model, request: ImageRequest) -> bool {
    model.image_states.is_current(request) && model.full_images.is_pending(request)
}

/// Touch an already loaded texture in the LRU, or queue it for background loading.
fn request_full_texture(model: &mut Model, idx: usize) {
    if idx >= model.image_paths.len() {
        return;
    }
    let now = Instant::now();
    let request = model.image_states.request(idx);
    if model.full_images.request(request, now) {
        if let Err(err) = model.full_req_tx.send(request) {
            model.full_images.failed(request, err.to_string(), now);
            eprintln!("failed to request full image load for index {idx}: {err}");
        }
    }
}

fn update_thumbnail_requests(
    app: &App,
    model: &mut Model,
    renderer: &Renderer,
    viewport_changed: bool,
) {
    if !matches!(model.mode, Mode::Thumbnails) {
        return;
    }
    let total = model.image_paths.len();
    if total == 0 {
        model.thumb_visible.clear();
        return;
    }
    let rect = app.rect();
    let grid = ThumbnailGrid::new(model, rect, app.scale_factor());
    let visible = grid.visible_indices();

    let scroll_changed = if (model.scroll_offset - model.prev_scroll).abs() > f32::EPSILON {
        model.prev_scroll = model.scroll_offset;
        true
    } else {
        false
    };
    if viewport_changed || scroll_changed {
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
        request,
        image,
        embedding,
    } = update;
    if !model
        .image_states
        .thumbnail_ready(request, embedding, model.clip_error.as_deref())
    {
        return;
    }
    let index = request.index;
    if let Some(error) = model.image_states.error(index) {
        eprintln!(
            "CLIP embedding unavailable for {}: {error}",
            model.image_paths[index].display()
        );
    }
    model.thumb_data.insert(index, ThumbnailEntry { image });
    update_search_with_image_embedding(app, model, index);
}

fn detect_file_changes(app: &App, model: &mut Model) -> bool {
    let total = model.image_paths.len();
    if total == 0 {
        return false;
    }
    let mut candidates: HashSet<usize> = HashSet::new();
    candidates.insert(model.current);
    if matches!(model.mode, Mode::Thumbnails) {
        for idx in ThumbnailGrid::new(model, app.rect(), app.scale_factor()).visible_indices() {
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
    let request = model.image_states.invalidate(idx);
    if let Some(search) = &mut model.search {
        search.invalidate_image(idx);
    }
    model.thumb_visible.remove(&idx);
    model.thumb_queue.enqueue(request);

    model.full_images.invalidate(idx);
    if matches!(model.mode, Mode::Single) && idx == model.current {
        request_full_texture(model, idx);
    }

    if matches!(model.mode, Mode::Thumbnails) && idx == model.current {
        model.selection_preload = SelectionPreload::after(Instant::now(), SELECTION_PRELOAD_DELAY);
    }
}

fn current_mod_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|meta| meta.modified()).ok()
}

/// Apply fit-to-window for current single-image view
fn apply_fit(app: &App, model: &mut Model) {
    model.fit_mode = true;
    if let Some(tex) = model.full_images.get(&model.current) {
        model.zoom = fit_image_zoom(app.rect(), tex.size(), app.scale_factor());
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
    fn binding_paths_remain_literal_shell_arguments() {
        for file in [
            "/photos/Michelin Challenge Design 2008/side.jpg",
            "/photos/it's a \"car\".jpg",
            "/photos/$(printf injected)`printf injected`;$HOME*?[a]\\\n車.jpg",
            "/photos/{file}.jpg",
        ] {
            let command = expand_binding_command("true && printf '%s\\0' {file} 3840 {file}", file);
            let output = std::process::Command::new("sh")
                .args(["-c", &command])
                .output()
                .unwrap();
            assert!(output.status.success());
            assert_eq!(
                output.stdout,
                format!("{file}\x003840\0{file}\0").as_bytes()
            );
        }
    }

    #[test]
    fn rapid_key_presses_are_all_handled() {
        let mut keys = Keys::default();
        let keys_in_order = [
            PhysicalKey::Code(KeyCode::KeyH),
            PhysicalKey::Code(KeyCode::KeyJ),
            PhysicalKey::Code(KeyCode::KeyK),
            PhysicalKey::Code(KeyCode::KeyL),
        ];

        for _ in 0..100 {
            for physical_key in keys_in_order {
                assert!(keys.should_handle(physical_key, ElementState::Pressed, false, false));
                assert!(!keys.should_handle(physical_key, ElementState::Released, false, false));
            }
        }
    }

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
