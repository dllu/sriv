use super::*;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::window::{Fullscreen, WindowId};

// Run against a disposable compositor, for example with scripts/test-wayland.py.
#[test]
#[ignore = "requires a native Wayland compositor and a graphics driver"]
fn wayland_surface_lifecycle() {
    let event_loop = EventLoop::builder()
        .with_wayland()
        .with_any_thread(true)
        .build()
        .unwrap();
    let mut app = TestApp {
        renderer: None,
        redraws: 0,
        deadline: Instant::now() + Duration::from_secs(30),
    };
    event_loop.run_app(&mut app).unwrap();
    assert_eq!(app.redraws, 64, "Wayland redraws stopped making progress");
}

struct TestApp {
    renderer: Option<Renderer>,
    redraws: usize,
    deadline: Instant,
}

impl ApplicationHandler for TestApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("sriv Wayland test"))
                .unwrap(),
        );
        let renderer = pollster::block_on(Renderer::new(window, event_loop)).unwrap();
        renderer.window.request_redraw();
        self.renderer = Some(renderer);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        assert!(Instant::now() < self.deadline, "Wayland redraw timed out");
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.deadline));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        match event {
            WindowEvent::Resized(size) => {
                renderer.resize(size.width, size.height);
                renderer.window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = renderer.window.inner_size();
                renderer.resize(size.width, size.height);
                renderer.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                draw(renderer);
                if self.redraws == 0 {
                    check_recovery(renderer);
                }
                self.redraws += 1;
                if self.redraws == 64 {
                    event_loop.exit();
                    return;
                }
                // Exercise compositor configure events as well as the injected statuses.
                if self.redraws % 8 == 0 {
                    renderer
                        .window
                        .set_fullscreen(Some(Fullscreen::Borderless(None)));
                } else if self.redraws % 8 == 4 {
                    renderer.window.set_fullscreen(None);
                    let _ = renderer.window.request_inner_size(PhysicalSize::new(
                        640 + self.redraws as u32 * 2,
                        480 + self.redraws as u32,
                    ));
                }
                renderer.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn draw(renderer: &mut Renderer) {
    let size = renderer.window.inner_size();
    let scale = renderer.window.scale_factor();
    let logical = size.to_logical::<f32>(scale);
    let mut scene = Scene::new(Rect::from_w_h(logical.width, logical.height));
    scene.rect().w_h(100.0, 100.0).color(WHITE);
    scene.text("Wayland surface recovery");
    renderer.render(&scene, scale).unwrap();
}

fn check_recovery(renderer: &mut Renderer) {
    use wgpu::CurrentSurfaceTexture as Current;

    let size = renderer.window.inner_size();
    for _ in 0..8 {
        let frame = renderer.acquire_surface_frame().unwrap().unwrap();
        // Inject the exact status from the crash with a real, live GPU surface frame.
        // This must return the usable frame without configuring or acquiring again.
        let frame = renderer
            .handle_surface_texture(Current::Suboptimal(frame))
            .unwrap()
            .unwrap();
        assert!(renderer.surface_needs_configure);
        // Resize notifications must also defer configuration while a frame is alive.
        renderer.resize(size.width + 16, size.height + 16);
        renderer.resize(size.width + 32, size.height + 32);
        renderer.resize(size.width, size.height);
        drop(frame);
        draw(renderer);
    }

    let configured = renderer.surface.get_configuration().unwrap();
    for (width, height) in [(0, 0), (0, size.height), (size.width, 0)] {
        renderer.resize(width, height);
        assert!(renderer.acquire_surface_frame().unwrap().is_none());
        draw(renderer);
        // A zero extent must never reach Surface::configure.
        let unchanged = renderer.surface.get_configuration().unwrap();
        assert_eq!(unchanged.width, configured.width);
        assert_eq!(unchanged.height, configured.height);
        renderer.resize(size.width, size.height);
        draw(renderer);
    }

    for status in [
        Current::Timeout,
        Current::Occluded,
        Current::Outdated,
        Current::Lost,
    ] {
        assert!(renderer.handle_surface_texture(status).unwrap().is_none());
        draw(renderer);
        // Verify recovery leaves the surface able to acquire another frame.
        drop(renderer.acquire_surface_frame().unwrap().unwrap());
    }
    assert!(renderer
        .handle_surface_texture(Current::Validation)
        .is_err());
    draw(renderer);
}
