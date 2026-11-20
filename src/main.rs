mod app;
mod clip;
mod color;
mod geometry;
mod grid;
mod image_io;
mod input;
mod library;
mod renderer;
mod source;
mod state;
mod thumbnails;
mod ui;

fn main() -> anyhow::Result<std::process::ExitCode> {
    app::run()
}
