mod app;
mod clip;
mod color;
mod geometry;
mod grid;
mod image_io;
mod input;
mod renderer;
mod state;
mod ui;

fn main() -> anyhow::Result<()> {
    app::run()
}
