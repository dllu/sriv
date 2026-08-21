use anyhow::Result;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::io::Read;
use std::process::Stdio;
use std::thread;

use crate::app::App;
use crate::geometry::{srgba, Rect, Rgba, WHITE};
use crate::grid::ThumbnailGrid;
use crate::image_io;
use crate::renderer::{Renderer, Scene};
use crate::state::{CommandEvent, Mode, Model, TerminalSession, TilePixelFormat};

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

pub(crate) fn sync_terminal_viewport(app: &App, model: &mut Model) {
    let (rows, cols) = terminal_grid_size(app.rect());
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

pub(crate) fn cycle_terminal_tab(model: &mut Model, delta: isize) {
    let len = model.terminal.sessions.len();
    if len == 0 {
        return;
    }
    let len = len as isize;
    let next = ((model.terminal.active as isize + delta).rem_euclid(len)) as usize;
    set_active_terminal(model, next);
}

pub(crate) fn scroll_active_terminal(model: &mut Model, delta: isize) {
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

pub(crate) fn close_active_terminal(model: &mut Model) {
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

pub(crate) fn launch_terminal_command(app: &App, model: &mut Model, command: String) {
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
                                        reader_proxy.wakeup();
                                    }
                                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                                    Err(err) => {
                                        let _ = reader_tx.send(CommandEvent::Failed {
                                            session_id,
                                            error: format!("Terminal read failed: {err}"),
                                        });
                                        reader_proxy.wakeup();
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
                            proxy.wakeup();
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

pub(crate) fn launch_detached_command(command: String) {
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
                draw.rect()
                    .x_y(x, y - TERMINAL_CELL_HEIGHT / 2.6)
                    .w_h(TERMINAL_CELL_WIDTH, 1.0)
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

fn prepare_current_full_textures(renderer: &Renderer, model: &mut Model) {
    let Some(tiled) = model.full_textures.get_mut(&model.current) else {
        return;
    };
    for tile in tiled.current_tiles_mut() {
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
                std::borrow::Cow::Owned(image_io::linear_rgba16_bytes_to_srgba8(&tile.pixel_data)),
            ),
            TilePixelFormat::Rgba16Float => (
                wgpu::TextureFormat::Rgba16Float,
                8 * tile.width,
                std::borrow::Cow::Borrowed(&tile.pixel_data),
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

pub(crate) fn render(app: &App, model: &mut Model, renderer: &mut Renderer) -> Result<()> {
    prepare_current_full_textures(renderer, model);
    let rect = app.rect();
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
                for tile in tex.current_tiles() {
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
                let info = if tex.frame_count() > 1 {
                    format!(
                        "{}×{}  {:.2}×  frame {}/{}",
                        full_w,
                        full_h,
                        model.zoom,
                        tex.current_frame_index() + 1,
                        tex.frame_count()
                    )
                } else {
                    format!("{}×{}  {:.2}×", full_w, full_h, model.zoom)
                };
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
    renderer.render(&draw, app.scale_factor())
}
