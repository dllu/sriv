use crate::state::Model;
use crate::THUMB_PREFETCH_ROWS;
use nannou::prelude::{vec2, Rect, Vec2};

#[derive(Debug, Clone, Copy)]
pub struct ThumbnailGrid {
    rect: Rect,
    cell: f32,
    cols: usize,
    rows: usize,
    half_gap: f32,
    thumb_size: f32,
    scroll: f32,
    total: usize,
}

impl ThumbnailGrid {
    pub fn new(model: &Model, rect: Rect) -> Self {
        let thumb_size = model.thumb_size as f32;
        let cell = thumb_size + model.gap;
        let mut cols = ((rect.w() + model.gap) / cell).floor() as isize;
        if cols < 1 {
            cols = 1;
        }
        let cols = cols as usize;
        let total = model.image_paths.len();
        let rows = if cols == 0 { 0 } else { total.div_ceil(cols) };
        let half_gap = model.gap / 2.0;
        Self {
            rect,
            cell,
            cols,
            rows,
            half_gap,
            thumb_size,
            scroll: model.scroll_offset,
            total,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0 || self.cols == 0
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn visible_rows(&self) -> Option<(usize, usize)> {
        if self.is_empty() || self.rows == 0 {
            return None;
        }
        let row_min_f = (self.scroll - self.thumb_size - self.half_gap) / self.cell;
        let row_max_f = (self.rect.h() + self.scroll - self.half_gap) / self.cell;
        let mut row_min = row_min_f.ceil() as isize - THUMB_PREFETCH_ROWS as isize;
        let mut row_max = row_max_f.floor() as isize + THUMB_PREFETCH_ROWS as isize;
        let max_row = self.rows.saturating_sub(1) as isize;
        if row_min < 0 {
            row_min = 0;
        }
        if row_max > max_row {
            row_max = max_row;
        }
        if row_max < row_min {
            row_max = row_min;
        }
        Some((row_min as usize, row_max as usize))
    }

    pub fn visible_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        if let Some((row_min, row_max)) = self.visible_rows() {
            for row in row_min..=row_max {
                let base = row * self.cols;
                for col in 0..self.cols {
                    let idx = base + col;
                    if idx >= self.total {
                        break;
                    }
                    indices.push(idx);
                }
            }
        }
        indices
    }

    pub fn index_center(&self, idx: usize) -> Option<Vec2> {
        if idx >= self.total || self.cols == 0 {
            return None;
        }
        let row = idx / self.cols;
        let col = idx % self.cols;
        let base_y = self.row_base_y(row);
        Some(vec2(self.col_center_x(col), base_y + self.scroll))
    }

    pub fn row_for_index(&self, idx: usize) -> Option<usize> {
        if self.cols == 0 || idx >= self.total {
            None
        } else {
            Some(idx / self.cols)
        }
    }

    pub fn max_scroll(&self) -> f32 {
        (self.rows as f32 * self.cell - self.rect.h()).max(0.0)
    }

    pub fn row_top(&self, row: usize) -> f32 {
        row as f32 * self.cell
    }

    pub fn row_bottom(&self, row: usize) -> f32 {
        self.row_top(row) + self.cell
    }

    pub fn row_length(&self, row: usize) -> usize {
        if self.cols == 0 || row >= self.rows {
            return 0;
        }
        let base = row * self.cols;
        let remaining = self.total.saturating_sub(base);
        remaining.min(self.cols)
    }

    pub fn viewport_priority(&self, idx: usize) -> f32 {
        if self.cols == 0 || idx >= self.total {
            return f32::MAX;
        }
        let row = idx / self.cols;
        let col = idx % self.cols;
        let row_center = row as f32 * self.cell + self.cell / 2.0;
        let viewport_center = self.scroll + self.rect.h() / 2.0;
        let vertical = (row_center - viewport_center).abs();
        let horizontal = self.col_center_x(col).abs();
        vertical + horizontal * 0.01
    }

    fn row_base_y(&self, row: usize) -> f32 {
        self.rect.h() / 2.0 - self.thumb_size / 2.0 - self.half_gap - (row as f32) * self.cell
    }

    fn col_center_x(&self, col: usize) -> f32 {
        -self.rect.w() / 2.0 + self.thumb_size / 2.0 + self.half_gap + (col as f32) * self.cell
    }
}
