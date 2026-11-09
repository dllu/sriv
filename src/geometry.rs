pub use glam::{vec2, Vec2};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    center: Vec2,
    size: Vec2,
}

impl Rect {
    pub fn from_w_h(width: f32, height: f32) -> Self {
        Self::from_x_y_w_h(0.0, 0.0, width, height)
    }

    pub fn from_x_y_w_h(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            center: vec2(x, y),
            size: vec2(width.max(0.0), height.max(0.0)),
        }
    }

    pub fn x(self) -> f32 {
        self.center.x
    }

    pub fn y(self) -> f32 {
        self.center.y
    }

    pub fn w(self) -> f32 {
        self.size.x
    }

    pub fn h(self) -> f32 {
        self.size.y
    }

    pub fn left(self) -> f32 {
        self.center.x - self.size.x / 2.0
    }

    pub fn right(self) -> f32 {
        self.center.x + self.size.x / 2.0
    }

    pub fn top(self) -> f32 {
        self.center.y + self.size.y / 2.0
    }

    pub fn bottom(self) -> f32 {
        self.center.y - self.size.y / 2.0
    }
}

/// Place image pixels in the logical coordinate system used by the UI.
/// A zoom of 1 means one image pixel per physical screen pixel, at any DPI.
pub fn image_rect(
    viewport: Rect,
    center: Vec2,
    size: [u32; 2],
    zoom: f32,
    scale_factor: f64,
) -> Rect {
    let scale = scale_factor as f32;
    let width = size[0] as f32 * zoom / scale;
    let height = size[1] as f32 * zoom / scale;
    let mut left = center.x - width / 2.0;
    let mut top = center.y + height / 2.0;
    if zoom == 1.0 {
        // Align the whole image to the framebuffer, including odd-sized images
        // and fractional DPI/scroll offsets, so linear sampling stays sharp.
        left = viewport.left() + ((left - viewport.left()) * scale).round() / scale;
        top = viewport.top() - ((viewport.top() - top) * scale).round() / scale;
    }
    Rect::from_x_y_w_h(left + width / 2.0, top - height / 2.0, width, height)
}

/// Fit an image to a logical viewport, returning a physical-pixel zoom ratio.
pub fn fit_image_zoom(viewport: Rect, size: [u32; 2], scale_factor: f64) -> f32 {
    (viewport.w() / size[0] as f32).min(viewport.h() / size[1] as f32) * scale_factor as f32
}

pub type Rgba = [f32; 4];

pub const WHITE: Rgba = [1.0, 1.0, 1.0, 1.0];

pub const fn srgba(red: f32, green: f32, blue: f32, alpha: f32) -> Rgba {
    [red, green, blue, alpha]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 0.001, "{actual} != {expected}");
    }

    #[test]
    fn native_images_and_thumbnails_keep_their_physical_pixel_size() {
        for scale in [1.0, 1.25, 1.5, 2.0, 3.0] {
            let viewport = Rect::from_w_h(1921.0 / scale as f32, 1081.0 / scale as f32);
            for size in [[601, 400], [256, 171], [171, 256], [2, 2]] {
                let bounds = image_rect(viewport, vec2(0.35, -17.25), size, 1.0, scale);
                close(bounds.w() * scale as f32, size[0] as f32);
                close(bounds.h() * scale as f32, size[1] as f32);
                // Integer pixel edges avoid blending adjacent source pixels, even
                // with odd viewport/image sizes and fractional DPI or scrolling.
                let left = (bounds.left() - viewport.left()) * scale as f32;
                let top = (viewport.top() - bounds.top()) * scale as f32;
                close(left, left.round());
                close(top, top.round());
            }
        }
    }

    #[test]
    fn fit_fills_the_window_at_every_dpi() {
        for scale in [1.0, 1.25, 1.5, 2.0, 3.0] {
            let viewport = Rect::from_w_h(1200.0 / scale as f32, 900.0 / scale as f32);
            for size in [[2400, 1200], [1200, 2400], [300, 200]] {
                let zoom = fit_image_zoom(viewport, size, scale);
                let bounds = image_rect(viewport, Vec2::ZERO, size, zoom, scale);
                close(zoom, (1200.0 / size[0] as f32).min(900.0 / size[1] as f32));
                assert!(bounds.w() <= viewport.w() + 0.001);
                assert!(bounds.h() <= viewport.h() + 0.001);
                assert!(
                    (bounds.w() - viewport.w()).abs() < 0.001
                        || (bounds.h() - viewport.h()).abs() < 0.001
                );
            }
        }
    }

    #[test]
    fn zoom_keeps_the_image_point_under_the_cursor_fixed() {
        let viewport = Rect::from_w_h(800.0, 600.0);
        let size = [601, 401];
        let pan = vec2(14.3, -51.2);
        let mouse = vec2(-100.25, 75.5);
        let old_zoom = 0.7;
        let new_zoom = 1.4;
        let new_pan = mouse + (pan - mouse) * (new_zoom / old_zoom);
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let before = image_rect(viewport, pan, size, old_zoom, scale);
            let after = image_rect(viewport, new_pan, size, new_zoom, scale);
            let before_x = (mouse.x - before.left()) * scale as f32 / old_zoom;
            let before_y = (before.top() - mouse.y) * scale as f32 / old_zoom;
            let after_x = (mouse.x - after.left()) * scale as f32 / new_zoom;
            let after_y = (after.top() - mouse.y) * scale as f32 / new_zoom;
            close(before_x, after_x);
            close(before_y, after_y);
        }
    }
}
