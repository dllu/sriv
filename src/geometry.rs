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

pub type Rgba = [f32; 4];

pub const WHITE: Rgba = [1.0, 1.0, 1.0, 1.0];

pub const fn srgba(red: f32, green: f32, blue: f32, alpha: f32) -> Rgba {
    [red, green, blue, alpha]
}
