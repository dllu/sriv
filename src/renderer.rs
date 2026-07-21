use crate::geometry::{Rect, Rgba, Vec2, WHITE};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glyphon::cosmic_text::Align as CosmicAlign;
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Color as TextColor, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use image::DynamicImage;
use std::mem;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

const QUAD_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.tex_coords = input.tex_coords;
    output.color = input.color;
    return output;
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, input.tex_coords) * input.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [f32; 4],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Debug)]
pub struct GpuTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    size: [u32; 2],
}

impl GpuTexture {
    pub fn size(&self) -> [u32; 2] {
        self.size
    }
}

#[derive(Clone, Copy)]
enum QuadImage<'a> {
    White,
    Texture(&'a GpuTexture),
}

#[derive(Clone, Copy)]
struct Quad<'a> {
    rect: Rect,
    color: Rgba,
    image: QuadImage<'a>,
}

#[derive(Clone, Copy)]
pub enum TextAlign {
    Left,
    Center,
    Right,
}

struct TextCommand {
    text: String,
    center: Vec2,
    size: [f32; 2],
    font_size: f32,
    color: Rgba,
    align: TextAlign,
}

pub struct Scene<'a> {
    rect: Rect,
    quads: Vec<Quad<'a>>,
    texts: Vec<TextCommand>,
}

impl<'a> Scene<'a> {
    pub fn new(rect: Rect) -> Self {
        Self {
            rect,
            quads: Vec::new(),
            texts: Vec::new(),
        }
    }

    pub fn rect(&mut self) -> RectBuilder<'_, 'a> {
        RectBuilder {
            scene: self,
            center: Vec2::ZERO,
            size: [1.0, 1.0],
            color: WHITE,
            fill: true,
            stroke: WHITE,
            stroke_weight: 1.0,
        }
    }

    pub fn line(&mut self) -> LineBuilder<'_, 'a> {
        LineBuilder {
            scene: self,
            start: Vec2::ZERO,
            end: Vec2::ZERO,
            weight: 1.0,
            color: WHITE,
        }
    }

    pub fn text(&mut self, text: impl AsRef<str>) -> TextBuilder<'_, 'a> {
        let width = self.rect.w();
        TextBuilder {
            scene: self,
            text: text.as_ref().to_owned(),
            center: Vec2::ZERO,
            size: [width, 24.0],
            font_size: 16.0,
            color: WHITE,
            align: TextAlign::Center,
        }
    }

    pub fn texture(&mut self, texture: &'a GpuTexture) -> TextureBuilder<'_, 'a> {
        TextureBuilder {
            scene: self,
            texture,
            center: Vec2::ZERO,
            size: [texture.size[0] as f32, texture.size[1] as f32],
            color: WHITE,
        }
    }

    fn push_solid_rect(&mut self, rect: Rect, color: Rgba) {
        self.quads.push(Quad {
            rect,
            color,
            image: QuadImage::White,
        });
    }
}

pub struct RectBuilder<'s, 'a> {
    scene: &'s mut Scene<'a>,
    center: Vec2,
    size: [f32; 2],
    color: Rgba,
    fill: bool,
    stroke: Rgba,
    stroke_weight: f32,
}

impl RectBuilder<'_, '_> {
    pub fn x_y(mut self, x: f32, y: f32) -> Self {
        self.center = Vec2::new(x, y);
        self
    }

    pub fn w_h(mut self, width: f32, height: f32) -> Self {
        self.size = [width, height];
        self
    }

    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    pub fn no_fill(mut self) -> Self {
        self.fill = false;
        self
    }

    pub fn stroke(mut self, color: Rgba) -> Self {
        self.stroke = color;
        self
    }

    pub fn stroke_weight(mut self, weight: f32) -> Self {
        self.stroke_weight = weight;
        self
    }
}

impl Drop for RectBuilder<'_, '_> {
    fn drop(&mut self) {
        let [width, height] = self.size;
        if self.fill {
            self.scene.push_solid_rect(
                Rect::from_x_y_w_h(self.center.x, self.center.y, width, height),
                self.color,
            );
            return;
        }

        let weight = self.stroke_weight.max(1.0);
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        self.scene.push_solid_rect(
            Rect::from_x_y_w_h(
                self.center.x,
                self.center.y + half_height,
                width + weight,
                weight,
            ),
            self.stroke,
        );
        self.scene.push_solid_rect(
            Rect::from_x_y_w_h(
                self.center.x,
                self.center.y - half_height,
                width + weight,
                weight,
            ),
            self.stroke,
        );
        self.scene.push_solid_rect(
            Rect::from_x_y_w_h(
                self.center.x - half_width,
                self.center.y,
                weight,
                height + weight,
            ),
            self.stroke,
        );
        self.scene.push_solid_rect(
            Rect::from_x_y_w_h(
                self.center.x + half_width,
                self.center.y,
                weight,
                height + weight,
            ),
            self.stroke,
        );
    }
}

pub struct LineBuilder<'s, 'a> {
    scene: &'s mut Scene<'a>,
    start: Vec2,
    end: Vec2,
    weight: f32,
    color: Rgba,
}

impl LineBuilder<'_, '_> {
    pub fn start(mut self, start: Vec2) -> Self {
        self.start = start;
        self
    }

    pub fn end(mut self, end: Vec2) -> Self {
        self.end = end;
        self
    }

    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }
}

impl Drop for LineBuilder<'_, '_> {
    fn drop(&mut self) {
        let midpoint = (self.start + self.end) / 2.0;
        let delta = self.end - self.start;
        // Every current call site draws horizontal underlines. Keeping the fallback
        // axis-aligned also makes one-pixel UI lines remain crisp.
        let (width, height) = if delta.x.abs() >= delta.y.abs() {
            (delta.x.abs().max(1.0), self.weight.max(1.0))
        } else {
            (self.weight.max(1.0), delta.y.abs().max(1.0))
        };
        self.scene.push_solid_rect(
            Rect::from_x_y_w_h(midpoint.x, midpoint.y, width, height),
            self.color,
        );
    }
}

pub struct TextureBuilder<'s, 'a> {
    scene: &'s mut Scene<'a>,
    texture: &'a GpuTexture,
    center: Vec2,
    size: [f32; 2],
    color: Rgba,
}

impl TextureBuilder<'_, '_> {
    pub fn x_y(mut self, x: f32, y: f32) -> Self {
        self.center = Vec2::new(x, y);
        self
    }

    pub fn w_h(mut self, width: f32, height: f32) -> Self {
        self.size = [width.max(0.0), height.max(0.0)];
        self
    }

    #[allow(dead_code)]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }
}

impl Drop for TextureBuilder<'_, '_> {
    fn drop(&mut self) {
        self.scene.quads.push(Quad {
            rect: Rect::from_x_y_w_h(self.center.x, self.center.y, self.size[0], self.size[1]),
            color: self.color,
            image: QuadImage::Texture(self.texture),
        });
    }
}

pub struct TextBuilder<'s, 'a> {
    scene: &'s mut Scene<'a>,
    text: String,
    center: Vec2,
    size: [f32; 2],
    font_size: f32,
    color: Rgba,
    align: TextAlign,
}

impl TextBuilder<'_, '_> {
    pub fn font_size(mut self, size: u32) -> Self {
        self.font_size = size as f32;
        self
    }

    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    pub fn w_h(mut self, width: f32, height: f32) -> Self {
        self.size = [width.max(1.0), height.max(1.0)];
        self
    }

    pub fn x_y(mut self, x: f32, y: f32) -> Self {
        self.center = Vec2::new(x, y);
        self
    }

    pub fn left_justify(mut self) -> Self {
        self.align = TextAlign::Left;
        self
    }

    pub fn right_justify(mut self) -> Self {
        self.align = TextAlign::Right;
        self
    }
}

impl Drop for TextBuilder<'_, '_> {
    fn drop(&mut self) {
        self.scene.texts.push(TextCommand {
            text: mem::take(&mut self.text),
            center: self.center,
            size: self.size,
            font_size: self.font_size,
            color: self.color,
            align: self.align,
        });
    }
}

pub struct Renderer {
    instance: wgpu::Instance,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    white_texture: GpuTexture,
    supports_rgba16: bool,
    font_system: FontSystem,
    font_family: Option<String>,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    window: Arc<Window>,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, event_loop: &ActiveEventLoop) -> Result<Self> {
        let physical_size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle(
            Box::new(event_loop.owned_display_handle()),
        ));
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create the window surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .context("failed to find a compatible graphics adapter")?;
        let supports_rgba16 = adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
        let required_features = if supports_rgba16 {
            wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sriv device"),
                required_features,
                ..Default::default()
            })
            .await
            .context("failed to create the graphics device")?;
        let mut surface_config = surface
            .get_default_config(
                &adapter,
                physical_size.width.max(1),
                physical_size.height.max(1),
            )
            .context("the graphics adapter cannot present to this window")?;
        if let Some(srgb) = surface
            .get_capabilities(&adapter)
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
        {
            surface_config.format = srgb;
        }
        surface.configure(&device, &surface_config);

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sriv texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sriv image sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sriv quad shader"),
            source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sriv quad pipeline layout"),
            bind_group_layouts: &[Some(&texture_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sriv quad pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::layout())],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let white_texture = Self::create_texture_inner(
            &device,
            &queue,
            &texture_layout,
            &sampler,
            "sriv white texture",
            1,
            1,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            4,
            &[255, 255, 255, 255],
        );

        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let viewport = Viewport::new(&device, &cache);
        let mut atlas = TextAtlas::new(&device, &queue, &cache, surface_config.format);
        let text_renderer =
            TextRenderer::new(&mut atlas, &device, wgpu::MultisampleState::default(), None);

        Ok(Self {
            instance,
            device,
            queue,
            surface,
            surface_config,
            pipeline,
            texture_layout,
            sampler,
            white_texture,
            supports_rgba16,
            font_system,
            font_family: None,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            window,
        })
    }

    pub fn load_ui_font(&mut self, path: Option<&Path>) {
        let Some(path) = path else {
            return;
        };
        let previous_count = self.font_system.db().faces().count();
        if let Err(error) = self.font_system.db_mut().load_font_file(path) {
            eprintln!("Failed to load ui_font_path {}: {error}", path.display());
            return;
        }
        self.font_family = self
            .font_system
            .db()
            .faces()
            .skip(previous_count)
            .find_map(|face| face.families.first().map(|(name, _)| name.clone()));
    }

    pub fn supports_rgba16(&self) -> bool {
        self.supports_rgba16
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn texture_from_image(&self, image: &DynamicImage) -> GpuTexture {
        let rgba = image.to_rgba8();
        self.create_texture(
            "sriv image",
            rgba.width(),
            rgba.height(),
            wgpu::TextureFormat::Rgba8UnormSrgb,
            4 * rgba.width(),
            rgba.as_raw(),
        )
    }

    pub fn create_texture(
        &self,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        bytes_per_row: u32,
        pixels: &[u8],
    ) -> GpuTexture {
        Self::create_texture_inner(
            &self.device,
            &self.queue,
            &self.texture_layout,
            &self.sampler,
            label,
            width,
            height,
            format,
            bytes_per_row,
            pixels,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_texture_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        bytes_per_row: u32,
        pixels: &[u8],
    ) -> GpuTexture {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        GpuTexture {
            _texture: texture,
            bind_group,
            size: [width, height],
        }
    }

    pub fn render(&mut self, scene: &Scene<'_>, scale_factor: f64) -> Result<()> {
        let logical_width = scene.rect.w().max(1.0);
        let logical_height = scene.rect.h().max(1.0);
        let mut vertices = Vec::with_capacity(scene.quads.len() * 4);
        let mut indices = Vec::with_capacity(scene.quads.len() * 6);
        for (quad_index, quad) in scene.quads.iter().enumerate() {
            let left = 2.0 * quad.rect.left() / logical_width;
            let right = 2.0 * quad.rect.right() / logical_width;
            let top = 2.0 * quad.rect.top() / logical_height;
            let bottom = 2.0 * quad.rect.bottom() / logical_height;
            vertices.extend_from_slice(&[
                Vertex {
                    position: [left, bottom],
                    tex_coords: [0.0, 1.0],
                    color: quad.color,
                },
                Vertex {
                    position: [right, bottom],
                    tex_coords: [1.0, 1.0],
                    color: quad.color,
                },
                Vertex {
                    position: [right, top],
                    tex_coords: [1.0, 0.0],
                    color: quad.color,
                },
                Vertex {
                    position: [left, top],
                    tex_coords: [0.0, 0.0],
                    color: quad.color,
                },
            ]);
            let base = (quad_index * 4) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let vertex_buffer = (!vertices.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sriv frame vertices"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
        });
        let index_buffer = (!indices.is_empty()).then(|| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sriv frame indices"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                })
        });

        let scale = scale_factor as f32;
        let family = self
            .font_family
            .as_deref()
            .map(Family::Name)
            .unwrap_or(Family::SansSerif);
        let mut text_buffers = Vec::with_capacity(scene.texts.len());
        for command in &scene.texts {
            let line_height = (command.font_size * 1.25).max(command.font_size + 1.0);
            let mut buffer = TextBuffer::new(
                &mut self.font_system,
                Metrics::new(command.font_size * scale, line_height * scale),
            );
            buffer.set_size(
                Some(command.size[0] * scale),
                Some(command.size[1].max(line_height) * scale),
            );
            buffer.set_text(
                &command.text,
                &Attrs::new().family(family),
                Shaping::Advanced,
                None,
            );
            let align = match command.align {
                TextAlign::Left => CosmicAlign::Left,
                TextAlign::Center => CosmicAlign::Center,
                TextAlign::Right => CosmicAlign::Right,
            };
            for line in &mut buffer.lines {
                line.set_align(Some(align));
            }
            buffer.shape_until_scroll(&mut self.font_system, false);
            text_buffers.push(buffer);
        }

        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.surface_config.width,
                height: self.surface_config.height,
            },
        );
        let text_areas = text_buffers
            .iter()
            .zip(&scene.texts)
            .map(|(buffer, command)| {
                let line_height = (command.font_size * 1.25).max(command.font_size + 1.0);
                let left = (logical_width / 2.0 + command.center.x - command.size[0] / 2.0) * scale;
                let top = (logical_height / 2.0 - command.center.y - line_height / 2.0) * scale;
                let bounds_left = left.floor() as i32;
                let bounds_top = ((logical_height / 2.0 - command.center.y - command.size[1] / 2.0)
                    * scale)
                    .floor() as i32;
                TextArea {
                    buffer,
                    left,
                    top,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: bounds_left,
                        top: bounds_top,
                        right: (left + command.size[0] * scale).ceil() as i32,
                        bottom: (bounds_top as f32 + command.size[1] * scale).ceil() as i32,
                    },
                    default_color: rgba_to_text_color(command.color),
                    custom_glyphs: &[],
                }
            });
        self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        )?;

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self.instance.create_surface(self.window.clone())?;
                self.surface.configure(&self.device, &self.surface_config);
                self.window.request_redraw();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                anyhow::bail!("surface validation failed while acquiring the next frame")
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sriv frame encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sriv frame pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let (Some(vertex_buffer), Some(index_buffer)) = (&vertex_buffer, &index_buffer) {
                pass.set_pipeline(&self.pipeline);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                for (quad_index, quad) in scene.quads.iter().enumerate() {
                    let bind_group = match quad.image {
                        QuadImage::White => &self.white_texture.bind_group,
                        QuadImage::Texture(texture) => &texture.bind_group,
                    };
                    pass.set_bind_group(0, bind_group, &[]);
                    let first = (quad_index * 6) as u32;
                    pass.draw_indexed(first..first + 6, 0, 0..1);
                }
            }
            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)?;
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
        self.atlas.trim();
        Ok(())
    }
}

fn rgba_to_text_color([red, green, blue, alpha]: Rgba) -> TextColor {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    TextColor::rgba(channel(red), channel(green), channel(blue), channel(alpha))
}
