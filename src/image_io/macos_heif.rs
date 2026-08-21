use super::{DecodedImage, EmbeddedColorProfile};
use anyhow::{anyhow, Result};
use image::{DynamicImage, RgbaImage};
use objc2_core_foundation::{CFData, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
};
use objc2_image_io::CGImageSource;
use std::ffi::c_void;
use std::fs;
use std::path::Path;

/// Decode HEIF with macOS ImageIO, which includes the system HEVC decoder.
pub(super) fn decode_heif_image(path: &Path) -> Result<DecodedImage> {
    let encoded = fs::read(path)?;
    let data = CFData::from_bytes(&encoded);
    // No options dictionary means there are no dynamically typed values for
    // ImageIO to interpret in these otherwise-unsafe APIs.
    let source = unsafe { CGImageSource::with_data(&data, None) }
        .ok_or_else(|| anyhow!("ImageIO could not open HEIF data"))?;
    let cg_image = unsafe { source.image_at_index(0, None) }
        .ok_or_else(|| anyhow!("ImageIO could not decode the primary HEIF image"))?;

    let width = CGImage::width(Some(&cg_image));
    let height = CGImage::height(Some(&cg_image));
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| anyhow!("HEIF row size overflows usize"))?;
    let len = row_bytes
        .checked_mul(height)
        .ok_or_else(|| anyhow!("HEIF image size overflows usize"))?;
    let width_u32 = u32::try_from(width).map_err(|_| anyhow!("HEIF image is too wide"))?;
    let height_u32 = u32::try_from(height).map_err(|_| anyhow!("HEIF image is too tall"))?;

    let source_space = CGImage::color_space(Some(&cg_image));
    let fallback_space;
    let color_space = match source_space.as_deref() {
        Some(space) => space,
        None => {
            fallback_space = CGColorSpace::new_device_rgb()
                .ok_or_else(|| anyhow!("could not create an RGB color space"))?;
            &fallback_space
        }
    };
    let embedded_color_profile = CGColorSpace::icc_data(Some(color_space))
        .map(|profile| EmbeddedColorProfile::Icc(profile.to_vec()));

    // CoreGraphics exposes a stable 8-bit RGBA layout for every image ImageIO
    // can decode. Convert its premultiplied alpha to `image`'s straight alpha.
    let mut pixels = vec![0_u8; len];
    // Default byte order with alpha last is the platform-independent RGBA layout.
    let bitmap_info = CGImageAlphaInfo::PremultipliedLast.0;
    let context = unsafe {
        CGBitmapContextCreate(
            pixels.as_mut_ptr().cast::<c_void>(),
            width,
            height,
            8,
            row_bytes,
            Some(color_space),
            bitmap_info,
        )
    }
    .ok_or_else(|| anyhow!("could not create a bitmap context for HEIF decoding"))?;
    CGContext::draw_image(
        Some(&context),
        CGRect::new(CGPoint::ZERO, CGSize::new(width as f64, height as f64)),
        Some(&cg_image),
    );
    drop(context);

    for pixel in pixels.chunks_exact_mut(4) {
        let alpha = pixel[3] as u32;
        if alpha != 0 && alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u32 * 255 + alpha / 2) / alpha).min(255) as u8;
            }
        }
    }

    let image = RgbaImage::from_raw(width_u32, height_u32, pixels)
        .ok_or_else(|| anyhow!("decoded HEIF data has the wrong length"))?;
    Ok(DecodedImage {
        image: DynamicImage::ImageRgba8(image),
        embedded_color_profile,
    })
}
