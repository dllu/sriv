use anyhow::{anyhow, Result};
use image as image_rs;
use image::{imageops::FilterType, DynamicImage, ImageDecoder, ImageReader, RgbImage, RgbaImage};
use libheif_rs::{ColorSpace as HeifColorSpace, HeifContext, LibHeif, Plane, RgbChroma};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::clip;
use crate::color::{self, EmbeddedColorProfile, OutputColorSpace};
use crate::state::{FullImageTile, TilePixelFormat};

struct DecodedImage {
    image: DynamicImage,
    embedded_color_profile: Option<EmbeddedColorProfile>,
}

const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "bmp", "tiff", "gif", "webp", "tif", "heif", "heic",
];

/// List of recognized raw file extensions for detecting XMP sidecars.
const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "bay", "cap", "cr2", "cr3", "crw", "cs1", "dcr", "dng", "erf", "fff",
    "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "ptx", "pxn",
    "raf", "raw", "rwl", "rw2", "rwz", "sr2", "srf", "srw", "x3f",
];

fn extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn is_heif_path(path: &Path) -> bool {
    matches!(extension_lower(path).as_deref(), Some("heif" | "heic"))
}

pub(crate) fn is_supported_image_path(path: &Path) -> bool {
    extension_lower(path)
        .as_deref()
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext))
}

/// Compute the cache path for an image based on a SHA1 of its path.
/// The cache suffix includes the output space so P3 pixels are never reused as sRGB (or vice versa).
fn thumbnail_cache_path(
    cache_base: &Path,
    image_path: &Path,
    output_color_space: OutputColorSpace,
) -> PathBuf {
    clip::cache_file_path(
        cache_base,
        image_path,
        &format!("{}.png", output_color_space.cache_tag()),
    )
}

fn orientation_from_tag_value(value: &rexif::TagValue) -> Option<u16> {
    let raw = match value {
        rexif::TagValue::U16(vals) => vals.first().copied(),
        rexif::TagValue::I16(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::U8(vals) => vals.first().map(|&v| v as u16),
        rexif::TagValue::I8(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::U32(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::I32(vals) => vals.first().and_then(|v| u16::try_from(*v).ok()),
        rexif::TagValue::URational(vals) => vals.first().and_then(|r| {
            let num = r.numerator;
            let den = r.denominator;
            if den == 0 || num % den != 0 {
                return None;
            }
            u16::try_from(num / den).ok()
        }),
        rexif::TagValue::IRational(vals) => vals.first().and_then(|r| {
            let num = r.numerator;
            let den = r.denominator;
            if den == 0 || num % den != 0 {
                return None;
            }
            u16::try_from(num / den).ok()
        }),
        _ => None,
    }?;

    (1..=8).contains(&raw).then_some(raw)
}

fn parse_exif_quiet(path: &Path) -> Option<rexif::ExifData> {
    let data = fs::read(path).ok()?;
    rexif::parse_buffer_quiet(&data).0.ok()
}

fn adjust_orientation_full(img: image_rs::DynamicImage, path: &Path) -> image_rs::DynamicImage {
    let mut oriented = img;
    if let Some(exif) = parse_exif_quiet(path) {
        for entry in exif.entries {
            if entry.tag == rexif::ExifTag::Orientation {
                if let Some(code) = orientation_from_tag_value(&entry.value) {
                    oriented = match code {
                        2 => oriented.fliph(),
                        3 => oriented.rotate180(),
                        4 => oriented.flipv(),
                        5 => oriented.rotate90().fliph(),
                        6 => oriented.rotate90(),
                        7 => oriented.rotate270().fliph(),
                        8 => oriented.rotate270(),
                        _ => oriented,
                    };
                }
                break;
            }
        }
    }
    oriented
}

fn srgb_u16_to_linear_u16(value: u16) -> u16 {
    static LUT: OnceLock<Vec<u16>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..=u16::MAX)
            .map(|encoded| {
                let encoded = encoded as f64 / u16::MAX as f64;
                let linear = if encoded <= 0.04045 {
                    encoded / 12.92
                } else {
                    ((encoded + 0.055) / 1.055).powf(2.4)
                };
                (linear * u16::MAX as f64).round() as u16
            })
            .collect()
    })[value as usize]
}

fn linear_u16_to_srgb_u8(value: u16) -> u8 {
    static LUT: OnceLock<Vec<u8>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..=u16::MAX)
            .map(|linear| {
                let linear = linear as f64 / u16::MAX as f64;
                let encoded = if linear <= 0.003_130_8 {
                    linear * 12.92
                } else {
                    1.055 * linear.powf(1.0 / 2.4) - 0.055
                };
                (encoded * u8::MAX as f64).round() as u8
            })
            .collect()
    })[value as usize]
}

fn push_linear_u16_rgba(pixel_data: &mut Vec<u8>, rgba: [u16; 4]) {
    for channel in rgba {
        pixel_data.extend_from_slice(&channel.to_ne_bytes());
    }
}

pub(crate) fn linear_rgba16_bytes_to_srgba8(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for (channel, chunk) in bytes.chunks_exact(2).enumerate() {
        let value = u16::from_ne_bytes([chunk[0], chunk[1]]);
        if channel % 4 == 3 {
            out.push(((value as u32 + 128) / 257) as u8);
        } else {
            out.push(linear_u16_to_srgb_u8(value));
        }
    }
    out
}

fn libheif() -> &'static LibHeif {
    static LIBHEIF: OnceLock<LibHeif> = OnceLock::new();
    LIBHEIF.get_or_init(LibHeif::new)
}

fn heif_target_color_space(has_alpha: bool, high_bit_depth: bool) -> HeifColorSpace {
    match (has_alpha, high_bit_depth, cfg!(target_endian = "little")) {
        (false, false, _) => HeifColorSpace::Rgb(RgbChroma::Rgb),
        (true, false, _) => HeifColorSpace::Rgb(RgbChroma::Rgba),
        (false, true, true) => HeifColorSpace::Rgb(RgbChroma::HdrRgbLe),
        (false, true, false) => HeifColorSpace::Rgb(RgbChroma::HdrRgbBe),
        (true, true, true) => HeifColorSpace::Rgb(RgbChroma::HdrRgbaLe),
        (true, true, false) => HeifColorSpace::Rgb(RgbChroma::HdrRgbaBe),
    }
}

fn heif_bytes_per_pixel(has_alpha: bool, high_bit_depth: bool) -> usize {
    let channels = if has_alpha { 4 } else { 3 };
    let bytes_per_channel = if high_bit_depth { 2 } else { 1 };
    channels * bytes_per_channel
}

fn copy_heif_interleaved_plane(plane: &Plane<&[u8]>, bytes_per_pixel: usize) -> Result<Vec<u8>> {
    if plane.stride == 0 {
        return Err(anyhow!("HEIF row stride is zero"));
    }
    let width = plane.width as usize;
    let height = plane.height as usize;
    let row_size = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| anyhow!("HEIF image row size overflows usize"))?;
    if row_size > plane.stride {
        return Err(anyhow!(
            "HEIF row stride {} is smaller than row size {}",
            plane.stride,
            row_size
        ));
    }
    let expected_len = row_size
        .checked_mul(height)
        .ok_or_else(|| anyhow!("HEIF image size overflows usize"))?;
    let mut pixels = Vec::with_capacity(expected_len);
    for row in plane.data.chunks_exact(plane.stride).take(height) {
        pixels.extend_from_slice(&row[..row_size]);
    }
    if pixels.len() != expected_len {
        return Err(anyhow!("HEIF pixel data ended before all rows were copied"));
    }
    Ok(pixels)
}

fn native_u16_samples(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn image_buffer_from_raw<P, Container>(
    width: u32,
    height: u32,
    buf: Container,
    format_name: &str,
) -> Result<image_rs::ImageBuffer<P, Container>>
where
    P: image_rs::Pixel,
    Container: std::ops::Deref<Target = [P::Subpixel]>,
{
    image_rs::ImageBuffer::from_raw(width, height, buf)
        .ok_or_else(|| anyhow!("decoded HEIF {format_name} data has the wrong length"))
}

fn decode_heif_image(path: &Path) -> Result<DecodedImage> {
    let lib_heif = libheif();
    let bytes = fs::read(path)?;
    let ctx = HeifContext::read_from_bytes(&bytes)?;
    let handle = ctx.primary_image_handle()?;
    let embedded_color_profile = handle
        .color_profile_raw()
        .map(|profile| EmbeddedColorProfile::Icc(profile.data))
        .or_else(|| {
            let profile = handle.color_profile_nclx()?;
            let color_primaries = profile.color_primaries();
            let transfer_characteristics = profile.transfer_characteristics();
            if matches!(
                color_primaries,
                libheif_rs::ColorPrimaries::Unknown | libheif_rs::ColorPrimaries::Unspecified
            ) || matches!(
                transfer_characteristics,
                libheif_rs::TransferCharacteristics::Unknown
                    | libheif_rs::TransferCharacteristics::Unspecified
            ) {
                return None;
            }
            Some(EmbeddedColorProfile::Cicp {
                color_primaries: color_primaries as u8,
                transfer_characteristics: transfer_characteristics as u8,
            })
        });
    let has_alpha = handle.has_alpha_channel();
    let high_bit_depth = handle.luma_bits_per_pixel() > 8 || handle.chroma_bits_per_pixel() > 8;
    let color_space = heif_target_color_space(has_alpha, high_bit_depth);
    let decoded = lib_heif.decode(&handle, color_space, None)?;
    let planes = decoded.planes();
    let plane = planes
        .interleaved
        .ok_or_else(|| anyhow!("decoded HEIF did not provide interleaved pixel data"))?;
    let width = plane.width;
    let height = plane.height;
    let bytes_per_pixel = heif_bytes_per_pixel(has_alpha, high_bit_depth);
    let pixels = copy_heif_interleaved_plane(&plane, bytes_per_pixel)?;

    let image = match (has_alpha, high_bit_depth) {
        (false, false) => {
            image_rs::DynamicImage::ImageRgb8(image_buffer_from_raw(width, height, pixels, "RGB8")?)
        }
        (true, false) => image_rs::DynamicImage::ImageRgba8(image_buffer_from_raw(
            width, height, pixels, "RGBA8",
        )?),
        (false, true) => image_rs::DynamicImage::ImageRgb16(image_buffer_from_raw(
            width,
            height,
            native_u16_samples(&pixels),
            "RGB16",
        )?),
        (true, true) => image_rs::DynamicImage::ImageRgba16(image_buffer_from_raw(
            width,
            height,
            native_u16_samples(&pixels),
            "RGBA16",
        )?),
    };
    Ok(DecodedImage {
        image,
        embedded_color_profile,
    })
}

fn decode_standard_image(path: &Path) -> Result<DecodedImage> {
    let mut reader = ImageReader::open(path)?;
    reader.no_limits();
    let mut decoder = reader.into_decoder()?;
    let embedded_color_profile = match decoder.icc_profile() {
        Ok(profile) => profile.map(EmbeddedColorProfile::Icc),
        Err(error) => {
            eprintln!(
                "Failed to read embedded color profile from {}: {error}",
                path.display()
            );
            None
        }
    };
    Ok(DecodedImage {
        image: DynamicImage::from_decoder(decoder)?,
        embedded_color_profile,
    })
}

fn convert_decoded_to_output(
    path: &Path,
    decoded: DecodedImage,
    output_color_space: OutputColorSpace,
) -> DynamicImage {
    let DecodedImage {
        mut image,
        embedded_color_profile,
    } = decoded;
    if embedded_color_profile.is_none() && output_color_space == OutputColorSpace::Srgb {
        return image;
    }
    match color::convert_to_output(
        &mut image,
        embedded_color_profile.as_ref(),
        output_color_space,
    ) {
        Ok(()) => image,
        Err(error) => {
            eprintln!(
                "Ignoring invalid or unsupported color profile in {}: {error:#}",
                path.display()
            );
            if embedded_color_profile.is_some() && output_color_space != OutputColorSpace::Srgb {
                // A broken embedded profile should not leave assumed-sRGB pixels mislabeled as P3.
                if color::convert_to_output(&mut image, None, output_color_space).is_ok() {
                    return image;
                }
            }
            image
        }
    }
}

fn decode_image_no_limits(
    path: &Path,
    output_color_space: OutputColorSpace,
) -> Result<image_rs::DynamicImage> {
    let is_heif = is_heif_path(path);
    let decoded = if is_heif {
        decode_heif_image(path)?
    } else {
        decode_standard_image(path)?
    };
    let image = convert_decoded_to_output(path, decoded, output_color_space);
    // libheif applies HEIF orientation transformations during decoding.
    Ok(if is_heif {
        image
    } else {
        adjust_orientation_full(image, path)
    })
}

fn collect_tiled_pixels<F>(
    full_w: u32,
    full_h: u32,
    format: TilePixelFormat,
    bytes_per_pixel: usize,
    mut fill_pixel: F,
) -> Vec<FullImageTile>
where
    F: FnMut(u32, u32, &mut Vec<u8>),
{
    const MAX_TILE_SIZE: u32 = 8192;
    let mut tiles = Vec::new();
    for y in (0..full_h).step_by(MAX_TILE_SIZE as usize) {
        for x in (0..full_w).step_by(MAX_TILE_SIZE as usize) {
            let tile_w = (full_w - x).min(MAX_TILE_SIZE);
            let tile_h = (full_h - y).min(MAX_TILE_SIZE);
            let mut pixel_data =
                Vec::with_capacity(tile_w as usize * tile_h as usize * bytes_per_pixel);
            for py in y..(y + tile_h) {
                for px in x..(x + tile_w) {
                    fill_pixel(px, py, &mut pixel_data);
                }
            }
            tiles.push((x, y, tile_w, tile_h, format, pixel_data));
        }
    }
    tiles
}

pub(crate) fn load_full_image_tiles(
    path: &Path,
    output_color_space: OutputColorSpace,
) -> Result<(u32, u32, Vec<FullImageTile>)> {
    let img = decode_image_no_limits(path, output_color_space)?;
    let (full_w, full_h) = (img.width(), img.height());
    let tiles = match img {
        image_rs::DynamicImage::ImageLuma8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [l] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[l, l, l, u8::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageLumaA8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [l, a] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[l, l, l, a]);
            },
        ),
        image_rs::DynamicImage::ImageRgb8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                let [r, g, b] = buf.get_pixel(px, py).0;
                pixel_data.extend_from_slice(&[r, g, b, u8::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageRgba8(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba8,
            4,
            |px, py, pixel_data| {
                pixel_data.extend_from_slice(&buf.get_pixel(px, py).0);
            },
        ),
        image_rs::DynamicImage::ImageLuma16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [l] = buf.get_pixel(px, py).0;
                let l = srgb_u16_to_linear_u16(l);
                push_linear_u16_rgba(pixel_data, [l, l, l, u16::MAX]);
            },
        ),
        image_rs::DynamicImage::ImageLumaA16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [l, a] = buf.get_pixel(px, py).0;
                let l = srgb_u16_to_linear_u16(l);
                push_linear_u16_rgba(pixel_data, [l, l, l, a]);
            },
        ),
        image_rs::DynamicImage::ImageRgb16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [r, g, b] = buf.get_pixel(px, py).0;
                push_linear_u16_rgba(
                    pixel_data,
                    [
                        srgb_u16_to_linear_u16(r),
                        srgb_u16_to_linear_u16(g),
                        srgb_u16_to_linear_u16(b),
                        u16::MAX,
                    ],
                );
            },
        ),
        image_rs::DynamicImage::ImageRgba16(buf) => collect_tiled_pixels(
            full_w,
            full_h,
            TilePixelFormat::Rgba16,
            8,
            |px, py, pixel_data| {
                let [r, g, b, a] = buf.get_pixel(px, py).0;
                push_linear_u16_rgba(
                    pixel_data,
                    [
                        srgb_u16_to_linear_u16(r),
                        srgb_u16_to_linear_u16(g),
                        srgb_u16_to_linear_u16(b),
                        a,
                    ],
                );
            },
        ),
        other => {
            let rgba = other.to_rgba8();
            collect_tiled_pixels(
                full_w,
                full_h,
                TilePixelFormat::Rgba8,
                4,
                |px, py, pixel_data| {
                    pixel_data.extend_from_slice(&rgba.get_pixel(px, py).0);
                },
            )
        }
    };
    Ok((full_w, full_h, tiles))
}

/// Load a display-ready thumbnail from the output-space-specific cache, or generate it.
pub(crate) fn load_thumbnail(
    cache_base: &Path,
    image_path: &Path,
    size: u32,
    output_color_space: OutputColorSpace,
) -> DynamicImage {
    let cache_path = thumbnail_cache_path(cache_base, image_path, output_color_space);
    if let (Ok(source), Ok(cached)) = (fs::metadata(image_path), fs::metadata(&cache_path)) {
        if matches!(
            (source.modified(), cached.modified()),
            (Ok(source_time), Ok(cache_time)) if cache_time >= source_time
        ) {
            if let Ok(image) = image_rs::open(&cache_path) {
                return DynamicImage::ImageRgba8(image.to_rgba8());
            }
        }
    }

    let Ok(image) = decode_image_no_limits(image_path, output_color_space) else {
        return fallback_thumbnail();
    };
    let mut thumbnail = image.thumbnail(size, size);
    let (width, height) = (thumbnail.width(), thumbnail.height());
    if width == 0 || height == 0 {
        return fallback_thumbnail();
    }
    if width < 2 || height < 2 {
        thumbnail = thumbnail.resize_exact(width.max(2), height.max(2), FilterType::Nearest);
    }
    let thumbnail = DynamicImage::ImageRgba8(thumbnail.to_rgba8());
    if let Some(parent) = cache_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = thumbnail.save(cache_path);
    thumbnail
}

pub(crate) fn thumbnail_for_clip(
    image: &DynamicImage,
    output_color_space: OutputColorSpace,
) -> Result<RgbImage> {
    if output_color_space == OutputColorSpace::Srgb {
        Ok(image.to_rgb8())
    } else {
        Ok(color::output_to_srgb(image, output_color_space)?.to_rgb8())
    }
}

fn fallback_thumbnail() -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        2,
        2,
        image_rs::Rgba([128, 128, 128, 255]),
    ))
}

/// Scan a directory for raw files that have matching XMP sidecars.
fn scan_raw_sidecars(dir: &Path) -> HashMap<String, bool> {
    let mut map = HashMap::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext_raw = match path.extension().and_then(|s| s.to_str()) {
                Some(ext) => ext,
                None => continue,
            };
            let ext_lower = ext_raw.to_ascii_lowercase();
            if !RAW_EXTENSIONS.contains(&ext_lower.as_str()) {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(stem) => stem,
                None => continue,
            };
            let mut has_xmp = false;
            // Typical variants: foo.RAF.xmp, foo.raf.xmp, foo.xmp
            let base_xmp = path.with_extension("xmp");
            let candidates = [
                path.with_extension(format!("{}.xmp", ext_raw)),
                path.with_extension(format!("{}.xmp", ext_lower)),
                base_xmp.clone(),
                path.parent()
                    .map(|parent| parent.join(format!("{}.xmp", stem)))
                    .unwrap_or_else(|| base_xmp.clone()),
            ];
            for candidate in candidates.iter() {
                if candidate.exists() {
                    has_xmp = true;
                    break;
                }
            }
            let key = stem.to_string();
            map.entry(key)
                .and_modify(|flag| *flag |= has_xmp)
                .or_insert(has_xmp);
        }
    }
    map
}

/// Determine which images have corresponding raw files with XMP sidecars.
pub(crate) fn detect_thumb_sidecars(image_paths: &[PathBuf]) -> Vec<bool> {
    let mut dir_cache: HashMap<PathBuf, HashMap<String, bool>> = HashMap::new();
    let mut flags = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem.to_string(),
            None => {
                flags.push(false);
                continue;
            }
        };
        let parent = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let entry = dir_cache
            .entry(parent.clone())
            .or_insert_with(|| scan_raw_sidecars(&parent));
        let flag = entry.get(&stem).copied().unwrap_or(false);
        flags.push(flag);
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_rs::{ExtendedColorType, ImageBuffer, ImageEncoder, Rgb};
    use moxcms::ColorProfile;
    use std::time::SystemTime;

    #[test]
    fn orientation_from_urational_rounds_down() {
        let value = rexif::TagValue::URational(vec![rexif::URational {
            numerator: 6,
            denominator: 2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), Some(3));
    }

    #[test]
    fn orientation_from_irational_with_negative_denominator() {
        let value = rexif::TagValue::IRational(vec![rexif::IRational {
            numerator: -12,
            denominator: -2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), Some(6));
    }

    #[test]
    fn orientation_from_irational_non_integer() {
        let value = rexif::TagValue::IRational(vec![rexif::IRational {
            numerator: 3,
            denominator: 2,
        }]);
        assert_eq!(orientation_from_tag_value(&value), None);
    }

    #[test]
    fn tiles_rgb16_images_as_rgba16_without_whole_image_rgba_conversion() {
        let img = image_rs::DynamicImage::ImageRgb16(ImageBuffer::from_fn(3, 2, |x, y| {
            Rgb([
                (x as u16) * 1000 + 1,
                (y as u16) * 1000 + 2,
                (x as u16 + y as u16) * 1000 + 3,
            ])
        }));
        let (full_w, full_h) = (img.width(), img.height());
        let tiles = match img {
            image_rs::DynamicImage::ImageRgb16(buf) => collect_tiled_pixels(
                full_w,
                full_h,
                TilePixelFormat::Rgba16,
                8,
                |px, py, pixel_data| {
                    let [r, g, b] = buf.get_pixel(px, py).0;
                    push_linear_u16_rgba(
                        pixel_data,
                        [
                            srgb_u16_to_linear_u16(r),
                            srgb_u16_to_linear_u16(g),
                            srgb_u16_to_linear_u16(b),
                            u16::MAX,
                        ],
                    );
                },
            ),
            _ => unreachable!(),
        };

        assert_eq!(tiles.len(), 1);
        let (_, _, width, height, format, pixel_data) = &tiles[0];
        assert_eq!((*width, *height), (3, 2));
        assert!(matches!(format, TilePixelFormat::Rgba16));
        assert_eq!(pixel_data.len(), 3 * 2 * 8);
        let expected = [
            srgb_u16_to_linear_u16(1).to_ne_bytes(),
            srgb_u16_to_linear_u16(2).to_ne_bytes(),
            srgb_u16_to_linear_u16(3).to_ne_bytes(),
            u16::MAX.to_ne_bytes(),
        ]
        .concat();
        assert_eq!(&pixel_data[0..8], expected);
    }

    #[test]
    fn linear_rgba16_fallback_encodes_srgb_and_preserves_alpha() {
        let bytes = [
            srgb_u16_to_linear_u16(0).to_ne_bytes(),
            srgb_u16_to_linear_u16(257).to_ne_bytes(),
            srgb_u16_to_linear_u16(32_768).to_ne_bytes(),
            u16::MAX.to_ne_bytes(),
        ]
        .concat();
        assert_eq!(linear_rgba16_bytes_to_srgba8(&bytes), vec![0, 1, 128, 255]);
    }

    #[test]
    fn supported_image_extensions_include_heif_and_heic() {
        assert!(is_supported_image_path(Path::new("photo.heif")));
        assert!(is_supported_image_path(Path::new("photo.HEIC")));
        assert!(is_heif_path(Path::new("photo.HEIF")));
        assert!(!is_supported_image_path(Path::new("notes.txt")));
    }

    #[test]
    fn thumbnail_caches_are_separate_for_each_output_space() {
        let cache = Path::new("/tmp/sriv-test-cache");
        let image = Path::new("/photos/example.jpg");
        let srgb = thumbnail_cache_path(cache, image, OutputColorSpace::Srgb);
        let display_p3 = thumbnail_cache_path(cache, image, OutputColorSpace::DisplayP3);
        assert_ne!(srgb, display_p3);
        assert_eq!(
            srgb.extension().and_then(|value| value.to_str()),
            Some("png")
        );
        assert_eq!(
            display_p3.extension().and_then(|value| value.to_str()),
            Some("png")
        );
    }

    #[test]
    fn standard_decoder_retains_embedded_icc_profile() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("sriv-icc-test-{}-{unique}.png", std::process::id()));
        let display_p3 = ColorProfile::new_display_p3().encode().unwrap();
        let file = fs::File::create(&path).unwrap();
        let mut encoder = image_rs::codecs::png::PngEncoder::new(file);
        encoder.set_icc_profile(display_p3.clone()).unwrap();
        encoder
            .write_image(&[128, 200, 50], 1, 1, ExtendedColorType::Rgb8)
            .unwrap();

        let decoded = decode_standard_image(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            decoded.embedded_color_profile,
            Some(EmbeddedColorProfile::Icc(ref profile)) if profile == &display_p3
        ));
        assert_eq!(decoded.image.to_rgb8().get_pixel(0, 0).0, [128, 200, 50]);
    }

    #[test]
    fn copy_heif_interleaved_plane_removes_row_padding() {
        let data = [1_u8, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 88, 88];
        let plane = Plane {
            data: &data[..],
            width: 2,
            height: 2,
            stride: 8,
            bits_per_pixel: 24,
            storage_bits_per_pixel: 24,
        };
        let pixels = copy_heif_interleaved_plane(&plane, 3).unwrap();
        assert_eq!(pixels, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
}
