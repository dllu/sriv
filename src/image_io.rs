use anyhow::{anyhow, Result};
use image as image_rs;
use image::codecs::{gif::GifDecoder, png::PngDecoder, webp::WebPDecoder};
use image::{
    imageops::FilterType, AnimationDecoder, DynamicImage, Frames, ImageDecoder, ImageFormat,
    ImageReader, RgbImage, RgbaImage,
};
use jxl_oxide::{EnumColourEncoding, JxlImage, Render, RenderingIntent};
#[cfg(not(target_os = "macos"))]
use libheif_rs::{ColorSpace as HeifColorSpace, HeifContext, LibHeif, Plane, RgbChroma};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crate::clip;
use crate::color::{self, EmbeddedColorProfile, OutputColorSpace};
use crate::state::{FullImageFrame, FullImageTile, TilePixelFormat};

#[cfg(target_os = "macos")]
mod macos_heif;

struct DecodedImage {
    image: DynamicImage,
    embedded_color_profile: Option<EmbeddedColorProfile>,
}

const FORMAT_DETECTION_BYTES: u64 = 4096;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DetectedImageFormat {
    Standard(ImageFormat),
    Heif,
    Jxl,
}

const JXL_CODESTREAM_SIGNATURE: &[u8] = b"\xff\x0a";
const JXL_CONTAINER_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a";

/// List of recognized raw file extensions for detecting XMP sidecars.
const RAW_EXTENSIONS: &[&str] = &[
    "3fr", "ari", "arw", "bay", "cap", "cr2", "cr3", "crw", "cs1", "dcr", "dng", "erf", "fff",
    "iiq", "k25", "kdc", "mdc", "mef", "mos", "mrw", "nef", "nrw", "orf", "pef", "ptx", "pxn",
    "raf", "raw", "rwl", "rw2", "rwz", "sr2", "srf", "srw", "x3f",
];

fn is_heif_brand(brand: &[u8]) -> bool {
    matches!(
        brand,
        b"heic"
            | b"heix"
            | b"hevc"
            | b"hevx"
            | b"heim"
            | b"heis"
            | b"hevm"
            | b"hevs"
            | b"mif1"
            | b"mif2"
            | b"msf1"
    )
}

fn has_heif_file_type_box(bytes: &[u8]) -> bool {
    if bytes.get(4..8) != Some(b"ftyp") {
        return false;
    }

    let Some(size_bytes) = bytes.get(0..4) else {
        return false;
    };
    let size = u32::from_be_bytes(size_bytes.try_into().unwrap());
    let (payload_start, declared_end) = if size == 1 {
        let Some(large_size) = bytes.get(8..16) else {
            return false;
        };
        let size = u64::from_be_bytes(large_size.try_into().unwrap());
        (16, usize::try_from(size).unwrap_or(usize::MAX))
    } else {
        (8, size as usize)
    };
    let box_end = if declared_end == 0 {
        bytes.len()
    } else {
        declared_end.min(bytes.len())
    };
    if box_end < payload_start + 8 {
        return false;
    }

    let major_brand = &bytes[payload_start..payload_start + 4];
    is_heif_brand(major_brand)
        || bytes[payload_start + 8..box_end]
            .chunks_exact(4)
            .any(is_heif_brand)
}

fn detect_supported_image_format_bytes(bytes: &[u8]) -> Option<DetectedImageFormat> {
    if bytes.starts_with(JXL_CODESTREAM_SIGNATURE) || bytes.starts_with(JXL_CONTAINER_SIGNATURE) {
        return Some(DetectedImageFormat::Jxl);
    }
    if has_heif_file_type_box(bytes) {
        return Some(DetectedImageFormat::Heif);
    }

    let format = image_rs::guess_format(bytes).ok()?;
    matches!(
        format,
        ImageFormat::Jpeg
            | ImageFormat::Png
            | ImageFormat::Bmp
            | ImageFormat::Gif
            | ImageFormat::Tiff
            | ImageFormat::WebP
    )
    .then_some(DetectedImageFormat::Standard(format))
}

fn detect_supported_image_format(path: &Path) -> Result<DetectedImageFormat> {
    let mut header = Vec::with_capacity(FORMAT_DETECTION_BYTES as usize);
    fs::File::open(path)?
        .take(FORMAT_DETECTION_BYTES)
        .read_to_end(&mut header)?;
    detect_supported_image_format_bytes(&header)
        .ok_or_else(|| anyhow!("unrecognized or unsupported image contents"))
}

pub(crate) fn is_supported_image_path(path: &Path) -> bool {
    detect_supported_image_format(path).is_ok()
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

fn orientation_code(path: &Path) -> Option<u16> {
    parse_exif_quiet(path)?
        .entries
        .into_iter()
        .find_map(|entry| {
            (entry.tag == rexif::ExifTag::Orientation)
                .then(|| orientation_from_tag_value(&entry.value))
                .flatten()
        })
}

fn apply_orientation(img: DynamicImage, orientation: Option<u16>) -> DynamicImage {
    match orientation {
        Some(2) => img.fliph(),
        Some(3) => img.rotate180(),
        Some(4) => img.flipv(),
        Some(5) => img.rotate90().fliph(),
        Some(6) => img.rotate90(),
        Some(7) => img.rotate270().fliph(),
        Some(8) => img.rotate270(),
        _ => img,
    }
}

fn adjust_orientation_full(img: DynamicImage, path: &Path) -> DynamicImage {
    apply_orientation(img, orientation_code(path))
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

#[cfg(not(target_os = "macos"))]
fn libheif() -> &'static LibHeif {
    static LIBHEIF: OnceLock<LibHeif> = OnceLock::new();
    LIBHEIF.get_or_init(LibHeif::new)
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(not(target_os = "macos"))]
fn heif_bytes_per_pixel(has_alpha: bool, high_bit_depth: bool) -> usize {
    let channels = if has_alpha { 4 } else { 3 };
    let bytes_per_channel = if high_bit_depth { 2 } else { 1 };
    channels * bytes_per_channel
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(any(not(target_os = "macos"), test))]
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
        .ok_or_else(|| anyhow!("decoded {format_name} data has the wrong length"))
}

#[cfg(not(target_os = "macos"))]
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

#[cfg(target_os = "macos")]
use macos_heif::decode_heif_image;

fn decode_standard_image(path: &Path, format: ImageFormat) -> Result<DecodedImage> {
    let mut reader = ImageReader::open(path)?;
    reader.set_format(format);
    reader.no_limits();
    let mut decoder = reader.into_decoder()?;
    let embedded_color_profile = read_embedded_color_profile(path, &mut decoder);
    Ok(DecodedImage {
        image: DynamicImage::from_decoder(decoder)?,
        embedded_color_profile,
    })
}

fn open_jxl_image(path: &Path, output_color_space: OutputColorSpace) -> Result<JxlImage> {
    let mut image = JxlImage::builder()
        .open(path)
        .map_err(|error| anyhow!("failed to decode JPEG XL image: {error}"))?;
    let rendering_intent = RenderingIntent::Relative;
    let encoding = if image.image_header().metadata.grayscale() {
        EnumColourEncoding::gray_srgb(rendering_intent)
    } else {
        match output_color_space {
            OutputColorSpace::Srgb => EnumColourEncoding::srgb(rendering_intent),
            OutputColorSpace::DisplayP3 => EnumColourEncoding::display_p3(rendering_intent),
        }
    };
    image.request_color_encoding(encoding);
    Ok(image)
}

fn jxl_needs_16_bit(image: &JxlImage) -> bool {
    let metadata = &image.image_header().metadata;
    metadata.bit_depth.bits_per_sample() > 8
        || metadata
            .ec_info
            .iter()
            .any(|channel| channel.bit_depth.bits_per_sample() > 8)
}

fn jxl_has_associated_alpha(image: &JxlImage) -> bool {
    image
        .image_header()
        .metadata
        .ec_info
        .iter()
        .find_map(|channel| channel.alpha_associated())
        .unwrap_or(false)
}

fn unpremultiply_u8(samples: &mut [u8], channels: usize) {
    for pixel in samples.chunks_exact_mut(channels) {
        let alpha = u16::from(pixel[channels - 1]);
        if alpha == 0 {
            continue;
        }
        for sample in &mut pixel[..channels - 1] {
            *sample = ((u16::from(*sample) * u16::from(u8::MAX) + alpha / 2) / alpha)
                .min(u16::from(u8::MAX)) as u8;
        }
    }
}

fn unpremultiply_u16(samples: &mut [u16], channels: usize) {
    for pixel in samples.chunks_exact_mut(channels) {
        let alpha = u32::from(pixel[channels - 1]);
        if alpha == 0 {
            continue;
        }
        for sample in &mut pixel[..channels - 1] {
            *sample = ((u32::from(*sample) * u32::from(u16::MAX) + alpha / 2) / alpha)
                .min(u32::from(u16::MAX)) as u16;
        }
    }
}

fn jxl_render_to_dynamic_image(
    render: &Render,
    high_bit_depth: bool,
    associated_alpha: bool,
) -> Result<DynamicImage> {
    let mut stream = render.stream();
    let width = stream.width();
    let height = stream.height();
    let channels = stream.channels() as usize;
    if !(1..=4).contains(&channels) {
        return Err(anyhow!(
            "unsupported decoded JPEG XL pixel format with {channels} channels"
        ));
    }
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| anyhow!("decoded JPEG XL dimensions overflow usize"))?;

    if high_bit_depth {
        let mut samples = vec![0_u16; sample_count];
        if stream.write_to_buffer(&mut samples) != sample_count {
            return Err(anyhow!("JPEG XL decoder returned too few pixel samples"));
        }
        if associated_alpha && matches!(channels, 2 | 4) {
            unpremultiply_u16(&mut samples, channels);
        }
        return Ok(match channels {
            1 => DynamicImage::ImageLuma16(image_buffer_from_raw(
                width,
                height,
                samples,
                "JPEG XL L16",
            )?),
            2 => DynamicImage::ImageLumaA16(image_buffer_from_raw(
                width,
                height,
                samples,
                "JPEG XL LA16",
            )?),
            3 => DynamicImage::ImageRgb16(image_buffer_from_raw(
                width,
                height,
                samples,
                "JPEG XL RGB16",
            )?),
            4 => DynamicImage::ImageRgba16(image_buffer_from_raw(
                width,
                height,
                samples,
                "JPEG XL RGBA16",
            )?),
            _ => unreachable!(),
        });
    }

    let mut samples = vec![0_u8; sample_count];
    if stream.write_to_buffer(&mut samples) != sample_count {
        return Err(anyhow!("JPEG XL decoder returned too few pixel samples"));
    }
    if associated_alpha && matches!(channels, 2 | 4) {
        unpremultiply_u8(&mut samples, channels);
    }
    Ok(match channels {
        1 => DynamicImage::ImageLuma8(image_buffer_from_raw(width, height, samples, "JPEG XL L8")?),
        2 => DynamicImage::ImageLumaA8(image_buffer_from_raw(
            width,
            height,
            samples,
            "JPEG XL LA8",
        )?),
        3 => DynamicImage::ImageRgb8(image_buffer_from_raw(
            width,
            height,
            samples,
            "JPEG XL RGB8",
        )?),
        4 => DynamicImage::ImageRgba8(image_buffer_from_raw(
            width,
            height,
            samples,
            "JPEG XL RGBA8",
        )?),
        _ => unreachable!(),
    })
}

fn decode_jxl_first_frame(
    path: &Path,
    output_color_space: OutputColorSpace,
) -> Result<DynamicImage> {
    let image = open_jxl_image(path, output_color_space)?;
    if image.num_loaded_keyframes() == 0 {
        return Err(anyhow!("JPEG XL image contains no displayable frames"));
    }
    let high_bit_depth = jxl_needs_16_bit(&image);
    let associated_alpha = jxl_has_associated_alpha(&image);
    let render = image
        .render_frame(0)
        .map_err(|error| anyhow!("failed to render JPEG XL frame: {error}"))?;
    jxl_render_to_dynamic_image(&render, high_bit_depth, associated_alpha)
}

fn read_embedded_color_profile(
    path: &Path,
    decoder: &mut impl ImageDecoder,
) -> Option<EmbeddedColorProfile> {
    match decoder.icc_profile() {
        Ok(profile) => profile.map(EmbeddedColorProfile::Icc),
        Err(error) => {
            eprintln!(
                "Failed to read embedded color profile from {}: {error}",
                path.display()
            );
            None
        }
    }
}

fn convert_decoded_to_output(
    path: &Path,
    decoded: DecodedImage,
    output_color_space: OutputColorSpace,
) -> DynamicImage {
    let DecodedImage {
        image,
        embedded_color_profile,
    } = decoded;
    convert_image_to_output(
        path,
        image,
        embedded_color_profile.as_ref(),
        output_color_space,
    )
}

fn convert_image_to_output(
    path: &Path,
    mut image: DynamicImage,
    embedded_color_profile: Option<&EmbeddedColorProfile>,
    output_color_space: OutputColorSpace,
) -> DynamicImage {
    if embedded_color_profile.is_none() && output_color_space == OutputColorSpace::Srgb {
        return image;
    }
    match color::convert_to_output(&mut image, embedded_color_profile, output_color_space) {
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
    let format = detect_supported_image_format(path)?;
    decode_image_no_limits_as(path, output_color_space, format)
}

fn decode_image_no_limits_as(
    path: &Path,
    output_color_space: OutputColorSpace,
    format: DetectedImageFormat,
) -> Result<image_rs::DynamicImage> {
    let decoded = match format {
        DetectedImageFormat::Heif => decode_heif_image(path)?,
        DetectedImageFormat::Standard(format) => decode_standard_image(path, format)?,
        DetectedImageFormat::Jxl => return decode_jxl_first_frame(path, output_color_space),
    };
    let image = convert_decoded_to_output(path, decoded, output_color_space);
    // libheif applies HEIF orientation transformations during decoding.
    Ok(if format == DetectedImageFormat::Heif {
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

fn tile_dynamic_image(img: DynamicImage) -> (u32, u32, Vec<FullImageTile>) {
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
    (full_w, full_h, tiles)
}

struct DecodedAnimation {
    dimensions: (u32, u32),
    embedded_color_profile: Option<EmbeddedColorProfile>,
    frames: Frames<'static>,
}

fn animation_from_decoder<D>(path: &Path, mut decoder: D) -> DecodedAnimation
where
    D: AnimationDecoder<'static> + ImageDecoder + 'static,
{
    let dimensions = decoder.dimensions();
    let embedded_color_profile = read_embedded_color_profile(path, &mut decoder);
    DecodedAnimation {
        dimensions,
        embedded_color_profile,
        frames: decoder.into_frames(),
    }
}

fn open_animation(path: &Path, format: DetectedImageFormat) -> Result<Option<DecodedAnimation>> {
    let reader = || -> Result<_> { Ok(BufReader::new(fs::File::open(path)?)) };
    match format {
        DetectedImageFormat::Standard(ImageFormat::Gif) => Ok(Some(animation_from_decoder(
            path,
            GifDecoder::new(reader()?)?,
        ))),
        DetectedImageFormat::Standard(ImageFormat::WebP) => {
            let decoder = WebPDecoder::new(reader()?)?;
            if decoder.has_animation() {
                Ok(Some(animation_from_decoder(path, decoder)))
            } else {
                Ok(None)
            }
        }
        DetectedImageFormat::Standard(ImageFormat::Png) => {
            let mut decoder = PngDecoder::new(reader()?)?;
            if !decoder.is_apng()? {
                return Ok(None);
            }
            let dimensions = decoder.dimensions();
            let embedded_color_profile = read_embedded_color_profile(path, &mut decoder);
            Ok(Some(DecodedAnimation {
                dimensions,
                embedded_color_profile,
                frames: decoder.apng()?.into_frames(),
            }))
        }
        _ => Ok(None),
    }
}

fn load_frame_sequence<I, F>(
    expected_dimensions: (u32, u32),
    decoded_frames: I,
    mut prepare_image: F,
) -> Result<(u32, u32, Vec<FullImageFrame>)>
where
    I: IntoIterator<Item = Result<(DynamicImage, Duration)>>,
    F: FnMut(DynamicImage) -> DynamicImage,
{
    let mut full_dimensions = None;
    let mut frames = Vec::new();
    for frame in decoded_frames {
        let (image, delay) = frame?;
        if (image.width(), image.height()) != expected_dimensions {
            return Err(anyhow!(
                "decoded frame is {}x{}, expected {}x{}",
                image.width(),
                image.height(),
                expected_dimensions.0,
                expected_dimensions.1
            ));
        }
        let image = prepare_image(image);
        let (frame_w, frame_h, tiles) = tile_dynamic_image(image);
        let dimensions = (frame_w, frame_h);
        if full_dimensions.is_some_and(|expected| expected != dimensions) {
            return Err(anyhow!("decoded frames have inconsistent dimensions"));
        }
        full_dimensions = Some(dimensions);
        frames.push(FullImageFrame { delay, tiles });
    }

    let (full_w, full_h) =
        full_dimensions.ok_or_else(|| anyhow!("image contains no animation frames"))?;
    Ok((full_w, full_h, frames))
}

fn load_animated_image(
    path: &Path,
    output_color_space: OutputColorSpace,
    animation: DecodedAnimation,
) -> Result<(u32, u32, Vec<FullImageFrame>)> {
    let DecodedAnimation {
        dimensions: expected_dimensions,
        embedded_color_profile,
        frames: decoded_frames,
    } = animation;
    let orientation = orientation_code(path);
    let frames = decoded_frames.map(|frame| -> Result<_> {
        let frame = frame?;
        let delay = Duration::from(frame.delay());
        Ok((DynamicImage::ImageRgba8(frame.into_buffer()), delay))
    });
    load_frame_sequence(expected_dimensions, frames, |image| {
        let image = convert_image_to_output(
            path,
            image,
            embedded_color_profile.as_ref(),
            output_color_space,
        );
        apply_orientation(image, orientation)
    })
}

fn jxl_frame_delay(ticks: u32, animation: Option<&jxl_oxide::image::AnimationHeader>) -> Duration {
    let Some(animation) = animation else {
        return Duration::ZERO;
    };
    if animation.tps_numerator == 0 {
        return Duration::ZERO;
    }
    let nanos = u128::from(ticks) * u128::from(animation.tps_denominator) * 1_000_000_000_u128
        / u128::from(animation.tps_numerator);
    Duration::new(
        (nanos / 1_000_000_000) as u64,
        (nanos % 1_000_000_000) as u32,
    )
}

fn load_jxl_image_frames(
    path: &Path,
    output_color_space: OutputColorSpace,
) -> Result<(u32, u32, Vec<FullImageFrame>)> {
    let image = open_jxl_image(path, output_color_space)?;
    let frame_count = image.num_loaded_keyframes();
    if frame_count == 0 {
        return Err(anyhow!("JPEG XL image contains no displayable frames"));
    }
    let dimensions = (image.width(), image.height());
    let high_bit_depth = jxl_needs_16_bit(&image);
    let associated_alpha = jxl_has_associated_alpha(&image);
    let animation = image.image_header().metadata.animation.as_ref();
    let frames = (0..frame_count).map(|frame_index| -> Result<_> {
        let render = image
            .render_frame(frame_index)
            .map_err(|error| anyhow!("failed to render JPEG XL frame: {error}"))?;
        let delay = jxl_frame_delay(render.duration(), animation);
        let image = jxl_render_to_dynamic_image(&render, high_bit_depth, associated_alpha)?;
        Ok((image, delay))
    });
    load_frame_sequence(dimensions, frames, std::convert::identity)
}

pub(crate) fn load_full_image_tiles(
    path: &Path,
    output_color_space: OutputColorSpace,
) -> Result<(u32, u32, Vec<FullImageFrame>)> {
    let format = detect_supported_image_format(path)?;
    if format == DetectedImageFormat::Jxl {
        return load_jxl_image_frames(path, output_color_space);
    }
    if let Some(animation) = open_animation(path, format)? {
        return load_animated_image(path, output_color_space, animation);
    }

    let img = decode_image_no_limits_as(path, output_color_space, format)?;
    let (full_w, full_h, tiles) = tile_dynamic_image(img);
    Ok((
        full_w,
        full_h,
        vec![FullImageFrame {
            delay: Duration::ZERO,
            tiles,
        }],
    ))
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
    use image_rs::codecs::gif::{GifEncoder, Repeat};
    use image_rs::codecs::webp::WebPEncoder;
    use image_rs::{Delay, ExtendedColorType, Frame, ImageBuffer, ImageEncoder, Rgb, Rgba};
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
    fn supported_formats_are_detected_from_magic_bytes() {
        assert_eq!(
            detect_supported_image_format_bytes(b"\x89PNG\r\n\x1a\nrest"),
            Some(DetectedImageFormat::Standard(ImageFormat::Png))
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"GIF89arest"),
            Some(DetectedImageFormat::Standard(ImageFormat::Gif))
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"\xff\xd8\xffrest"),
            Some(DetectedImageFormat::Standard(ImageFormat::Jpeg))
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"\0\0\0\x10ftypheic\0\0\0\0"),
            Some(DetectedImageFormat::Heif)
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"\0\0\0\x14ftypisom\0\0\0\0heix"),
            Some(DetectedImageFormat::Heif)
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"\xff\x0aJPEG XL codestream"),
            Some(DetectedImageFormat::Jxl)
        );
        assert_eq!(
            detect_supported_image_format_bytes(b"\0\0\0\x0cJXL \r\n\x87\nrest"),
            Some(DetectedImageFormat::Jxl)
        );
        assert_eq!(detect_supported_image_format_bytes(b"plain text"), None);
    }

    #[test]
    fn supported_file_detection_ignores_the_extension() {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let image_path = std::env::temp_dir().join(format!(
            "sriv-extensionless-image-test-{}-{unique}",
            std::process::id()
        ));
        let fake_path = image_path.with_extension("jpg");
        image_rs::codecs::png::PngEncoder::new(fs::File::create(&image_path).unwrap())
            .write_image(&[10, 20, 30, 255], 1, 1, ExtendedColorType::Rgba8)
            .unwrap();
        fs::write(&fake_path, b"not really a JPEG").unwrap();

        assert!(is_supported_image_path(&image_path));
        assert!(!is_supported_image_path(&fake_path));
        let (width, height, frames) =
            load_full_image_tiles(&image_path, OutputColorSpace::Srgb).unwrap();

        fs::remove_file(&image_path).unwrap();
        fs::remove_file(&fake_path).unwrap();
        assert_eq!((width, height), (1, 1));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].delay, Duration::ZERO);
        assert_eq!(frames[0].tiles[0].5, [10, 20, 30, 255]);
    }

    const RED_FRAME: [u8; 8] = [255, 0, 0, 255, 255, 0, 0, 255];
    const BLUE_FRAME: [u8; 8] = [0, 0, 255, 255, 0, 0, 255, 255];

    fn animation_test_path(format: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sriv-animated-{format}-test-{}-{unique}.wrong-extension",
            std::process::id(),
        ))
    }

    fn assert_test_animation(path: &Path) {
        let (width, height, frames) = load_full_image_tiles(path, OutputColorSpace::Srgb).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!((width, height), (2, 1));
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].delay, Duration::from_millis(40));
        assert_eq!(frames[1].delay, Duration::from_millis(70));
        assert_eq!(frames[0].tiles.len(), 1);
        assert_eq!(frames[1].tiles.len(), 1);
        assert_eq!(frames[0].tiles[0].5, RED_FRAME);
        assert_eq!(frames[1].tiles[0].5, BLUE_FRAME);
    }

    #[test]
    fn full_gif_loader_uses_shared_animation_path() {
        let path = animation_test_path("gif");
        {
            let file = fs::File::create(&path).unwrap();
            let mut encoder = GifEncoder::new(file);
            encoder.set_repeat(Repeat::Infinite).unwrap();
            encoder
                .encode_frame(Frame::from_parts(
                    RgbaImage::from_pixel(2, 1, Rgba([255, 0, 0, 255])),
                    0,
                    0,
                    Delay::from_numer_denom_ms(40, 1),
                ))
                .unwrap();
            encoder
                .encode_frame(Frame::from_parts(
                    RgbaImage::from_pixel(2, 1, Rgba([0, 0, 255, 255])),
                    0,
                    0,
                    Delay::from_numer_denom_ms(70, 1),
                ))
                .unwrap();
        }

        assert_test_animation(&path);
    }

    #[test]
    fn full_apng_loader_uses_shared_animation_path() {
        let path = animation_test_path("apng");
        {
            let mut encoder = png::Encoder::new(fs::File::create(&path).unwrap(), 2, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.set_animated(2, 0).unwrap();
            encoder.set_frame_delay(4, 100).unwrap();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&RED_FRAME).unwrap();
            writer.set_frame_delay(7, 100).unwrap();
            writer.write_image_data(&BLUE_FRAME).unwrap();
            writer.finish().unwrap();
        }

        assert_test_animation(&path);
    }

    fn push_webp_chunk(output: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
        output.extend_from_slice(fourcc);
        output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        output.extend_from_slice(payload);
        if !payload.len().is_multiple_of(2) {
            output.push(0);
        }
    }

    fn push_le_u24(output: &mut Vec<u8>, value: u32) {
        output.extend_from_slice(&value.to_le_bytes()[..3]);
    }

    fn lossless_webp_chunk(rgb: &[u8]) -> Vec<u8> {
        let mut encoded = Vec::new();
        WebPEncoder::new_lossless(&mut encoded)
            .write_image(rgb, 2, 1, ExtendedColorType::Rgb8)
            .unwrap();
        assert_eq!(&encoded[..4], b"RIFF");
        assert_eq!(&encoded[8..12], b"WEBP");
        encoded[12..].to_vec()
    }

    fn push_webp_animation_frame(output: &mut Vec<u8>, rgb: &[u8], delay_ms: u32) {
        let mut payload = Vec::new();
        push_le_u24(&mut payload, 0); // x / 2
        push_le_u24(&mut payload, 0); // y / 2
        push_le_u24(&mut payload, 1); // width - 1
        push_le_u24(&mut payload, 0); // height - 1
        push_le_u24(&mut payload, delay_ms);
        payload.push(0b10); // replace the canvas; do not dispose
        payload.extend_from_slice(&lossless_webp_chunk(rgb));
        push_webp_chunk(output, b"ANMF", &payload);
    }

    #[test]
    fn full_animated_webp_loader_uses_shared_animation_path() {
        let path = animation_test_path("webp");
        let mut chunks = Vec::new();
        push_webp_chunk(&mut chunks, b"VP8X", &[0b10, 0, 0, 0, 1, 0, 0, 0, 0, 0]);
        push_webp_chunk(&mut chunks, b"ANIM", &[0; 6]);
        push_webp_animation_frame(&mut chunks, &[255, 0, 0, 255, 0, 0], 40);
        push_webp_animation_frame(&mut chunks, &[0, 0, 255, 0, 0, 255], 70);

        let mut encoded = b"RIFF\0\0\0\0WEBP".to_vec();
        encoded.extend_from_slice(&chunks);
        let riff_size = (encoded.len() - 8) as u32;
        encoded[4..8].copy_from_slice(&riff_size.to_le_bytes());
        fs::write(&path, encoded).unwrap();

        assert_test_animation(&path);
    }

    fn decode_hex_fixture(encoded: &str) -> Vec<u8> {
        assert!(encoded.len().is_multiple_of(2));
        encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|digits| u8::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
            .collect()
    }

    fn wrap_jxl_container(codestream: &[u8]) -> Vec<u8> {
        let mut container = JXL_CONTAINER_SIGNATURE.to_vec();
        container.extend_from_slice(&20_u32.to_be_bytes());
        container.extend_from_slice(b"ftypjxl \0\0\0\0jxl ");
        container.extend_from_slice(&(codestream.len() as u32 + 8).to_be_bytes());
        container.extend_from_slice(b"jxlc");
        container.extend_from_slice(codestream);
        container
    }

    #[test]
    fn full_animated_jxl_loader_uses_shared_animation_path() {
        // Lossless 2x1 RGBA animation generated with the libjxl reference encoder.
        const ANIMATED_JXL: &str =
            "ff0a00704100d60408082001000034004b188b15c249411e4084fefa030808e041000034004b188b15c249411e40a43ffa03";
        let path = animation_test_path("jxl");
        fs::write(&path, decode_hex_fixture(ANIMATED_JXL)).unwrap();

        assert_test_animation(&path);
    }

    #[test]
    fn non_animated_jxl_has_one_zero_delay_frame() {
        const STILL_JXL: &str =
            "ff0a305410090806010078004b38413cb63a51fe00471ea085b8271a4845841b714fa83e8e3003928401";
        let path = animation_test_path("still-jxl-container");
        let codestream = decode_hex_fixture(STILL_JXL);
        fs::write(&path, wrap_jxl_container(&codestream)).unwrap();

        let still = decode_image_no_limits(&path, OutputColorSpace::Srgb).unwrap();
        assert_eq!((still.width(), still.height()), (240, 135));
        assert_eq!(still.to_rgba8().get_pixel(0, 0).0, [6, 6, 6, 255]);

        let (width, height, frames) = load_full_image_tiles(&path, OutputColorSpace::Srgb).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!((width, height), (240, 135));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].delay, Duration::ZERO);
        assert_eq!(&frames[0].tiles[0].5[..4], &[6, 6, 6, 255]);
    }

    #[test]
    fn jxl_preserves_high_bit_depth_pixels() {
        // Lossless 1x1 RGBA16 image generated with the libjxl reference encoder.
        const RGBA16_JXL: &str =
            "0000000c4a584c200d0a870a00000014667479706a786c20000000006a786c20000000096a786c6c0a000000276a786c63ff0a0010fc087e80040808100040004b188b15428a8c021cc069fcff01e200";
        let path = animation_test_path("rgba16-jxl");
        fs::write(&path, decode_hex_fixture(RGBA16_JXL)).unwrap();

        let (width, height, frames) = load_full_image_tiles(&path, OutputColorSpace::Srgb).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!((width, height), (1, 1));
        assert_eq!(frames.len(), 1);
        let (_, _, _, _, format, pixels) = &frames[0].tiles[0];
        assert_eq!(*format, TilePixelFormat::Rgba16);
        assert_eq!(
            native_u16_samples(pixels),
            [
                srgb_u16_to_linear_u16(0),
                srgb_u16_to_linear_u16(32_768),
                srgb_u16_to_linear_u16(65_535),
                40_000,
            ]
        );
    }

    #[test]
    fn non_animated_webp_stays_on_the_still_image_path() {
        let path = animation_test_path("still-webp");
        WebPEncoder::new_lossless(fs::File::create(&path).unwrap())
            .write_image(&RED_FRAME, 2, 1, ExtendedColorType::Rgba8)
            .unwrap();

        let (width, height, frames) = load_full_image_tiles(&path, OutputColorSpace::Srgb).unwrap();
        fs::remove_file(&path).unwrap();

        assert_eq!((width, height), (2, 1));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].delay, Duration::ZERO);
        assert_eq!(frames[0].tiles[0].5, RED_FRAME);
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

        let decoded = decode_standard_image(&path, ImageFormat::Png).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            decoded.embedded_color_profile,
            Some(EmbeddedColorProfile::Icc(ref profile)) if profile == &display_p3
        ));
        assert_eq!(decoded.image.to_rgb8().get_pixel(0, 0).0, [128, 200, 50]);
    }

    #[cfg(not(target_os = "macos"))]
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
