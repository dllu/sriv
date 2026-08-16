use anyhow::{anyhow, Context, Result};
use image::{DynamicImage, ImageBuffer, Pixel};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, Layout, MatrixCoefficients,
    TransferCharacteristics, TransformOptions,
};

/// Standardized output spaces that sriv can advertise to the presentation system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputColorSpace {
    Srgb,
    DisplayP3,
}

impl OutputColorSpace {
    fn profile(self) -> ColorProfile {
        match self {
            Self::Srgb => ColorProfile::new_srgb(),
            Self::DisplayP3 => ColorProfile::new_display_p3(),
        }
    }

    pub const fn cache_tag(self) -> &'static str {
        match self {
            Self::Srgb => "srgb-v1",
            Self::DisplayP3 => "display-p3-v1",
        }
    }
}

/// Color information attached to decoded pixels.
#[derive(Debug)]
pub enum EmbeddedColorProfile {
    /// A complete ICC profile, as embedded in JPEG, PNG, TIFF, WebP, or HEIF.
    Icc(Vec<u8>),
    /// HEIF commonly stores CICP/NCLX identifiers instead of a full ICC profile.
    Cicp {
        color_primaries: u8,
        transfer_characteristics: u8,
    },
}

fn source_profile(profile: &EmbeddedColorProfile) -> Result<ColorProfile> {
    match profile {
        EmbeddedColorProfile::Icc(bytes) => {
            ColorProfile::new_from_slice(bytes).context("failed to parse the embedded ICC profile")
        }
        EmbeddedColorProfile::Cicp {
            color_primaries,
            transfer_characteristics,
        } => {
            let color_primaries = CicpColorPrimaries::try_from(*color_primaries)
                .context("unsupported CICP color primaries")?;
            let transfer_characteristics =
                TransferCharacteristics::try_from(*transfer_characteristics)
                    .context("unsupported CICP transfer characteristics")?;
            if matches!(
                color_primaries,
                CicpColorPrimaries::Reserved | CicpColorPrimaries::Unspecified
            ) || matches!(
                transfer_characteristics,
                TransferCharacteristics::Reserved | TransferCharacteristics::Unspecified
            ) {
                return Err(anyhow!("incomplete CICP color profile"));
            }
            // libheif has already converted YCbCr to RGB. The remaining transform
            // therefore describes RGB code values, irrespective of the source matrix.
            Ok(ColorProfile::new_from_cicp(CicpProfile {
                color_primaries,
                transfer_characteristics,
                matrix_coefficients: MatrixCoefficients::Identity,
                full_range: true,
            }))
        }
    }
}

fn output_len(width: u32, height: u32, channels: usize) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| anyhow!("color-converted image dimensions overflow usize"))
}

fn image_buffer<P>(
    width: u32,
    height: u32,
    pixels: Vec<P::Subpixel>,
) -> Result<ImageBuffer<P, Vec<P::Subpixel>>>
where
    P: Pixel + 'static,
{
    ImageBuffer::from_raw(width, height, pixels)
        .ok_or_else(|| anyhow!("color transform returned the wrong number of samples"))
}

/// Convert decoded pixels into values encoded for the selected output color space.
///
/// RGB buffers are transformed in place to avoid doubling peak memory for very large images.
pub fn convert_to_output(
    image: &mut DynamicImage,
    embedded_profile: Option<&EmbeddedColorProfile>,
    output_color_space: OutputColorSpace,
) -> Result<()> {
    let source = match embedded_profile {
        Some(profile) => source_profile(profile)?,
        None => ColorProfile::new_srgb(),
    };
    let destination = output_color_space.profile();
    convert_between_profiles(image, &source, &destination)
}

/// Convert a display-ready thumbnail back to sRGB for models that define sRGB inputs.
pub fn output_to_srgb(
    image: &DynamicImage,
    output_color_space: OutputColorSpace,
) -> Result<DynamicImage> {
    let mut converted = image.clone();
    if output_color_space != OutputColorSpace::Srgb {
        convert_between_profiles(
            &mut converted,
            &output_color_space.profile(),
            &ColorProfile::new_srgb(),
        )?;
    }
    Ok(converted)
}

fn convert_between_profiles(
    image: &mut DynamicImage,
    source: &ColorProfile,
    destination: &ColorProfile,
) -> Result<()> {
    let options = TransformOptions::default();
    let (width, height) = (image.width(), image.height());

    let replacement = match image {
        DynamicImage::ImageLuma8(buffer) => {
            let transform = source
                .create_transform_8bit(Layout::Gray, destination, Layout::Rgb, options)
                .context("embedded profile does not describe the decoded grayscale pixels")?;
            let mut pixels = vec![0; output_len(width, height, 3)?];
            transform.transform(buffer.as_raw(), &mut pixels)?;
            Some(DynamicImage::ImageRgb8(image_buffer(
                width, height, pixels,
            )?))
        }
        DynamicImage::ImageLumaA8(buffer) => {
            let transform = source
                .create_transform_8bit(Layout::GrayAlpha, destination, Layout::Rgba, options)
                .context("embedded profile does not describe the decoded grayscale pixels")?;
            let mut pixels = vec![0; output_len(width, height, 4)?];
            transform.transform(buffer.as_raw(), &mut pixels)?;
            Some(DynamicImage::ImageRgba8(image_buffer(
                width, height, pixels,
            )?))
        }
        DynamicImage::ImageRgb8(buffer) => {
            let transform = source
                .create_in_place_transform_8bit(Layout::Rgb, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        DynamicImage::ImageRgba8(buffer) => {
            let transform = source
                .create_in_place_transform_8bit(Layout::Rgba, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        DynamicImage::ImageLuma16(buffer) => {
            let transform = source
                .create_transform_16bit(Layout::Gray, destination, Layout::Rgb, options)
                .context("embedded profile does not describe the decoded grayscale pixels")?;
            let mut pixels = vec![0; output_len(width, height, 3)?];
            transform.transform(buffer.as_raw(), &mut pixels)?;
            Some(DynamicImage::ImageRgb16(image_buffer(
                width, height, pixels,
            )?))
        }
        DynamicImage::ImageLumaA16(buffer) => {
            let transform = source
                .create_transform_16bit(Layout::GrayAlpha, destination, Layout::Rgba, options)
                .context("embedded profile does not describe the decoded grayscale pixels")?;
            let mut pixels = vec![0; output_len(width, height, 4)?];
            transform.transform(buffer.as_raw(), &mut pixels)?;
            Some(DynamicImage::ImageRgba16(image_buffer(
                width, height, pixels,
            )?))
        }
        DynamicImage::ImageRgb16(buffer) => {
            let transform = source
                .create_in_place_transform_16bit(Layout::Rgb, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        DynamicImage::ImageRgba16(buffer) => {
            let transform = source
                .create_in_place_transform_16bit(Layout::Rgba, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        DynamicImage::ImageRgb32F(buffer) => {
            let transform = source
                .create_in_place_transform_f32(Layout::Rgb, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        DynamicImage::ImageRgba32F(buffer) => {
            let transform = source
                .create_in_place_transform_f32(Layout::Rgba, destination, options)
                .context("embedded profile does not describe the decoded RGB pixels")?;
            transform.transform(buffer.as_mut())?;
            None
        }
        _ => {
            return Err(anyhow!(
                "unsupported decoded pixel format for color conversion"
            ))
        }
    };
    if let Some(replacement) = replacement {
        *image = replacement;
    }
    Ok(())
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert an sRGB UI color to linear values in the surface's output gamut.
pub fn srgba_to_output_linear(
    [red, green, blue, alpha]: [f32; 4],
    output_color_space: OutputColorSpace,
) -> [f32; 4] {
    let linear_srgb = [
        srgb_to_linear(red),
        srgb_to_linear(green),
        srgb_to_linear(blue),
    ];
    let [red, green, blue] = match output_color_space {
        OutputColorSpace::Srgb => linear_srgb,
        OutputColorSpace::DisplayP3 => {
            // Linear-light sRGB to linear-light Display P3, both with a D65 white point.
            let [red, green, blue] = linear_srgb;
            [
                0.822_461_96 * red + 0.177_538_04 * green,
                0.033_194_2 * red + 0.966_805_8 * green,
                0.017_082_63 * red + 0.072_397_44 * green + 0.910_519_96 * blue,
            ]
        }
    };
    [red, green, blue, alpha]
}

/// Convert an sRGB UI color to encoded values in the surface's output gamut.
pub fn srgba_to_output_encoded(color: [f32; 4], output_color_space: OutputColorSpace) -> [f32; 4] {
    if output_color_space == OutputColorSpace::Srgb {
        return color;
    }
    let [red, green, blue, alpha] = srgba_to_output_linear(color, output_color_space);
    [
        linear_to_srgb(red).clamp(0.0, 1.0),
        linear_to_srgb(green).clamp(0.0, 1.0),
        linear_to_srgb(blue).clamp(0.0, 1.0),
        alpha,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgba};

    #[test]
    fn display_p3_is_converted_to_srgb_without_changing_alpha() {
        let source = ColorProfile::new_display_p3();
        let profile = EmbeddedColorProfile::Icc(source.encode().unwrap());
        let image =
            DynamicImage::ImageRgba8(ImageBuffer::from_pixel(1, 1, Rgba([128, 200, 50, 77])));

        let mut converted = image;
        convert_to_output(&mut converted, Some(&profile), OutputColorSpace::Srgb).unwrap();
        let converted = converted.to_rgba8();
        let pixel = converted.get_pixel(0, 0).0;
        assert_ne!(&pixel[..3], &[128, 200, 50]);
        assert_eq!(pixel[3], 77);
    }

    #[test]
    fn heif_cicp_display_p3_matches_an_icc_transform() {
        let image = DynamicImage::ImageRgba16(ImageBuffer::from_pixel(
            1,
            1,
            Rgba([32_768, 48_000, 12_000, 42_000]),
        ));
        let icc = EmbeddedColorProfile::Icc(ColorProfile::new_display_p3().encode().unwrap());
        // H.273: SMPTE EG 432-1 primaries (12), IEC 61966-2-1 transfer (13).
        let cicp = EmbeddedColorProfile::Cicp {
            color_primaries: 12,
            transfer_characteristics: 13,
        };

        let mut from_icc = image.clone();
        convert_to_output(&mut from_icc, Some(&icc), OutputColorSpace::Srgb).unwrap();
        let from_icc = from_icc.to_rgba16();
        let mut from_cicp = image;
        convert_to_output(&mut from_cicp, Some(&cicp), OutputColorSpace::Srgb).unwrap();
        let from_cicp = from_cicp.to_rgba16();
        for (actual, expected) in from_cicp
            .get_pixel(0, 0)
            .0
            .iter()
            .zip(from_icc.get_pixel(0, 0).0)
        {
            // Serializing the generated ICC profile quantizes its matrix/TRC tags slightly.
            assert!(actual.abs_diff(expected) <= 16, "{actual} != {expected}");
        }
    }

    #[test]
    fn output_gamut_conversion_keeps_neutral_ui_colors_neutral() {
        let converted = srgba_to_output_encoded([0.5, 0.5, 0.5, 0.25], OutputColorSpace::DisplayP3);
        for channel in &converted[..3] {
            assert!((*channel - 0.5).abs() < 1e-5);
        }
        assert_eq!(converted[3], 0.25);
    }
}
