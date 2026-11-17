//! The pixels Mirage looks at.
//!
//! ## Overview
//!
//! The engine takes one layout and nothing else: packed 8-bit RGB, three bytes per pixel, rows
//! top to bottom, with no padding between rows. [`Image`] is that buffer with its width and
//! height, checked once so that a mismatch is an [`Error::InvalidImage`] here and never a read
//! past the end of the buffer inside the engine.
//!
//! Any size is accepted. The engine fits the image onto its own 512-pixel canvas, and reports
//! every position on a 0 to 1000 scale relative to the image it was given, so the result does not
//! depend on the canvas.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::mirage::Image;
//!
//! // A 2 x 1 image: one red pixel, one blue.
//! let pixels = [255, 0, 0, 0, 0, 255];
//! let image = Image::from_rgb(&pixels, 2, 1)?;
//! assert_eq!((image.width(), image.height()), (2, 1));
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! With the `image` feature, [`Image::open`] and [`Image::decode`] read PNG and JPEG files and
//! apply the orientation a camera recorded:
//!
//! ```no_run
//! # #[cfg(feature = "image")]
//! # fn main() -> Result<(), cactus_rs::Error> {
//! use cactus_rs::mirage::Image;
//!
//! let image = Image::open("photo.jpg")?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "image"))]
//! # fn main() {}
//! ```

use std::borrow::Cow;
use std::fmt;

use crate::error::{Error, Result};
use crate::ffi::Pixels;

/// A packed RGB image whose buffer is known to match its dimensions.
///
/// Borrowed from your own buffer by [`Image::from_rgb`], or owning its pixels after
/// [`Image::from_rgb_vec`], [`Image::from_rgba`] or, behind the `image` feature, [`Image::open`]
/// and [`Image::decode`].
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Image;
///
/// let image = Image::from_rgb_vec(vec![0; 4 * 3 * 3], 4, 3)?;
/// assert_eq!(image.as_rgb().len(), 36);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Image<'a> {
    rgb: Cow<'a, [u8]>,
    width: u32,
    height: u32,
}

impl<'a> Image<'a> {
    /// Borrows packed RGB pixels: `width * height * 3` bytes, rows top to bottom.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// let pixels = vec![128_u8; 640 * 480 * 3];
    /// let image = Image::from_rgb(&pixels, 640, 480)?;
    /// assert_eq!(image.width(), 640);
    ///
    /// // One byte short is refused here, not read past in the engine.
    /// assert!(Image::from_rgb(&pixels[1..], 640, 480).is_err());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImage`] when a dimension is zero, when the buffer is not exactly
    /// `width * height * 3` bytes, or when that count exceeds what the C API can address
    /// (2 147 483 647 bytes).
    pub fn from_rgb(rgb: &'a [u8], width: u32, height: u32) -> Result<Self> {
        Pixels::new(rgb, width, height)?;
        Ok(Image {
            rgb: Cow::Borrowed(rgb),
            width,
            height,
        })
    }

    /// Takes ownership of packed RGB pixels, for an image that must outlive its source.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// fn blank(width: u32, height: u32) -> cactus_rs::Result<Image<'static>> {
    ///     Image::from_rgb_vec(vec![255; (width * height * 3) as usize], width, height)
    /// }
    ///
    /// assert_eq!(blank(8, 8)?.height(), 8);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImage`] on the same conditions as [`Image::from_rgb`].
    pub fn from_rgb_vec(rgb: Vec<u8>, width: u32, height: u32) -> Result<Image<'static>> {
        Pixels::new(&rgb, width, height)?;
        Ok(Image {
            rgb: Cow::Owned(rgb),
            width,
            height,
        })
    }

    /// Converts packed RGBA pixels, four bytes per pixel, by dropping the alpha channel.
    ///
    /// Transparency is not blended against anything: a fully transparent pixel keeps whatever
    /// colour its other three bytes hold.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// let image = Image::from_rgba(&[10, 20, 30, 255, 40, 50, 60, 0], 2, 1)?;
    /// assert_eq!(image.as_rgb(), [10, 20, 30, 40, 50, 60]);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImage`] when a dimension is zero or the buffer is not exactly
    /// `width * height * 4` bytes.
    pub fn from_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Image<'static>> {
        // 128 bits: two `u32` dimensions times four do not fit in 64.
        if u128::from(width) * u128::from(height) * 4 != rgba.len() as u128 {
            return Err(Error::InvalidImage {
                width,
                height,
                len: rgba.len(),
            });
        }
        let rgb = rgba
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        Image::from_rgb_vec(rgb, width, height)
    }

    /// The width in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// assert_eq!(Image::from_rgb(&[0; 6], 2, 1)?.width(), 2);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The height in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// assert_eq!(Image::from_rgb(&[0; 6], 2, 1)?.height(), 1);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The packed RGB bytes, `width * height * 3` of them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Image;
    ///
    /// let image = Image::from_rgb(&[1, 2, 3], 1, 1)?;
    /// assert_eq!(image.as_rgb(), [1, 2, 3]);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn as_rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// The pixels in the shape the FFI layer takes.
    pub(crate) fn pixels(&self) -> Result<Pixels<'_>> {
        // Checked at construction; checked again so that `Pixels` has one constructor.
        Pixels::new(&self.rgb, self.width, self.height)
    }
}

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
impl Image<'static> {
    /// Reads and decodes a PNG or JPEG file.
    ///
    /// The orientation recorded in the file's EXIF data is applied, so a photo taken with the
    /// camera turned comes out upright, as an image viewer shows it. Any colour type is
    /// converted to 8-bit RGB; alpha is dropped.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::Image;
    ///
    /// let image = Image::open("photo.jpg")?;
    /// println!("{} x {}", image.width(), image.height());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be read and [`Error::ImageDecode`] when its
    /// contents are not an image this build can decode.
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<std::path::Path>,
    {
        Self::decode(&std::fs::read(path.as_ref())?)
    }

    /// Decodes a PNG or JPEG image held in memory.
    ///
    /// Behaves as [`Image::open`] does on the file's contents.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::Image;
    ///
    /// let bytes = std::fs::read("photo.png")?;
    /// let image = Image::decode(&bytes)?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::ImageDecode`] when the bytes are not an image this build can decode.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        use image::ImageDecoder as _;

        let decode = |error: image::ImageError| Error::ImageDecode {
            detail: error.to_string(),
        };
        let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()?
            .into_decoder()
            .map_err(decode)?;
        // A file without EXIF data, or with data that cannot be read, is taken as upright.
        let orientation = decoder
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut decoded = image::DynamicImage::from_decoder(decoder).map_err(decode)?;
        decoded.apply_orientation(orientation);

        let rgb = decoded.into_rgb8();
        let (width, height) = rgb.dimensions();
        Image::from_rgb_vec(rgb.into_raw(), width, height)
    }
}

/// Prints the dimensions, never the pixels.
impl fmt::Debug for Image<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_buffer_must_match_its_dimensions() {
        assert!(Image::from_rgb(&[0; 12], 2, 2).is_ok());
        for (len, width, height) in [(11, 2, 2), (13, 2, 2), (0, 0, 0), (3, 1, 0), (3, 0, 1)] {
            let buffer = vec![0_u8; len];
            let error = Image::from_rgb(&buffer, width, height).expect_err("a mismatch");
            assert!(matches!(error, Error::InvalidImage { len: l, .. } if l == len));
        }
    }

    #[test]
    fn rgba_loses_its_alpha_and_nothing_else() {
        let image = Image::from_rgba(&[1, 2, 3, 4, 5, 6, 7, 8], 1, 2).expect("eight bytes");
        assert_eq!(image.as_rgb(), [1, 2, 3, 5, 6, 7]);
        assert_eq!((image.width(), image.height()), (1, 2));

        // RGB handed in as RGBA is a length mismatch, reported with the length given.
        let error = Image::from_rgba(&[0; 6], 2, 1).expect_err("six bytes are not 2 x 1 RGBA");
        assert!(matches!(error, Error::InvalidImage { len: 6, .. }));
    }

    #[test]
    fn an_owned_image_equals_the_borrowed_one() {
        let pixels = [9_u8; 27];
        let borrowed = Image::from_rgb(&pixels, 3, 3).expect("27 bytes");
        let owned = Image::from_rgb_vec(pixels.to_vec(), 3, 3).expect("27 bytes");
        assert_eq!(borrowed, owned);
    }

    #[test]
    fn debug_does_not_dump_the_pixels() {
        let image = Image::from_rgb(&[7; 12], 2, 2).expect("12 bytes");
        assert_eq!(format!("{image:?}"), "Image { width: 2, height: 2 }");
    }

    #[cfg(feature = "image")]
    #[test]
    fn bytes_that_are_no_image_are_a_decode_error() {
        let error = Image::decode(b"not an image").expect_err("no format matches");
        assert!(matches!(error, Error::ImageDecode { .. }));
    }
}
