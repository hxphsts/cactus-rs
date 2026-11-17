//! The vision model: one image in, one description or one set of answers out.
//!
//! ## Overview
//!
//! [`Mirage`] holds the engine's vision slot. [`Mirage::describe`] runs the whole model over an
//! [`Image`] and returns a [`Description`]; [`Mirage::extract`] answers a [`Form`] and returns
//! its [`Answers`]. Both are one blocking engine call under the lock every model shares, so a
//! description waits for a transcription in flight and the other way round.
//!
//! Nothing is kept between calls: each image is described on its own.

use std::fmt;

use super::description::Description;
use super::form::{Answers, Form};
use super::image::Image;
use super::options::{DescribeOptions, Encoded};
use super::weights::Weights;
use crate::error::{Error, Result};
use crate::ffi::{self, Kind, Slot};

/// The output buffer a call starts with.
///
/// A full description of a simple scene is under 2 KiB and one of a busy scene a few times
/// that; only bitmap masks, which are written at the size of the image, go far beyond.
const INITIAL_CAPACITY: usize = 64 * 1024;

/// The largest output buffer a call will grow to: what upstream's own client allocates.
const MAX_CAPACITY: usize = 16 * 1024 * 1024;

/// What the engine says, before a byte count, when the result does not fit the buffer.
const TOO_SMALL: &str = "output buffer is too small: ";

/// Collects the default options, then takes the vision model.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::mirage::{DescribeOptions, Mirage, Section, Weights};
///
/// let mirage = Mirage::builder(Weights::from_file("mirage.cact")?)
///     .options(DescribeOptions::new().with_sections([Section::Objects, Section::Text]))
///     .build()?;
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct MirageBuilder {
    weights: Weights,
    options: DescribeOptions,
}

impl MirageBuilder {
    /// Sets the options [`Mirage::describe`] uses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{DescribeOptions, Language, Mirage, Weights};
    ///
    /// let mirage = Mirage::builder(Weights::from_file("mirage.cact")?)
    ///     .options(DescribeOptions::new().with_language(Language::De))
    ///     .build()?;
    /// assert_eq!(mirage.options().language(), Some(Language::De));
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn options(mut self, options: DescribeOptions) -> Self {
        self.options = options;
        self
    }

    /// Checks the options, takes the vision model, and loads the weights if they are not loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{Mirage, Weights};
    ///
    /// let mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] or [`Error::InteriorNul`] when a keyword in the options
    /// cannot be passed on, [`Error::EngineBusy`] when another [`Mirage`] is alive,
    /// [`Error::WeightsAlreadyLoaded`] when the process already loaded a different Mirage
    /// archive, [`Error::WrongModel`] when the archive holds another model, and
    /// [`Error::LoadFailed`] when the engine rejects it.
    pub fn build(self) -> Result<Mirage> {
        // Checked first: a bad keyword should not cost a 21.6 MB load.
        let encoded = self.options.encode()?;

        // The slot releases the vision model on every path out of this function. It is declared
        // before the guard so that it drops after it: `Slot`'s `Drop` takes the lock.
        let slot = Slot::take(Kind::Vision)?;
        let mut guard = ffi::lock();
        slot.load(&mut guard, self.weights.as_bytes())?;
        drop(guard);
        // The engine copied the archive; 21.6 MB need not stay resident on our side.
        drop(self.weights);

        Ok(Mirage {
            _slot: slot,
            options: self.options,
            encoded,
            out: Vec::new(),
        })
    }
}

/// The vision model, held by exactly one value in the process.
///
/// Build one through [`Mirage::builder`]. Dropping it frees the vision model for the next
/// [`MirageBuilder::build`]; the weights stay loaded for the life of the process, because the C
/// API offers no way to unload them. A [`Mirage`] lives beside a
/// [`Needle`](crate::needle::Needle) and a [`Whistle`](crate::whistle::Whistle): the engine
/// holds one model of each kind.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::mirage::{Image, Mirage, Weights};
///
/// let mut mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
/// let pixels = vec![0_u8; 640 * 480 * 3];
/// let description = mirage.describe(&Image::from_rgb(&pixels, 640, 480)?)?;
/// for object in description.objects() {
///     println!("{} {:?}", object.name(), object.bbox());
/// }
/// # Ok::<(), cactus_rs::Error>(())
/// ```
///
/// A `Mirage` is [`Send`] and [`Sync`]. Neither makes the engine concurrent: every engine call
/// takes one process-wide lock. A shared `&Mirage` can read [`Mirage::options`] and nothing
/// else.
///
/// ```rust
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<cactus_rs::mirage::Mirage>();
/// ```
pub struct Mirage {
    /// Released when the `Mirage` drops; there is nothing to rewind first.
    _slot: Slot,
    options: DescribeOptions,
    /// `options`, checked and encoded once at build time.
    encoded: Encoded,
    /// Reused across images so describing does not allocate 64 KiB per call.
    out: Vec<u8>,
}

// `Mirage` is `Send + Sync` by auto trait, on the same grounds as `Needle` and `Whistle`: every C
// call happens under the FFI layer's one process-wide lock, and the value itself is plain data.

impl Mirage {
    /// Starts building the one [`Mirage`] this process may hold.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{Mirage, Weights};
    ///
    /// let builder = Mirage::builder(Weights::from_file("mirage.cact")?);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn builder(weights: Weights) -> MirageBuilder {
        MirageBuilder {
            weights,
            options: DescribeOptions::new(),
        }
    }

    /// Describes an image with the options the [`Mirage`] was built with.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{Image, Mirage, Weights};
    ///
    /// let mut mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
    /// let pixels = vec![0_u8; 320 * 240 * 3];
    /// let description = mirage.describe(&Image::from_rgb(&pixels, 320, 240)?)?;
    /// for line in description.text() {
    ///     println!("{}", line.string());
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Describe`] when the engine refuses the options or fails,
    /// [`Error::Truncated`] when the result would not fit in 16 MiB, and [`Error::Envelope`]
    /// when the engine writes something this crate cannot read.
    pub fn describe(&mut self, image: &Image<'_>) -> Result<Description> {
        let (tokens, text) = run(&mut self.out, image, &self.encoded)?;
        Ok(text.parse::<Description>()?.with_tokens(tokens))
    }

    /// Describes an image with `options` instead of the ones the [`Mirage`] was built with.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{DescribeOptions, Image, Mirage, Section, Weights};
    ///
    /// let mut mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
    /// let pixels = vec![0_u8; 320 * 240 * 3];
    /// let text_only = DescribeOptions::new().with_sections([Section::Text]);
    /// let description =
    ///     mirage.describe_with_options(&Image::from_rgb(&pixels, 320, 240)?, &text_only)?;
    /// assert!(description.objects().is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] or [`Error::InteriorNul`] when a keyword in `options`
    /// cannot be passed on, and otherwise fails as [`Mirage::describe`] does.
    pub fn describe_with_options(
        &mut self,
        image: &Image<'_>,
        options: &DescribeOptions,
    ) -> Result<Description> {
        let encoded = options.encode()?;
        let (tokens, text) = run(&mut self.out, image, &encoded)?;
        Ok(text.parse::<Description>()?.with_tokens(tokens))
    }

    /// Answers a form about an image.
    ///
    /// The form path does not produce a description, and the options the [`Mirage`] was built
    /// with do not apply to it.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{Form, Image, Mirage, Weights};
    ///
    /// let mut mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
    /// let pixels = vec![0_u8; 320 * 240 * 3];
    /// let form = Form::new().choice("sign", ["stop", "yield", "speed limit"]);
    ///
    /// let answers = mirage.extract(&Image::from_rgb(&pixels, 320, 240)?, &form)?;
    /// match answers.choice("sign") {
    ///     Some(sign) => println!("a {sign} sign"),
    ///     None => println!("none of those"),
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Describe`] when the engine refuses the form, naming the field, or fails;
    /// [`Error::InteriorNul`] when a name or a choice holds a NUL byte; and [`Error::Envelope`]
    /// when the engine's answer does not fit the form.
    pub fn extract(&mut self, image: &Image<'_>, form: &Form) -> Result<Answers> {
        let options = std::ffi::CString::new(form.document())
            .map_err(|_| Error::InteriorNul { field: "form" })?;
        let encoded = Encoded {
            options,
            language: None,
            keywords: None,
        };
        let (tokens, text) = run(&mut self.out, image, &encoded)?;
        Ok(Answers::parse(form, &text)?.with_tokens(tokens))
    }

    /// The options [`Mirage::describe`] uses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::{Mirage, Weights};
    ///
    /// let mirage = Mirage::builder(Weights::from_file("mirage.cact")?).build()?;
    /// assert_eq!(mirage.options().sections(), None);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn options(&self) -> &DescribeOptions {
        &self.options
    }
}

/// Runs one `cactus_describe` and returns the token count and the document it wrote.
///
/// The engine does not truncate: a result that does not fit fails and says how many bytes it
/// needs. The call is then repeated once with that much room, up to [`MAX_CAPACITY`].
fn run(out: &mut Vec<u8>, image: &Image<'_>, encoded: &Encoded) -> Result<(u32, String)> {
    let pixels = image.pixels()?;
    if out.len() < INITIAL_CAPACITY {
        out.resize(INITIAL_CAPACITY, 0);
    }

    let mut guard = ffi::lock();
    loop {
        // A failed call writes nothing; clearing the first byte keeps the previous image's
        // document from reading back as this one's.
        out[0] = 0;
        let rc = guard.describe(
            pixels,
            &encoded.options,
            encoded.language.as_deref(),
            encoded.keywords.as_deref(),
            out.as_mut_slice(),
        );
        if rc >= 0 {
            drop(guard);
            let end = out.iter().position(|byte| *byte == 0).unwrap_or(out.len());
            let text = String::from_utf8_lossy(&out[..end]).into_owned();
            return Ok((u32::try_from(rc).unwrap_or(0), text));
        }

        // Copied while the guard is held: the string is valid only until the next call.
        let detail = ffi::or_unexplained(guard.last_error(), "cactus_describe", rc);
        match needed(&detail) {
            Some(bytes) if bytes > out.len() && bytes <= MAX_CAPACITY => out.resize(bytes, 0),
            Some(bytes) if bytes > MAX_CAPACITY => {
                return Err(Error::Truncated {
                    capacity: MAX_CAPACITY,
                });
            }
            // Any other failure, or a size that would not help: the engine's words stand.
            _ => return Err(Error::Describe { detail }),
        }
    }
}

/// The buffer size the engine asks for in a "too small" failure, with room for the NUL.
fn needed(detail: &str) -> Option<usize> {
    let rest = detail.strip_prefix(TOO_SMALL)?;
    let digits = rest.split(|c: char| !c.is_ascii_digit()).next()?;
    digits.parse::<usize>().ok()?.checked_add(1)
}

/// Prints the configuration, never an image or a description.
impl fmt::Debug for Mirage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mirage")
            .field("options", &self.options)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sound because every engine call takes the FFI layer's lock; see the note above
    // `impl Mirage`.
    #[test]
    fn test_mirage_is_send_and_sync() {
        fn assert_bounds<T: Send + Sync>() {}
        assert_bounds::<Mirage>();
        assert_bounds::<MirageBuilder>();
    }

    #[test]
    fn a_bad_keyword_fails_the_build_before_the_engine_is_touched() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05]).expect("the tag");
        let error = Mirage::builder(weights)
            .options(DescribeOptions::new().with_keyword("a\nb"))
            .build()
            .expect_err("a line break inside a keyword");
        assert!(matches!(error, Error::InvalidKeyword { .. }));
    }

    #[test]
    fn the_engines_size_request_is_read_from_its_words() {
        // Measured: a 64-byte buffer for a 1678-byte document.
        assert_eq!(
            needed("output buffer is too small: 1679 bytes needed"),
            Some(1680)
        );
        assert_eq!(needed("output buffer is too small: "), None);
        assert_eq!(
            needed("output buffer is too small: many bytes needed"),
            None
        );
        // The other "too small" messages ask for nothing in particular.
        assert_eq!(
            needed("output buffer is too small for count probabilities"),
            None
        );
        assert_eq!(needed("invalid options: conf must be a number"), None);
        assert_eq!(needed(""), None);
    }

    #[test]
    fn the_buffer_limits_are_the_ones_documented() {
        assert_eq!(INITIAL_CAPACITY, 65_536);
        assert_eq!(MAX_CAPACITY, 16_777_216);
    }
}
