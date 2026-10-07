//! The speech model itself: acquiring it, and transcribing and embedding clips with it.
//!
//! ## Overview
//!
//! The engine holds one speech model per process, beside one text model, behind one C API with
//! no handles. [`Whistle`] is that speech model expressed as a Rust value:
//!
//! - **One per process.** [`WhistleBuilder::build`] takes the speech model and returns
//!   [`Error::EngineBusy`] while another [`Whistle`] is alive. Dropping one frees the slot. A
//!   [`Needle`](crate::needle::Needle) holds the other slot, so one of each can live together.
//! - **Weights are permanent.** The first successful load owns the process. Building again from
//!   the same archive reuses it; a different archive returns [`Error::WeightsAlreadyLoaded`],
//!   and a Needle archive returns [`Error::WrongModel`].
//! - **Calls do not overlap.** Every engine call takes one process-wide lock, so a [`Whistle`]
//!   and a [`Needle`](crate::needle::Needle) on different threads take turns. [`Whistle`] is
//!   [`Send`] and [`Sync`]; its methods that reach the engine take `&mut self`.
//! - **No state between clips.** Each transcription stands alone, with the options it is given;
//!   there is no conversation to reset, so dropping a [`Whistle`] only frees the slot.
//! - **No streaming.** A clip of up to 30 seconds goes in and its whole transcript comes back.
//!
//! This module has no `unsafe` in it: all of the crate's lives in its private FFI layer, which
//! also holds the lock.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::whistle::{Language, TranscribeOptions, Weights, Whistle};
//!
//! let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?)
//!     .options(TranscribeOptions::new().with_language(Language::En))
//!     .build()?;
//!
//! let pcm = vec![0.0_f32; 16_000]; // one second of 16 kHz mono silence
//! let transcript = whistle.transcribe(&pcm)?;
//! assert!(transcript.is_empty());
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::fmt;

use super::options::TranscribeOptions;
use super::transcript::Transcript;
use super::weights::Weights;
use super::{FRAME_SAMPLES, frame_count, validate};
use crate::error::{Error, Result};
use crate::ffi::{self, EncodedAudio, Input, Kind, Slot};

/// Size of the transcription buffer, as upstream's own binding uses.
///
/// A 30 second clip with word timestamps measured at a few kilobytes, so this never fills in
/// practice; [`Error::Truncated`] covers the case where it does.
const TRANSCRIPT_CAPACITY: usize = 256 * 1024;

/// What the engine said about a failure, or a stand-in when it said nothing.
fn reason(detail: String) -> String {
    if detail.is_empty() {
        "the engine gave no reason".to_owned()
    } else {
        detail
    }
}

/// Collects the default transcription options, then takes the speech model.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::whistle::{TranscribeOptions, Weights, Whistle};
///
/// let whistle = Whistle::builder(Weights::from_file("whistle.cact")?)
///     .options(TranscribeOptions::new().with_word_timestamps(true))
///     .build()?;
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct WhistleBuilder {
    weights: Weights,
    options: TranscribeOptions,
}

impl WhistleBuilder {
    /// Sets the options [`Whistle::transcribe`] uses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Language, TranscribeOptions, Weights, Whistle};
    ///
    /// let whistle = Whistle::builder(Weights::from_file("whistle.cact")?)
    ///     .options(TranscribeOptions::new().with_language(Language::De))
    ///     .build()?;
    /// assert_eq!(whistle.options().language(), Some(Language::De));
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn options(mut self, options: TranscribeOptions) -> Self {
        self.options = options;
        self
    }

    /// Checks the options, takes the speech model, and loads the weights if they are not loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] or [`Error::InteriorNul`] when a keyword in the options
    /// cannot be passed on, [`Error::EngineBusy`] when another [`Whistle`] is alive,
    /// [`Error::WeightsAlreadyLoaded`] when the process already loaded a different Whistle
    /// archive, [`Error::WrongModel`] when the archive is a Needle archive, and [`Error::LoadFailed`]
    /// when the engine rejects it.
    pub fn build(self) -> Result<Whistle> {
        // Checked first: a bad keyword should not cost a 16.9 MB load.
        let audio = self.options.encode()?;

        // The slot releases the speech model on every path out of this function. It is declared
        // before the guard so that it drops after it: `Slot`'s `Drop` takes the lock.
        let slot = Slot::take(Kind::Speech)?;
        let mut guard = ffi::lock();
        slot.load(&mut guard, self.weights.as_bytes())?;
        drop(guard);
        // The engine copied the archive; 16.9 MB need not stay resident on our side.
        drop(self.weights);

        Ok(Whistle {
            _slot: slot,
            options: self.options,
            audio,
            out: Vec::new(),
            width: None,
        })
    }
}

/// The speech model, held by exactly one value in the process.
///
/// Build one through [`Whistle::builder`]. Dropping it frees the speech model for the next
/// [`WhistleBuilder::build`]; the weights stay loaded for the life of the process, because the C
/// API offers no way to unload them. That is also what lets
/// [`Needle::complete_audio`](crate::needle::Needle::complete_audio) keep working after the
/// [`Whistle`] that loaded them is gone.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::whistle::{Weights, Whistle};
///
/// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
/// let pcm = vec![0.0_f32; 16_000];
/// println!("{}", whistle.transcribe(&pcm)?.text());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
///
/// A `Whistle` is [`Send`] and [`Sync`]. Neither makes the engine concurrent: every engine call
/// takes one process-wide lock. A shared `&Whistle` can read [`Whistle::options`] and nothing
/// else.
///
/// ```rust
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<cactus_rs::whistle::Whistle>();
/// ```
pub struct Whistle {
    /// Released when the `Whistle` drops; there is nothing to rewind first.
    _slot: Slot,
    options: TranscribeOptions,
    /// `options`, checked and encoded once at build time.
    audio: EncodedAudio,
    /// Reused across clips so transcribing does not allocate 256 KiB per call.
    out: Vec<u8>,
    width: Option<usize>,
}

// `Whistle` is `Send + Sync` by auto trait, on the same grounds as `Needle`: every C call happens
// under the FFI layer's one process-wide lock, and the engine has no thread affinity (measured on
// the pinned Linux x86_64 archive: a model loaded and used on one thread transcribes identically
// on two others after it exits).

impl Whistle {
    /// Starts building the one [`Whistle`] this process may hold.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let builder = Whistle::builder(Weights::from_file("whistle.cact")?);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn builder(weights: Weights) -> WhistleBuilder {
        WhistleBuilder {
            weights,
            options: TranscribeOptions::new(),
        }
    }

    /// Transcribes one clip with the options the builder was given.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let pcm = vec![0.0_f32; 16_000]; // 16 kHz mono, in [-1, 1]
    ///
    /// let transcript = whistle.transcribe(&pcm)?;
    /// println!("{} [{}]", transcript.text(), transcript.language_code());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// See [`Whistle::transcribe_with_options`].
    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<Transcript> {
        transcribe(&mut self.out, pcm, &self.audio)
    }

    /// Transcribes one clip with the options given here, leaving the builder's untouched.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{TranscribeOptions, Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let pcm = vec![0.0_f32; 16_000];
    ///
    /// let options = TranscribeOptions::new()
    ///     .with_keywords(["Siobhan", "Krzysztof"])
    ///     .with_word_timestamps(true);
    /// for word in whistle.transcribe_with_options(&pcm, &options)?.words() {
    ///     println!("{:.2}-{:.2} {}", word.start(), word.end(), word.word());
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] or [`Error::InteriorNul`] when a keyword cannot be
    /// passed on, [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`](super::MAX_SAMPLES),
    /// [`Error::NonFiniteSample`] for a NaN or an infinity, [`Error::Transcribe`] with the
    /// engine's reason when it fails, [`Error::Truncated`] when the transcript filled the output
    /// buffer, and [`Error::Envelope`] when the response is not the JSON this crate expects.
    pub fn transcribe_with_options(
        &mut self,
        pcm: &[f32],
        options: &TranscribeOptions,
    ) -> Result<Transcript> {
        let audio = options.encode()?;
        transcribe(&mut self.out, pcm, &audio)
    }

    /// Embeds one clip: one row of [`Whistle::embedding_width`] floats per 80 ms frame,
    /// flattened.
    ///
    /// The result holds [`frame_count`]`(pcm.len())` rows, so an empty clip still gives one.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle, frame_count};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let pcm = vec![0.0_f32; 16_000];
    ///
    /// let width = whistle.embedding_width()?;
    /// let embedding = whistle.embed(&pcm)?;
    /// assert_eq!(embedding.len(), frame_count(pcm.len()) * width);
    ///
    /// for (frame, row) in embedding.chunks_exact(width).enumerate() {
    ///     println!("{} ms: {:?}", frame * 80, &row[..4]);
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`](super::MAX_SAMPLES),
    /// [`Error::NonFiniteSample`] for a NaN or an infinity, and [`Error::EmbedAudio`] with the
    /// engine's reason when it fails.
    pub fn embed(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        validate(pcm)?;

        let mut guard = ffi::lock();
        let wanted = guard.embed_len(Input::Audio(pcm))?;
        let Ok(len) = usize::try_from(wanted) else {
            return Err(Error::EmbedAudio {
                detail: reason(guard.last_error()),
            });
        };

        let mut embedding = vec![0.0_f32; len];
        let written = guard.embed(Input::Audio(pcm), &mut embedding)?;
        if written != wanted {
            let detail = if written < 0 {
                reason(guard.last_error())
            } else {
                format!("the engine wrote {written} floats after asking for {wanted}")
            };
            return Err(Error::EmbedAudio { detail });
        }

        Ok(embedding)
    }

    /// The width of one row of [`Whistle::embed`]'s output: 512 for the pinned Whistle.
    ///
    /// Asked of the engine once, by sizing the embedding of one frame, and cached.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// assert_eq!(whistle.embedding_width()?, 512);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbedAudio`] when the engine does not report a width.
    pub fn embedding_width(&mut self) -> Result<usize> {
        if let Some(width) = self.width {
            return Ok(width);
        }

        // A null buffer asks for the float count without computing anything.
        let frame = [0.0_f32; FRAME_SAMPLES];
        let mut guard = ffi::lock();
        let rc = guard.embed_len(Input::Audio(&frame))?;
        let Ok(count) = usize::try_from(rc) else {
            return Err(Error::EmbedAudio {
                detail: reason(guard.last_error()),
            });
        };
        drop(guard);

        let frames = frame_count(FRAME_SAMPLES);
        if count == 0 || count % frames != 0 {
            return Err(Error::EmbedAudio {
                detail: format!("the engine sized {FRAME_SAMPLES} samples at {count} floats"),
            });
        }

        let width = count / frames;
        self.width = Some(width);
        Ok(width)
    }

    /// The options [`Whistle::transcribe`] uses.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// assert!(whistle.options().keywords().is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn options(&self) -> &TranscribeOptions {
        &self.options
    }
}

/// Runs one transcription into `out`, which is grown to [`TRANSCRIPT_CAPACITY`] and reused.
fn transcribe(out: &mut Vec<u8>, pcm: &[f32], audio: &EncodedAudio) -> Result<Transcript> {
    // Checked before the lock: a bad clip should not wait for, or hold up, anyone else.
    validate(pcm)?;

    out.resize(TRANSCRIPT_CAPACITY, 0);
    // A failed call writes nothing; clearing the first byte keeps the previous clip's text from
    // reading back as this one's.
    out[0] = 0;

    let mut guard = ffi::lock();
    // The options go with every call: `needle_transcribe` never reads `needle_set_audio`'s.
    let rc = guard.transcribe(pcm, audio, out)?;
    if rc < 0 {
        // A failed transcription writes no envelope; the reason is only in the error slot, and
        // it must be copied before the lock is released.
        return Err(Error::Transcribe {
            detail: reason(guard.last_error()),
        });
    }
    drop(guard);

    let capacity = out.len();
    let end = out.iter().position(|byte| *byte == 0).unwrap_or(capacity);
    // The engine truncates to `capacity - 1` bytes and says nothing; a full buffer is the only
    // signal there is.
    if end >= capacity - 1 {
        return Err(Error::Truncated { capacity });
    }

    String::from_utf8_lossy(&out[..end]).parse()
}

/// Prints the configuration, never a transcript.
impl fmt::Debug for Whistle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Whistle")
            .field("options", &self.options)
            .field("embedding_width", &self.width)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sound because every engine call takes the FFI layer's lock; see the note above
    // `impl Whistle`.
    #[test]
    fn test_whistle_is_send_and_sync() {
        fn assert_bounds<T: Send + Sync>() {}
        assert_bounds::<Whistle>();
        assert_bounds::<WhistleBuilder>();
    }

    #[test]
    fn a_bad_keyword_fails_the_build_before_the_engine_is_touched() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05]).expect("the tag");
        let error = Whistle::builder(weights)
            .options(TranscribeOptions::new().with_keyword("a\nb"))
            .build()
            .expect_err("a line break inside a keyword");
        assert!(matches!(error, Error::InvalidKeyword { .. }));
    }

    #[test]
    fn a_reason_is_never_empty() {
        assert_eq!(reason(String::new()), "the engine gave no reason");
        assert_eq!(
            reason("audio limit is 30 s".to_owned()),
            "audio limit is 30 s"
        );
    }

    #[test]
    fn the_buffer_is_the_size_upstream_uses() {
        assert_eq!(TRANSCRIPT_CAPACITY, 262_144);
    }
}
