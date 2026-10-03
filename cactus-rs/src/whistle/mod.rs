//! Whistle: speech-to-text, word timestamps and speech embedding on the Needle engine.
//!
//! ## Overview
//!
//! Whistle is not a second engine. It is a 17 MB speech model loaded into the same runtime as
//! Needle, which holds one model per kind, so a [`Whistle`] and a
//! [`Needle`](crate::needle::Needle) live side by side in one process. Five types carry the
//! whole API:
//!
//! | Type | What it is |
//! | --- | --- |
//! | [`Weights`] | the 17 MB `whistle.cact` archive, validated |
//! | [`TranscribeOptions`] | language, keywords to bias towards, and word timestamps |
//! | [`Whistle`] | the speech model, held by exactly one value in the process |
//! | [`Transcript`] | one clip's text and [`Language`], with its [`Word`]s when asked for |
//! | [`Language`] | the seven languages Whistle transcribes |
//!
//! [`WhistleBuilder`] sets the options a [`Whistle`] transcribes with by default.
//!
//! Audio is 16 kHz mono `f32` PCM in `[-1, 1]` ([`SAMPLE_RATE`]), at most 30 seconds
//! ([`MAX_SAMPLES`]) per call. Clips are plain `&[f32]` slices, so the output of a WAV reader,
//! a resampler or a microphone callback goes in as it is. Every clip is checked before it reaches
//! the engine: a longer one is [`Error::AudioTooLong`], and a NaN or
//! an infinity is [`Error::NonFiniteSample`]. An empty clip is
//! not an error: it transcribes to an empty [`Transcript`], as silence does.
//!
//! The speech model also runs inside the text model:
//! [`Needle::complete_audio`](crate::needle::Needle::complete_audio) transcribes a clip and
//! answers the transcript with tool calls in one engine call, once a [`Whistle`] has loaded the
//! speech weights.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::whistle::{Language, TranscribeOptions, Weights, Whistle};
//!
//! let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
//!
//! // 16 kHz mono samples in [-1, 1], from a WAV reader or a microphone.
//! let pcm = vec![0.0_f32; 16_000];
//!
//! let transcript = whistle.transcribe(&pcm)?;
//! println!("{} ({})", transcript.text(), transcript.language_code());
//!
//! let options = TranscribeOptions::new()
//!     .with_language(Language::En)
//!     .with_keywords(["Siobhan", "Krzysztof"])
//!     .with_word_timestamps(true);
//! for word in whistle.transcribe_with_options(&pcm, &options)?.words() {
//!     println!("{:>6.2}s {}", word.start(), word.word());
//! }
//!
//! let embedding = whistle.embed(&pcm)?;
//! assert_eq!(
//!     embedding.len(),
//!     cactus_rs::whistle::frame_count(pcm.len()) * whistle.embedding_width()?
//! );
//! # Ok::<(), cactus_rs::Error>(())
//! ```

pub mod engine;
pub mod options;
pub mod transcript;
pub mod weights;

pub use engine::{Whistle, WhistleBuilder};
pub use options::{Language, TranscribeOptions};
pub use transcript::{Transcript, Word};
pub use weights::Weights;

use crate::error::{Error, Result};

/// The sample rate Whistle expects, in hertz. Clips are mono.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{MAX_SAMPLES, MAX_SECONDS, SAMPLE_RATE};
///
/// assert_eq!(SAMPLE_RATE as usize * MAX_SECONDS as usize, MAX_SAMPLES);
/// ```
pub const SAMPLE_RATE: u32 = 16_000;

/// The longest clip one call accepts, in seconds.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::whistle::MAX_SECONDS, 30);
/// ```
pub const MAX_SECONDS: u32 = 30;

/// The longest clip one call accepts, in samples: 30 seconds at 16 kHz.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::whistle::MAX_SAMPLES, 480_000);
/// ```
pub const MAX_SAMPLES: usize = crate::ffi::MAX_SAMPLES;

/// The span of audio one speech embedding row covers, in milliseconds.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE};
///
/// assert_eq!(FRAME_SAMPLES, (SAMPLE_RATE / 1000 * FRAME_MS) as usize);
/// ```
pub const FRAME_MS: u32 = 80;

/// The span of audio one speech embedding row covers, in samples: 80 ms at 16 kHz.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::whistle::FRAME_SAMPLES, 1_280);
/// ```
pub const FRAME_SAMPLES: usize = 1_280;

/// How many rows [`Whistle::embed`] returns for a clip of `samples` samples.
///
/// The engine frames audio with a 10 ms hop and pools eight hops into one 80 ms row, keeping a
/// trailing part-row once it holds more than one hop beyond the last full row. That works out
/// to `max(1, (samples + 1120) / 1280)`, measured against the pinned engine for every length
/// from 0 to [`MAX_SAMPLES`]: even an empty clip gets one row, and a full 30 s clip gets 375.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{MAX_SAMPLES, frame_count};
///
/// assert_eq!(frame_count(0), 1);
/// assert_eq!(frame_count(1_440), 2);
/// assert_eq!(frame_count(16_000), 13);
/// assert_eq!(frame_count(MAX_SAMPLES), 375);
/// ```
#[inline]
#[must_use]
pub const fn frame_count(samples: usize) -> usize {
    let frames = samples.saturating_add(1_120) / FRAME_SAMPLES;
    if frames == 0 { 1 } else { frames }
}

/// Checks a clip before it reaches the engine: length first, then every sample.
///
/// # Errors
///
/// Returns [`Error::AudioTooLong`] for more than [`MAX_SAMPLES`] samples, and
/// [`Error::NonFiniteSample`] naming the first NaN or infinity.
pub(crate) fn validate(pcm: &[f32]) -> Result<()> {
    if pcm.len() > MAX_SAMPLES {
        return Err(Error::AudioTooLong { samples: pcm.len() });
    }
    match pcm.iter().position(|sample| !sample.is_finite()) {
        Some(index) => Err(Error::NonFiniteSample { index }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_the_measured_engine() {
        // Float counts the pinned engine reported, divided by its 512-wide rows.
        for (samples, frames) in [
            (0, 1),
            (1, 1),
            (1_279, 1),
            (1_280, 1),
            (1_281, 1),
            (1_439, 1),
            (1_440, 2),
            (2_720, 3),
            (16_000, 13),
            (160_000, 125),
            (480_000, 375),
        ] {
            assert_eq!(frame_count(samples), frames, "{samples} samples");
        }
    }

    #[test]
    fn frame_count_never_overflows() {
        assert_eq!(frame_count(usize::MAX), usize::MAX / FRAME_SAMPLES);
    }

    #[test]
    fn the_constants_agree() {
        assert_eq!(SAMPLE_RATE as usize * MAX_SECONDS as usize, MAX_SAMPLES);
        assert_eq!((SAMPLE_RATE / 1000 * FRAME_MS) as usize, FRAME_SAMPLES);
    }

    #[test]
    fn thirty_seconds_is_accepted_and_one_sample_more_is_not() {
        assert!(validate(&vec![0.0; MAX_SAMPLES]).is_ok());
        let error = validate(&vec![0.0; MAX_SAMPLES + 1]).expect_err("over 30 s");
        assert!(matches!(error, Error::AudioTooLong { samples } if samples == MAX_SAMPLES + 1));
    }

    #[test]
    fn an_empty_clip_is_valid() {
        assert!(validate(&[]).is_ok());
    }

    #[test]
    fn the_first_non_finite_sample_is_named() {
        let error = validate(&[0.0, 1.0, f32::NAN, f32::INFINITY]).expect_err("a NaN");
        assert!(matches!(error, Error::NonFiniteSample { index: 2 }));

        let error = validate(&[f32::NEG_INFINITY]).expect_err("an infinity");
        assert!(matches!(error, Error::NonFiniteSample { index: 0 }));

        // Out of range but finite is the engine's business, not a validation error.
        assert!(validate(&[-1.0, 1.0, 1.5, f32::MAX]).is_ok());
    }

    #[test]
    fn length_is_checked_before_the_samples() {
        let mut clip = vec![0.0; MAX_SAMPLES + 1];
        clip[0] = f32::NAN;
        assert!(matches!(validate(&clip), Err(Error::AudioTooLong { .. })));
    }
}
