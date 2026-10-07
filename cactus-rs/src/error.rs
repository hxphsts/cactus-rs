//! Error type for every fallible operation in this crate.
//!
//! ## Overview
//!
//! One [`Error`] enum covers the whole crate, and [`Result<T>`] is the alias every fallible
//! function returns. The variants split into five groups:
//!
//! - **Engine ownership**: [`Error::EngineBusy`], [`Error::WeightsAlreadyLoaded`],
//!   [`Error::WrongModel`]: the engine is one process-global runtime holding one model per kind,
//!   so these report a rule of the C API rather than a failure
//! - **Engine calls**: [`Error::LoadFailed`], [`Error::InitFailed`], [`Error::Complete`],
//!   [`Error::Truncated`], [`Error::EmbedFailed`], [`Error::InteriorNul`]
//! - **Audio**: [`Error::SpeechModelNotLoaded`], [`Error::AudioTooLong`],
//!   [`Error::NonFiniteSample`], [`Error::Transcribe`], [`Error::EmbedAudio`],
//!   [`Error::UnsupportedLanguage`], [`Error::InvalidKeyword`]
//! - **Encoding and decoding**: [`Error::Tools`], [`Error::Envelope`], [`Error::Arguments`]
//! - **Weights acquisition**: [`Error::UnsupportedWeights`], [`Error::Io`], [`Error::Download`],
//!   [`Error::ChecksumMismatch`]
//!
//! [`Error::Load`], [`Error::Init`] and [`Error::Embed`] are kept so that code matching on them
//! still compiles, but nothing returns them since 0.2.1: the `*Failed` variants replaced them and
//! carry the engine's own reason, which [`Error::detail`] reads from any variant that has one.
//!
//! Every message is written for the person who will read it in a terminal: what happened, then a
//! final `Help:` line saying what to do about it.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::Error;
//!
//! let err = Error::UnsupportedWeights { tag: 0x05E1_2A83 };
//! assert!(err.to_string().contains("Help:"));
//! ```

use thiserror::Error as ThisError;

use crate::archive::MAGIC;
use crate::model::Model;

/// The number of bytes of a bad engine envelope kept in [`Error::Envelope`].
///
/// Envelopes run to a few hundred bytes of JSON and a truncated head is enough to recognise
/// what came back; keeping all of it would put model output into log lines.
const ENVELOPE_EXCERPT: usize = 200;

/// What [`Error::UnsupportedWeights`] says about its tag.
fn unsupported(tag: u32) -> String {
    if tag == MAGIC {
        format!(
            "magic tag {tag:#010X} is right, but the tensor directory after it is truncated or \
             malformed"
        )
    } else {
        format!("magic tag {tag:#010X} is not a Needle 3 or Whistle archive")
    }
}

/// The weights type each model is built from, as [`Error::WrongModel`] names it.
const fn weights_type(model: Model) -> &'static str {
    match model {
        Model::Needle => "needle::Weights",
        Model::Whistle => "whistle::Weights",
    }
}

/// Everything that can go wrong while loading weights or driving the engine.
///
/// The enum is `#[non_exhaustive]`: matching it needs a `_` arm, and new variants are not a
/// breaking change.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::Error;
///
/// let err = Error::Truncated { capacity: 65_536 };
/// assert!(err.to_string().contains("65536"));
/// assert!(err.to_string().contains("Help:"));
/// ```
#[derive(Debug, ThisError)]
#[non_exhaustive]
pub enum Error {
    /// Another instance of the same model is alive in this process: a second
    /// [`Needle`](crate::needle::Needle) while one exists, or a second Whistle while one exists.
    #[error(
        "the engine slot for this model is already in use by another instance in this process\n\
         The engine is one process-global runtime with no handles that holds one model per \
         kind, so one Needle and one Whistle can exist at a time, not two of either.\n\
         Help: drop the existing instance before building another of the same model, or share \
         it behind a mutex."
    )]
    EngineBusy,

    /// Weights of the same kind were already loaded, and the new archive is a different one.
    #[error(
        "different weights of the same kind are already loaded in this process\n\
         The engine copies the archive on the first load of each kind and offers no way to \
         unload it.\n\
         Help: build every instance of a model from the same weights, or run the second archive \
         in its own process."
    )]
    WeightsAlreadyLoaded,

    /// The archive holds the other kind of model than the one being built.
    ///
    /// Needle 3 and Whistle archives carry the same magic tag, so the kind is read from the
    /// archive's tensor directory before the engine sees it. A wrong archive is never loaded:
    /// the engine would have swapped it in over the model of that kind.
    #[error(
        "the weights are not a {expected} archive\n\
         Needle 3 and Whistle archives share one magic tag, and this one holds the other model; \
         it was not loaded.\n\
         Help: build a {expected} from {} through {}, and hand this archive to the other \
         model's builder.",
        .expected.archive(),
        weights_type(*.expected)
    )]
    WrongModel {
        /// The model that was being built.
        expected: Model,
    },

    /// The archive is not a Needle 3 or Whistle `.cact` file: a wrong magic tag, or the right
    /// one over a tensor directory that cannot be read.
    #[error(
        "unsupported weights: {}\n\
         Needle 3 and Whistle archives start with the little-endian tag 0x05E12A84; 0x05E12A83 \
         is a Needle 2 archive, which the linked engine cannot read.\n\
         Help: download needle3.cact from Cactus-Compute/needle3 or whistle.cact from \
         Cactus-Compute/whistle, or call Weights::fetch().",
        unsupported(*.tag)
    )]
    UnsupportedWeights {
        /// The little-endian `u32` read from the first four bytes of the archive. It is the
        /// right tag, 0x05E12A84, when the tensor directory after it is truncated or malformed.
        tag: u32,
    },

    /// Not returned since 0.2.1; see [`Error::LoadFailed`], which carries the engine's reason.
    #[error(
        "the engine rejected the weight archive\n\
         The magic tag was right but the engine could not load the bytes as a Needle 3 or \
         Whistle model, which usually means a truncated or corrupted download.\n\
         Help: delete the cached archive and fetch it again, or verify its SHA-256."
    )]
    Load,

    /// `needle_load` rejected the archive; `detail` is the engine's own reason.
    #[error(
        "the engine rejected the {model} archive: {detail}\n\
         The magic tag was right but the engine could not load the bytes, which usually means a \
         truncated or corrupted download.\n\
         Help: delete the cached archive and fetch it again, or verify its SHA-256."
    )]
    LoadFailed {
        /// The model that was being built.
        model: Model,
        /// The engine's own reason, from `needle_last_error`.
        detail: String,
    },

    /// Not returned since 0.2.1; see [`Error::InitFailed`], which carries the engine's reason.
    #[error(
        "the engine could not initialise the conversation prefix\n\
         needle_init returned a non-positive token count, which happens when the weights are not \
         loaded or the tool index path does not exist.\n\
         Help: check that the tool index path exists and that the weights loaded successfully."
    )]
    Init,

    /// `needle_init` failed; `detail` includes the engine's reason (e.g. the measured token count
    /// when the prefix overflows the context).
    #[error(
        "the engine could not initialise the conversation prefix: {detail}\n\
         The system prompt and the tool declarations must fit the context window together, and a \
         tool index path must exist.\n\
         Help: shorten the system prompt or declare fewer tools, and check that the tool index \
         path exists."
    )]
    InitFailed {
        /// The engine's own reason, from `needle_last_error`, with the count `needle_init`
        /// returned.
        detail: String,
    },

    /// `needle_complete` failed, with whatever detail the engine wrote into the buffer.
    #[error(
        "the engine failed to complete the request: {detail}\n\
         Help: reset the conversation, shorten the input, or raise max_new_tokens."
    )]
    Complete {
        /// The `error` field of the engine's error envelope, or the raw text when it had none.
        detail: String,
    },

    /// The engine filled the output buffer and silently cut the envelope short.
    #[error(
        "the engine truncated its response to fill the {capacity} byte output buffer\n\
         The engine truncates without reporting it, so the JSON envelope is incomplete.\n\
         Help: lower max_new_tokens, ask for fewer tool calls in one turn, or turn word \
         timestamps off."
    )]
    Truncated {
        /// The capacity of the output buffer, in bytes, that the response filled.
        capacity: usize,
    },

    /// Not returned since 0.2.1; see [`Error::EmbedFailed`], which carries the engine's reason.
    #[error(
        "the engine failed to embed the input\n\
         needle_embed returned a negative value, which happens when no weights are loaded.\n\
         Help: build the Needle from valid weights before embedding."
    )]
    Embed,

    /// `needle_embed` failed on text: the engine's reason, or a float-count mismatch ("engine
    /// wrote N floats, expected D").
    #[error(
        "the engine failed to embed the input: {detail}\n\
         Help: build the Needle from valid weights before embedding, and report the reason above \
         if they are."
    )]
    EmbedFailed {
        /// The engine's own reason, from `needle_last_error`, or the float-count mismatch.
        detail: String,
    },

    /// A string bound for the C API contains an interior NUL byte.
    #[error(
        "the {field} contains an interior NUL byte\n\
         The engine takes NUL-terminated C strings, so a NUL inside the text would silently cut \
         it short.\n\
         Help: strip '\\0' from the {field} before passing it in."
    )]
    InteriorNul {
        /// Which input held the NUL: `"system prompt"`, `"tools"`, `"input"`, `"tool index"` or
        /// `"keywords"`.
        field: &'static str,
    },

    /// An audio completion was asked for, and no speech model is loaded.
    #[error(
        "no speech model is loaded in this process\n\
         An audio completion transcribes the clip with Whistle before Needle answers it, and \
         the engine holds no Whistle weights yet.\n\
         Help: build a Whistle once in this process before completing audio; its weights stay \
         loaded after it is dropped."
    )]
    SpeechModelNotLoaded,

    /// The clip is longer than the engine accepts.
    #[error(
        "the clip is {samples} samples long, more than the 480000 the engine accepts\n\
         The engine takes at most 30 seconds of 16 kHz mono audio in one call.\n\
         Help: split the clip into chunks of at most 30 seconds and send them one at a time."
    )]
    AudioTooLong {
        /// The number of samples in the clip.
        samples: usize,
    },

    /// A sample of the clip is NaN or infinite.
    #[error(
        "sample {index} of the clip is not a finite number\n\
         The engine expects 16 kHz mono PCM in [-1, 1], and a NaN or an infinity would poison \
         every frame it touches.\n\
         Help: check the decoder or resampler that produced the clip, or replace non-finite \
         samples with 0.0."
    )]
    NonFiniteSample {
        /// The position of the first non-finite sample.
        index: usize,
    },

    /// `needle_transcribe` failed, with the engine's reason.
    #[error(
        "the engine failed to transcribe the clip: {detail}\n\
         Help: check that the clip is 16 kHz mono PCM in [-1, 1], and that the Whistle weights \
         loaded successfully."
    )]
    Transcribe {
        /// The engine's own reason, from `needle_last_error`.
        detail: String,
    },

    /// `needle_embed` failed on a clip, with the engine's reason.
    #[error(
        "the engine failed to embed the clip: {detail}\n\
         Help: check that the clip is 16 kHz mono PCM in [-1, 1], and that the Whistle weights \
         loaded successfully."
    )]
    EmbedAudio {
        /// The engine's own reason, from `needle_last_error`.
        detail: String,
    },

    /// A language code is not one Whistle transcribes.
    #[error(
        "unsupported language {code:?}\n\
         Whistle transcribes English, German, French, Spanish, Italian, Dutch and Polish.\n\
         Help: use one of en, de, fr, es, it, nl or pl, or leave the language unset to detect it."
    )]
    UnsupportedLanguage {
        /// The code that was given.
        code: String,
    },

    /// A keyword holds a line break.
    #[error(
        "the keyword {keyword:?} contains a line break\n\
         Keywords reach the engine as one newline-separated list, so a line break inside one \
         would silently split it in two.\n\
         Help: pass each word or phrase as its own keyword."
    )]
    InvalidKeyword {
        /// The keyword as it was given.
        keyword: String,
    },

    /// The engine's response was not the JSON envelope this crate expects.
    #[error(
        "the engine returned an envelope this crate could not parse: {source}\n\
         Response: {text}\n\
         Help: this is a bug or an engine newer than the pinned one; please report the response \
         above."
    )]
    Envelope {
        /// The first few hundred bytes of the response, for the report.
        text: String,
        /// The `serde_json` error that rejected it.
        #[source]
        source: serde_json::Error,
    },

    /// A tool call's arguments did not fit the requested Rust type.
    #[error(
        "the tool call arguments do not fit the requested type: {source}\n\
         Help: the model fills arguments from the JSON schema you declared; relax the type, or \
         tighten the schema so the engine cannot produce this shape."
    )]
    Arguments {
        /// The `serde_json` error that rejected the arguments.
        #[source]
        source: serde_json::Error,
    },

    /// The tool declarations could not be written as JSON for the engine.
    #[error(
        "the tool declarations could not be serialised: {source}\n\
         Help: check the `parameters` value of each Tool; it must be plain JSON."
    )]
    Tools {
        /// The `serde_json` error that rejected the declarations.
        #[source]
        source: serde_json::Error,
    },

    /// Reading or writing a file failed.
    #[error(
        "filesystem error: {0}\n\
         Help: check that the path exists and that this process may read and write it."
    )]
    Io(#[from] std::io::Error),

    /// Downloading the weight archive failed.
    #[error(
        "could not download weights from {url}: {detail}\n\
         Help: retry, or download the archive by hand and point CACTUS_NEEDLE_WEIGHTS (Needle) \
         or CACTUS_WHISTLE_WEIGHTS (Whistle) at it."
    )]
    Download {
        /// The URL that was being fetched.
        url: String,
        /// What went wrong, as reported by the HTTP client.
        detail: String,
    },

    /// A downloaded archive did not match its pinned SHA-256.
    #[error(
        "the downloaded weights do not match their pinned checksum\n\
         Expected sha256 {expected}, got {actual}.\n\
         Help: the download was corrupted or the remote file changed; retry, and report the \
         mismatch if it persists."
    )]
    ChecksumMismatch {
        /// The SHA-256 this crate pins, lowercase hex.
        expected: String,
        /// The SHA-256 of the bytes that arrived, lowercase hex.
        actual: String,
    },
}

impl Error {
    /// The underlying reason in the error's own words, for the variants that carry one.
    ///
    /// That is the engine's text for [`Error::Complete`], [`Error::Transcribe`],
    /// [`Error::EmbedAudio`], [`Error::LoadFailed`], [`Error::InitFailed`] and
    /// [`Error::EmbedFailed`], and the HTTP client's for [`Error::Download`]. Every other
    /// variant returns `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::Error;
    ///
    /// let err = Error::InitFailed {
    ///     detail: "prefix of 40000 tokens exceeds the context".to_owned(),
    /// };
    /// assert_eq!(err.detail(), Some("prefix of 40000 tokens exceeds the context"));
    /// assert_eq!(Error::EngineBusy.detail(), None);
    /// ```
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        match self {
            Error::Complete { detail }
            | Error::Transcribe { detail }
            | Error::EmbedAudio { detail }
            | Error::LoadFailed { detail, .. }
            | Error::InitFailed { detail }
            | Error::EmbedFailed { detail }
            | Error::Download { detail, .. } => Some(detail),
            _ => None,
        }
    }

    /// Builds an [`Error::Envelope`] with the response text truncated to a loggable excerpt.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::Error;
    ///
    /// let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
    /// let err = Error::envelope(&"x".repeat(1000), source);
    /// assert!(err.to_string().contains("(truncated)"));
    /// ```
    #[must_use]
    pub fn envelope(text: &str, source: serde_json::Error) -> Self {
        let excerpt = match text.char_indices().nth(ENVELOPE_EXCERPT) {
            Some((cut, _)) => format!("{}... (truncated)", &text[..cut]),
            None => text.to_owned(),
        };
        Error::Envelope {
            text: excerpt,
            source,
        }
    }
}

/// The result type every fallible function in this crate returns.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::{Error, Result};
///
/// fn refuse() -> Result<()> {
///     Err(Error::EngineBusy)
/// }
///
/// assert!(refuse().is_err());
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    // Errors cross threads with the `Needle` that produced them, and end up in `Box<dyn Error>`.
    #[test]
    fn test_error_is_send_sync_static() {
        fn assert_bounds<T: Send + Sync + 'static>() {}
        assert_bounds::<Error>();
    }

    /// How many variants [`Error`] has. Bump it with a new arm in [`variant_index`].
    const VARIANTS: usize = 26;

    /// A distinct number per variant.
    ///
    /// The `match` has no `_` arm, so a new variant does not compile until it is listed here,
    /// and [`every_message_ends_with_a_help_line`] fails until it has a sample.
    fn variant_index(error: &Error) -> usize {
        match error {
            Error::EngineBusy => 0,
            Error::WeightsAlreadyLoaded => 1,
            Error::WrongModel { .. } => 2,
            Error::UnsupportedWeights { .. } => 3,
            Error::Load => 4,
            Error::Init => 5,
            Error::Complete { .. } => 6,
            Error::Truncated { .. } => 7,
            Error::Embed => 8,
            Error::InteriorNul { .. } => 9,
            Error::SpeechModelNotLoaded => 10,
            Error::AudioTooLong { .. } => 11,
            Error::NonFiniteSample { .. } => 12,
            Error::Transcribe { .. } => 13,
            Error::EmbedAudio { .. } => 14,
            Error::UnsupportedLanguage { .. } => 15,
            Error::InvalidKeyword { .. } => 16,
            Error::Envelope { .. } => 17,
            Error::Arguments { .. } => 18,
            Error::Tools { .. } => 19,
            Error::Io(_) => 20,
            Error::Download { .. } => 21,
            Error::ChecksumMismatch { .. } => 22,
            Error::LoadFailed { .. } => 23,
            Error::InitFailed { .. } => 24,
            Error::EmbedFailed { .. } => 25,
        }
    }

    /// One value of every variant.
    fn samples() -> Vec<Error> {
        let json = || serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        vec![
            Error::EngineBusy,
            Error::WeightsAlreadyLoaded,
            Error::WrongModel {
                expected: Model::Needle,
            },
            Error::WrongModel {
                expected: Model::Whistle,
            },
            Error::UnsupportedWeights { tag: 0 },
            Error::UnsupportedWeights { tag: MAGIC },
            Error::Load,
            Error::LoadFailed {
                model: Model::Whistle,
                detail: "x".to_owned(),
            },
            Error::Init,
            Error::InitFailed {
                detail: "x".to_owned(),
            },
            Error::Complete {
                detail: "x".to_owned(),
            },
            Error::Truncated { capacity: 1 },
            Error::Embed,
            Error::EmbedFailed {
                detail: "x".to_owned(),
            },
            Error::InteriorNul { field: "input" },
            Error::InteriorNul { field: "keywords" },
            Error::SpeechModelNotLoaded,
            Error::AudioTooLong { samples: 480_001 },
            Error::NonFiniteSample { index: 7 },
            Error::Transcribe {
                detail: "x".to_owned(),
            },
            Error::EmbedAudio {
                detail: "x".to_owned(),
            },
            Error::UnsupportedLanguage {
                code: "xx".to_owned(),
            },
            Error::InvalidKeyword {
                keyword: "a\nb".to_owned(),
            },
            Error::envelope("{", json()),
            Error::Arguments { source: json() },
            Error::Tools { source: json() },
            Error::Io(std::io::Error::other("x")),
            Error::Download {
                url: "u".to_owned(),
                detail: "d".to_owned(),
            },
            Error::ChecksumMismatch {
                expected: "a".to_owned(),
                actual: "b".to_owned(),
            },
        ]
    }

    #[test]
    fn every_message_ends_with_a_help_line() {
        let samples = samples();

        let mut covered = [false; VARIANTS];
        for error in &samples {
            covered[variant_index(error)] = true;
        }
        assert!(
            covered.iter().all(|seen| *seen),
            "a variant has no sample: {covered:?}"
        );

        for error in samples {
            let message = error.to_string();
            let last = message.lines().last().unwrap_or_default();
            assert!(last.starts_with("Help: "), "no help line in: {message}");
            // What happened, then why or what to do: never a bare one-liner.
            assert!(message.lines().count() >= 2, "one line only: {message}");
        }
    }

    #[test]
    fn detail_is_the_carried_reason_and_nothing_else() {
        for error in samples() {
            let carries = matches!(
                error,
                Error::Complete { .. }
                    | Error::Transcribe { .. }
                    | Error::EmbedAudio { .. }
                    | Error::LoadFailed { .. }
                    | Error::InitFailed { .. }
                    | Error::EmbedFailed { .. }
                    | Error::Download { .. }
            );
            match error.detail() {
                Some(detail) => {
                    assert!(carries, "unexpected detail on {error:?}");
                    assert!(error.to_string().contains(detail), "{error:?}");
                }
                None => assert!(!carries, "no detail on {error:?}"),
            }
        }
    }

    #[test]
    fn the_failed_variants_carry_the_engines_words() {
        let load = Error::LoadFailed {
            model: Model::Needle,
            detail: "bad tensor table".to_owned(),
        };
        assert_eq!(load.detail(), Some("bad tensor table"));
        assert!(load.to_string().starts_with("the engine rejected the "));
        assert!(load.to_string().contains("bad tensor table"));

        let init = Error::InitFailed {
            detail: "needle_init returned -1: 40000 tokens".to_owned(),
        };
        assert_eq!(init.detail(), Some("needle_init returned -1: 40000 tokens"));

        let embed = Error::EmbedFailed {
            detail: "engine wrote 12 floats, expected 3072".to_owned(),
        };
        assert_eq!(
            embed.detail(),
            Some("engine wrote 12 floats, expected 3072")
        );
        for error in [Error::Load, Error::Init, Error::Embed] {
            assert_eq!(error.detail(), None);
        }
    }

    #[test]
    fn wrong_model_names_both_the_model_and_its_archive() {
        let message = Error::WrongModel {
            expected: Model::Whistle,
        }
        .to_string();
        assert!(message.starts_with("the weights are not a Whistle archive"));
        assert!(message.contains("whistle.cact"));
        assert!(message.contains("whistle::Weights"));
        assert!(message.contains("it was not loaded"));
        assert!(!message.contains("stays loaded"));
    }

    #[test]
    fn unsupported_weights_tells_a_bad_tag_from_a_bad_directory() {
        let tag = Error::UnsupportedWeights { tag: 0x05E1_2A83 }.to_string();
        assert!(tag.contains("0x05E12A83 is not a Needle 3 or Whistle archive"));

        let directory = Error::UnsupportedWeights { tag: MAGIC }.to_string();
        assert!(directory.contains("tensor directory"));
        assert!(!directory.contains("is not a Needle 3"));
    }

    #[test]
    fn download_help_names_both_variables() {
        let message = Error::Download {
            url: "u".to_owned(),
            detail: "d".to_owned(),
        }
        .to_string();
        assert!(message.contains("CACTUS_NEEDLE_WEIGHTS"));
        assert!(message.contains("CACTUS_WHISTLE_WEIGHTS"));
    }

    #[test]
    fn envelope_excerpt_is_bounded() {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let err = Error::envelope(&"a".repeat(4096), source);
        let Error::Envelope { text, .. } = &err else {
            panic!("expected an envelope error");
        };
        assert!(text.len() < 4096);
    }

    #[test]
    fn short_envelopes_are_kept_whole() {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let err = Error::envelope("{\"type\":\"nope\"", source);
        let Error::Envelope { text, .. } = &err else {
            panic!("expected an envelope error");
        };
        assert_eq!(text, "{\"type\":\"nope\"");
    }
}
