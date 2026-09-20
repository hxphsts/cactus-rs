//! Error type for every fallible operation in this crate.
//!
//! ## Overview
//!
//! One [`Error`] enum covers the whole crate, and [`Result<T>`] is the alias every fallible
//! function returns. The variants split into four groups:
//!
//! - **Engine ownership**: [`Error::EngineBusy`], [`Error::WeightsAlreadyLoaded`]: the engine is
//!   one process-global model, so these report a rule of the C API rather than a failure
//! - **Engine calls**: [`Error::Load`], [`Error::Init`], [`Error::Complete`],
//!   [`Error::Truncated`], [`Error::Embed`], [`Error::InteriorNul`]
//! - **Encoding and decoding**: [`Error::Tools`], [`Error::Envelope`], [`Error::Arguments`]
//! - **Weights acquisition**: [`Error::UnsupportedWeights`], [`Error::Io`], [`Error::Download`],
//!   [`Error::ChecksumMismatch`]
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

/// The number of bytes of a bad engine envelope kept in [`Error::Envelope`].
///
/// Envelopes run to a few hundred bytes of JSON and a truncated head is enough to recognise
/// what came back; keeping all of it would put model output into log lines.
const ENVELOPE_EXCERPT: usize = 200;

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
    /// Another [`Needle`](crate::needle::Needle) is alive in this process.
    #[error(
        "the Needle engine is already in use by another instance in this process\n\
         The engine is one process-global model with no handles, so only one Needle can exist \
         at a time.\n\
         Help: drop the existing Needle before building another, or share it behind a mutex."
    )]
    EngineBusy,

    /// Weights were already loaded, and the new archive is a different one.
    #[error(
        "different weights are already loaded in this process\n\
         The engine copies the archive on the first load and offers no way to unload it.\n\
         Help: build every Needle from the same weights, or run the second model in its own \
         process."
    )]
    WeightsAlreadyLoaded,

    /// The archive is not a Needle 3 `.cact` file.
    #[error(
        "unsupported weights: magic tag {tag:#010X} is not a Needle 3 archive\n\
         Needle 3 archives start with the little-endian tag 0x05E12A84; 0x05E12A83 is a \
         Needle 2 archive, which the linked engine cannot read.\n\
         Help: download needle3.cact from the Cactus-Compute/needle3 repository, or call \
         Weights::fetch()."
    )]
    UnsupportedWeights {
        /// The little-endian `u32` read from the first four bytes of the archive.
        tag: u32,
    },

    /// `needle_load` rejected the archive.
    #[error(
        "the engine rejected the weight archive\n\
         The magic tag was right but the engine could not load the bytes, which usually means a \
         truncated or corrupted download.\n\
         Help: delete the cached archive and fetch it again, or verify its SHA-256."
    )]
    Load,

    /// `needle_init` failed to build the conversation prefix.
    #[error(
        "the engine could not initialise the conversation prefix\n\
         needle_init returned a non-positive token count, which happens when the weights are not \
         loaded or the tool index path does not exist.\n\
         Help: check that the tool index path exists and that the weights loaded successfully."
    )]
    Init,

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
         needle_complete truncates without reporting it, so the JSON envelope is incomplete.\n\
         Help: lower max_new_tokens, or ask for fewer tool calls in one turn."
    )]
    Truncated {
        /// The capacity of the output buffer, in bytes, that the response filled.
        capacity: usize,
    },

    /// `needle_embed` failed.
    #[error(
        "the engine failed to embed the input\n\
         needle_embed returned a negative value, which happens when no weights are loaded.\n\
         Help: build the Needle from valid weights before embedding."
    )]
    Embed,

    /// A string bound for the C API contains an interior NUL byte.
    #[error(
        "the {field} contains an interior NUL byte\n\
         The engine takes NUL-terminated C strings, so a NUL inside the text would silently cut \
         it short.\n\
         Help: strip '\\0' from the {field} before passing it in."
    )]
    InteriorNul {
        /// Which input held the NUL: `"system prompt"`, `"tools"`, `"input"` or `"tool index"`.
        field: &'static str,
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
         Help: retry, or download the archive by hand and point CACTUS_NEEDLE_WEIGHTS at it."
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

    #[test]
    fn every_message_ends_with_a_help_line() {
        let messages = [
            Error::EngineBusy.to_string(),
            Error::WeightsAlreadyLoaded.to_string(),
            Error::UnsupportedWeights { tag: 0 }.to_string(),
            Error::Load.to_string(),
            Error::Init.to_string(),
            Error::Complete {
                detail: "x".to_owned(),
            }
            .to_string(),
            Error::Truncated { capacity: 1 }.to_string(),
            Error::Embed.to_string(),
            Error::InteriorNul { field: "input" }.to_string(),
            Error::Download {
                url: "u".to_owned(),
                detail: "d".to_owned(),
            }
            .to_string(),
            Error::ChecksumMismatch {
                expected: "a".to_owned(),
                actual: "b".to_owned(),
            }
            .to_string(),
        ];

        for message in messages {
            let last = message.lines().last().unwrap_or_default();
            assert!(last.starts_with("Help: "), "no help line in: {message}");
        }
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
