//! The JSON envelope a transcription answers with, as Rust types.
//!
//! ## Overview
//!
//! Every [`needle_transcribe`](cactus_sys::needle_transcribe) call writes one JSON object, and
//! [`Transcript`] is that object parsed:
//!
//! ```json
//! {"text":"Turn off the kitchen lights.","language":"en","ttft_ms":52.4,"decode_tps":202.2}
//! ```
//!
//! With word timestamps on, a `"words"` array of [`Word`]s sits between `language` and
//! `ttft_ms`. Silence, steady noise and an empty clip give an empty `text` and an empty
//! `language`, and zero timings.
//!
//! The same fields come back inside an audio
//! [`Completion`](crate::needle::Completion), under an `audio_` prefix, as
//! [`Completion::audio`](crate::needle::Completion::audio).
//!
//! Every field is `#[serde(default)]` and both types are `#[non_exhaustive]`: an engine that
//! grows a field, or drops one, still parses.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::whistle::{Language, Transcript};
//!
//! let transcript: Transcript =
//!     r#"{"text":"Turn the hall light on.","language":"en","ttft_ms":23.2,"decode_tps":274.8}"#
//!         .parse()?;
//!
//! assert_eq!(transcript.text(), "Turn the hall light on.");
//! assert_eq!(transcript.language(), Some(Language::En));
//! assert!(transcript.words().is_empty());
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::options::Language;
use crate::error::{Error, Result};

/// One recognised word, with when it was said and how sure the model was.
///
/// Times are seconds from the start of the clip. The word keeps the punctuation the model
/// attached to it, so the words joined with spaces give back the transcript's text.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::Word;
///
/// let word: Word = serde_json::from_str(
///     r#"{"word":"kitchen","start":1.36,"end":1.84,"probability":0.928}"#,
/// )?;
///
/// assert_eq!(word.word(), "kitchen");
/// assert!(word.end() > word.start());
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Word {
    #[serde(default)]
    word: String,
    #[serde(default)]
    start: f32,
    #[serde(default)]
    end: f32,
    #[serde(default)]
    probability: f32,
}

impl Word {
    /// The word as the model wrote it, punctuation included.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Word;
    ///
    /// let word: Word = serde_json::from_str(r#"{"word":"lights."}"#)?;
    /// assert_eq!(word.word(), "lights.");
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn word(&self) -> &str {
        &self.word
    }

    /// When the word starts, in seconds from the start of the clip.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Word;
    ///
    /// let word: Word = serde_json::from_str(r#"{"word":"Turn","start":0.24}"#)?;
    /// assert_eq!(word.start(), 0.24);
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn start(&self) -> f32 {
        self.start
    }

    /// When the word ends, in seconds from the start of the clip.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Word;
    ///
    /// let word: Word = serde_json::from_str(r#"{"word":"Turn","end":0.56}"#)?;
    /// assert_eq!(word.end(), 0.56);
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn end(&self) -> f32 {
        self.end
    }

    /// How sure the model was of the word, from zero to one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Word;
    ///
    /// let word: Word = serde_json::from_str(r#"{"word":"Turn","probability":0.887}"#)?;
    /// assert_eq!(word.probability(), 0.887);
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn probability(&self) -> f32 {
        self.probability
    }
}

/// One clip's transcription.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{Language, Transcript};
///
/// let transcript: Transcript = r#"{"text":"Mach das Licht in der Kitche aus.",
///     "language":"de","ttft_ms":57.4,"decode_tps":182.4}"#
///     .parse()?;
///
/// assert_eq!(transcript.language(), Some(Language::De));
/// assert!(!transcript.is_empty());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Transcript {
    #[serde(default)]
    text: String,
    // Kept as the engine wrote it: empty for silence, and possibly a code this crate does not
    // know from a newer engine.
    #[serde(default)]
    language: String,
    #[serde(default)]
    words: Vec<Word>,
    #[serde(default)]
    ttft_ms: f32,
    #[serde(default)]
    decode_tps: f32,
}

impl Transcript {
    /// A transcript from its parts, for the `audio_*` fields of a completion.
    pub(crate) fn from_parts(
        text: String,
        language: String,
        words: Vec<Word>,
        ttft_ms: f32,
        decode_tps: f32,
    ) -> Self {
        Transcript {
            text,
            language,
            words,
            ttft_ms,
            decode_tps,
        }
    }

    /// What was said, as one string. Empty for silence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let transcript: Transcript = r#"{"text":"Turn off the kitchen lights."}"#.parse()?;
    /// assert_eq!(transcript.text(), "Turn off the kitchen lights.");
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The language the clip was transcribed as: the forced one, or the one detected.
    ///
    /// `None` for silence, which has no language, and for a code this crate does not know;
    /// [`Transcript::language_code`] still has that code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::{Language, Transcript};
    ///
    /// let spoken: Transcript = r#"{"text":"Hallo.","language":"de"}"#.parse()?;
    /// assert_eq!(spoken.language(), Some(Language::De));
    ///
    /// let silence: Transcript = r#"{"text":"","language":""}"#.parse()?;
    /// assert_eq!(silence.language(), None);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn language(&self) -> Option<Language> {
        self.language.parse().ok()
    }

    /// The language code exactly as the engine wrote it, empty for silence.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let transcript: Transcript = r#"{"text":"Hello.","language":"en"}"#.parse()?;
    /// assert_eq!(transcript.language_code(), "en");
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn language_code(&self) -> &str {
        &self.language
    }

    /// Every word with its timing, when word timestamps were asked for; empty otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let transcript: Transcript = r#"{"text":"On.","language":"en",
    ///     "words":[{"word":"On.","start":0.16,"end":0.4,"probability":0.85}]}"#
    ///     .parse()?;
    /// assert_eq!(transcript.words()[0].word(), "On.");
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Milliseconds until the first token was decoded, which is mostly the encoder's time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let transcript: Transcript = r#"{"text":"On.","ttft_ms":23.2}"#.parse()?;
    /// assert_eq!(transcript.ttft_ms(), 23.2);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn ttft_ms(&self) -> f32 {
        self.ttft_ms
    }

    /// Tokens per second while decoding the text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let transcript: Transcript = r#"{"text":"On.","decode_tps":274.8}"#.parse()?;
    /// assert_eq!(transcript.decode_tps(), 274.8);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn decode_tps(&self) -> f32 {
        self.decode_tps
    }

    /// Whether nothing was recognised: silence, noise or an empty clip.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Transcript;
    ///
    /// let silence: Transcript = r#"{"text":"","language":""}"#.parse()?;
    /// assert!(silence.is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Parses a transcription envelope.
impl FromStr for Transcript {
    type Err = Error;

    /// # Errors
    ///
    /// Returns [`Error::Envelope`] when the text is not the JSON this crate expects.
    fn from_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|error| Error::envelope(s, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the pinned engine answered for one second of zeros, with word timestamps on.
    const SILENCE: &str = r#"{"text":"","language":"","words":[],"ttft_ms":0.0,"decode_tps":0.0}"#;

    /// The pinned engine on `lights_en.wav`, with word timestamps on.
    const LIGHTS: &str = concat!(
        r#"{"text":"Turn off the kitchen lights.","language":"en","words":["#,
        r#"{"word":"Turn","start":0.24,"end":0.56,"probability":0.887},"#,
        r#"{"word":"off","start":0.56,"end":0.96,"probability":0.985},"#,
        r#"{"word":"the","start":0.96,"end":1.36,"probability":0.993},"#,
        r#"{"word":"kitchen","start":1.36,"end":1.84,"probability":0.928},"#,
        r#"{"word":"lights.","start":1.84,"end":2.56,"probability":0.993}],"#,
        r#""ttft_ms":53.6,"decode_tps":203.9}"#,
    );

    #[test]
    fn silence_parses_to_an_empty_transcript() {
        let transcript: Transcript = SILENCE.parse().expect("the silence envelope parses");
        assert!(transcript.is_empty());
        assert_eq!(transcript.text(), "");
        assert_eq!(transcript.language(), None);
        assert_eq!(transcript.language_code(), "");
        assert!(transcript.words().is_empty());
        assert_eq!(transcript.ttft_ms(), 0.0);
        assert_eq!(transcript.decode_tps(), 0.0);
    }

    #[test]
    fn a_timestamped_envelope_parses() {
        let transcript: Transcript = LIGHTS.parse().expect("the sample envelope parses");
        assert_eq!(transcript.text(), "Turn off the kitchen lights.");
        assert_eq!(transcript.language(), Some(Language::En));
        assert_eq!(transcript.ttft_ms(), 53.6);
        assert_eq!(transcript.decode_tps(), 203.9);

        let words = transcript.words();
        assert_eq!(words.len(), 5);
        assert_eq!(words[3].word(), "kitchen");
        assert_eq!(words[3].start(), 1.36);
        assert_eq!(words[3].end(), 1.84);
        assert_eq!(words[3].probability(), 0.928);

        let joined: Vec<&str> = words.iter().map(Word::word).collect();
        assert_eq!(joined.join(" "), transcript.text());
        assert!(
            words
                .windows(2)
                .all(|pair| pair[0].end() <= pair[1].start())
        );
    }

    #[test]
    fn an_unknown_language_keeps_its_code() {
        let transcript: Transcript = r#"{"text":"Hej.","language":"sv"}"#
            .parse()
            .expect("an unknown code is not a parse failure");
        assert_eq!(transcript.language(), None);
        assert_eq!(transcript.language_code(), "sv");
    }

    #[test]
    fn a_missing_field_is_not_a_parse_failure() {
        let transcript: Transcript = "{}".parse().expect("defaults fill in");
        assert_eq!(transcript, Transcript::default());
        assert!(transcript.is_empty());
    }

    #[test]
    fn junk_is_reported_with_its_text() {
        let error = "not json"
            .parse::<Transcript>()
            .expect_err("junk is not a transcript");
        let Error::Envelope { text, .. } = &error else {
            panic!("expected Error::Envelope");
        };
        assert_eq!(text, "not json");
    }

    #[test]
    fn a_transcript_survives_serde() {
        let transcript: Transcript = LIGHTS.parse().expect("the sample envelope parses");
        let wire = serde_json::to_string(&transcript).expect("serialises");
        let back: Transcript = wire.parse().expect("parses back");
        assert_eq!(back, transcript);
    }
}
