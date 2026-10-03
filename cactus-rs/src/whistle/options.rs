//! What a transcription is asked to do: which language, which words to listen for, and whether
//! to time every word.
//!
//! ## Overview
//!
//! [`TranscribeOptions`] is three settings, all off by default:
//!
//! - **Language**: one of the seven [`Language`]s, or `None` to let Whistle detect it. Forcing a
//!   language steers the decoder and labels the [`Transcript`](super::Transcript); it does not
//!   translate, so an English clip forced to German still comes back in English, labelled `de`.
//! - **Keywords**: words and phrases to bias the decoder towards, such as names and product
//!   terms it would otherwise spell by ear. Each keyword is trimmed and empty ones are dropped.
//! - **Word timestamps**: every [`Word`](super::Word) with its start, end and probability.
//!
//! Options are checked when they are used, not when they are set: a keyword holding a line
//! break is [`Error::InvalidKeyword`] and one holding a NUL byte is [`Error::InteriorNul`],
//! from [`WhistleBuilder::build`](super::WhistleBuilder::build) or from the call that was given
//! them.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::whistle::{Language, TranscribeOptions};
//!
//! let options = TranscribeOptions::new()
//!     .with_language(Language::De)
//!     .with_keyword("Küche")
//!     .with_word_timestamps(true);
//!
//! assert_eq!(options.language(), Some(Language::De));
//! assert_eq!(options.keywords(), ["Küche"]);
//! assert!(options.word_timestamps());
//!
//! let language: Language = "fr".parse()?;
//! assert_eq!(language.to_string(), "fr");
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::ffi::EncodedAudio;

/// A language Whistle transcribes.
///
/// Serialises as its lowercase ISO 639-1 code, the same text as [`Language::as_str`]. The enum
/// is `#[non_exhaustive]`: matching it needs a `_` arm, and a language added by a later model is
/// not a breaking change.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::Language;
///
/// assert_eq!(Language::De.as_str(), "de");
/// assert_eq!("pl".parse::<Language>()?, Language::Pl);
/// assert!("xx".parse::<Language>().is_err());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Language {
    /// English, `en`.
    En,
    /// German, `de`.
    De,
    /// French, `fr`.
    Fr,
    /// Spanish, `es`.
    Es,
    /// Italian, `it`.
    It,
    /// Dutch, `nl`.
    Nl,
    /// Polish, `pl`.
    Pl,
}

impl Language {
    /// Every language Whistle transcribes, in the order upstream lists them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Language;
    ///
    /// assert_eq!(Language::ALL.len(), 7);
    /// assert_eq!(Language::ALL[0], Language::En);
    /// ```
    pub const ALL: [Language; 7] = [
        Language::En,
        Language::De,
        Language::Fr,
        Language::Es,
        Language::It,
        Language::Nl,
        Language::Pl,
    ];

    /// The lowercase ISO 639-1 code the engine takes and reports.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::Language;
    ///
    /// assert_eq!(Language::Nl.as_str(), "nl");
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Language::En => "en",
            Language::De => "de",
            Language::Fr => "fr",
            Language::Es => "es",
            Language::It => "it",
            Language::Nl => "nl",
            Language::Pl => "pl",
        }
    }
}

impl Language {
    /// The code as the C string the engine takes.
    const fn as_c_str(self) -> &'static CStr {
        match self {
            Language::En => c"en",
            Language::De => c"de",
            Language::Fr => c"fr",
            Language::Es => c"es",
            Language::It => c"it",
            Language::Nl => c"nl",
            Language::Pl => c"pl",
        }
    }
}

/// Writes the language's code, the same text as [`Language::as_str`].
impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parses a language code.
///
/// The engine only takes lowercase codes; this accepts any ASCII case, since the code it passes
/// on is always [`Language::as_str`].
impl FromStr for Language {
    type Err = Error;

    /// # Errors
    ///
    /// Returns [`Error::UnsupportedLanguage`] for anything but the seven codes.
    fn from_str(s: &str) -> Result<Self> {
        Language::ALL
            .into_iter()
            .find(|language| language.as_str().eq_ignore_ascii_case(s))
            .ok_or_else(|| Error::UnsupportedLanguage { code: s.to_owned() })
    }
}

/// How one clip is transcribed.
///
/// The defaults detect the language, use no keywords and leave word timestamps off, which is
/// what [`Whistle::transcribe`](super::Whistle::transcribe) uses unless the builder was given
/// other options.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{Language, TranscribeOptions};
///
/// let options = TranscribeOptions::new()
///     .with_language(Language::En)
///     .with_keywords(["Siobhan", "Krzysztof"]);
///
/// assert_eq!(options.keywords().len(), 2);
/// assert!(!options.word_timestamps());
/// assert_eq!(TranscribeOptions::new(), TranscribeOptions::default());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct TranscribeOptions {
    language: Option<Language>,
    keywords: Vec<String>,
    word_timestamps: bool,
}

impl TranscribeOptions {
    /// The defaults: detect the language, no keywords, no word timestamps.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// let options = TranscribeOptions::new();
    /// assert_eq!(options.language(), None);
    /// assert!(options.keywords().is_empty());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        TranscribeOptions {
            language: None,
            keywords: Vec::new(),
            word_timestamps: false,
        }
    }

    /// Forces the language, or with `None` goes back to detecting it.
    ///
    /// Forcing steers the decoder and sets the transcript's language; it does not translate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::{Language, TranscribeOptions};
    ///
    /// let forced = TranscribeOptions::new().with_language(Language::De);
    /// assert_eq!(forced.language(), Some(Language::De));
    ///
    /// let detected = forced.with_language(None);
    /// assert_eq!(detected.language(), None);
    /// ```
    #[must_use]
    pub fn with_language<L>(mut self, language: L) -> Self
    where
        L: Into<Option<Language>>,
    {
        self.language = language.into();
        self
    }

    /// Adds one word or phrase to bias the decoder towards.
    ///
    /// Leading and trailing whitespace is trimmed when the options are used, and a keyword that
    /// is empty after trimming is dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// let options = TranscribeOptions::new().with_keyword("Krzysztof");
    /// assert_eq!(options.keywords(), ["Krzysztof"]);
    /// ```
    #[must_use]
    pub fn with_keyword<S>(mut self, keyword: S) -> Self
    where
        S: Into<String>,
    {
        self.keywords.push(keyword.into());
        self
    }

    /// Adds several keywords at once.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// let options = TranscribeOptions::new().with_keywords(["Siobhan", "Krzysztof"]);
    /// assert_eq!(options.keywords(), ["Siobhan", "Krzysztof"]);
    /// ```
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords.extend(keywords.into_iter().map(Into::into));
        self
    }

    /// Asks for every word's start, end and probability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// let options = TranscribeOptions::new().with_word_timestamps(true);
    /// assert!(options.word_timestamps());
    /// ```
    #[must_use]
    pub const fn with_word_timestamps(mut self, word_timestamps: bool) -> Self {
        self.word_timestamps = word_timestamps;
        self
    }

    /// The forced language, or `None` when it is detected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::{Language, TranscribeOptions};
    ///
    /// let options = TranscribeOptions::new().with_language(Language::It);
    /// assert_eq!(options.language(), Some(Language::It));
    /// ```
    #[inline]
    #[must_use]
    pub const fn language(&self) -> Option<Language> {
        self.language
    }

    /// The keywords as they were given, before trimming.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// let options = TranscribeOptions::new().with_keyword(" Siobhan ");
    /// assert_eq!(options.keywords(), [" Siobhan "]);
    /// ```
    #[inline]
    #[must_use]
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }

    /// Whether word timestamps are asked for.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::whistle::TranscribeOptions;
    ///
    /// assert!(!TranscribeOptions::new().word_timestamps());
    /// ```
    #[inline]
    #[must_use]
    pub const fn word_timestamps(&self) -> bool {
        self.word_timestamps
    }

    /// The options in the shape the C API takes.
    ///
    /// Keywords are trimmed, empty ones dropped, and the rest joined with `\n`; no keywords at
    /// all is a null pointer rather than an empty string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] for a keyword with a line break inside it, and
    /// [`Error::InteriorNul`] for one with a NUL byte.
    pub(crate) fn encode(&self) -> Result<EncodedAudio> {
        let mut joined = String::new();
        for keyword in &self.keywords {
            let trimmed = keyword.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.contains(['\n', '\r']) {
                return Err(Error::InvalidKeyword {
                    keyword: keyword.clone(),
                });
            }
            if !joined.is_empty() {
                joined.push('\n');
            }
            joined.push_str(trimmed);
        }

        let keywords = if joined.is_empty() {
            None
        } else {
            Some(CString::new(joined).map_err(|_| Error::InteriorNul { field: "keywords" })?)
        };

        let language = self.language.map(|language| language.as_c_str().to_owned());

        Ok(EncodedAudio::new(language, keywords, self.word_timestamps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The keywords an encoding carries, split back apart.
    fn decoded(options: &TranscribeOptions) -> Vec<String> {
        options
            .encode()
            .expect("valid keywords encode")
            .keywords()
            .map(|joined| {
                joined
                    .to_str()
                    .expect("keywords are UTF-8")
                    .split('\n')
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn c_codes_match_the_codes() {
        for language in Language::ALL {
            assert_eq!(language.as_c_str().to_str().ok(), Some(language.as_str()));
        }
    }

    #[test]
    fn every_code_parses_back_to_its_language() {
        for language in Language::ALL {
            assert_eq!(language.as_str().parse::<Language>().ok(), Some(language));
            assert_eq!(language.to_string(), language.as_str());
        }
    }

    #[test]
    fn parsing_ignores_ascii_case() {
        assert_eq!("EN".parse::<Language>().ok(), Some(Language::En));
        assert_eq!("De".parse::<Language>().ok(), Some(Language::De));
    }

    #[test]
    fn an_unknown_code_is_named() {
        for code in ["", "xx", "eng", " en", "zh"] {
            let error = code
                .parse::<Language>()
                .expect_err("not a Whistle language");
            assert!(matches!(error, Error::UnsupportedLanguage { code: ref c } if c == code));
        }
    }

    #[test]
    fn languages_serialise_as_their_codes() {
        let wire = serde_json::to_string(&Language::ALL).expect("serialises");
        assert_eq!(wire, r#"["en","de","fr","es","it","nl","pl"]"#);
        let back: Vec<Language> = serde_json::from_str(&wire).expect("deserialises");
        assert_eq!(back, Language::ALL);
    }

    #[test]
    fn defaults_encode_to_nulls() {
        let encoded = TranscribeOptions::new().encode().expect("defaults encode");
        assert_eq!(encoded, EncodedAudio::default());
    }

    #[test]
    fn keywords_are_trimmed_and_empties_dropped() {
        let options =
            TranscribeOptions::new().with_keywords([" Siobhan ", "", "  ", "\tKrzysztof\n"]);
        assert_eq!(decoded(&options), ["Siobhan", "Krzysztof"]);

        let blank = TranscribeOptions::new().with_keywords(["", " \r\n "]);
        assert_eq!(blank.encode().expect("blank keywords").keywords(), None);
    }

    #[test]
    fn a_line_break_inside_a_keyword_is_refused() {
        for keyword in ["new\nyork", "new\ryork", " a\r\nb "] {
            let options = TranscribeOptions::new().with_keyword(keyword);
            let error = options.encode().expect_err("a line break inside");
            assert!(matches!(error, Error::InvalidKeyword { keyword: ref k } if k == keyword));
        }
    }

    #[test]
    fn a_nul_inside_a_keyword_is_refused() {
        let options = TranscribeOptions::new().with_keyword("kit\0chen");
        let error = options.encode().expect_err("a NUL inside");
        assert!(matches!(error, Error::InteriorNul { field: "keywords" }));
    }

    #[test]
    fn language_and_timestamps_are_carried() {
        let options = TranscribeOptions::new()
            .with_language(Language::Pl)
            .with_word_timestamps(true);
        let encoded = options.encode().expect("encodes");
        assert_eq!(encoded.language(), Some(c"pl"));
        assert!(encoded.word_timestamps());
    }

    proptest! {
        #[test]
        fn keywords_round_trip_through_the_newline_list(
            keywords in prop::collection::vec("[^\n\r\0]{0,12}", 0..8),
        ) {
            let options = TranscribeOptions::new().with_keywords(keywords.clone());
            let expected: Vec<String> = keywords
                .iter()
                .map(|keyword| keyword.trim())
                .filter(|keyword| !keyword.is_empty())
                .map(str::to_owned)
                .collect();
            prop_assert_eq!(decoded(&options), expected);
        }

        #[test]
        fn a_line_break_between_words_is_always_refused(
            head in "[a-zA-Z]{1,8}",
            tail in "[a-zA-Z]{1,8}",
            brk in prop::sample::select(vec!["\n", "\r", "\r\n"]),
        ) {
            let keyword = format!("{head}{brk}{tail}");
            let options = TranscribeOptions::new().with_keywords(["fine", keyword.as_str()]);
            let refused = matches!(
                options.encode(),
                Err(Error::InvalidKeyword { keyword: ref k }) if *k == keyword
            );
            prop_assert!(refused);
        }

        #[test]
        fn language_codes_round_trip(index in 0..Language::ALL.len()) {
            let language = Language::ALL[index];
            prop_assert_eq!(language.as_str().parse::<Language>().ok(), Some(language));
            let upper = language.as_str().to_uppercase();
            prop_assert_eq!(upper.parse::<Language>().ok(), Some(language));
            let wire = serde_json::to_string(&language).expect("serialises");
            prop_assert_eq!(serde_json::from_str::<Language>(&wire).ok(), Some(language));
        }
    }
}
