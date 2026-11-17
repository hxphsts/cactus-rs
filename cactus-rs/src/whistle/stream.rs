//! Live transcription: audio in as it arrives, words out as the model settles on them.
//!
//! ## Overview
//!
//! [`Whistle::stream`](super::Whistle::stream) opens a [`TranscriptStream`]. Audio is pushed in
//! pieces of any size; about once per second of audio the engine re-transcribes what it has
//! heard so far and **commits** the words that two passes in a row agree on. Each
//! [`Update`] carries the words committed since the last one, and the **pending** tail: what
//! the model currently hears after them, which may still change.
//!
//! ```text
//! push 1 s   committed: ""                    pending: "And so..."
//! push 1 s   committed: "And"                 pending: "so my fellow marriage"
//! push 1 s   committed: ""                    pending: "so, my fellow Americans!"
//! ...
//! finish     committed: "country."            pending: ""
//! ```
//!
//! Committed words never change, so they are safe to act on. The pending tail arrives a second
//! or more earlier and is usually right for a short command, which makes it useful for getting
//! ready: route it, look up the tool, and act once the same words are committed or the speaker
//! stops.
//!
//! ## One-second passes
//!
//! The engine expects about a second of audio per call. Fed 100 ms at a time it commits words
//! it has barely heard, and the transcript of the same recording comes out wrong. A
//! [`TranscriptStream`] therefore collects what is pushed and runs one pass per whole second
//! ([`PASS_SAMPLES`]); a push that does not complete a second returns an update with no new
//! words and the pending tail unchanged. [`TranscriptStream::finish`] transcribes whatever is
//! left.
//!
//! ## Between pushes
//!
//! The stream borrows its [`Whistle`], so the same `Whistle` cannot start a
//! second stream or transcribe a clip until this one is finished or dropped. The engine lock is
//! taken for each pass and released after it, so a [`Needle`](crate::needle::Needle) or a
//! [`Mirage`](crate::mirage::Mirage) can run between pushes: that is how a pending tail is
//! routed to a tool call while the audio keeps coming.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::whistle::{Weights, Whistle};
//!
//! let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
//! let microphone: Vec<Vec<f32>> = Vec::new(); // 16 kHz mono chunks, as they arrive
//!
//! let mut stream = whistle.stream()?;
//! for chunk in &microphone {
//!     let update = stream.push(chunk)?;
//!     if !update.is_empty() {
//!         println!("{} [{}]", update.text(), update.pending());
//!     }
//! }
//! println!("{}", stream.finish()?.text());
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use super::engine::Whistle;
use super::options::Language;
use super::transcript::Word;
use super::{SAMPLE_RATE, validate_finite};
use crate::error::{Error, Result};
use crate::ffi::{self, EncodedAudio};

/// How much audio one pass covers, in samples: one second at 16 kHz.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{PASS_SAMPLES, SAMPLE_RATE};
///
/// assert_eq!(PASS_SAMPLES, SAMPLE_RATE as usize);
/// ```
pub const PASS_SAMPLES: usize = SAMPLE_RATE as usize;

/// Size of the buffer one pass writes into.
///
/// A pass that commits a whole 30 second window with every word's times measured at under
/// 8 KiB. The engine does not truncate: a pass that did not fit would fail and its words would
/// be lost, so the buffer is far larger than anything measured.
const UPDATE_CAPACITY: usize = 256 * 1024;

/// What one pass, or several, added to a live transcription.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::whistle::{Language, Update};
///
/// let update: Update = r#"{"text":"And","words":[{"word":"And","start":0.0,"end":0.72,
///     "probability":0.981}],"pending":"so my fellow","language":"en","received":2.0,
///     "pass_ms":18.0}"#
///     .parse()?;
///
/// assert_eq!(update.text(), "And");
/// assert_eq!(update.words().len(), 1);
/// assert_eq!(update.pending(), "so my fellow");
/// assert_eq!(update.language(), Some(Language::En));
/// assert_eq!(update.received(), 2.0);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Update {
    #[serde(default)]
    text: String,
    #[serde(default)]
    words: Vec<Word>,
    #[serde(default)]
    pending: String,
    // Kept as the engine wrote it: empty until speech is heard.
    #[serde(default)]
    language: String,
    #[serde(default)]
    received: f32,
    #[serde(default)]
    pass_ms: f32,
}

impl Update {
    /// The words committed by this update, as text. Empty when none were.
    ///
    /// Joining the text of every update of a stream, with spaces, gives its transcript.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The words committed by this update, with their times in seconds from the start of the
    /// stream.
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// What the model currently hears after the committed words. It may change with the next
    /// pass, and is empty after [`TranscriptStream::finish`].
    #[inline]
    #[must_use]
    pub fn pending(&self) -> &str {
        &self.pending
    }

    /// The language of the stream, once speech has been heard and the code is one this crate
    /// knows.
    #[inline]
    #[must_use]
    pub fn language(&self) -> Option<Language> {
        self.language.parse().ok()
    }

    /// The language code as the engine wrote it: empty until speech is heard.
    #[inline]
    #[must_use]
    pub fn language_code(&self) -> &str {
        &self.language
    }

    /// How much audio the engine has taken in so far, in seconds.
    #[inline]
    #[must_use]
    pub const fn received(&self) -> f32 {
        self.received
    }

    /// How long the engine spent on this update's passes, in milliseconds.
    ///
    /// A pass re-transcribes the audio that is not yet committed, so it grows with a long
    /// pending tail: 12 ms after the first second of one recording and 53 ms after the eleventh.
    #[inline]
    #[must_use]
    pub const fn pass_ms(&self) -> f32 {
        self.pass_ms
    }

    /// Whether this update committed no words.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty() && self.text.is_empty()
    }

    /// Folds a later pass into this update: its words are appended, and its tail, language and
    /// position replace this one's.
    fn absorb(&mut self, later: Update) {
        if !later.text.is_empty() {
            if !self.text.is_empty() {
                self.text.push(' ');
            }
            self.text.push_str(&later.text);
        }
        self.words.extend(later.words);
        self.pending = later.pending;
        if !later.language.is_empty() {
            self.language = later.language;
        }
        self.received = later.received;
        self.pass_ms += later.pass_ms;
    }
}

/// Parses the document one streaming pass writes.
///
/// # Errors
///
/// Returns [`Error::Envelope`] when the text is not that document.
impl std::str::FromStr for Update {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|source| Error::envelope(text, source))
    }
}

/// A live transcription in progress.
///
/// Created by [`Whistle::stream`](super::Whistle::stream) or
/// [`Whistle::stream_with_options`](super::Whistle::stream_with_options). Dropping it ends the
/// stream in the engine and discards the words that were still pending; call
/// [`TranscriptStream::finish`] to get them.
pub struct TranscriptStream<'a> {
    /// Held so that the `Whistle` can do nothing else while the stream is open.
    whistle: &'a mut Whistle,
    audio: EncodedAudio,
    /// Audio pushed but not yet handed to the engine: always less than one pass.
    waiting: Vec<f32>,
    /// The tail and position of the last pass, repeated by a push that runs none.
    last: Update,
    /// Set once the engine's stream has been stopped, so `Drop` does not stop it again.
    stopped: bool,
}

impl<'a> TranscriptStream<'a> {
    /// Opens a stream on `whistle`, ending any the engine still has open.
    pub(super) fn open(whistle: &'a mut Whistle, audio: EncodedAudio) -> Result<Self> {
        // The engine has one stream. A previous one that was leaked instead of dropped would
        // otherwise continue here, with its audio in front of this one's.
        whistle.out.resize(UPDATE_CAPACITY, 0);
        let mut guard = ffi::lock();
        let rc = guard.stream_stop(&mut whistle.out);
        if rc < 0 {
            return Err(Error::Transcribe {
                detail: ffi::or_unexplained(
                    guard.last_error(),
                    "cactus_stream_transcribe_stop",
                    rc,
                ),
            });
        }
        drop(guard);

        Ok(TranscriptStream {
            whistle,
            audio,
            waiting: Vec::with_capacity(PASS_SAMPLES),
            last: Update::default(),
            stopped: false,
        })
    }

    /// Adds audio: 16 kHz mono samples in `[-1, 1]`, in pieces of any size.
    ///
    /// One pass runs for every whole second that has accumulated. The update holds the words
    /// those passes committed and the pending tail after the last of them. A push that does not
    /// complete a second runs no pass: its update has no words and repeats the last tail.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let mut stream = whistle.stream()?;
    ///
    /// let second = vec![0.0_f32; 16_000];
    /// let update = stream.push(&second)?;
    /// assert_eq!(update.received(), 1.0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::NonFiniteSample`] for a NaN or an infinity, with its index in this
    /// push, before any of the push is taken; and [`Error::Transcribe`] with the engine's
    /// reason when a pass fails. After a failed pass the stream should be dropped: the audio of
    /// that pass may or may not have been taken.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Update> {
        validate_finite(pcm)?;
        self.waiting.extend_from_slice(pcm);

        let mut update = self.idle();
        let whole = self.waiting.len() / PASS_SAMPLES * PASS_SAMPLES;
        let mut ran = false;
        for pass in self.waiting[..whole].chunks_exact(PASS_SAMPLES) {
            let later = process(&mut self.whistle.out, pass, &self.audio)?;
            if ran {
                update.absorb(later);
            } else {
                update = later;
                ran = true;
            }
        }
        self.waiting.drain(..whole);

        if ran {
            self.last = self.repeatable(&update);
        }
        Ok(update)
    }

    /// Ends the stream and returns its last words.
    ///
    /// Audio that was pushed but had not filled a second is transcribed first. Then the pending
    /// tail is committed as it stands. The update holds those words and an empty tail.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let mut stream = whistle.stream()?;
    /// stream.push(&vec![0.0_f32; 24_000])?;
    /// let last = stream.finish()?;
    /// assert_eq!(last.pending(), "");
    /// // The `Whistle` is free again.
    /// whistle.transcribe(&[])?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Transcribe`] with the engine's reason when the last pass or the stop
    /// fails.
    pub fn finish(mut self) -> Result<Update> {
        let mut update = self.idle();
        update.pending.clear();

        let rest = std::mem::take(&mut self.waiting);
        if !rest.is_empty() {
            update = process(&mut self.whistle.out, &rest, &self.audio)?;
        }

        // Marked first: whether the stop succeeds or not, there is nothing left to stop.
        self.stopped = true;
        let tail = stop(&mut self.whistle.out)?;
        update.absorb(tail);
        Ok(update)
    }

    /// How much pushed audio is waiting for the rest of its second, in samples.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::whistle::{Weights, Whistle};
    ///
    /// let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
    /// let mut stream = whistle.stream()?;
    /// stream.push(&vec![0.0_f32; 20_000])?;
    /// assert_eq!(stream.waiting(), 4_000);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// The update of a push that runs no pass: no words, the last tail, language and position.
    fn idle(&self) -> Update {
        self.last.clone()
    }

    /// What of `update` a later idle push repeats.
    fn repeatable(&self, update: &Update) -> Update {
        Update {
            text: String::new(),
            words: Vec::new(),
            pending: update.pending.clone(),
            language: update.language.clone(),
            received: update.received,
            pass_ms: 0.0,
        }
    }
}

/// Ends the engine's stream if [`TranscriptStream::finish`] did not.
impl Drop for TranscriptStream<'_> {
    fn drop(&mut self) {
        if !self.stopped {
            // The pending words are discarded with the stream; an error here has no one to go to.
            let _ = stop(&mut self.whistle.out);
        }
    }
}

/// Prints where the stream is, never the audio.
impl fmt::Debug for TranscriptStream<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TranscriptStream")
            .field("received", &self.last.received)
            .field("waiting", &self.waiting.len())
            .finish()
    }
}

/// Runs one pass over `pcm` and parses what it committed.
fn process(out: &mut Vec<u8>, pcm: &[f32], audio: &EncodedAudio) -> Result<Update> {
    out.resize(UPDATE_CAPACITY, 0);
    out[0] = 0;
    let mut guard = ffi::lock();
    let rc = guard.stream_process(pcm, audio, out)?;
    if rc < 0 {
        // Copied while the guard is held: the string is valid only until the next call.
        return Err(Error::Transcribe {
            detail: ffi::or_unexplained(guard.last_error(), "cactus_stream_transcribe_process", rc),
        });
    }
    drop(guard);
    written(out).parse()
}

/// Stops the engine's stream and parses its tail.
fn stop(out: &mut Vec<u8>) -> Result<Update> {
    out.resize(UPDATE_CAPACITY, 0);
    out[0] = 0;
    let mut guard = ffi::lock();
    let rc = guard.stream_stop(out);
    if rc < 0 {
        return Err(Error::Transcribe {
            detail: ffi::or_unexplained(guard.last_error(), "cactus_stream_transcribe_stop", rc),
        });
    }
    drop(guard);
    written(out).parse()
}

/// The NUL-terminated text the engine left in `out`.
fn written(out: &[u8]) -> String {
    let end = out.iter().position(|byte| *byte == 0).unwrap_or(out.len());
    String::from_utf8_lossy(&out[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Updates the pinned engine wrote for consecutive seconds of `jfk.wav`.
    const FIRST: &str = r#"{"text":"","words":[],"pending":"And so...","language":"en","received":1.00,"pass_ms":12.0}"#;
    const SECOND: &str = r#"{"text":"And","words":[{"word":"And","start":0.00,"end":0.72,"probability":0.981}],"pending":"so my fellow marriage","language":"en","received":2.00,"pass_ms":18.0}"#;
    const LATER: &str = r#"{"text":"country can","words":[{"word":"country","start":6.00,"end":6.40,"probability":0.998},{"word":"can","start":6.40,"end":6.72,"probability":0.997}],"pending":"do for you.","language":"en","received":8.00,"pass_ms":39.1}"#;
    const IDLE: &str =
        r#"{"text":"","words":[],"pending":"","language":"","received":0.00,"pass_ms":0.0}"#;

    #[test]
    fn the_measured_updates_parse() {
        let first: Update = FIRST.parse().expect("the engine's own output");
        assert!(first.is_empty());
        assert_eq!(first.pending(), "And so...");
        assert_eq!(first.language(), Some(Language::En));
        assert_eq!((first.received(), first.pass_ms()), (1.0, 12.0));

        let second: Update = SECOND.parse().expect("the engine's own output");
        assert!(!second.is_empty());
        assert_eq!(second.text(), "And");
        assert_eq!(second.words()[0].word(), "And");
        assert_eq!(second.words()[0].end(), 0.72);

        let idle: Update = IDLE.parse().expect("the engine's own output");
        assert!(idle.is_empty());
        assert_eq!((idle.language(), idle.language_code()), (None, ""));
    }

    #[test]
    fn absorbing_appends_words_and_replaces_the_tail() {
        let mut update: Update = SECOND.parse().expect("the engine's own output");
        update.absorb(LATER.parse().expect("the engine's own output"));
        assert_eq!(update.text(), "And country can");
        assert_eq!(update.words().len(), 3);
        assert_eq!(update.pending(), "do for you.");
        assert_eq!(update.received(), 8.0);
        assert!((update.pass_ms() - 57.1).abs() < 1e-3);

        // A pass that commits nothing adds no stray space and keeps the language.
        update.absorb(IDLE.parse().expect("the engine's own output"));
        assert_eq!(update.text(), "And country can");
        assert_eq!(update.language_code(), "en");
        assert_eq!(update.pending(), "");

        let mut empty = Update::default();
        empty.absorb(SECOND.parse().expect("the engine's own output"));
        assert_eq!(empty.text(), "And");
    }

    #[test]
    fn text_that_is_not_an_update_is_an_envelope_error() {
        for text in ["", "not json", r#"{"words":{}}"#] {
            assert!(matches!(
                text.parse::<Update>(),
                Err(Error::Envelope { .. })
            ));
        }
    }

    #[test]
    fn a_pass_is_one_second() {
        assert_eq!(PASS_SAMPLES, 16_000);
        assert_eq!(UPDATE_CAPACITY, 262_144);
    }
}
