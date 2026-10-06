//! The engine itself: acquiring it, configuring it, and driving one conversation.
//!
//! ## Overview
//!
//! The C API has no handles. There is one text model per process, the runtime is not
//! thread-safe, and its weights can never be unloaded. [`Needle`] is that singleton expressed as
//! a Rust value:
//!
//! - **One per process.** [`NeedleBuilder::build`] takes the engine and returns
//!   [`Error::EngineBusy`] while another [`Needle`] is alive. Dropping one frees the slot.
//!   A second `build()` after that succeeds, and starts from a fresh conversation.
//! - **Weights are permanent.** The first successful load owns the process. Building again from
//!   the same archive reuses it; a different archive returns [`Error::WeightsAlreadyLoaded`].
//! - **Calls do not overlap.** Every engine call takes one process-wide lock, so a [`Needle`]
//!   and a Whistle on different threads never overlap. Methods that reach the engine also take
//!   `&mut self`, because a conversation is one sequence of turns. [`Needle`] is [`Send`] and
//!   [`Sync`]: it can move to another thread, and a shared `&Needle` can only read
//!   [`Needle::prefix_tokens`].
//! - **No streaming.** [`Needle::complete`] returns when the turn is finished. There is no token
//!   callback in the C API to expose.
//! - **The conversation accumulates.** Each `complete` appends a turn, so later turns see earlier
//!   ones until [`Needle::reset`] rewinds to the prefix that [`NeedleBuilder::build`] set up.
//!
//! This module has no `unsafe` in it: all of the crate's lives in its private FFI layer, which
//! also holds the lock.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::needle::{CompleteOptions, Needle, Tool, Weights};
//! use serde_json::json;
//!
//! let mut needle = Needle::builder(Weights::from_file("needle3.cact")?)
//!     .system("You control the lights.")
//!     .tool(Tool::new(
//!         "set_light",
//!         "Turn a room light on or off",
//!         json!({ "type": "object", "properties": { "room": { "type": "string" } } }),
//!     ))
//!     .build()?;
//!
//! let completion = needle.complete("turn the kitchen light on")?;
//! for call in completion.calls() {
//!     println!("{} {}", call.name(), call.arguments_json());
//! }
//!
//! // A longer answer needs a bigger budget; the output buffer is sized from it.
//! let long = needle.complete_with_options("and the hall", CompleteOptions::new().with_max_new_tokens(512))?;
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::ffi::{CString, c_int};
use std::fmt;
use std::path::PathBuf;

use super::completion::Completion;
use super::tool::Tool;
use super::weights::Weights;
use crate::error::{Error, Result};
use crate::ffi::{self, EncodedAudio, Input, Kind, Slot};
use crate::whistle::TranscribeOptions;

/// Default token budget for one turn.
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;

/// Floor on the output buffer. Tool-call envelopes measured at a few hundred bytes; this leaves
/// room for a turn that calls many tools at once without ever growing the buffer.
const MIN_OUTPUT_CAPACITY: usize = 64 * 1024;

/// Extra output room for an audio turn: the transcript and, when asked for, every word with its
/// timing ride in the same envelope as the calls.
const AUDIO_HEADROOM: usize = 64 * 1024;

/// Bytes of envelope kept when the engine reports a failure in prose rather than JSON.
const DETAIL_EXCERPT: usize = 200;

/// Turns a Rust string into a C string, naming the field when it holds an interior NUL.
fn c_string(text: &str, field: &'static str) -> Result<CString> {
    CString::new(text).map_err(|_| Error::InteriorNul { field })
}

/// How much room a turn is given to answer.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::CompleteOptions;
///
/// assert_eq!(CompleteOptions::new().max_new_tokens(), 256);
/// assert_eq!(
///     CompleteOptions::new().with_max_new_tokens(512).max_new_tokens(),
///     512
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompleteOptions {
    max_new_tokens: u32,
}

impl Default for CompleteOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl CompleteOptions {
    /// The defaults: 256 new tokens, which covers a turn of several tool calls.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::CompleteOptions;
    ///
    /// assert_eq!(CompleteOptions::new(), CompleteOptions::default());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        CompleteOptions {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }

    /// Sets the token budget for the turn.
    ///
    /// The budget also sizes the output buffer, so raising it costs memory as well as time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::CompleteOptions;
    ///
    /// let options = CompleteOptions::new().with_max_new_tokens(1024);
    /// assert_eq!(options.max_new_tokens(), 1024);
    /// ```
    #[must_use]
    pub const fn with_max_new_tokens(mut self, max_new_tokens: u32) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// The token budget for the turn.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::CompleteOptions;
    ///
    /// assert_eq!(CompleteOptions::default().max_new_tokens(), 256);
    /// ```
    #[inline]
    #[must_use]
    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    /// The output buffer size this budget needs, in bytes.
    ///
    /// The engine truncates its envelope silently, so the buffer is sized from the budget rather
    /// than grown on demand: 64 bytes per token is well above what any observed envelope used,
    /// with a 64 KiB floor and an `i32::MAX` ceiling because the C API takes an `int`.
    fn output_capacity(self) -> usize {
        // Saturating throughout: on the 32-bit targets upstream ships, a large budget times 64
        // does not fit in a `usize`.
        let per_token = (self.max_new_tokens as usize).saturating_mul(64);
        let wanted = 4096_usize.saturating_add(per_token);
        wanted.max(MIN_OUTPUT_CAPACITY).min(i32::MAX as usize)
    }
}

/// Collects the conversation prefix, then takes the engine.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::needle::{Needle, Tool, Weights};
/// use serde_json::json;
///
/// let needle = Needle::builder(Weights::from_file("needle3.cact")?)
///     .system("You control the lights.")
///     .tools([
///         Tool::new("set_light", "Turn a light on or off", json!({ "type": "object" })),
///         Tool::new("dim_light", "Dim a light", json!({ "type": "object" })),
///     ])
///     .build()?;
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct NeedleBuilder {
    weights: Weights,
    system: Option<String>,
    tools: Vec<Tool>,
    tool_index: Option<PathBuf>,
}

impl NeedleBuilder {
    /// Sets the system prompt.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let needle = Needle::builder(Weights::from_file("needle3.cact")?)
    ///     .system("You control the lights.")
    ///     .build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn system<S>(mut self, system: S) -> Self
    where
        S: Into<String>,
    {
        self.system = Some(system.into());
        self
    }

    /// Declares one tool.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Tool, Weights};
    /// use serde_json::json;
    ///
    /// let needle = Needle::builder(Weights::from_file("needle3.cact")?)
    ///     .tool(Tool::new("ping", "Check a host", json!({ "type": "object" })))
    ///     .build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    /// Declares several tools at once.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Tool, Weights};
    /// use serde_json::json;
    ///
    /// let tools = vec![Tool::new("ping", "Check a host", json!({ "type": "object" }))];
    /// let needle = Needle::builder(Weights::from_file("needle3.cact")?).tools(tools).build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn tools<I>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
    {
        self.tools.extend(tools);
        self
    }

    /// Points the engine at a tool index file it built earlier.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let needle = Needle::builder(Weights::from_file("needle3.cact")?)
    ///     .tool_index("/var/lib/lights/tools.index")
    ///     .build()?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn tool_index<P>(mut self, path: P) -> Self
    where
        P: Into<PathBuf>,
    {
        self.tool_index = Some(path.into());
        self
    }

    /// Takes the engine, loads the weights if they are not loaded, and builds the prefix.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// assert!(needle.prefix_tokens() > 0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::EngineBusy`] when another [`Needle`] is alive,
    /// [`Error::WeightsAlreadyLoaded`] when the process already loaded a different archive,
    /// [`Error::LoadFailed`] when the engine rejects the archive, [`Error::Tools`] when the tool
    /// declarations cannot be serialised, [`Error::InteriorNul`] when the
    /// system prompt, the tool declarations or the tool index path contain a NUL byte, and
    /// [`Error::InitFailed`] with the engine's reason when it cannot build the prefix, such as a
    /// prefix too long for the context window.
    pub fn build(self) -> Result<Needle> {
        // Taken first: the slot releases the engine on every path out of this function. It is
        // declared before the guard so that it drops after it: `Slot`'s `Drop` takes the lock.
        let slot = Slot::take(Kind::Text)?;
        let mut guard = ffi::lock();
        slot.load(&mut guard, self.weights.as_bytes())?;
        // The engine copied the archive; 35 MB need not stay resident on our side.
        drop(self.weights);

        let system = self
            .system
            .as_deref()
            .map(|text| c_string(text, "system prompt"))
            .transpose()?;

        // Never fall back to "[]": a model that silently lost its tools answers as if it had none.
        let tools_json =
            serde_json::to_string(&self.tools).map_err(|source| Error::Tools { source })?;
        let tools = c_string(&tools_json, "tools")?;

        let tool_index = self
            .tool_index
            .as_ref()
            .map(|path| c_string(&path.to_string_lossy(), "tool index"))
            .transpose()?;

        let rc = guard.init(system.as_deref(), &tools, tool_index.as_deref());
        if rc <= 0 {
            // Copied while the guard is held: the string is valid only until the next call.
            let reason = guard.last_error();
            let detail = if reason.is_empty() {
                format!("needle_init returned {rc} and gave no reason")
            } else {
                format!("{reason} (needle_init returned {rc})")
            };
            return Err(Error::InitFailed { detail });
        }

        Ok(Needle {
            _slot: slot,
            prefix_tokens: u32::try_from(rc).unwrap_or_default(),
            out: Vec::new(),
            dimension: None,
        })
    }
}

/// The engine, held by exactly one value in the process.
///
/// Build one through [`Needle::builder`]. Dropping it rewinds the conversation and frees the
/// engine for the next [`NeedleBuilder::build`]; the weights stay loaded for the life of the
/// process, because the C API offers no way to unload them.
///
/// # Examples
///
/// ```no_run
/// use cactus_rs::needle::{Needle, Weights};
///
/// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
/// let completion = needle.complete("hello")?;
/// assert!(completion.kind().is_respond() || completion.kind().is_call());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
///
/// A `Needle` is [`Send`] and [`Sync`]. Neither makes the engine concurrent: every engine call
/// takes one process-wide lock, and every method that reaches the engine takes `&mut self`, so a
/// conversation's turns stay in order. A shared `&Needle` can read [`Needle::prefix_tokens`] and
/// nothing else.
///
/// ```rust
/// fn assert_send_sync<T: Send + Sync>() {}
/// assert_send_sync::<cactus_rs::needle::Needle>();
/// ```
pub struct Needle {
    /// Released after `Drop` rewinds the conversation.
    _slot: Slot,
    prefix_tokens: u32,
    /// Reused across turns so a conversation does not allocate 64 KiB per call.
    out: Vec<u8>,
    dimension: Option<usize>,
}

// `Needle` is `Send + Sync` by auto trait. The header's only requirement is that engine calls
// never overlap, and the crate's FFI layer enforces that itself: every C call happens under its
// one process-wide lock, whichever model or thread it comes from. `&mut self` on the methods that
// reach the engine keeps one conversation's turns in order; it is no longer what soundness rests
// on.
//
// `Send` also needs the engine to have no thread affinity. The archive references thread-local
// symbols, so that was checked rather than assumed: against the pinned macOS arm64 archive, load +
// init on one thread followed by complete, embed, reset and re-init on four other threads, after
// the first had exited, gave the same results as a single-threaded run.

impl Needle {
    /// Starts building the one [`Needle`] this process may hold.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let builder = Needle::builder(Weights::from_file("needle3.cact")?);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn builder(weights: Weights) -> NeedleBuilder {
        NeedleBuilder {
            weights,
            system: None,
            tools: Vec::new(),
            tool_index: None,
        }
    }

    /// Runs one turn with the default [`CompleteOptions`].
    ///
    /// The turn is appended to the conversation: the next `complete` sees this one until
    /// [`Needle::reset`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// let completion = needle.complete("turn the kitchen light on")?;
    ///
    /// for call in completion.calls() {
    ///     println!("{} {}", call.name(), call.arguments_json());
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// See [`Needle::complete_with_options`].
    pub fn complete<S>(&mut self, input: S) -> Result<Completion>
    where
        S: AsRef<str>,
    {
        self.complete_with_options(input, CompleteOptions::default())
    }

    /// Runs one turn with an explicit token budget.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{CompleteOptions, Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// let options = CompleteOptions::new().with_max_new_tokens(512);
    /// let completion = needle.complete_with_options("summarise the day", options)?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InteriorNul`] when the input holds a NUL byte, [`Error::Complete`] when
    /// the engine reports a failure, [`Error::Truncated`] when the answer filled the output
    /// buffer, and [`Error::Envelope`] when the response is not the JSON this crate expects.
    pub fn complete_with_options<S>(
        &mut self,
        input: S,
        options: CompleteOptions,
    ) -> Result<Completion>
    where
        S: AsRef<str>,
    {
        let input = c_string(input.as_ref(), "input")?;
        self.turn(
            Input::Text(&input),
            options.output_capacity(),
            options,
            None,
        )
    }

    /// Runs one turn on a speech clip with the default options.
    ///
    /// The engine transcribes the clip with the speech model, answers the transcript as it would
    /// a typed turn, and reports the transcript in [`Completion::audio`]. The language is
    /// detected, no keywords are used and no word timestamps are reported.
    ///
    /// The speech model must be loaded: build a [`Whistle`](crate::whistle::Whistle) once in
    /// this process first. It need not stay alive, since weights are never unloaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    /// use cactus_rs::whistle::{self, Whistle};
    ///
    /// // Loading Whistle's weights is what lets Needle hear.
    /// let whistle = Whistle::builder(whistle::Weights::from_file("whistle.cact")?).build()?;
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    ///
    /// let pcm = vec![0.0_f32; 16_000]; // 16 kHz mono, in [-1, 1]
    /// let completion = needle.complete_audio(&pcm)?;
    ///
    /// if let Some(heard) = completion.audio() {
    ///     println!("heard: {}", heard.text());
    /// }
    /// for call in completion.calls() {
    ///     println!("{} {}", call.name(), call.arguments_json());
    /// }
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// See [`Needle::complete_audio_with_options`].
    pub fn complete_audio(&mut self, pcm: &[f32]) -> Result<Completion> {
        self.complete_audio_with_options(pcm, CompleteOptions::default(), &TranscribeOptions::new())
    }

    /// Runs one turn on a speech clip with an explicit token budget and transcription options.
    ///
    /// The transcription options are handed to the engine on every call, so one turn's language
    /// or keywords never leak into the next. The output buffer gets 64 KiB on top of what the
    /// token budget needs, for the transcript and its word timestamps.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{CompleteOptions, Needle, Weights};
    /// use cactus_rs::whistle::{self, Language, TranscribeOptions, Whistle};
    ///
    /// let _whistle = Whistle::builder(whistle::Weights::from_file("whistle.cact")?).build()?;
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    ///
    /// let pcm = vec![0.0_f32; 16_000];
    /// let audio = TranscribeOptions::new()
    ///     .with_language(Language::De)
    ///     .with_word_timestamps(true);
    /// let completion = needle.complete_audio_with_options(&pcm, CompleteOptions::new(), &audio)?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] or [`Error::InteriorNul`] when a keyword cannot be
    /// passed on, [`Error::AudioTooLong`] for a clip over
    /// [`MAX_SAMPLES`](crate::whistle::MAX_SAMPLES), [`Error::NonFiniteSample`] for a NaN or an
    /// infinity, [`Error::SpeechModelNotLoaded`] when no Whistle weights are loaded,
    /// [`Error::Complete`] when the engine reports a failure, [`Error::Truncated`] when the
    /// answer filled the output buffer, and [`Error::Envelope`] when the response is not the
    /// JSON this crate expects.
    pub fn complete_audio_with_options(
        &mut self,
        pcm: &[f32],
        options: CompleteOptions,
        audio: &TranscribeOptions,
    ) -> Result<Completion> {
        let audio = audio.encode()?;
        crate::whistle::validate(pcm)?;
        let capacity = options
            .output_capacity()
            .saturating_add(AUDIO_HEADROOM)
            .min(i32::MAX as usize);
        self.turn(Input::Audio(pcm), capacity, options, Some(&audio))
    }

    /// Runs one turn of either kind into the reused output buffer.
    ///
    /// With `audio`, checks that the speech model is loaded and configures its transcription
    /// under the same lock as the completion, so no other call can change it in between.
    fn turn(
        &mut self,
        input: Input<'_>,
        capacity: usize,
        options: CompleteOptions,
        audio: Option<&EncodedAudio>,
    ) -> Result<Completion> {
        let budget = c_int::try_from(options.max_new_tokens()).unwrap_or(c_int::MAX);

        self.out.resize(capacity, 0);
        // The engine NUL-terminates what it writes; clearing the first byte means a call that
        // writes nothing at all reads back as empty rather than as the previous turn.
        self.out[0] = 0;

        let mut guard = ffi::lock();
        if let Some(audio) = audio {
            // Without the speech model the engine fails with a message about passing a file to
            // `needle_load`; the typed error says what to do from Rust instead.
            if guard.models() & Kind::Speech.bit() == 0 {
                return Err(Error::SpeechModelNotLoaded);
            }
            // The setting is process-global and survives `needle_reset` and `needle_init`, so
            // it is set on every audio turn rather than trusted from the last one.
            guard.set_audio(audio);
        }
        let rc = guard.complete(input, budget, &mut self.out)?;

        let end = self
            .out
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(capacity);
        let text = String::from_utf8_lossy(&self.out[..end]);

        if rc < 0 {
            // The envelope carries the engine's words; when it is empty, the error slot may
            // still, and it must be copied before the lock is released.
            let detail = if text.is_empty() {
                Some(guard.last_error()).filter(|reason| !reason.is_empty())
            } else {
                None
            };
            return Err(Error::Complete {
                detail: detail.unwrap_or_else(|| failure_detail(&text)),
            });
        }
        drop(guard);

        // The engine truncates to `capacity - 1` bytes and says nothing; a full buffer is the
        // only signal there is. `>=` also covers a buffer it filled without terminating.
        if end >= capacity - 1 {
            return Err(Error::Truncated { capacity });
        }

        text.parse()
    }

    /// Embeds one input into the model's vector space.
    ///
    /// The dimension is asked for once and cached, so this is a single engine call after the
    /// first.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// let vector = needle.embed("kitchen")?;
    /// assert_eq!(vector.len(), needle.embedding_dimension()?);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InteriorNul`] when the input holds a NUL byte, and [`Error::EmbedFailed`]
    /// with the engine's reason when it refuses the call.
    pub fn embed<S>(&mut self, input: S) -> Result<Vec<f32>>
    where
        S: AsRef<str>,
    {
        let dimension = self.embedding_dimension()?;
        let input = c_string(input.as_ref(), "input")?;

        let mut vector = vec![0.0_f32; dimension];
        let mut guard = ffi::lock();
        let rc = guard.embed(Input::Text(&input), &mut vector)?;
        // Success is the dimension coming back; anything else means the vector was not filled.
        if usize::try_from(rc) != Ok(dimension) {
            let detail = if rc > 0 {
                format!("engine wrote {rc} floats, expected {dimension}")
            } else {
                ffi::or_unexplained(guard.last_error(), "needle_embed", rc)
            };
            return Err(Error::EmbedFailed { detail });
        }
        drop(guard);

        Ok(vector)
    }

    /// The width of the vectors [`Needle::embed`] returns: 3072 for Needle 3.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// assert_eq!(needle.embedding_dimension()?, 3072);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmbedFailed`] with the engine's reason when it does not report a
    /// dimension.
    pub fn embedding_dimension(&mut self) -> Result<usize> {
        if let Some(dimension) = self.dimension {
            return Ok(dimension);
        }

        // A null buffer is the documented way to ask for the count without computing anything.
        // The engine needs one input, and an empty text is the cheapest one there is: measured
        // against the pinned engine, it returns 3072 in about a microsecond.
        let mut guard = ffi::lock();
        let rc = guard.embed_len(Input::Text(c""))?;
        let dimension = match usize::try_from(rc) {
            Ok(dimension) if dimension > 0 => dimension,
            _ => {
                return Err(Error::EmbedFailed {
                    detail: ffi::or_unexplained(guard.last_error(), "needle_embed", rc),
                });
            }
        };
        drop(guard);

        self.dimension = Some(dimension);
        Ok(dimension)
    }

    /// Rewinds the conversation to the prefix the builder set up.
    ///
    /// The system prompt and the tool declarations survive; the turns taken since do not.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
    /// needle.complete("turn the kitchen light on")?;
    /// needle.reset(); // the next turn does not see the kitchen
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    pub fn reset(&mut self) {
        ffi::lock().reset();
    }

    /// How many tokens the conversation prefix costs.
    ///
    /// The system prompt and tool declarations are read once, and every turn pays for them, so
    /// this is the standing cost of the configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let needle = Needle::builder(Weights::from_file("needle3.cact")?)
    ///     .system("Be terse.")
    ///     .build()?;
    /// println!("prefix: {} tokens", needle.prefix_tokens());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn prefix_tokens(&self) -> u32 {
        self.prefix_tokens
    }
}

/// Pulls the engine's own words out of a failed response, or keeps an excerpt of the text.
fn failure_detail(text: &str) -> String {
    let reported = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|envelope| {
            envelope
                .get("error")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        });

    reported.unwrap_or_else(|| match text.char_indices().nth(DETAIL_EXCERPT) {
        Some((cut, _)) => format!("{}... (truncated)", &text[..cut]),
        None if text.is_empty() => "the engine wrote nothing".to_owned(),
        None => text.to_owned(),
    })
}

/// Rewinds the conversation, then frees the engine for the next [`NeedleBuilder::build`].
impl Drop for Needle {
    fn drop(&mut self) {
        // The guard is a temporary, released at the end of this statement. The slot is released
        // afterwards, when the `Slot` field drops (taking the lock again), so nothing can take
        // the text model in between.
        ffi::lock().reset();
    }
}

/// Prints the configuration, never the conversation.
impl fmt::Debug for Needle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Needle")
            .field("prefix_tokens", &self.prefix_tokens)
            .field("embedding_dimension", &self.dimension)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sound because every engine call takes the FFI layer's lock; see the note above
    // `impl Needle`.
    #[test]
    fn test_needle_is_send_and_sync() {
        fn assert_bounds<T: Send + Sync>() {}
        assert_bounds::<Needle>();
        assert_bounds::<NeedleBuilder>();
    }

    #[test]
    fn the_buffer_floor_holds_for_small_budgets() {
        assert_eq!(
            CompleteOptions::new()
                .with_max_new_tokens(1)
                .output_capacity(),
            MIN_OUTPUT_CAPACITY
        );
    }

    #[test]
    fn the_buffer_grows_with_the_budget() {
        let capacity = CompleteOptions::new()
            .with_max_new_tokens(8192)
            .output_capacity();
        assert_eq!(capacity, 4096 + 8192 * 64);
    }

    #[test]
    fn the_buffer_never_exceeds_an_int() {
        let capacity = CompleteOptions::new()
            .with_max_new_tokens(u32::MAX)
            .output_capacity();
        assert_eq!(capacity, i32::MAX as usize);
    }

    #[test]
    fn an_audio_turn_gets_headroom_within_an_int() {
        let small = CompleteOptions::new().output_capacity() + AUDIO_HEADROOM;
        assert_eq!(small, MIN_OUTPUT_CAPACITY + 64 * 1024);

        let huge = CompleteOptions::new()
            .with_max_new_tokens(u32::MAX)
            .output_capacity()
            .saturating_add(AUDIO_HEADROOM)
            .min(i32::MAX as usize);
        assert_eq!(huge, i32::MAX as usize);
    }

    #[test]
    fn an_interior_nul_names_its_field() {
        let error = c_string("kit\0chen", "input").expect_err("a NUL is not a C string");
        assert!(matches!(error, Error::InteriorNul { field: "input" }));
    }

    #[test]
    fn a_failure_detail_prefers_the_engines_own_words() {
        let detail = failure_detail(r#"{"type":"error","error":"needle_init not called"}"#);
        assert_eq!(detail, "needle_init not called");
    }

    #[test]
    fn a_failure_detail_falls_back_to_the_text() {
        assert_eq!(failure_detail("segfault imminent"), "segfault imminent");
        assert_eq!(failure_detail(""), "the engine wrote nothing");
        assert!(failure_detail(&"x".repeat(4096)).ends_with("... (truncated)"));
    }
}
