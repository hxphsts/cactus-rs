//! The engine itself: acquiring it, configuring it, and driving one conversation.
//!
//! ## Overview
//!
//! The C API has no handles. There is one model per process, it is not thread-safe, and its
//! weights can never be unloaded. [`Needle`] is that singleton expressed as a Rust value:
//!
//! - **One per process.** [`NeedleBuilder::build`] takes the engine and returns
//!   [`Error::EngineBusy`] while another [`Needle`] is alive. Dropping one frees the slot.
//!   A second `build()` after that succeeds, and starts from a fresh conversation.
//! - **Weights are permanent.** The first successful load owns the process. Building again from
//!   the same archive reuses it; a different archive returns [`Error::WeightsAlreadyLoaded`].
//! - **Calls do not overlap.** Every method takes `&mut self`, which is what makes the engine's
//!   lack of thread safety a compile error rather than a crash. [`Needle`] is [`Send`], so it can
//!   move to another thread, and not [`Sync`], so it cannot be shared with one.
//! - **No streaming.** [`Needle::complete`] returns when the turn is finished. There is no token
//!   callback in the C API to expose.
//! - **The conversation accumulates.** Each `complete` appends a turn, so later turns see earlier
//!   ones until [`Needle::reset`] rewinds to the prefix that [`NeedleBuilder::build`] set up.
//!
//! This is the only module in the crate with `unsafe` in it.
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

use std::ffi::{CString, c_char, c_int, c_ulonglong};
use std::fmt;
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use super::completion::Completion;
use super::tool::Tool;
use super::weights::Weights;
use crate::error::{Error, Result};

/// Whether a [`Needle`] is alive in this process.
static ENGINE_TAKEN: AtomicBool = AtomicBool::new(false);

/// Fingerprint of the weights the engine loaded, set once and never cleared.
static LOADED: OnceLock<u64> = OnceLock::new();

/// Default token budget for one turn.
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;

/// Floor on the output buffer. Tool-call envelopes measured at a few hundred bytes; this leaves
/// room for a turn that calls many tools at once without ever growing the buffer.
const MIN_OUTPUT_CAPACITY: usize = 64 * 1024;

/// Bytes of envelope kept when the engine reports a failure in prose rather than JSON.
const DETAIL_EXCERPT: usize = 200;

/// Exclusive claim on the process-global engine, released on drop.
///
/// It exists so that an error between taking the engine and building the [`Needle`] still frees
/// the slot: the guard is dropped on the way out of `build()` whichever path is taken.
struct Slot;

impl Slot {
    /// Takes the engine, or reports that someone else has it.
    fn take() -> Result<Self> {
        match ENGINE_TAKEN.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => Ok(Slot),
            Err(_) => Err(Error::EngineBusy),
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        ENGINE_TAKEN.store(false, Ordering::Release);
    }
}

/// Word-wise FNV-1a over the archive, enough to tell two weight files apart.
///
/// A 64-bit non-cryptographic hash, so this is not a security check (the archive has already
/// been verified by whoever produced it),
/// only a way to answer "are these the bytes already loaded?" without keeping 35 MB alive or
/// depending on a hash crate the `download` feature happens to pull in.
fn fingerprint(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    // Eight bytes per round rather than one: every `build()` pays for this pass over a 35 MB
    // archive, and byte-at-a-time FNV made it the slowest step of building a `Needle`.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for word in &mut words {
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(word);
        hash = (hash ^ u64::from_le_bytes(buffer)).wrapping_mul(PRIME);
    }
    for byte in words.remainder() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(PRIME);
    }
    hash
}

/// Loads `weights` unless the same archive is already in the engine.
///
/// The caller must hold the [`Slot`], which is what makes the load non-concurrent.
fn load_once(slot: &Slot, weights: &Weights) -> Result<()> {
    let _ = slot;
    let wanted = fingerprint(weights.as_bytes());

    if let Some(loaded) = LOADED.get() {
        return if *loaded == wanted {
            Ok(())
        } else {
            Err(Error::WeightsAlreadyLoaded)
        };
    }

    let bytes = weights.as_bytes();
    // SAFETY: `bytes` is valid for reads of `bytes.len()` bytes for the whole call, and the
    // caller holds the engine slot, so no other engine call can be in flight. The engine copies
    // what it needs before returning, so the archive may be dropped afterwards.
    let rc = unsafe { cactus_sys::needle_load(bytes.as_ptr(), bytes.len() as c_ulonglong) };
    if rc < 0 {
        return Err(Error::Load);
    }

    // Only the slot holder reaches this line, so the set cannot lose a race.
    let _ = LOADED.set(wanted);
    Ok(())
}

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
/// let needle = Needle::builder(Weights::fetch()?)
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
    /// let needle = Needle::builder(Weights::fetch()?)
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
    /// let needle = Needle::builder(Weights::fetch()?)
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
    /// let needle = Needle::builder(Weights::fetch()?).tools(tools).build()?;
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
    /// let needle = Needle::builder(Weights::fetch()?)
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
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
    /// assert!(needle.prefix_tokens() > 0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::EngineBusy`] when another [`Needle`] is alive,
    /// [`Error::WeightsAlreadyLoaded`] when the process already loaded a different archive,
    /// [`Error::Load`] when the engine rejects the archive, [`Error::Tools`] when the tool
    /// declarations cannot be serialised, [`Error::InteriorNul`] when the
    /// system prompt, the tool declarations or the tool index path contain a NUL byte, and
    /// [`Error::Init`] when the engine cannot build the prefix.
    pub fn build(self) -> Result<Needle> {
        // Taken first: the guard releases the engine on every path out of this function.
        let slot = Slot::take()?;
        load_once(&slot, &self.weights)?;
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

        let system_ptr = system.as_ref().map_or(ptr::null(), |text| text.as_ptr());
        let index_ptr = tool_index
            .as_ref()
            .map_or(ptr::null(), |text| text.as_ptr());

        // SAFETY: the three pointers are either null or point at NUL-terminated C strings that
        // outlive the call, weights are loaded, and the slot guard means no other engine call is
        // in flight.
        let rc = unsafe { cactus_sys::needle_init(system_ptr, tools.as_ptr(), index_ptr) };
        if rc <= 0 {
            return Err(Error::Init);
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
/// let mut needle = Needle::builder(Weights::fetch()?).build()?;
/// let completion = needle.complete("hello")?;
/// assert!(completion.kind().is_respond() || completion.kind().is_call());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
///
/// A `Needle` is [`Send`] and [`Sync`]. Neither makes the engine concurrent: every method that
/// reaches it takes `&mut self`, so calls are serialised by the borrow checker, or by whatever lock
/// the value is shared behind. A shared `&Needle` can read [`Needle::prefix_tokens`] and nothing
/// else.
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

// `Needle` is `Send + Sync` by auto trait, and both rest on one rule that the compiler cannot
// check: EVERY METHOD THAT CALLS INTO `cactus_sys` TAKES `&mut self`. The header's only requirement
// is that engine calls never overlap; `&mut self` makes overlap impossible however the value is
// moved or shared, and a `&self` method that called the engine would turn `Sync` into a data race.
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
    /// let builder = Needle::builder(Weights::fetch()?);
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
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
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
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
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
        let capacity = options.output_capacity();

        self.out.resize(capacity, 0);
        // The engine NUL-terminates what it writes; clearing the first byte means a call that
        // writes nothing at all reads back as empty rather than as the previous turn.
        self.out[0] = 0;

        let budget = c_int::try_from(options.max_new_tokens()).unwrap_or(c_int::MAX);
        let out_capacity = c_int::try_from(capacity).unwrap_or(c_int::MAX);

        // SAFETY: `input` is a NUL-terminated C string that outlives the call, `self.out` is
        // valid for writes of `capacity` bytes and `out_capacity` is that same length as an
        // `int`. `&mut self` means no other engine call is in flight, and
        // `build()` proved the weights are loaded and the prefix initialised.
        let rc = unsafe {
            cactus_sys::needle_complete(
                input.as_ptr(),
                budget,
                self.out.as_mut_ptr().cast::<c_char>(),
                out_capacity,
            )
        };

        let end = self
            .out
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(capacity);
        let text = String::from_utf8_lossy(&self.out[..end]);

        if rc < 0 {
            return Err(Error::Complete {
                detail: failure_detail(&text),
            });
        }

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
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
    /// let vector = needle.embed("kitchen")?;
    /// assert_eq!(vector.len(), needle.embedding_dimension()?);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InteriorNul`] when the input holds a NUL byte, and [`Error::Embed`] when
    /// the engine refuses the call.
    pub fn embed<S>(&mut self, input: S) -> Result<Vec<f32>>
    where
        S: AsRef<str>,
    {
        let dimension = self.embedding_dimension()?;
        let input = c_string(input.as_ref(), "input")?;

        let mut vector = vec![0.0_f32; dimension];
        let capacity = c_int::try_from(dimension).unwrap_or(c_int::MAX);

        // SAFETY: `input` is a NUL-terminated C string that outlives the call, and `vector` is
        // valid for writes of `dimension` floats, which is what `capacity` says. `&mut self`
        // serialises this against every other engine call.
        let rc = unsafe { cactus_sys::needle_embed(input.as_ptr(), vector.as_mut_ptr(), capacity) };
        // Success is the dimension coming back; anything else means the vector was not filled.
        if usize::try_from(rc) != Ok(dimension) {
            return Err(Error::Embed);
        }

        Ok(vector)
    }

    /// The width of the vectors [`Needle::embed`] returns: 3072 for Needle 3.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::{Needle, Weights};
    ///
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
    /// assert_eq!(needle.embedding_dimension()?, 3072);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the engine does not report a dimension.
    pub fn embedding_dimension(&mut self) -> Result<usize> {
        if let Some(dimension) = self.dimension {
            return Ok(dimension);
        }

        // SAFETY: a null buffer with capacity zero is the documented way to ask for the
        // dimension without computing anything, and `&mut self` serialises the call.
        let rc = unsafe { cactus_sys::needle_embed(ptr::null(), ptr::null_mut(), 0) };
        let dimension = usize::try_from(rc).map_err(|_| Error::Embed)?;
        if dimension == 0 {
            return Err(Error::Embed);
        }

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
    /// let mut needle = Needle::builder(Weights::fetch()?).build()?;
    /// needle.complete("turn the kitchen light on")?;
    /// needle.reset(); // the next turn does not see the kitchen
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    pub fn reset(&mut self) {
        // SAFETY: `&mut self` means no other engine call is in flight, which is this function's
        // only requirement.
        unsafe { cactus_sys::needle_reset() };
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
    /// let needle = Needle::builder(Weights::fetch()?).system("Be terse.").build()?;
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
        // SAFETY: `&mut self` in `drop` means no other engine call is in flight. The slot is
        // released afterwards, when the `Slot` field drops, so nothing can take the engine in
        // between.
        unsafe { cactus_sys::needle_reset() };
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

    // Sound only while every engine call takes `&mut self`; see the note above `impl Needle`.
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
    fn different_archives_fingerprint_differently() {
        assert_ne!(fingerprint(b"needle-a"), fingerprint(b"needle-b"));
        assert_ne!(fingerprint(b"needle"), fingerprint(b"needle\0"));
        assert_eq!(fingerprint(b"needle"), fingerprint(b"needle"));
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
