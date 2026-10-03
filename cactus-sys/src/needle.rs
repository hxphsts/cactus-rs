//! Raw declarations for the Needle 3 engine C API (`include/needle.h`).
//!
//! ## Overview
//!
//! Nine functions, no handles, no opaque types. The engine owns one process-global model per
//! kind: a text model ([`NEEDLE_TEXT`], Needle 3) and a speech model ([`NEEDLE_SPEECH`],
//! Whistle). [`needle_load`] reads whichever kind an archive holds and keeps the other, so both
//! can live in one process; [`needle_models`] reports which are loaded.
//!
//! The text model is configured with [`needle_init`], answers with [`needle_complete`] and is
//! rewound by [`needle_reset`]. The speech model transcribes with [`needle_transcribe`], and also
//! runs inside [`needle_complete`] when it is handed a clip instead of a string, configured by
//! [`needle_set_audio`]. [`needle_embed`] embeds either kind of input.
//!
//! The declarations are written by hand rather than generated. Nine stable entry points do not
//! justify a `bindgen` build dependency, and hand-written signatures let every function carry the
//! `# Safety` contract that was measured against the shipped archive.
//!
//! Every function that can fail returns a negative value when it does, and [`needle_last_error`]
//! then names the reason. That string is one process-global slot shared by both kinds, owned by
//! the runtime and overwritten by the next call: copy it before calling anything else.
//!
//! Audio is 16 kHz mono `f32` PCM in `[-1, 1]`, at most 30 seconds (480 000 samples) per call.
//!
//! ## Usage
//!
//! ```no_run
//! use core::ffi::{c_char, c_int, c_ulonglong};
//!
//! use cactus_sys::{NEEDLE_TEXT, needle_complete, needle_init, needle_load, needle_models};
//!
//! let weights: &[u8] = b"...contents of needle3.cact...";
//! let mut out = [0u8; 64 * 1024];
//!
//! // SAFETY: every call below happens on one thread, in order, and the engine copies the
//! // archive during `needle_load`, so `weights` needs no particular lifetime afterwards.
//! unsafe {
//!     assert_eq!(needle_load(weights.as_ptr(), weights.len() as c_ulonglong), 0);
//!     assert_ne!(needle_models() & NEEDLE_TEXT, 0);
//!     assert!(needle_init(c"You are terse.".as_ptr(), c"[]".as_ptr(), core::ptr::null()) > 0);
//!
//!     let tokens = needle_complete(
//!         c"hello".as_ptr(),
//!         core::ptr::null(),
//!         0,
//!         64,
//!         out.as_mut_ptr().cast::<c_char>(),
//!         out.len() as c_int,
//!     );
//!     assert!(tokens >= 0);
//! }
//! ```

use core::ffi::{c_char, c_float, c_int, c_uchar, c_ulonglong};

/// Commit of the `Cactus-Compute/needle3` Hugging Face repository this crate is pinned to.
///
/// It is the `commit` field of `cactus-sys/prebuilt.toml`, the same revision the
/// `download-binaries` feature fetches archives from, threaded through by the build script so
/// that the pin has exactly one source of truth.
///
/// # Examples
///
/// ```
/// assert_eq!(cactus_sys::NEEDLE_ENGINE_COMMIT.len(), 40);
/// ```
pub const NEEDLE_ENGINE_COMMIT: &str = env!("NEEDLE_ENGINE_COMMIT");

/// Bit set in [`needle_models`] when a text model (Needle 3) is loaded.
pub const NEEDLE_TEXT: c_int = 1;

/// Bit set in [`needle_models`] when a speech model (Whistle) is loaded.
pub const NEEDLE_SPEECH: c_int = 2;

unsafe extern "C" {
    /// Configures the conversation prefix: system prompt, tool declarations and tool index.
    ///
    /// Returns the number of tokens in the resulting prefix, always greater than zero, or a
    /// negative value when it fails, most commonly because no text model has been loaded yet;
    /// [`needle_last_error`] then names the reason. Calling it again replaces the previous
    /// configuration.
    ///
    /// Tool declarations are flat objects (`{name, description, parameters, triggers?}`), not the
    /// OpenAI-style nested form. **The engine does not validate `tools_json`**: malformed JSON
    /// still returns a success value, and the damage only shows up later as a model that ignores
    /// the tools. Validate the document before handing it over.
    ///
    /// # Safety
    ///
    /// - `system_prompt`, `tools_json` and `tool_index_path` must each be either null or a
    ///   pointer to a NUL-terminated C string that stays valid for the duration of the call.
    ///   `system_prompt` and `tool_index_path` are documented as optional; passing null for
    ///   `tools_json` declares no tools.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model. No
    ///   call in this module may run concurrently with any other; the caller must serialise
    ///   all of them.
    /// - A text model must have been loaded with [`needle_load`] first, otherwise this call
    ///   fails. Loading a text archive again, even the same one, discards the configuration:
    ///   call this again afterwards.
    pub fn needle_init(
        system_prompt: *const c_char,
        tools_json: *const c_char,
        tool_index_path: *const c_char,
    ) -> c_int;

    /// Runs one completion on text or on a speech clip and writes a JSON envelope into `out`.
    ///
    /// Exactly one of `input` and `pcm` is non-null. Given `pcm`, the engine transcribes the
    /// clip with the speech model, as configured by [`needle_set_audio`], answers the
    /// transcript with the text model, and merges the speech fields into the same JSON object
    /// under an `audio_` prefix (`audio_text`, `audio_language`, ...).
    ///
    /// Returns a non-negative **token count** on success (not the number of bytes written) and
    /// a negative value on failure, with the reason in [`needle_last_error`]. On failure `out`
    /// also holds a NUL-terminated error envelope carrying the same text, such as
    /// `{"type":"error","error":"needle_init not called"}`. A failed audio completion may still
    /// carry the `audio_*` fields of a transcription that ran before the failure.
    ///
    /// `out` is always NUL-terminated, and the envelope is **silently truncated** to
    /// `out_capacity - 1` bytes when it does not fit: there is no error and no length hint, so a
    /// short buffer surfaces as JSON that fails to parse. An `out_capacity` of `0` is tolerated
    /// and writes nothing. Size the buffer generously; 64 KiB is comfortable for tool calls, and
    /// an audio completion with word timestamps needs room for every word on top.
    ///
    /// # Safety
    ///
    /// - Exactly one of `input` and `pcm` must be non-null.
    /// - `input` must be null or a pointer to a NUL-terminated C string valid for the call.
    /// - `pcm` must be null, or valid for reads of `samples` `f32` values for the duration of
    ///   the call, holding 16 kHz mono audio in `[-1, 1]`. `samples` must not be negative and
    ///   must not exceed 480 000 (30 seconds); with `pcm` null it is ignored and `0` is
    ///   conventional.
    /// - `out` must be valid for writes of `out_capacity` bytes, and `out_capacity` must not be
    ///   negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    /// - A text model must be loaded and [`needle_init`] must have succeeded first, and a
    ///   speech model must also be loaded when `pcm` is non-null; otherwise this fails.
    pub fn needle_complete(
        input: *const c_char,
        pcm: *const c_float,
        samples: c_int,
        max_new_tokens: c_int,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Configures the transcription that [`needle_complete`] runs for itself on a clip.
    ///
    /// `language`, `keywords` and `word_timestamps` mean what they mean in
    /// [`needle_transcribe`]. The setting is process-global and stays in force until the next
    /// call; the defaults are to detect the language, use no keywords and report no word times.
    /// The engine **copies** both strings, so they may be freed as soon as this returns. It
    /// cannot fail and returns nothing. The setting survives [`needle_reset`] and
    /// [`needle_init`], and [`needle_transcribe`] ignores it: that function takes its own.
    ///
    /// # Safety
    ///
    /// - `language` and `keywords` must each be either null or a pointer to a NUL-terminated C
    ///   string valid for the duration of the call.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn needle_set_audio(
        language: *const c_char,
        keywords: *const c_char,
        word_timestamps: c_int,
    );

    /// Transcribes one speech clip and writes a JSON envelope into `out`.
    ///
    /// Returns the number of tokens generated on success and a negative value on failure, with
    /// the reason in [`needle_last_error`]. The envelope has the shape
    /// `{"text":"...","language":"en","ttft_ms":0.0,"decode_tps":0.0}`; a non-zero
    /// `word_timestamps` adds `"words":[{"word":"...","start":0.00,"end":0.00,"probability":0.000}]`
    /// with times in seconds. Silence and steady noise give an empty `text` and `language`.
    ///
    /// `language` is one of `"en"`, `"de"`, `"fr"`, `"es"`, `"it"`, `"nl"` or `"pl"`, or null to
    /// detect it. Codes are case-sensitive, the empty string also means detect, and any other
    /// code fails with `unknown language <code>`. Forcing a language steers the decoder and
    /// labels the result; it does not translate. `keywords` is null or newline-separated words
    /// and phrases to bias the decoder towards. An empty clip (`samples` 0, `pcm` non-null) is
    /// not an error: it gives the silence envelope.
    ///
    /// `out` follows the rules of [`needle_complete`]: always NUL-terminated and **silently
    /// truncated** to `out_capacity - 1` bytes when it does not fit, with the token count still
    /// returned. Unlike [`needle_complete`], an `out_capacity` of `0` or a null `out` fails with
    /// "no output buffer". A failed call writes no envelope; the reason is only in
    /// [`needle_last_error`].
    ///
    /// # Safety
    ///
    /// - `pcm` must be valid for reads of `samples` `f32` values for the duration of the call,
    ///   holding 16 kHz mono audio in `[-1, 1]`. `samples` must not be negative and must not
    ///   exceed 480 000 (30 seconds).
    /// - `language` and `keywords` must each be either null or a pointer to a NUL-terminated C
    ///   string valid for the duration of the call.
    /// - `out` must be null or valid for writes of `out_capacity` bytes, and `out_capacity`
    ///   must not be negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    /// - A speech model must have been loaded with [`needle_load`] first, otherwise this call
    ///   fails.
    pub fn needle_transcribe(
        pcm: *const c_float,
        samples: c_int,
        language: *const c_char,
        keywords: *const c_char,
        word_timestamps: c_int,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Embeds text or a speech clip, or reports how many floats that takes.
    ///
    /// Exactly one of `input` and `pcm` is non-null. Text yields one vector of the model's
    /// embedding dimension (3072 for Needle 3, and asking with an empty string returns it
    /// without computing). Speech yields one row of the speech model's width (512 for the
    /// pinned Whistle, not the text dimension) per 80 ms frame, flattened, with
    /// `frames = max(1, (samples + 1120) / 1280)`: an empty clip still gives one row. With
    /// `out` null, returns that float count without computing anything. With a buffer, returns
    /// the same count after filling it, or a negative value when `out_capacity` is too small or
    /// the call fails, with the reason in
    /// [`needle_last_error`]. Query the count first and size the buffer from the answer.
    ///
    /// # Safety
    ///
    /// - Exactly one of `input` and `pcm` must be non-null.
    /// - `input` must be null or a pointer to a NUL-terminated C string valid for the call.
    /// - `pcm` must be null, or valid for reads of `samples` `f32` values for the duration of
    ///   the call, holding 16 kHz mono audio in `[-1, 1]`. `samples` must not be negative and
    ///   must not exceed 480 000 (30 seconds); with `pcm` null it is ignored and `0` is
    ///   conventional.
    /// - `out` must be null, or valid for writes of `out_capacity` `f32` values; `out_capacity`
    ///   must not be negative or exceed the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    /// - A model of the input's kind must have been loaded with [`needle_load`] first.
    pub fn needle_embed(
        input: *const c_char,
        pcm: *const c_float,
        samples: c_int,
        out: *mut c_float,
        out_capacity: c_int,
    ) -> c_int;

    /// Rewinds the conversation to the prefix that [`needle_init`] built.
    ///
    /// Tool declarations and the system prompt survive; only the turns taken since are dropped.
    /// It cannot fail and it does not unload weights. Nothing in this API does.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module. In particular this must not run while
    ///   any other call, of either kind, is in flight.
    pub fn needle_reset();

    /// Loads a `.cact` weight archive from memory.
    ///
    /// The engine reads whichever kind of model the archive holds, text or speech, and keeps a
    /// model of the other kind already loaded, so a process with both takes one call per
    /// archive. Both kinds carry the same magic tag: check [`needle_models`] afterwards to learn
    /// which one was loaded.
    ///
    /// Returns `0` on success and a negative value on failure, including for a null pointer or
    /// bytes that are not a valid archive, with the reason in [`needle_last_error`]. The engine
    /// **copies** what it needs during the call, so the buffer may be freed as soon as it
    /// returns.
    ///
    /// Weights cannot be unloaded. Loading a text archive again, even the one already loaded,
    /// discards the [`needle_init`] configuration; the speech model has none to lose.
    ///
    /// # Safety
    ///
    /// - `cact` must be null, or valid for reads of `n` bytes for the duration of the call.
    /// - `n` must be the length of that buffer in bytes.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn needle_load(cact: *const c_uchar, n: c_ulonglong) -> c_int;

    /// Reports which kinds of model this process has loaded.
    ///
    /// Returns a bitmask of [`NEEDLE_TEXT`] and [`NEEDLE_SPEECH`]: `0` before any successful
    /// [`needle_load`], and never fewer bits afterwards, since nothing unloads weights. It cannot
    /// fail.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn needle_models() -> c_int;

    /// Returns the reason the last failing call failed.
    ///
    /// The result is a NUL-terminated C string **owned by the runtime**. It is one
    /// process-global slot shared by both kinds of model, and it is valid only until the next
    /// call into this module, which may overwrite or free it: copy it out before calling
    /// anything else, `needle_last_error` included.
    ///
    /// # Safety
    ///
    /// - The returned pointer may be null; check before dereferencing it.
    /// - It must not be written through, freed, or read after the next call into this module.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module, and must copy the string before
    ///   releasing whatever serialises them.
    pub fn needle_last_error() -> *const c_char;
}
