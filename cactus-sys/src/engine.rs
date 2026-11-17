//! Raw declarations for the Cactus engine C API (`include/cactus.h`).
//!
//! ## Overview
//!
//! Thirteen functions, no handles, no opaque types. The engine owns one process-global model per
//! kind: a text model ([`CACTUS_TEXT`], Needle 3), a speech model ([`CACTUS_SPEECH`], Whistle)
//! and a vision model ([`CACTUS_VISION`], Mirage). [`cactus_load`] reads whichever kind an
//! archive holds and keeps the others, so all can live in one process; [`cactus_models`] reports
//! which are loaded.
//!
//! The text model is configured with [`cactus_init`], answers with [`cactus_complete`] and is
//! rewound by [`cactus_reset`]. The speech model transcribes with [`cactus_transcribe`], and also
//! runs inside [`cactus_complete`] when it is handed a clip instead of a string, configured by
//! [`cactus_set_audio`]. [`cactus_embed`] embeds either kind of input.
//!
//! The vision model describes an image with [`cactus_describe`].
//!
//! [`cactus_stream_transcribe_process`] and [`cactus_stream_transcribe_stop`] transcribe audio as
//! it arrives. [`cactus_set_threads`] is declared so that this module matches the header
//! function for function; no effect of it could be measured.
//!
//! The declarations are written by hand rather than generated. Thirteen stable entry points do
//! not justify a `bindgen` build dependency, and hand-written signatures let every function
//! carry the `# Safety` contract that was measured against the shipped archive. The header
//! itself carries no comments since engine 4.0, so these doc comments are the written contract;
//! `tests/abi.rs` checks the signatures against the header.
//!
//! Every function that can fail returns a negative value when it does, and [`cactus_last_error`]
//! then names the reason. That string is one process-global slot shared by both kinds, owned by
//! the runtime and overwritten by the next call: copy it before calling anything else.
//!
//! Audio is 16 kHz mono `f32` PCM in `[-1, 1]`. Transcription takes any length; an embedding or
//! an audio completion at most 30 seconds (480 000 samples) per call.
//!
//! ## Usage
//!
//! ```no_run
//! use core::ffi::{c_char, c_int, c_ulonglong};
//!
//! use cactus_sys::{CACTUS_TEXT, cactus_complete, cactus_init, cactus_load, cactus_models};
//!
//! let weights: &[u8] = b"...contents of needle3.cact...";
//! let mut out = [0u8; 64 * 1024];
//!
//! // SAFETY: every call below happens on one thread, in order, and the engine copies the
//! // archive during `cactus_load`, so `weights` needs no particular lifetime afterwards.
//! unsafe {
//!     assert_eq!(cactus_load(weights.as_ptr(), weights.len() as c_ulonglong), 0);
//!     assert_ne!(cactus_models() & CACTUS_TEXT, 0);
//!     assert!(cactus_init(c"You are terse.".as_ptr(), c"[]".as_ptr(), core::ptr::null()) > 0);
//!
//!     let tokens = cactus_complete(
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

/// Commit of the `Cactus-Compute/cactus` Hugging Face repository this crate is pinned to.
///
/// It is the `commit` field of `cactus-sys/prebuilt.toml`, the same revision the
/// `download-binaries` feature fetches archives from, threaded through by the build script so
/// that the pin has exactly one source of truth.
///
/// # Examples
///
/// ```
/// assert_eq!(cactus_sys::CACTUS_ENGINE_COMMIT.len(), 40);
/// ```
pub const CACTUS_ENGINE_COMMIT: &str = env!("CACTUS_ENGINE_COMMIT");

/// Engine release this crate is pinned to, such as `4.1.1`.
///
/// It is the `release` field of `cactus-sys/prebuilt.toml`: the folder under `releases/` at
/// [`CACTUS_ENGINE_COMMIT`] that the archives come from.
///
/// # Examples
///
/// ```
/// assert!(cactus_sys::CACTUS_ENGINE_VERSION.starts_with("4."));
/// ```
pub const CACTUS_ENGINE_VERSION: &str = env!("CACTUS_ENGINE_VERSION");

/// Bit set in [`cactus_models`] when a text model (Needle 3) is loaded.
pub const CACTUS_TEXT: c_int = 1;

/// Bit set in [`cactus_models`] when a speech model (Whistle) is loaded.
pub const CACTUS_SPEECH: c_int = 2;

/// Bit set in [`cactus_models`] when a vision model (Mirage) is loaded.
pub const CACTUS_VISION: c_int = 4;

unsafe extern "C" {
    /// Configures the conversation prefix: system prompt, tool declarations and tool index.
    ///
    /// Returns the number of tokens in the resulting prefix, always greater than zero, or a
    /// negative value when it fails, most commonly because no text model has been loaded yet;
    /// [`cactus_last_error`] then names the reason. Calling it again replaces the previous
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
    /// - A text model must have been loaded with [`cactus_load`] first, otherwise this call
    ///   fails. Loading a text archive again, even the same one, discards the configuration:
    ///   call this again afterwards.
    pub fn cactus_init(
        system_prompt: *const c_char,
        tools_json: *const c_char,
        tool_index_path: *const c_char,
    ) -> c_int;

    /// Runs one completion on text or on a speech clip and writes a JSON envelope into `out`.
    ///
    /// Exactly one of `input` and `pcm` is non-null. Given `pcm`, the engine transcribes the
    /// clip with the speech model, as configured by [`cactus_set_audio`], answers the
    /// transcript with the text model, and merges the speech fields into the same JSON object
    /// under an `audio_` prefix (`audio_text`, `audio_language`, ...).
    ///
    /// Returns a non-negative **token count** on success (not the number of bytes written) and
    /// a negative value on failure, with the reason in [`cactus_last_error`]. On failure `out`
    /// also holds a NUL-terminated error envelope carrying the same text, such as
    /// `{"type":"error","error":"cactus_init not called"}`. A failed audio completion may still
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
    /// - A text model must be loaded and [`cactus_init`] must have succeeded first, and a
    ///   speech model must also be loaded when `pcm` is non-null; otherwise this fails.
    pub fn cactus_complete(
        input: *const c_char,
        pcm: *const c_float,
        samples: c_int,
        max_new_tokens: c_int,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Configures the transcription that [`cactus_complete`] runs for itself on a clip.
    ///
    /// `language`, `keywords` and `word_timestamps` mean what they mean in
    /// [`cactus_transcribe`]. The setting is process-global and stays in force until the next
    /// call; the defaults are to detect the language, use no keywords and report no word times.
    /// The engine **copies** both strings, so they may be freed as soon as this returns. It
    /// cannot fail and returns nothing. The setting survives [`cactus_reset`] and
    /// [`cactus_init`], and [`cactus_transcribe`] ignores it: that function takes its own.
    ///
    /// # Safety
    ///
    /// - `language` and `keywords` must each be either null or a pointer to a NUL-terminated C
    ///   string valid for the duration of the call.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn cactus_set_audio(
        language: *const c_char,
        keywords: *const c_char,
        word_timestamps: c_int,
    );

    /// Transcribes one speech clip and writes a JSON envelope into `out`.
    ///
    /// Returns the number of tokens generated on success and a negative value on failure, with
    /// the reason in [`cactus_last_error`]. The envelope has the shape
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
    /// A clip may be any length: up to 30 seconds runs as one pass, and a longer one as
    /// consecutive 30 second windows, each cut at its last sentence end, with word times counted
    /// from the start of the clip.
    ///
    /// **`out` is never truncated** (it was, before engine 4). A transcript that does not fit
    /// fails with `output buffer is too small: <n> bytes needed`, where `n` includes the
    /// terminating NUL, and leaves an empty string in `out`. An `out_capacity` of `0` or a null
    /// `out` fails with "no output buffer". A failed call writes no envelope; the reason is only
    /// in [`cactus_last_error`].
    ///
    /// # Safety
    ///
    /// - `pcm` must be valid for reads of `samples` `f32` values for the duration of the call,
    ///   holding 16 kHz mono audio in `[-1, 1]`. `samples` must not be negative.
    /// - `language` and `keywords` must each be either null or a pointer to a NUL-terminated C
    ///   string valid for the duration of the call.
    /// - `out` must be null or valid for writes of `out_capacity` bytes, and `out_capacity`
    ///   must not be negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    /// - A speech model must have been loaded with [`cactus_load`] first, otherwise this call
    ///   fails.
    pub fn cactus_transcribe(
        pcm: *const c_float,
        samples: c_int,
        language: *const c_char,
        keywords: *const c_char,
        word_timestamps: c_int,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Feeds audio to the live transcription and writes what this call committed into `out`.
    ///
    /// Appends 16 kHz mono float PCM to the stream, re-transcribes the audio that is not yet
    /// committed, and commits the words two consecutive passes agree on. Returns the number of
    /// words this call committed, zero or more, and a negative value on failure with the reason
    /// in [`cactus_last_error`]. Measured against the pinned archive by `tests/probe_stream.rs`.
    ///
    /// `out` receives
    /// `{"text":"...","words":[{"word":"...","start":0.00,"end":0.00,"probability":0.000}],"pending":"...","language":"en","received":0.00,"pass_ms":0.0}`:
    /// the committed words as text and one by one, with times in seconds from the start of the
    /// stream; `pending`, the unconfirmed tail, which the next call may change; the language;
    /// the seconds of audio taken in so far; and the time this pass took.
    ///
    /// **Feed about one second at a time.** Fed 100 ms at a time, the same recording committed
    /// words it had barely heard and came out wrong. A whole clip in one feed commits nothing:
    /// all of it is pending until [`cactus_stream_transcribe_stop`]. An empty feed is not an
    /// error.
    ///
    /// There is one stream per process. It starts with the first feed and ends with
    /// [`cactus_stream_transcribe_stop`]. A [`cactus_transcribe`] in between does not disturb
    /// it. `language` and `keywords` are as in [`cactus_transcribe`], with the same
    /// `unknown language <code>` failure.
    ///
    /// `out` is never truncated: a result that does not fit fails with
    /// `output buffer is too small: <n> bytes needed`, and an `out_capacity` of `0` with
    /// "no output buffer".
    ///
    /// # Safety
    ///
    /// - `pcm` must be valid for reads of `samples` `f32` values for the duration of the call,
    ///   holding 16 kHz mono audio in `[-1, 1]`, and `samples` must not be negative.
    /// - `language` and `keywords` must each be either null or a pointer to a NUL-terminated C
    ///   string valid for the duration of the call.
    /// - `out` must be valid for writes of `out_capacity` bytes, and `out_capacity` must not be
    ///   negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for every kind of model; the
    ///   caller must serialise every call in this module.
    /// - A speech model must have been loaded with [`cactus_load`] first, otherwise this call
    ///   fails.
    pub fn cactus_stream_transcribe_process(
        pcm: *const c_float,
        samples: c_int,
        language: *const c_char,
        keywords: *const c_char,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Ends the live transcription and writes its tail into `out`.
    ///
    /// The pending tail is committed as it stands: `out` receives the same document as
    /// [`cactus_stream_transcribe_process`], with the tail as its words and an empty `pending`,
    /// and the return value is the number of words. The next feed starts a new stream. With no
    /// stream open this is not an error: it writes an update with no words and returns `0`.
    ///
    /// **Give it room the first time.** A tail that does not fit fails with
    /// `output buffer is too small: <n> bytes needed`, and the stream is ended all the same:
    /// those words are lost, and calling again returns the empty update.
    ///
    /// # Safety
    ///
    /// - `out` must be valid for writes of `out_capacity` bytes, and `out_capacity` must not be
    ///   negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for every kind of model; the
    ///   caller must serialise every call in this module.
    /// - A speech model must have been loaded with [`cactus_load`] first, otherwise this call
    ///   fails.
    pub fn cactus_stream_transcribe_stop(out: *mut c_char, out_capacity: c_int) -> c_int;

    /// Describes an image with the vision model and writes a JSON document into `out`.
    ///
    /// Returns the number of tokens generated, greater than zero, on success, and a negative
    /// value on failure with the reason in [`cactus_last_error`]. Upstream documents none of
    /// what follows; it was measured against the pinned archive by `tests/probe_vision.rs`.
    ///
    /// `rgb` is packed 8-bit RGB, `width * height * 3` bytes, rows top to bottom, of any
    /// positive size; the engine fits it to its own 512-pixel canvas. A non-positive dimension
    /// fails with `invalid image: rgb must hold width x height x 3 bytes`. The engine cannot
    /// check the length of `rgb`: it reads what the dimensions say.
    ///
    /// By default `out` receives
    /// `{"scene":{...},"objects":[...],"regions":[...],"text":[...],"relations":[...],"events":[...],"reasoning":"..."}`.
    /// An object is `{"id","name","bbox":[left,top,right,bottom],"score","mask","position","description":{...}}`
    /// and a line of text `{"string","bbox","score"}`. Every position is on a scale from 0 to
    /// 1000 along each axis of the image given. The same image and options give the same bytes.
    ///
    /// `options` is null or a JSON object. Unknown keys are ignored, **and so is a string that
    /// is not JSON**: the image is then described with the defaults and no error is reported.
    /// Recognised keys, each range-checked with a failure of the form
    /// `invalid options: <key> must be ...`:
    ///
    /// - `sections`: a list of `scene`, `objects`, `regions`, `text`, `relations`, `events`,
    ///   `reasoning`; only these are written. `fields`: a list of `bbox`, `score`, `mask`,
    ///   `point`, `position`, `description`, `count`; only these are written on each object.
    /// - `conf`, `text_conf`, `iou`, `mask_threshold`: numbers from 0 to 1. `mask_tolerance`:
    ///   0 or more. `keyword_bias`: 0 to 100.
    /// - `max_objects`, `max_text_lines`, `max_tokens`: whole numbers of at least 1. A token
    ///   budget that runs out is not a failure; the document is closed where the model stopped.
    /// - `nms`, `agnostic_nms`: booleans.
    /// - `mask_format`: `largest` (the default) or `merged`, one outline as a flat
    ///   `[x,y,x,y,...]` list; `polygons`, a list of such lists; or `bitmap`,
    ///   `{"size":[height,width],"counts":[...]}` at the size of the image, run-length encoded
    ///   column by column starting with a run outside the mask.
    /// - `form`: an object of fields, answered in order instead of describing the image; `out`
    ///   then receives only `{"form":{<name>:{"value":...,...}}}`. A field is a choice
    ///   (`{"enum":[...]}`, answered with `value` and a `probabilities` object holding every
    ///   label and `-` for none of them), a count
    ///   (`{"type":"integer","minimum":a,"maximum":b}`, with one probability per number), text
    ///   (`{"type":"string"}`, optionally with `maxLength`, answered with `value` and
    ///   `probability`), or a box (`{"x-source":"bbox","x-of":"<an earlier choice field>"}`,
    ///   answered with a `[left,top,right,bottom]` value and `probability`). A `value` of `null`
    ///   means the model chose none. A form the engine cannot use fails with
    ///   `invalid form: ...`, naming the field.
    ///
    /// `language` is one of the seven codes [`cactus_transcribe`] takes, or null or empty for
    /// English; it sets the language of names and descriptions, and any other code fails with
    /// `unsupported language: <code>`. `keywords` is as in [`cactus_transcribe`].
    ///
    /// **`out` is never truncated.** A document that does not fit fails with
    /// `output buffer is too small: <n> bytes needed`, where `n` includes the terminating NUL,
    /// and leaves an empty string in `out`. An `out_capacity` of `0` fails with
    /// `invalid output buffer`.
    ///
    /// # Safety
    ///
    /// - `rgb` must be valid for reads of `width * height * 3` bytes for the duration of the
    ///   call, and `width` and `height` must be positive.
    /// - `options`, `language` and `keywords` must each be either null or a pointer to a
    ///   NUL-terminated C string valid for the duration of the call.
    /// - `out` must be valid for writes of `out_capacity` bytes, and `out_capacity` must not be
    ///   negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe runtime for every kind of model; the
    ///   caller must serialise every call in this module.
    /// - A vision model must have been loaded with [`cactus_load`] first.
    pub fn cactus_describe(
        rgb: *const c_uchar,
        width: c_int,
        height: c_int,
        options: *const c_char,
        language: *const c_char,
        keywords: *const c_char,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Sets how many threads the engine computes with, by upstream's name for it.
    ///
    /// The setting is process-global and returns nothing. Any value is accepted, zero and
    /// negative ones included. On the machine `tests/probe_stream.rs` was written on, values
    /// from 1 to 8 changed neither the output nor the time of a transcription, a completion or
    /// a description, so `cactus-rs` does not expose it.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe runtime for every kind of model; the
    ///   caller must serialise every call in this module. In particular this must not run while
    ///   any other call is in flight.
    pub fn cactus_set_threads(threads: c_int);

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
    /// [`cactus_last_error`]. Query the count first and size the buffer from the answer.
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
    /// - A model of the input's kind must have been loaded with [`cactus_load`] first.
    pub fn cactus_embed(
        input: *const c_char,
        pcm: *const c_float,
        samples: c_int,
        out: *mut c_float,
        out_capacity: c_int,
    ) -> c_int;

    /// Rewinds the conversation to the prefix that [`cactus_init`] built.
    ///
    /// Tool declarations and the system prompt survive; only the turns taken since are dropped.
    /// It cannot fail and it does not unload weights. Nothing in this API does.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module. In particular this must not run while
    ///   any other call, of either kind, is in flight.
    pub fn cactus_reset();

    /// Loads a `.cact` weight archive from memory.
    ///
    /// The engine reads whichever kind of model the archive holds, text or speech, and keeps a
    /// model of the other kind already loaded, so a process with both takes one call per
    /// archive. Both kinds carry the same magic tag: check [`cactus_models`] afterwards to learn
    /// which one was loaded.
    ///
    /// Returns `0` on success and a negative value on failure, including for a null pointer or
    /// bytes that are not a valid archive, with the reason in [`cactus_last_error`]. The engine
    /// **copies** what it needs during the call, so the buffer may be freed as soon as it
    /// returns.
    ///
    /// Weights cannot be unloaded. Loading a text archive again, even the one already loaded,
    /// discards the [`cactus_init`] configuration; the speech model has none to lose.
    ///
    /// # Safety
    ///
    /// - `cact` must be null, or valid for reads of `n` bytes for the duration of the call.
    /// - `n` must be the length of that buffer in bytes.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn cactus_load(cact: *const c_uchar, n: c_ulonglong) -> c_int;

    /// Reports which kinds of model this process has loaded.
    ///
    /// Returns a bitmask of [`CACTUS_TEXT`] and [`CACTUS_SPEECH`]: `0` before any successful
    /// [`cactus_load`], and never fewer bits afterwards, since nothing unloads weights. It cannot
    /// fail.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module.
    pub fn cactus_models() -> c_int;

    /// Returns the reason the last failing call failed.
    ///
    /// The result is a NUL-terminated C string **owned by the runtime**. It is one
    /// process-global slot shared by both kinds of model, and it is valid only until the next
    /// call into this module, which may overwrite or free it: copy it out before calling
    /// anything else, `cactus_last_error` included.
    ///
    /// # Safety
    ///
    /// - The returned pointer may be null; check before dereferencing it.
    /// - It must not be written through, freed, or read after the next call into this module.
    /// - The engine is one process-global, non-thread-safe runtime for both kinds of model; the
    ///   caller must serialise every call in this module, and must copy the string before
    ///   releasing whatever serialises them.
    pub fn cactus_last_error() -> *const c_char;
}
