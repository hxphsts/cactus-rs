//! Raw declarations for the Needle 3 engine C API (`include/needle.h`).
//!
//! ## Overview
//!
//! Five functions, no handles, no opaque types. They drive one process-global model that the
//! engine owns: weights are loaded once with [`needle_load`], a conversation is configured with
//! [`needle_init`], and then [`needle_complete`] and [`needle_embed`] run against it until
//! [`needle_reset`] rewinds it.
//!
//! The declarations are written by hand rather than generated. Five stable entry points do not
//! justify a `bindgen` build dependency, and hand-written signatures let every function carry the
//! `# Safety` contract that was measured against the shipped archive.
//!
//! Every function returns `-1` on failure. There is no error-string API: when
//! [`needle_complete`] fails it writes a NUL-terminated JSON envelope of the shape
//! `{"type":"error","error":"..."}` into the caller's buffer instead.
//!
//! ## Usage
//!
//! ```no_run
//! use core::ffi::{c_char, c_int, c_ulonglong};
//!
//! use cactus_sys::{needle_complete, needle_init, needle_load};
//!
//! let weights: &[u8] = b"...contents of needle3.cact...";
//! let mut out = [0u8; 64 * 1024];
//!
//! // SAFETY: every call below happens on one thread, in order, and the engine copies the
//! // archive during `needle_load`, so `weights` needs no particular lifetime afterwards.
//! unsafe {
//!     assert_eq!(needle_load(weights.as_ptr(), weights.len() as c_ulonglong), 0);
//!     assert!(needle_init(c"You are terse.".as_ptr(), c"[]".as_ptr(), core::ptr::null()) > 0);
//!
//!     let tokens = needle_complete(
//!         c"hello".as_ptr(),
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

unsafe extern "C" {
    /// Configures the conversation prefix: system prompt, tool declarations and tool index.
    ///
    /// Returns the number of tokens in the resulting prefix, always greater than zero, or `-1`
    /// when it fails, most commonly because no weights have been loaded yet. Calling it again
    /// replaces the previous configuration.
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
    /// - The engine is one process-global, non-thread-safe model. No call in this module may run
    ///   concurrently with any other; the caller must serialise all of them.
    /// - [`needle_load`] must have succeeded first, otherwise this call returns `-1`.
    pub fn needle_init(
        system_prompt: *const c_char,
        tools_json: *const c_char,
        tool_index_path: *const c_char,
    ) -> c_int;

    /// Runs one completion and writes a JSON envelope into `out`.
    ///
    /// Returns a non-negative **token count** on success (not the number of bytes written) and
    /// `-1` on failure. On failure `out` holds a NUL-terminated error envelope such as
    /// `{"type":"error","error":"needle_init not called"}`, which is the only error detail the
    /// engine offers.
    ///
    /// `out` is always NUL-terminated, and the envelope is **silently truncated** to
    /// `out_capacity - 1` bytes when it does not fit: there is no error and no length hint, so a
    /// short buffer surfaces as JSON that fails to parse. An `out_capacity` of `0` is tolerated
    /// and writes nothing. Size the buffer generously; 64 KiB is comfortable for tool calls.
    ///
    /// # Safety
    ///
    /// - `input` must be null or a pointer to a NUL-terminated C string valid for the call.
    /// - `out` must be valid for writes of `out_capacity` bytes, and `out_capacity` must not be
    ///   negative or larger than the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe model; the caller must serialise
    ///   every call in this module.
    /// - [`needle_load`] and [`needle_init`] must have succeeded first; otherwise this returns
    ///   `-1` with the error envelope described above.
    pub fn needle_complete(
        input: *const c_char,
        max_new_tokens: c_int,
        out: *mut c_char,
        out_capacity: c_int,
    ) -> c_int;

    /// Embeds `input`, or reports the embedding dimension.
    ///
    /// With `out` null and `out_capacity` `0`, returns the model's embedding dimension without
    /// computing anything (3072 for Needle 3). With a buffer, returns that same dimension after
    /// filling the first `dimension` floats, or `-1` when `out_capacity` is smaller than the
    /// dimension. Query the dimension first and size the buffer from the answer.
    ///
    /// # Safety
    ///
    /// - `input` must be null or a pointer to a NUL-terminated C string valid for the call.
    /// - `out` must be null, or valid for writes of `out_capacity` `f32` values; `out_capacity`
    ///   must not be negative or exceed the allocation behind `out`.
    /// - The engine is one process-global, non-thread-safe model; the caller must serialise
    ///   every call in this module.
    /// - [`needle_load`] must have succeeded first.
    pub fn needle_embed(input: *const c_char, out: *mut c_float, out_capacity: c_int) -> c_int;

    /// Rewinds the conversation to the prefix that [`needle_init`] built.
    ///
    /// Tool declarations and the system prompt survive; only the turns taken since are dropped.
    /// It cannot fail and it does not unload weights. Nothing in this API does.
    ///
    /// # Safety
    ///
    /// - The engine is one process-global, non-thread-safe model; the caller must serialise
    ///   every call in this module. In particular this must not run while a
    ///   [`needle_complete`] or [`needle_embed`] call is in flight.
    pub fn needle_reset();

    /// Loads a `.cact` weight archive from memory.
    ///
    /// Returns `0` on success and `-1` on failure, including for a null pointer or bytes that
    /// are not a valid archive. The engine **copies** what it needs during the call, so the
    /// buffer may be freed as soon as it returns. Loading the same archive again returns `0`.
    ///
    /// Weights cannot be unloaded; the first successful load owns the process.
    ///
    /// # Safety
    ///
    /// - `cact` must be null, or valid for reads of `n` bytes for the duration of the call.
    /// - `n` must be the length of that buffer in bytes.
    /// - The engine is one process-global, non-thread-safe model; the caller must serialise
    ///   every call in this module.
    pub fn needle_load(cact: *const c_uchar, n: c_ulonglong) -> c_int;
}
