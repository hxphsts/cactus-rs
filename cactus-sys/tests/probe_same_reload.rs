//! Probe: what `cactus_load` of the very archive already loaded does to the text model.
//!
//! Its own test binary, so its own process: engine state is global and cannot be undone. Skips
//! unless `CACTUS_NEEDLE_WEIGHTS` names a `needle3.cact`. Every measurement is printed and the
//! observed behaviour asserted, so an engine that changes it turns this red. This is why
//! `Slot::load` in `cactus-rs` answers an identical fingerprint without calling the engine.

#![cfg(feature = "engine")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};
use std::ffi::CString;

use cactus_sys::{
    CACTUS_TEXT, cactus_complete, cactus_embed, cactus_init, cactus_last_error, cactus_load,
    cactus_models,
};

// ---- Shared probe helpers, duplicated in each probe file on purpose: every probe is its own
// test binary (so its own process), because engine state is global and cannot be undone. ----

/// The NUL-terminated output buffer size every probe hands the engine.
const OUT: usize = 256 * 1024;

/// Reads the archive named by `var`, or `None` (and a "skipping" line) when it is unset.
fn archive(var: &str) -> Option<Vec<u8>> {
    let Some(path) = std::env::var_os(var) else {
        println!("skipping: set {var} to run this probe");
        return None;
    };
    Some(std::fs::read(&path).unwrap_or_else(|error| panic!("{var}: {error}")))
}

/// Copies [`cactus_last_error`] out before the next call can invalidate it.
///
/// # Safety
///
/// Every engine call must be serialised, as each probe's single test function does.
unsafe fn last_error() -> String {
    // SAFETY: serialised by the caller; the pointer is checked and copied before returning.
    let ptr = unsafe { cactus_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: non-null, NUL-terminated, and valid until the next engine call.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// `cactus_load` of `bytes`: the return code, the error text when it failed, and the models after.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn load(label: &str, bytes: &[u8]) -> (c_int, String, c_int) {
    // SAFETY: `bytes` is valid for its length for the call; serialised by the caller.
    let rc = unsafe { cactus_load(bytes.as_ptr(), bytes.len() as c_ulonglong) };
    // SAFETY: serialised by the caller.
    let error = if rc < 0 {
        unsafe { last_error() }
    } else {
        String::new()
    };
    // SAFETY: serialised by the caller.
    let models = unsafe { cactus_models() };
    println!("{label}: cactus_load rc {rc}, last_error {error:?}, cactus_models {models}");
    (rc, error, models)
}

/// One tool, in the flat wire form the engine reads.
const TOOLS: &CStr = c"[{\"name\":\"control_lights\",\"description\":\"Turn lights on or off in a room.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\"},\"action\":{\"type\":\"string\",\"enum\":[\"on\",\"off\"]}},\"required\":[\"room\",\"action\"]}}]";

/// `cactus_init` with one tool: the prefix token count, or the error text.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn init() -> (c_int, String) {
    // SAFETY: both strings are NUL-terminated statics; serialised by the caller.
    let rc = unsafe {
        cactus_init(
            c"You control a smart home.".as_ptr(),
            TOOLS.as_ptr(),
            core::ptr::null(),
        )
    };
    // SAFETY: serialised by the caller.
    let error = if rc < 0 {
        unsafe { last_error() }
    } else {
        String::new()
    };
    println!("cactus_init: rc {rc}, last_error {error:?}");
    (rc, error)
}

/// `cactus_complete` on `input`: the return code and the envelope.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn complete(label: &str, input: &str) -> (c_int, String) {
    let input = CString::new(input).unwrap();
    let mut out = vec![0u8; OUT];
    // SAFETY: `input` is NUL-terminated, `out` holds `OUT` bytes; serialised by the caller.
    let rc = unsafe {
        cactus_complete(
            input.as_ptr(),
            core::ptr::null(),
            0,
            64,
            out.as_mut_ptr().cast::<c_char>(),
            OUT as c_int,
        )
    };
    // SAFETY: serialised by the caller.
    let error = if rc < 0 {
        unsafe { last_error() }
    } else {
        String::new()
    };
    let envelope = CStr::from_bytes_until_nul(&out)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    println!("{label}: cactus_complete rc {rc}, last_error {error:?}, envelope {envelope}");
    (rc, envelope)
}

/// `cactus_embed` of `input`: the whole vector.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn embed(label: &str, input: &str) -> Vec<f32> {
    let input = CString::new(input).unwrap();
    // SAFETY: a null output only asks for the count; serialised by the caller.
    let count = unsafe {
        cactus_embed(
            input.as_ptr(),
            core::ptr::null(),
            0,
            core::ptr::null_mut(),
            0,
        )
    };
    assert!(count > 0, "embed count {count}: {}", unsafe {
        last_error()
    });
    let mut out = vec![0f32; count as usize];
    // SAFETY: `out` holds `count` floats; serialised by the caller.
    let rc = unsafe {
        cactus_embed(
            input.as_ptr(),
            core::ptr::null(),
            0,
            out.as_mut_ptr(),
            count,
        )
    };
    assert_eq!(rc, count, "embed: {}", unsafe { last_error() });
    println!(
        "{label}: cactus_embed({input:?}) dim {count}, first 8 {:?}",
        &out[..8]
    );
    out
}

#[test]
fn reloading_the_same_text_archive_drops_init() {
    let Some(text) = archive("CACTUS_NEEDLE_WEIGHTS") else {
        return;
    };

    // SAFETY: every call in this test runs on one thread, in order.
    unsafe {
        assert_eq!(load("text", &text), (0, String::new(), CACTUS_TEXT));
        assert!(init().0 > 0);
        assert!(complete("before", "turn on the kitchen lights").0 >= 0);
        let before = embed("before", "kitchen");

        let (rc, error, models) = load("same bytes again", &text);
        let (complete_rc, envelope) =
            complete("after reload, no re-init", "turn on the kitchen lights");
        let survived = complete_rc >= 0;
        println!("init survived the identical reload: {survived}");
        let after = embed("after reload", "kitchen");
        println!("embedding changed: {}", before != after);

        // Measured at c7c415a3: the identical archive is accepted, the weights are the same,
        // and the init configuration is gone, as the header says.
        assert_eq!(rc, 0, "reload: {error}");
        assert_eq!(models, CACTUS_TEXT);
        assert!(!survived);
        assert!(envelope.contains("cactus_init not called"), "{envelope}");
        assert_eq!(before, after);
    }
}
