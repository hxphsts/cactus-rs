//! Link smoke test - proves the archive links and answers, without needing any weights.

#![cfg(feature = "needle")]

use core::ffi::{CStr, c_char, c_int};

use cactus_sys::{
    needle_complete, needle_last_error, needle_load, needle_models, needle_set_audio,
    needle_transcribe,
};

/// Copies [`needle_last_error`] out before the next call can invalidate it.
///
/// # Safety
///
/// Must be called with every other engine call serialised, as the test below does.
unsafe fn last_error() -> Option<String> {
    // SAFETY: serialised by the caller; the pointer is checked and copied before returning.
    let ptr = unsafe { needle_last_error() };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: non-null, NUL-terminated, and valid until the next engine call.
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

// One test function on purpose: the engine is process-global and not thread-safe, and the test
// harness runs separate `#[test]` functions on separate threads.
#[test]
fn test_engine_links_and_reports_errors() {
    // SAFETY: every call in this test runs on one thread, in order.
    unsafe {
        assert_eq!(needle_models(), 0);

        let mut out = [0u8; 256];
        let rc = needle_complete(
            c"hello".as_ptr(),
            core::ptr::null(),
            0,
            8,
            out.as_mut_ptr().cast::<c_char>(),
            out.len() as c_int,
        );
        let error = last_error();
        let envelope = CStr::from_bytes_until_nul(&out).unwrap().to_str().unwrap();
        println!("complete: rc {rc}, envelope {envelope:?}, last error {error:?}");
        assert!(rc < 0);
        // Measured at c7c415a3: the engine reports this failure on both channels, the
        // `{"type":"error","error":"needle_init not called"}` envelope in `out` and the same
        // "needle_init not called" text in `needle_last_error`.
        assert!(
            envelope.contains("needle_init not called")
                || error.as_deref().is_some_and(|e| !e.is_empty())
        );

        let junk = [0u8; 64];
        let rc = needle_load(junk.as_ptr(), junk.len() as u64);
        let error = last_error();
        println!("junk load: rc {rc}, last error {error:?}");
        assert!(rc < 0);
        assert!(error.is_some_and(|e| !e.is_empty()));
        assert_eq!(needle_models(), 0);

        let pcm = [0.0f32; 1600];
        let mut out = [0u8; 1024];
        let rc = needle_transcribe(
            pcm.as_ptr(),
            pcm.len() as c_int,
            core::ptr::null(),
            core::ptr::null(),
            0,
            out.as_mut_ptr().cast::<c_char>(),
            out.len() as c_int,
        );
        let error = last_error();
        let envelope = CStr::from_bytes_until_nul(&out).unwrap().to_str().unwrap();
        println!("transcribe: rc {rc}, envelope {envelope:?}, last error {error:?}");
        assert!(rc < 0);
        assert!(error.is_some_and(|e| !e.is_empty()));

        needle_set_audio(core::ptr::null(), core::ptr::null(), 0);
    }
}
