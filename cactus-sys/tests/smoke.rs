//! Link smoke test - proves the archive links and answers, without needing any weights.

#![cfg(feature = "needle")]

use core::ffi::{CStr, c_char, c_int};

use cactus_sys::{needle_complete, needle_load};

// One test function on purpose: the engine is process-global and not thread-safe, and the test
// harness runs separate `#[test]` functions on separate threads.
#[test]
fn test_engine_links_and_reports_errors() {
    let mut out = [0u8; 256];

    // SAFETY: single-threaded, `out` is valid for `out.len()` bytes, the input is a C string.
    let rc = unsafe {
        needle_complete(
            c"hello".as_ptr(),
            8,
            out.as_mut_ptr().cast::<c_char>(),
            out.len() as c_int,
        )
    };
    assert_eq!(rc, -1);

    let envelope = CStr::from_bytes_until_nul(&out).unwrap().to_str().unwrap();
    println!("envelope: {envelope}");
    assert!(envelope.contains("needle_init not called"));

    let junk = [0u8; 64];
    // SAFETY: single-threaded, `junk` is valid for reads of `junk.len()` bytes.
    let rc = unsafe { needle_load(junk.as_ptr(), junk.len() as u64) };
    assert_eq!(rc, -1);
}
