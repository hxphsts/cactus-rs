//! Probe: what `needle_load` of a second, different text archive does to a loaded text model.
//!
//! Its own test binary, so its own process: engine state is global and cannot be undone. Skips
//! unless `CACTUS_NEEDLE_WEIGHTS` names a `needle3.cact`. Every measurement is printed and the
//! observed behaviour asserted, so an engine that changes it turns this red.

#![cfg(feature = "needle")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};
use std::ffi::CString;

use cactus_sys::{
    NEEDLE_TEXT, needle_complete, needle_embed, needle_init, needle_last_error, needle_load,
    needle_models,
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

/// Copies [`needle_last_error`] out before the next call can invalidate it.
///
/// # Safety
///
/// Every engine call must be serialised, as each probe's single test function does.
unsafe fn last_error() -> String {
    // SAFETY: serialised by the caller; the pointer is checked and copied before returning.
    let ptr = unsafe { needle_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: non-null, NUL-terminated, and valid until the next engine call.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// `needle_load` of `bytes`: the return code, the error text when it failed, and the models after.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn load(label: &str, bytes: &[u8]) -> (c_int, String, c_int) {
    // SAFETY: `bytes` is valid for its length for the call; serialised by the caller.
    let rc = unsafe { needle_load(bytes.as_ptr(), bytes.len() as c_ulonglong) };
    // SAFETY: serialised by the caller.
    let error = if rc < 0 {
        unsafe { last_error() }
    } else {
        String::new()
    };
    // SAFETY: serialised by the caller.
    let models = unsafe { needle_models() };
    println!("{label}: needle_load rc {rc}, last_error {error:?}, needle_models {models}");
    (rc, error, models)
}

/// One record of the nameless tensor directory: dtype, shape, absolute offset and byte size.
#[derive(Debug, Clone, Copy)]
struct Tensor {
    dtype: u8,
    ndim: u8,
    dims: [u32; 4],
    offset: u64,
    size: u64,
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// Parses the directory: header field 1 is the tensor count, field 2 the codebook length in
/// `u32`s, and `n` records of 44 bytes (`<BBHIIIIQQII`) follow the 196-byte header and codebook.
fn directory(bytes: &[u8]) -> Vec<Tensor> {
    let count = u32_at(bytes, 4) as usize;
    let start = 196 + u32_at(bytes, 8) as usize * 4;
    (0..count)
        .map(|index| {
            let at = start + index * 44;
            Tensor {
                dtype: bytes[at],
                ndim: bytes[at + 1],
                dims: [
                    u32_at(bytes, at + 4),
                    u32_at(bytes, at + 8),
                    u32_at(bytes, at + 12),
                    u32_at(bytes, at + 16),
                ],
                offset: u64_at(bytes, at + 20),
                size: u64_at(bytes, at + 28),
            }
        })
        .collect()
}

/// Prints the tensor count and the first records, the material for a header discriminator.
fn describe(label: &str, bytes: &[u8]) -> Vec<Tensor> {
    let tensors = directory(bytes);
    let header: Vec<u32> = (0..48).map(|field| u32_at(bytes, field * 4)).collect();
    println!(
        "{label}: {} bytes, header u32 fields {header:?}",
        bytes.len()
    );
    println!("{label}: {} tensors", tensors.len());
    for (index, tensor) in tensors.iter().take(4).enumerate() {
        println!("{label}: tensor {index} {tensor:?}");
    }
    tensors
}

/// A copy of `bytes` with the sign bit of every value of the largest FP16 tensor flipped.
///
/// dtype 1 is FP16: every such record's byte size is twice the product of its dims.
fn perturbed(bytes: &[u8]) -> Vec<u8> {
    let tensors = directory(bytes);
    let (index, largest) = tensors
        .iter()
        .enumerate()
        .filter(|(_, tensor)| tensor.dtype == 1)
        .max_by_key(|(index, tensor)| (tensor.size, std::cmp::Reverse(*index)))
        .map(|(index, tensor)| (index, *tensor))
        .expect("an FP16 tensor");
    let elements: u64 = largest.dims[..usize::from(largest.ndim)]
        .iter()
        .map(|&dim| u64::from(dim))
        .product();
    assert_eq!(largest.size, 2 * elements, "dtype 1 is two bytes per value");
    println!("perturbing tensor {index} {largest:?}: sign bit of {elements} FP16 values");

    let mut copy = bytes.to_vec();
    let start = largest.offset as usize;
    for value in copy[start..start + largest.size as usize].chunks_exact_mut(2) {
        value[1] ^= 0x80;
    }
    copy
}

/// One tool, in the flat wire form the engine reads.
const TOOLS: &CStr = c"[{\"name\":\"control_lights\",\"description\":\"Turn lights on or off in a room.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"room\":{\"type\":\"string\"},\"action\":{\"type\":\"string\",\"enum\":[\"on\",\"off\"]}},\"required\":[\"room\",\"action\"]}}]";

/// `needle_init` with one tool: the prefix token count, or the error text.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn init() -> (c_int, String) {
    // SAFETY: both strings are NUL-terminated statics; serialised by the caller.
    let rc = unsafe {
        needle_init(
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
    println!("needle_init: rc {rc}, last_error {error:?}");
    (rc, error)
}

/// `needle_complete` on `input`: the return code and the envelope.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn complete(label: &str, input: &str) -> (c_int, String) {
    let input = CString::new(input).unwrap();
    let mut out = vec![0u8; OUT];
    // SAFETY: `input` is NUL-terminated, `out` holds `OUT` bytes; serialised by the caller.
    let rc = unsafe {
        needle_complete(
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
    println!("{label}: needle_complete rc {rc}, last_error {error:?}, envelope {envelope}");
    (rc, envelope)
}

/// `needle_embed` of `input`: the whole vector.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn embed(label: &str, input: &str) -> Vec<f32> {
    let input = CString::new(input).unwrap();
    // SAFETY: a null output only asks for the count; serialised by the caller.
    let count = unsafe {
        needle_embed(
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
        needle_embed(
            input.as_ptr(),
            core::ptr::null(),
            0,
            out.as_mut_ptr(),
            count,
        )
    };
    assert_eq!(rc, count, "embed: {}", unsafe { last_error() });
    println!(
        "{label}: needle_embed({input:?}) dim {count}, first 8 {:?}",
        &out[..8]
    );
    out
}

#[test]
fn a_second_text_archive_replaces_or_is_ignored() {
    let Some(original) = archive("CACTUS_NEEDLE_WEIGHTS") else {
        return;
    };
    describe("needle3", &original);

    // SAFETY: every call in this test runs on one thread, in order.
    unsafe {
        let (rc, _, models) = load("original", &original);
        assert_eq!(rc, 0);
        assert_eq!(models, NEEDLE_TEXT);
        assert!(init().0 > 0);
        assert!(complete("before", "turn on the kitchen lights").0 >= 0);
        let before = embed("before", "kitchen");

        let altered = perturbed(&original);
        let (rc, error, models) = load("perturbed", &altered);
        let (complete_rc, envelope) = complete("after, no re-init", "turn on the kitchen lights");
        let survived = complete_rc >= 0;
        println!("init survived the perturbed load: {survived}");
        let after = embed("after", "kitchen");
        let changed = before != after;
        println!("embedding changed (model replaced): {changed}");

        // Measured at c7c415a3: a different text archive is accepted, REPLACES the loaded text
        // model (the embedding moves) and drops the init configuration, exactly as reloading
        // the same archive does.
        assert_eq!(rc, 0, "perturbed load: {error}");
        assert_eq!(models, NEEDLE_TEXT);
        assert!(!survived);
        assert!(envelope.contains("needle_init not called"), "{envelope}");
        assert!(changed, "a different text archive is no longer ignored");

        // Re-initialising brings the replaced model back into service.
        assert!(init().0 > 0);
        assert!(complete("after re-init", "turn on the kitchen lights").0 >= 0);

        // A Needle 2 tag in front of junk: rejected, and the loaded model is untouched, init
        // configuration included.
        let mut retagged = original[..4096].to_vec();
        retagged[..4].copy_from_slice(&0x05E1_2A83_u32.to_le_bytes());
        retagged.extend(std::iter::repeat_n(0xA5, 4096));
        let (rc, error, models) = load("needle 2 tag + junk", &retagged);
        assert!(rc < 0);
        assert_eq!(error, "invalid .cact model");
        assert_eq!(models, NEEDLE_TEXT);
        let survived = complete("after rejected load", "turn on the kitchen lights").0 >= 0;
        println!("init survived the rejected load: {survived}");
        assert!(survived);
        assert_eq!(embed("after rejected load", "kitchen"), after);
    }
}
