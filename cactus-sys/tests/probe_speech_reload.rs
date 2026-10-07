//! Probe: what `needle_load` of a second, different speech archive does to a loaded speech model.
//!
//! Its own test binary, so its own process: engine state is global and cannot be undone. Skips
//! unless `CACTUS_WHISTLE_WEIGHTS` names a `whistle.cact`. Every measurement is printed and the
//! observed behaviour asserted, so an engine that changes it turns this red.

#![cfg(feature = "needle")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};

use cactus_sys::{
    NEEDLE_SPEECH, needle_embed, needle_last_error, needle_load, needle_models, needle_transcribe,
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

/// One second of a 440 Hz sine at 16 kHz, half amplitude.
fn sine() -> Vec<f32> {
    (0..16_000)
        .map(|n| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16_000.0).sin())
        .collect()
}

/// `tests/data/clips/lights_en.wav` from the `cactus-rs` crate, as 16 kHz mono `f32` PCM.
///
/// The clip is a plain 44-byte-header PCM16 WAV, so its header is checked and skipped by hand
/// rather than pulling in a WAV crate.
fn lights_en() -> Vec<f32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cactus-rs/tests/data/clips/lights_en.wav"
    );
    let bytes = std::fs::read(path).expect("the cactus-rs test clip");
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1, "mono");
    assert_eq!(u32_at(&bytes, 24), 16_000, "16 kHz");
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16, "PCM16");
    let length = u32_at(&bytes, 40) as usize;
    bytes[44..44 + length]
        .chunks_exact(2)
        .map(|sample| f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32_768.0)
        .collect()
}

/// `needle_embed` of `pcm`: the whole flattened frame matrix.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn embed_pcm(label: &str, pcm: &[f32]) -> Vec<f32> {
    let samples = pcm.len() as c_int;
    // SAFETY: a null output only asks for the count; serialised by the caller.
    let count = unsafe {
        needle_embed(
            core::ptr::null(),
            pcm.as_ptr(),
            samples,
            core::ptr::null_mut(),
            0,
        )
    };
    assert!(count > 0, "embed count {count}: {}", unsafe {
        last_error()
    });
    let mut out = vec![0f32; count as usize];
    // SAFETY: `pcm` holds `samples` floats, `out` holds `count`; serialised by the caller.
    let rc = unsafe {
        needle_embed(
            core::ptr::null(),
            pcm.as_ptr(),
            samples,
            out.as_mut_ptr(),
            count,
        )
    };
    assert_eq!(rc, count, "embed: {}", unsafe { last_error() });
    println!(
        "{label}: needle_embed(pcm) {count} floats, first 8 {:?}",
        &out[..8]
    );
    out
}

/// `needle_transcribe` of `pcm` in English: the return code and the envelope.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn transcribe(label: &str, pcm: &[f32]) -> (c_int, String) {
    let mut out = vec![0u8; OUT];
    // SAFETY: `pcm` holds its length in floats, `out` holds `OUT` bytes; serialised by the caller.
    let rc = unsafe {
        needle_transcribe(
            pcm.as_ptr(),
            pcm.len() as c_int,
            c"en".as_ptr(),
            core::ptr::null(),
            0,
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
    println!("{label}: needle_transcribe rc {rc}, last_error {error:?}, envelope {envelope}");
    (rc, envelope)
}

#[test]
fn a_second_speech_archive_replaces_or_is_ignored() {
    let Some(original) = archive("CACTUS_WHISTLE_WEIGHTS") else {
        return;
    };
    describe("whistle", &original);
    let tone = sine();
    let clip = lights_en();

    // SAFETY: every call in this test runs on one thread, in order.
    unsafe {
        let (rc, _, models) = load("original", &original);
        assert_eq!(rc, 0);
        assert_eq!(models, NEEDLE_SPEECH);
        let before = embed_pcm("before", &tone);
        let (transcribe_rc_before, text_before) = transcribe("before", &clip);
        assert!(transcribe_rc_before >= 0);

        let altered = perturbed(&original);
        let (rc, error, models) = load("perturbed", &altered);
        let after = embed_pcm("after", &tone);
        let changed = before != after;
        println!("embedding changed (model replaced): {changed}");
        let (transcribe_rc, text_after) = transcribe("after", &clip);
        println!("transcript changed: {}", text_before != text_after);

        // Measured at c7c415a3: a different speech archive is accepted and REPLACES the loaded
        // speech model; the embedding of the same tone moves.
        assert_eq!(rc, 0, "perturbed load: {error}");
        assert_eq!(models, NEEDLE_SPEECH);
        assert!(changed, "a different speech archive is no longer ignored");
        assert!(transcribe_rc >= 0);

        // A Needle 2 tag in front of junk: rejected, and the loaded model is untouched.
        let mut retagged = original[..4096].to_vec();
        retagged[..4].copy_from_slice(&0x05E1_2A83_u32.to_le_bytes());
        retagged.extend(std::iter::repeat_n(0xA5, 4096));
        let (rc, error, models) = load("needle 2 tag + junk", &retagged);
        assert!(rc < 0);
        assert_eq!(error, "invalid .cact model");
        assert_eq!(models, NEEDLE_SPEECH);
        assert_eq!(embed_pcm("after rejected load", &tone), after);

        // Loading the original again restores it exactly: the replacement is complete.
        assert_eq!(load("original again", &original).0, 0);
        assert_eq!(embed_pcm("restored", &tone), before);
        assert_eq!(transcribe("restored", &clip).0, transcribe_rc_before);
    }
}
