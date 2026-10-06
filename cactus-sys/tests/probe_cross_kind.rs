//! Probe: what loading a speech archive does to an initialised text model, and back.
//!
//! Its own test binary, so its own process: engine state is global and cannot be undone. Skips
//! unless both `CACTUS_NEEDLE_WEIGHTS` and `CACTUS_WHISTLE_WEIGHTS` are set. Every measurement
//! is printed and the observed behaviour asserted, so an engine that changes it turns this red.
//! It also prints both archives' headers and first directory records, the evidence for telling
//! the two kinds apart without the engine.

#![cfg(feature = "needle")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};
use std::ffi::CString;

use cactus_sys::{
    NEEDLE_SPEECH, NEEDLE_TEXT, needle_complete, needle_embed, needle_init, needle_last_error,
    needle_load, needle_models, needle_transcribe,
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

/// The audio manifest a Whistle archive carries and a Needle 3 archive does not: a 1-D FP32
/// (dtype 2) tensor of 13 values whose tenth is the sample rate, 16 000.
fn audio_manifest(bytes: &[u8], tensors: &[Tensor]) -> Option<(usize, Vec<f32>)> {
    tensors.iter().enumerate().find_map(|(index, tensor)| {
        if tensor.dtype != 2 || tensor.ndim != 1 || tensor.dims[0] != 13 || tensor.size != 52 {
            return None;
        }
        let start = tensor.offset as usize;
        let values: Vec<f32> = bytes[start..start + 52]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect();
        (values[9] == 16_000.0).then_some((index, values))
    })
}

#[test]
fn a_speech_load_keeps_the_text_model_and_its_init() {
    let (Some(text), Some(speech)) = (
        archive("CACTUS_NEEDLE_WEIGHTS"),
        archive("CACTUS_WHISTLE_WEIGHTS"),
    ) else {
        return;
    };

    // The header discriminator: what tells the two archives apart before the engine sees them.
    let text_tensors = describe("needle3", &text);
    let speech_tensors = describe("whistle", &speech);
    let text_manifest = audio_manifest(&text, &text_tensors);
    let speech_manifest = audio_manifest(&speech, &speech_tensors);
    println!("needle3 audio manifest: {text_manifest:?}");
    println!("whistle audio manifest: {speech_manifest:?}");
    assert_eq!(text_tensors.len(), 581);
    assert_eq!(speech_tensors.len(), 681);
    assert!(text_manifest.is_none());
    assert!(speech_manifest.is_some());
    let tone = sine();

    // SAFETY: every call in this test runs on one thread, in order.
    unsafe {
        assert_eq!(load("text", &text), (0, String::new(), NEEDLE_TEXT));
        assert!(init().0 > 0);
        assert!(complete("text only", "turn on the kitchen lights").0 >= 0);
        let words = embed("text only", "kitchen");

        let (rc, error, models) = load("speech after text", &speech);
        let (complete_rc, _) = complete(
            "after speech load, no re-init",
            "turn on the kitchen lights",
        );
        let survived = complete_rc >= 0;
        println!("text init survived the speech load: {survived}");
        let words_after = embed("after speech load", "kitchen");
        println!("text embedding changed: {}", words != words_after);
        let sound = embed_pcm("after speech load", &tone);

        // Measured at c7c415a3: a speech archive leaves the text model, and its init, alone.
        assert_eq!(rc, 0, "speech load: {error}");
        assert_eq!(models, NEEDLE_TEXT | NEEDLE_SPEECH);
        assert!(survived);
        assert_eq!(words, words_after);

        let (rc, error, models) = load("text again", &text);
        let (complete_rc, envelope) = complete(
            "after text reload, no re-init",
            "turn on the kitchen lights",
        );
        let survived = complete_rc >= 0;
        println!("text init survived the text reload: {survived}");
        let sound_after = embed_pcm("after text reload", &tone);
        println!("speech embedding changed: {}", sound != sound_after);
        let (transcribe_rc, _) = transcribe("after text reload", &lights_en());

        // The speech bit is kept and the speech model untouched; the text reload drops init.
        assert_eq!(rc, 0, "text reload: {error}");
        assert_eq!(models, NEEDLE_TEXT | NEEDLE_SPEECH);
        assert!(!survived);
        assert!(envelope.contains("needle_init not called"), "{envelope}");
        assert_eq!(sound, sound_after);
        assert!(transcribe_rc >= 0);
    }
}
