//! Probe: what the live transcription calls do, what `cactus_transcribe` does with a long clip
//! and a small buffer, and whether `cactus_set_threads` changes anything.
//!
//! Upstream's header has no comments since engine 4.0, so the contract `cactus-rs` builds on is
//! the one measured here. Its own test binary, so its own process: engine state is global and
//! cannot be undone. Skips unless `CACTUS_WHISTLE_WEIGHTS` names a `whistle.cact`. Every
//! measurement is printed and the observed behaviour asserted, so an engine that changes it
//! turns this red.

#![cfg(feature = "engine")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};

use cactus_sys::{
    cactus_embed, cactus_last_error, cactus_load, cactus_set_threads,
    cactus_stream_transcribe_process, cactus_stream_transcribe_stop, cactus_transcribe,
};

// ---- Shared probe helpers, duplicated in each probe file on purpose: every probe is its own
// test binary (so its own process), because engine state is global and cannot be undone. ----

/// The NUL-terminated output buffer size the probe hands the engine unless it is testing the
/// buffer itself.
const OUT: usize = 256 * 1024;

/// What a call reports as written when the engine left the buffer untouched.
const UNTOUCHED: &str = "<no NUL written>";

/// The update a stream writes when it has nothing: no words, no tail, no language. The time the
/// pass took follows and varies.
const NOTHING: &str = r#"{"text":"","words":[],"pending":"","language":"","received":0.00,"#;

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
/// Every engine call must be serialised, as the probe's single test function does.
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

/// `tests/data/clips/jfk.wav` from the `cactus-rs` crate, as 16 kHz mono `f32` PCM: 11 seconds.
///
/// The clip is a plain 44-byte-header PCM16 WAV, so its header is skipped by hand rather than
/// pulling in a WAV crate.
fn jfk() -> Vec<f32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../cactus-rs/tests/data/clips/jfk.wav"
    );
    let bytes = std::fs::read(path).expect("the cactus-rs test clip");
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[36..40], b"data");
    let length = u32::from_le_bytes(bytes[40..44].try_into().expect("four bytes")) as usize;
    bytes[44..44 + length]
        .chunks_exact(2)
        .map(|sample| f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32_768.0)
        .collect()
}

/// What a call wrote into a buffer pre-filled with `0x55`.
fn written(out: &[u8]) -> String {
    match out.iter().position(|byte| *byte == 0) {
        Some(end) => String::from_utf8_lossy(&out[..end]).into_owned(),
        None => String::from(UNTOUCHED),
    }
}

/// One engine call's return code, what it wrote, and its error when it failed.
struct Outcome {
    rc: c_int,
    text: String,
    error: String,
}

/// Wraps up a call: reads the error if it failed, prints, returns.
///
/// # Safety
///
/// Every engine call must be serialised; this must run right after the call it reports.
unsafe fn outcome(label: &str, rc: c_int, out: &[u8]) -> Outcome {
    let error = if rc < 0 {
        // SAFETY: serialised by the caller.
        unsafe { last_error() }
    } else {
        String::new()
    };
    let text = written(out);
    let shown: String = text.chars().take(200).collect();
    println!("{label}: rc {rc}, error {error:?}\n  {shown}");
    Outcome { rc, text, error }
}

/// `cactus_stream_transcribe_process` of `pcm` with a buffer of `capacity` bytes.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn process(label: &str, pcm: &[f32], language: Option<&CStr>, capacity: usize) -> Outcome {
    let mut out = vec![0x55_u8; capacity.max(1)];
    // SAFETY: `pcm` is readable for its length, the language is null or NUL-terminated, and
    // `out` is writable for `capacity` bytes (0 passes a one-byte buffer and says 0);
    // serialised by the caller.
    let rc = unsafe {
        cactus_stream_transcribe_process(
            pcm.as_ptr(),
            pcm.len() as c_int,
            language.map_or(core::ptr::null(), CStr::as_ptr),
            core::ptr::null(),
            out.as_mut_ptr().cast::<c_char>(),
            capacity as c_int,
        )
    };
    // SAFETY: serialised by the caller, right after the call.
    unsafe { outcome(label, rc, &out) }
}

/// `cactus_stream_transcribe_stop` with a buffer of `capacity` bytes.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn stop(label: &str, capacity: usize) -> Outcome {
    let mut out = vec![0x55_u8; capacity.max(1)];
    // SAFETY: `out` is writable for `capacity` bytes; serialised by the caller.
    let rc = unsafe {
        cactus_stream_transcribe_stop(out.as_mut_ptr().cast::<c_char>(), capacity as c_int)
    };
    // SAFETY: serialised by the caller, right after the call.
    unsafe { outcome(label, rc, &out) }
}

/// `cactus_transcribe` of `pcm` with a buffer of `capacity` bytes.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn transcribe(label: &str, pcm: &[f32], words: bool, capacity: usize) -> Outcome {
    let mut out = vec![0x55_u8; capacity.max(1)];
    // SAFETY: `pcm` is readable for its length and `out` writable for `capacity` bytes;
    // serialised by the caller.
    let rc = unsafe {
        cactus_transcribe(
            pcm.as_ptr(),
            pcm.len() as c_int,
            core::ptr::null(),
            core::ptr::null(),
            c_int::from(words),
            out.as_mut_ptr().cast::<c_char>(),
            capacity as c_int,
        )
    };
    // SAFETY: serialised by the caller, right after the call.
    unsafe { outcome(label, rc, &out) }
}

/// The value of a string field of a flat JSON document, read without a JSON crate.
fn field<'a>(document: &'a str, key: &str) -> &'a str {
    let start = document
        .find(&format!("\"{key}\":\""))
        .unwrap_or_else(|| panic!("no `{key}` in {document}"))
        + key.len()
        + 4;
    let end = document[start..].find('"').expect("a closing quote");
    &document[start..start + end]
}

/// Lower-case words of `text`, without punctuation.
fn plain(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn probe_stream() {
    let Some(whistle) = archive("CACTUS_WHISTLE_WEIGHTS") else {
        return;
    };
    let jfk = jfk();
    let second = 16_000;

    // SAFETY: one thread, every call in order; each buffer outlives its call.
    unsafe {
        // ---- Before a speech model is loaded both calls fail with a reason.
        for outcome in [
            process("process before load", &jfk[..second], None, OUT),
            stop("stop before load", OUT),
        ] {
            assert!(outcome.rc < 0);
            assert_eq!(
                outcome.error,
                "no speech model loaded; pass a Whistle .cact to cactus_load"
            );
        }

        assert_eq!(
            cactus_load(whistle.as_ptr(), whistle.len() as c_ulonglong),
            0
        );

        // ---- With no stream open, stop is not an error: it writes the empty update.
        let idle = stop("stop with no stream", OUT);
        assert!(
            idle.rc == 0 && idle.text.starts_with(NOTHING),
            "{}",
            idle.text
        );
        // Nor is an empty feed.
        let empty = process("empty feed", &jfk[..0], None, OUT);
        assert!(
            empty.rc == 0 && empty.text.starts_with(NOTHING),
            "{}",
            empty.text
        );

        // ---- One-shot, for comparison.
        let whole = transcribe("one-shot", &jfk, false, OUT);
        assert!(whole.rc > 0);
        let heard = plain(field(&whole.text, "text"));
        assert!(heard.starts_with("and so my fellow americans"), "{heard}");

        // ---- A second at a time: the return value is the number of words this call
        // committed, `text` is those words, `pending` the unconfirmed tail, `received` the
        // seconds taken in so far. Stop commits the tail and empties it.
        let mut texts = Vec::new();
        let mut committed = 0;
        let mut tail_seen = false;
        for (index, chunk) in jfk.chunks(second).enumerate() {
            let update = process(&format!("second {}", index + 1), chunk, None, OUT);
            assert!(update.rc >= 0);
            committed += update.rc;
            let text = field(&update.text, "text");
            assert_eq!(
                text.split_whitespace().count() as c_int,
                update.rc,
                "{}",
                update.text
            );
            assert_eq!(update.rc == 0, update.text.contains("\"words\":[]"));
            if !text.is_empty() {
                texts.push(text.to_owned());
            }
            tail_seen |= !field(&update.text, "pending").is_empty();
            let seconds = (index * second + chunk.len()) as f32 / second as f32;
            assert!(update.text.contains(&format!("\"received\":{seconds:.2},")));
            assert!(update.text.contains("\"language\":\"en\""));
        }
        assert!(tail_seen, "no update had a pending tail");
        let last = stop("stop", OUT);
        assert!(last.rc >= 0);
        committed += last.rc;
        texts.push(field(&last.text, "text").to_owned());
        assert_eq!(field(&last.text, "pending"), "");
        // The stream heard what the one-shot call heard, word for word; punctuation differs.
        assert_eq!(plain(&texts.join(" ")), heard);
        assert_eq!(committed as usize, heard.split(' ').count());
        // Stop ended the stream: a second stop has nothing.
        let again = stop("second stop", OUT);
        assert!(
            again.rc == 0 && again.text.starts_with(NOTHING),
            "{}",
            again.text
        );

        // ---- Much shorter feeds commit words the model has barely heard, and the transcript
        // is wrong. This is why `cactus-rs` collects audio into whole seconds.
        let mut texts = Vec::new();
        for chunk in jfk.chunks(second / 10) {
            let update = process("100 ms", chunk, None, OUT);
            assert!(update.rc >= 0);
            texts.push(field(&update.text, "text").to_owned());
        }
        texts.push(field(&stop("stop after 100 ms feeds", OUT).text, "text").to_owned());
        let choppy = plain(&texts.join(" "));
        println!("100 ms feeds heard: {choppy}");
        assert_ne!(
            choppy, heard,
            "100 ms feeds now transcribe as well as 1 s ones"
        );

        // ---- A whole clip in one feed commits nothing; everything is pending until stop.
        let all = process("the whole clip in one feed", &jfk, None, OUT);
        assert_eq!(all.rc, 0);
        assert_eq!(plain(field(&all.text, "pending")), heard);
        let flushed = stop("stop after one feed", OUT);
        assert_eq!(plain(field(&flushed.text, "text")), heard);

        // ---- A one-shot transcription in the middle of a stream does not disturb it.
        let half = jfk.len() / 2;
        assert!(process("first half", &jfk[..half], None, OUT).rc >= 0);
        assert!(transcribe("one-shot mid-stream", &jfk[..second], false, OUT).rc >= 0);
        let rest = process("second half", &jfk[half..], None, OUT);
        let end = stop("stop after the interruption", OUT);
        let both = format!("{} {}", field(&rest.text, "text"), field(&end.text, "text"));
        assert_eq!(plain(&both), heard);

        // ---- The buffer. Neither call truncates: a result that does not fit fails with the
        // size it needs. A stop that fails this way still ends the stream, so its words are
        // lost; a caller must give stop enough room the first time.
        let small = process("process, 16-byte buffer", &jfk[..2 * second], None, 16);
        assert!(small.rc < 0);
        assert!(
            small.error.starts_with("output buffer is too small: ")
                && small.error.ends_with(" bytes needed"),
            "{}",
            small.error
        );
        assert_eq!(small.text, "");
        let none = process("process, no buffer", &jfk[..second], None, 0);
        assert!(none.rc < 0);
        assert_eq!(none.error, "no output buffer");
        assert_eq!(none.text, UNTOUCHED);
        let lost = stop("stop, 8-byte buffer", 8);
        assert!(lost.rc < 0 && lost.error.starts_with("output buffer is too small: "));
        let after = stop("stop after the failed stop", OUT);
        assert!(
            after.rc == 0 && after.text.starts_with(NOTHING),
            "{}",
            after.text
        );

        // ---- Languages are checked as `cactus_transcribe` checks them.
        let refused = process("language xx", &jfk[..second], Some(c"xx"), OUT);
        assert!(refused.rc < 0);
        assert_eq!(refused.error, "unknown language xx");
        assert!(process("language en", &jfk[..second], Some(c"en"), OUT).rc >= 0);
        assert!(stop("stop", OUT).rc >= 0);

        // ---- `cactus_transcribe` since engine 4: any length, and no truncation.
        let short = transcribe("one-shot, 32-byte buffer", &jfk, false, 32);
        assert!(short.rc < 0);
        assert_eq!(
            short.error,
            format!(
                "output buffer is too small: {} bytes needed",
                whole.text.len() + 1
            )
        );
        assert_eq!(short.text, "");

        // 33 seconds: three windows' worth of the same sentence, with word times that run on
        // from the start of the clip.
        let long: Vec<f32> = jfk.iter().chain(&jfk).chain(&jfk).copied().collect();
        assert!(long.len() > 480_000);
        let thrice = transcribe("33 seconds", &long, true, OUT);
        assert!(thrice.rc > 0);
        let text = plain(field(&thrice.text, "text"));
        assert_eq!(
            text.matches("ask not what your country").count(),
            3,
            "{text}"
        );
        assert!(thrice.text.contains("\"start\":3"), "no word after 30 s");

        // An embedding is still bounded at 30 seconds.
        let floats = cactus_embed(
            core::ptr::null(),
            long.as_ptr(),
            long.len() as c_int,
            core::ptr::null_mut(),
            0,
        );
        assert!(floats < 0);
        assert_eq!(last_error(), "embedding audio limit is 30 s");

        // ---- Threads. Any value is accepted, and the transcript does not depend on it. On the
        // machine this was written on the time did not either, so `cactus-rs` exposes no knob.
        for threads in [1, 2, 8, 0, -1] {
            cactus_set_threads(threads);
            let started = std::time::Instant::now();
            let same = transcribe(&format!("{threads} thread(s)"), &jfk, false, OUT);
            println!("  in {} ms", started.elapsed().as_millis());
            assert_eq!(field(&same.text, "text"), field(&whole.text, "text"));
        }
    }
}
