//! Probe: what `cactus_describe` accepts, writes and refuses, and what loading a vision model
//! does to a loaded text model.
//!
//! Upstream publishes the signature of `cactus_describe` and nothing else, so the contract
//! `cactus-rs` builds on is the one measured here. Its own test binary, so its own process:
//! engine state is global and cannot be undone. Skips unless `CACTUS_MIRAGE_WEIGHTS` names a
//! `mirage.cact`; the text-model half also needs `CACTUS_NEEDLE_WEIGHTS`. Every measurement is
//! printed and the observed behaviour asserted, so an engine that changes it turns this red.
//!
//! The image is drawn here: a red square on blue. The probe is about the interface, not about
//! what the model recognises, which `cactus-rs/tests/mirage_tests.rs` covers.

#![cfg(feature = "engine")]

use core::ffi::{CStr, c_char, c_int, c_ulonglong};
use std::ffi::CString;

use cactus_sys::{
    CACTUS_TEXT, CACTUS_VISION, cactus_complete, cactus_describe, cactus_init, cactus_last_error,
    cactus_load, cactus_models,
};

// ---- Shared probe helpers, duplicated in each probe file on purpose: every probe is its own
// test binary (so its own process), because engine state is global and cannot be undone. ----

/// The output buffer size the probe hands the engine unless it is testing the buffer itself.
const OUT: usize = 1024 * 1024;

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

/// The width of the probe's image.
const WIDTH: usize = 320;

/// The height of the probe's image.
const HEIGHT: usize = 240;

/// The image: blue, with a red square in the middle.
fn image() -> Vec<u8> {
    let mut rgb = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let red = (110..210).contains(&x) && (70..170).contains(&y);
            rgb.extend_from_slice(if red {
                &[200, 16, 46]
            } else {
                &[143, 180, 217]
            });
        }
    }
    rgb
}

/// One `cactus_describe`: the return code, what was written, and the error when it failed.
struct Described {
    rc: c_int,
    text: String,
    error: String,
}

/// What [`describe`] reports as written when the engine left the buffer untouched.
const UNTOUCHED: &str = "<no NUL written>";

/// Runs `cactus_describe` on `rgb` with a buffer of `capacity` bytes pre-filled with `0x55`.
///
/// # Safety
///
/// Every engine call must be serialised. `rgb` must hold at least `width * height * 3` bytes
/// whenever both are positive.
unsafe fn describe(
    label: &str,
    rgb: &[u8],
    (width, height): (c_int, c_int),
    options: Option<&str>,
    language: Option<&str>,
    capacity: usize,
) -> Described {
    // Not zeroed, so that "the engine wrote nothing" and "the engine wrote an empty string"
    // can be told apart.
    let mut out = vec![0x55_u8; capacity.max(1)];
    let options = options.map(|text| CString::new(text).expect("no NUL in the options"));
    let language = language.map(|text| CString::new(text).expect("no NUL in the language"));
    // SAFETY: `rgb` is readable for the bytes the dimensions describe, per the caller; the
    // strings are null or NUL-terminated and outlive the call; `out` is writable for `capacity`
    // bytes (a capacity of 0 passes a one-byte buffer and says 0); serialised by the caller.
    let rc = unsafe {
        cactus_describe(
            rgb.as_ptr(),
            width,
            height,
            options.as_ref().map_or(core::ptr::null(), |o| o.as_ptr()),
            language.as_ref().map_or(core::ptr::null(), |l| l.as_ptr()),
            core::ptr::null(),
            out.as_mut_ptr().cast::<c_char>(),
            capacity as c_int,
        )
    };
    let error = if rc < 0 {
        // SAFETY: serialised by the caller.
        unsafe { last_error() }
    } else {
        String::new()
    };
    let text = match out.iter().position(|byte| *byte == 0) {
        Some(end) => String::from_utf8_lossy(&out[..end]).into_owned(),
        None => String::from(UNTOUCHED),
    };
    let shown: String = text.chars().take(160).collect();
    println!(
        "{label}: rc {rc}, {} bytes, error {error:?}\n  {shown}",
        text.len()
    );
    Described { rc, text, error }
}

/// One short text completion: the return code and the envelope.
///
/// # Safety
///
/// Every engine call must be serialised.
unsafe fn complete(label: &str) -> (c_int, String) {
    let mut out = vec![0_u8; OUT];
    // SAFETY: the input is a NUL-terminated literal, `pcm` is null with `samples` 0, and `out`
    // is writable for its length; serialised by the caller.
    let rc = unsafe {
        cactus_complete(
            c"hello".as_ptr(),
            core::ptr::null(),
            0,
            16,
            out.as_mut_ptr().cast::<c_char>(),
            out.len() as c_int,
        )
    };
    let end = out.iter().position(|byte| *byte == 0).unwrap_or(0);
    let text = String::from_utf8_lossy(&out[..end]).into_owned();
    println!("{label}: cactus_complete rc {rc}\n  {text}");
    (rc, text)
}

#[test]
fn probe_vision() {
    let Some(mirage) = archive("CACTUS_MIRAGE_WEIGHTS") else {
        return;
    };
    let rgb = image();
    let size = (WIDTH as c_int, HEIGHT as c_int);

    // SAFETY: one thread, every call in order; each buffer outlives its call, and `rgb` holds
    // WIDTH x HEIGHT x 3 bytes, which is at least what every call below describes.
    unsafe {
        // ---- Before any vision model is loaded: a failure with a reason, not a crash.
        let before = describe("before load", &rgb, size, None, None, OUT);
        assert!(before.rc < 0);
        assert_eq!(
            before.error,
            "no vision model loaded; pass a Mirage .cact to cactus_load"
        );

        // ---- A text model first, when there is one, to see what a vision load does to it.
        let needle = archive("CACTUS_NEEDLE_WEIGHTS");
        if let Some(needle) = &needle {
            assert_eq!(cactus_load(needle.as_ptr(), needle.len() as c_ulonglong), 0);
            assert!(
                cactus_init(
                    c"You are terse.".as_ptr(),
                    c"[]".as_ptr(),
                    core::ptr::null()
                ) > 0
            );
            let (rc, text) = complete("text before the vision load");
            assert!(rc >= 0 && !text.contains("cactus_init not called"));
        }

        // ---- The load: it sets the vision bit and keeps whatever else is loaded.
        let rc = cactus_load(mirage.as_ptr(), mirage.len() as c_ulonglong);
        let models = cactus_models();
        println!("mirage: cactus_load rc {rc}, cactus_models {models}");
        assert_eq!(rc, 0, "{}", last_error());
        assert_ne!(models & CACTUS_VISION, 0);
        if needle.is_some() {
            assert_ne!(models & CACTUS_TEXT, 0);
            // Unlike a text load, a vision load does not drop the text model's init.
            let (rc, text) = complete("text after the vision load");
            assert!(
                rc >= 0 && !text.contains("cactus_init not called"),
                "{text}"
            );
        }

        // ---- A description: a positive token count and one NUL-terminated JSON object with
        // the seven sections in a fixed order.
        let full = describe("default", &rgb, size, None, None, OUT);
        assert!(full.rc > 0);
        assert!(full.text.starts_with("{\"scene\":{") && full.text.ends_with('}'));
        let mut at = 0;
        for key in [
            "scene",
            "objects",
            "regions",
            "text",
            "relations",
            "events",
            "reasoning",
        ] {
            let found = full.text[at..]
                .find(&format!("\"{key}\":"))
                .unwrap_or_else(|| panic!("no `{key}` after byte {at}"));
            at += found;
        }

        // The same image gives the same bytes: nothing is sampled.
        let again = describe("default again", &rgb, size, None, None, OUT);
        assert_eq!((again.rc, &again.text), (full.rc, &full.text));

        // Null options, an empty object, an empty language and unknown keys all mean defaults.
        for (label, options, language) in [
            ("empty options", Some("{}"), None),
            ("unknown key", Some(r#"{"bogus":1}"#), None),
            ("empty language", None, Some("")),
        ] {
            let same = describe(label, &rgb, size, options, language, OUT);
            assert_eq!(same.text, full.text, "{label}");
        }

        // Options that are not JSON are NOT an error: they are ignored. A caller that builds
        // the string by hand and gets it wrong is silently given the defaults.
        let ignored = describe("not json", &rgb, size, Some("sections=text"), None, OUT);
        assert_eq!((ignored.rc, &ignored.text), (full.rc, &full.text));

        // ---- The buffer. A result that does not fit fails, names the size it needs (with the
        // NUL) and leaves an empty string; it is never truncated.
        let needed = full.text.len() + 1;
        let small = describe("small buffer", &rgb, size, None, None, 64);
        assert!(small.rc < 0);
        assert_eq!(
            small.error,
            format!("output buffer is too small: {needed} bytes needed")
        );
        assert_eq!(
            small.text, "",
            "an empty string is written, not a partial document"
        );
        let one_short = describe("one byte short", &rgb, size, None, None, needed - 1);
        assert!(one_short.rc < 0);
        let exact = describe("exactly enough", &rgb, size, None, None, needed);
        assert_eq!((exact.rc, &exact.text), (full.rc, &full.text));
        let zero = describe("zero capacity", &rgb, size, None, None, 0);
        assert!(zero.rc < 0);
        assert_eq!(zero.error, "invalid output buffer");

        // ---- The image. Dimensions are checked; any positive size is accepted.
        for bad in [(0, HEIGHT as c_int), (WIDTH as c_int, 0), (-1, 1)] {
            let refused = describe("bad dimensions", &rgb, bad, None, None, OUT);
            assert!(refused.rc < 0);
            assert_eq!(
                refused.error,
                "invalid image: rgb must hold width x height x 3 bytes"
            );
        }
        assert!(describe("one pixel", &rgb[..3], (1, 1), None, None, OUT).rc > 0);

        // ---- Sections and fields: a list of known names, or a refusal that names the stranger.
        let text = describe(
            "text only",
            &rgb,
            size,
            Some(r#"{"sections":["text"]}"#),
            None,
            OUT,
        );
        assert!(text.text.starts_with("{\"text\":[") && !text.text.contains("\"scene\""));
        let none = describe(
            "no sections",
            &rgb,
            size,
            Some(r#"{"sections":[]}"#),
            None,
            OUT,
        );
        assert_eq!(none.text, "{}");
        for (options, error) in [
            (
                r#"{"sections":["trace"]}"#,
                "invalid options: sections has an unknown name trace",
            ),
            (
                r#"{"sections":"text"}"#,
                "invalid options: sections must be a list of names",
            ),
            (
                r#"{"fields":["name"]}"#,
                "invalid options: fields has an unknown name name",
            ),
        ] {
            let refused = describe(options, &rgb, size, Some(options), None, OUT);
            assert!(refused.rc < 0);
            assert_eq!(refused.error, error);
        }
        for field in [
            "bbox",
            "score",
            "mask",
            "point",
            "position",
            "description",
            "count",
        ] {
            let options = format!(r#"{{"sections":["objects"],"fields":["{field}"]}}"#);
            assert!(describe(field, &rgb, size, Some(&options), None, OUT).rc > 0);
        }

        // ---- Every numeric option is range-checked, in these words.
        for (options, error) in [
            (r#"{"conf":2}"#, "conf must be a number from 0.0 to 1.0"),
            (
                r#"{"text_conf":-1}"#,
                "text_conf must be a number from 0.0 to 1.0",
            ),
            (r#"{"iou":7}"#, "iou must be a number from 0.0 to 1.0"),
            (
                r#"{"mask_threshold":7}"#,
                "mask_threshold must be a number from 0.0 to 1.0",
            ),
            (
                r#"{"mask_tolerance":-1}"#,
                "mask_tolerance must be a number from 0.0 up",
            ),
            (
                r#"{"keyword_bias":101}"#,
                "keyword_bias must be a number from 0.0 to 100.0",
            ),
            (r#"{"conf":null}"#, "conf must be a number from 0.0 to 1.0"),
            (
                r#"{"max_objects":0}"#,
                "max_objects must be a whole number of at least 1",
            ),
            (
                r#"{"max_text_lines":0}"#,
                "max_text_lines must be a whole number of at least 1",
            ),
            (
                r#"{"max_tokens":0}"#,
                "max_tokens must be a whole number of at least 1",
            ),
            (r#"{"nms":1}"#, "nms must be true or false"),
            (
                r#"{"agnostic_nms":1}"#,
                "agnostic_nms must be true or false",
            ),
            (
                r#"{"mask_format":"png"}"#,
                "mask_format must be largest, merged, polygons or bitmap",
            ),
        ] {
            let refused = describe(options, &rgb, size, Some(options), None, OUT);
            assert!(refused.rc < 0, "{options}");
            assert_eq!(refused.error, format!("invalid options: {error}"));
        }

        // A token budget that runs out is not an error: the object is still closed.
        let short = describe(
            "eight tokens",
            &rgb,
            size,
            Some(r#"{"max_tokens":8}"#),
            None,
            OUT,
        );
        assert_eq!(short.rc, 8);
        assert!(short.text.starts_with('{') && short.text.ends_with('}'));

        // ---- Masks. A bitmap is `[height, width]` of the image given, not of the canvas.
        let bitmap = describe(
            "bitmap",
            &rgb,
            size,
            Some(r#"{"sections":["objects"],"fields":["mask"],"mask_format":"bitmap"}"#),
            None,
            OUT,
        );
        assert!(bitmap.rc > 0);
        if bitmap.text.contains("\"mask\"") {
            let expected = format!("\"mask\":{{\"size\":[{HEIGHT},{WIDTH}],\"counts\":[");
            assert!(bitmap.text.contains(&expected), "{}", bitmap.text);
        }

        // ---- Languages: the seven codes, and a refusal for any other.
        for code in ["en", "de", "fr", "es", "it", "nl", "pl"] {
            let scene = Some(r#"{"sections":["scene"]}"#);
            assert!(describe(code, &rgb, size, scene, Some(code), OUT).rc > 0);
        }
        let refused = describe("language xx", &rgb, size, None, Some("xx"), OUT);
        assert!(refused.rc < 0);
        assert_eq!(refused.error, "unsupported language: xx");

        // ---- Forms: a separate mode that writes only `form`, in the order asked.
        let form = describe(
            "form",
            &rgb,
            size,
            Some(concat!(
                r#"{"form":{"colour":{"enum":["red","green"]},"#,
                r#""n":{"type":"integer","minimum":0,"maximum":2},"#,
                r#""note":{"type":"string","maxLength":8},"#,
                r#""where":{"x-source":"bbox","x-of":"colour"}},"sections":["scene"]}"#
            )),
            None,
            OUT,
        );
        assert!(form.rc > 0);
        assert!(form.text.starts_with("{\"form\":{\"colour\":{\"value\":"));
        assert!(!form.text.contains("\"scene\""), "sections apply to a form");
        // A choice: every label and the abstain label `-`, in order.
        let labels = form
            .text
            .find("\"probabilities\":{\"red\":")
            .expect("red first");
        let green = form.text[labels..].find(",\"green\":").expect("then green");
        assert!(form.text[labels + green..].contains(",\"-\":"));
        // A count: one probability per number in the range.
        assert!(form.text.contains("\"n\":{\"value\":"));
        assert!(form.text.contains("\"probabilities\":{\"0\":"));
        // Text and a box: one value and one probability.
        assert!(form.text.contains("\"note\":{\"value\":"));
        assert!(form.text.contains("\"where\":{\"value\":"));
        assert!(form.text.ends_with("}}}"));

        for (form, error) in [
            (r#"{"form":{}}"#, "form has no fields"),
            (r#"{"form":[]}"#, "form must be an object of fields"),
            (
                r#"{"form":{"a":{"enum":[]}}}"#,
                "field a needs distinct choices without ;|=#@$* or a newline",
            ),
            (
                r#"{"form":{"a":{"type":"integer"}}}"#,
                "field a needs an integer minimum and maximum",
            ),
            (
                r#"{"form":{"a":{"type":"boolean"}}}"#,
                "field a is not a choice, count, box or text field",
            ),
            (
                r#"{"form":{"a":{"type":"string","maxLength":0}}}"#,
                "field a needs a positive maxLength",
            ),
            (
                r#"{"form":{"a;b":{"enum":["x","y"]}}}"#,
                "field name a;b is empty or uses one of ;|=#@$* or a newline",
            ),
            // A box needs both keys, and an earlier choice to point at.
            (
                r#"{"form":{"c":{"enum":["x","y"]},"w":{"x-of":"c"}}}"#,
                "field w is not a choice, count, box or text field",
            ),
        ] {
            let refused = describe(form, &rgb, size, Some(form), None, OUT);
            assert!(refused.rc < 0, "{form}");
            assert_eq!(refused.error, format!("invalid form: {error}"));
        }
        for form in [
            r#"{"form":{"n":{"type":"integer","minimum":0,"maximum":1},"n":{"type":"integer","minimum":0,"maximum":1}}}"#,
            r#"{"form":{"w":{"x-source":"bbox","x-of":"nothing"}}}"#,
            r#"{"form":{"n":{"type":"integer","minimum":3,"maximum":1}}}"#,
        ] {
            let refused = describe(form, &rgb, size, Some(form), None, OUT);
            assert!(refused.rc < 0, "{form}");
            assert!(
                refused.error.starts_with("invalid form: "),
                "{}",
                refused.error
            );
        }

        // ---- A refusal changes nothing: the model still describes, and as it did.
        let last = describe("after the refusals", &rgb, size, None, None, OUT);
        assert_eq!((last.rc, &last.text), (full.rc, &full.text));
    }
}
