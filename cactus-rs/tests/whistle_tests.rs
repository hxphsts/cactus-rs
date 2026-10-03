//! Whistle against the live engine, alone and beside Needle.
//!
//! These tests are about what the speech model hears in real audio and the rules the C API
//! imposes on it: one speech model per process, clips of at most 30 seconds, one engine shared
//! with the text model, and transcription settings that must not leak from one audio turn into
//! the next.
//!
//! The engine is one process-global runtime, and `cargo test` runs a file's tests on parallel
//! threads, so every test that touches it holds [`ENGINE`] for its whole life and lets its
//! [`Whistle`] and [`Needle`] drop before the guard does. Files run in separate processes, so
//! this lock only has to cover this one.
//!
//! The clips are vendored under `tests/data/clips/`; `tests/data/README.md` records where each
//! came from and what it says. Tests that need the speech model skip when
//! `CACTUS_WHISTLE_WEIGHTS` names no readable archive, and those that also need the text model
//! skip when `CACTUS_NEEDLE_WEIGHTS` does not. Nothing here downloads anything.

use std::env;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;

use cactus_rs::needle::{self, CompleteOptions, Needle, Tool};
use cactus_rs::whistle::{
    self, Language, MAX_SAMPLES, SAMPLE_RATE, TranscribeOptions, Whistle, frame_count,
};
use cactus_rs::{Error, Model};
use serde_json::{Value, json};

/// Serialises every engine-touching test in this binary.
static ENGINE: Mutex<()> = Mutex::new(());

/// What Whistle hears in `jfk.wav`, as upstream's own Python binding transcribes it.
const JFK: &str = "And so, my fellow Americans, ask not what your country can do for you, ask \
                   what you can do for your country.";

/// The length of `jfk.wav`, in seconds, which no word may end after (plus a frame's slack).
const JFK_SECONDS: f32 = 11.01;

/// Takes the engine lock, ignoring a poisoning left by an earlier failed test.
///
/// A panicking test leaves nothing behind in the engine that the next test depends on: every
/// transcription stands alone, and every Needle here rewinds before it answers.
fn lock() -> MutexGuard<'static, ()> {
    ENGINE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The Whistle archive, or `None` when this machine has none to hand.
///
/// Only `CACTUS_WHISTLE_WEIGHTS` is consulted: `Weights::fetch()` would reach for the network,
/// and a test suite should not start a download.
fn whistle_weights() -> Option<whistle::Weights> {
    let path = env::var_os("CACTUS_WHISTLE_WEIGHTS")?;
    whistle::Weights::from_file(path).ok()
}

/// The Needle archive, or `None` when this machine has none to hand.
fn needle_weights() -> Option<needle::Weights> {
    let path = env::var_os("CACTUS_NEEDLE_WEIGHTS")?;
    needle::Weights::from_file(path).ok()
}

/// Where `CACTUS_NEEDLE_WEIGHTS` points, for reading the Needle archive as the wrong model.
fn needle_path() -> std::ffi::OsString {
    env::var_os("CACTUS_NEEDLE_WEIGHTS").expect("checked by needle_weights")
}

/// Where `CACTUS_WHISTLE_WEIGHTS` points, for reading the Whistle archive as the wrong model.
fn whistle_path() -> std::ffi::OsString {
    env::var_os("CACTUS_WHISTLE_WEIGHTS").expect("checked by whistle_weights")
}

/// The line a skipped test prints, so a green run with no engine is not mistaken for a real one.
fn skipped(what: &str) {
    println!("skipping {what}: set CACTUS_WHISTLE_WEIGHTS to a whistle.cact to run it");
}

/// The line a skipped test that needs both models prints.
fn skipped_both(what: &str) {
    println!(
        "skipping {what}: set CACTUS_WHISTLE_WEIGHTS to a whistle.cact and \
         CACTUS_NEEDLE_WEIGHTS to a needle3.cact to run it"
    );
}

/// Reads `tests/data/clips/<name>.wav` as the `f32` samples Whistle takes.
///
/// Every vendored clip is 16 kHz mono 16-bit PCM, so this checks that rather than converting.
fn clip(name: &str) -> Vec<f32> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "data", "clips"]
        .iter()
        .collect::<PathBuf>()
        .join(format!("{name}.wav"));
    let mut reader = hound::WavReader::open(&path)
        .unwrap_or_else(|error| panic!("{} opens: {error}", path.display()));

    let spec = reader.spec();
    assert_eq!(spec.sample_rate, SAMPLE_RATE, "{name} is 16 kHz");
    assert_eq!(spec.channels, 1, "{name} is mono");
    assert_eq!(spec.bits_per_sample, 16, "{name} is 16-bit");
    assert_eq!(
        spec.sample_format,
        hound::SampleFormat::Int,
        "{name} is PCM"
    );

    reader
        .samples::<i16>()
        .map(|sample| f32::from(sample.expect("a whole sample")) / 32768.0)
        .collect()
}

/// The Whistle these tests transcribe with, built from `weights` with default options.
fn build(weights: whistle::Weights) -> Whistle {
    Whistle::builder(weights)
        .build()
        .expect("the speech model is free")
}

/// The lights Needle from the README: one tool, one system prompt.
fn lights(weights: needle::Weights) -> Needle {
    Needle::builder(weights)
        .system("You control the lights.")
        .tool(Tool::new(
            "set_light",
            "Turn a room light on or off",
            json!({
                "type": "object",
                "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
                "required": ["room", "on"]
            }),
        ))
        .build()
        .expect("the text model is free")
}

/// Each call of a completion as `(name, arguments)`, for comparing whole turns.
fn calls(completion: &needle::Completion) -> Vec<(String, Value)> {
    completion
        .calls()
        .iter()
        .map(|call| (call.name().to_owned(), call.arguments().clone()))
        .collect()
}

/// Transcribes `jfk.wav` as upstream's Python binding does.
///
/// Measured on 2026-10-03 against Hugging Face commit `c7c415a3` of the engine, Linux x86_64;
/// the engine inside `cactus-needle` 3.1.0's wheel gave the same text. Kernels differ by
/// architecture, so elsewhere the check is the phrase that matters rather than every comma.
#[test]
fn jfk_transcribes_as_upstream_does() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the jfk parity test");
        return;
    };

    let mut whistle = build(weights);
    let transcript = whistle
        .transcribe(&clip("jfk"))
        .expect("the clip transcribes");

    if cfg!(target_arch = "x86_64") {
        assert_eq!(transcript.text(), JFK);
    } else {
        assert!(
            transcript
                .text()
                .to_lowercase()
                .contains("ask not what your country can do for you"),
            "got {:?}",
            transcript.text()
        );
    }
    assert_eq!(transcript.language(), Some(Language::En));
    assert_eq!(transcript.language_code(), "en");
    assert!(
        transcript.words().is_empty(),
        "no timestamps were asked for"
    );
}

#[test]
fn a_forced_language_relabels_the_transcript() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the forced-language test");
        return;
    };

    let mut whistle = build(weights);
    let jfk = clip("jfk");
    let detected = whistle.transcribe(&jfk).expect("the clip transcribes");
    let forced = whistle
        .transcribe_with_options(&jfk, &TranscribeOptions::new().with_language(Language::De))
        .expect("the clip transcribes");

    assert_eq!(forced.language(), Some(Language::De));
    assert_eq!(
        forced.text(),
        detected.text(),
        "the words are still English"
    );
}

#[test]
fn silence_transcribes_to_nothing() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the silence test");
        return;
    };

    let mut whistle = build(weights);
    let timestamps = TranscribeOptions::new().with_word_timestamps(true);

    for (what, pcm) in [
        ("silence.wav", clip("silence")),
        ("a second of zeros", vec![0.0; SAMPLE_RATE as usize]),
    ] {
        let transcript = whistle
            .transcribe_with_options(&pcm, &timestamps)
            .expect("silence is not an error");
        assert!(transcript.is_empty(), "{what}: got {:?}", transcript.text());
        assert_eq!(transcript.language(), None, "{what}");
        assert!(transcript.words().is_empty(), "{what}");
    }
}

#[test]
fn an_empty_clip_is_not_an_error() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the empty-clip test");
        return;
    };

    let mut whistle = build(weights);
    let transcript = whistle.transcribe(&[]).expect("an empty clip transcribes");

    assert!(transcript.is_empty(), "got {:?}", transcript.text());
    assert_eq!(transcript.language(), None);
}

#[test]
fn thirty_seconds_is_the_limit_and_nan_is_refused() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the clip-limit test");
        return;
    };

    let mut whistle = build(weights);

    let longest = vec![0.0; MAX_SAMPLES];
    whistle
        .transcribe(&longest)
        .expect("thirty seconds is accepted");

    let too_long = vec![0.0; MAX_SAMPLES + 1];
    let error = whistle
        .transcribe(&too_long)
        .expect_err("a sample more is not");
    assert!(
        matches!(error, Error::AudioTooLong { samples } if samples == MAX_SAMPLES + 1),
        "got {error:?}"
    );

    let mut poisoned = clip("lights_en");
    poisoned[1_000] = f32::NAN;
    let error = whistle
        .transcribe(&poisoned)
        .expect_err("a NaN is not audio");
    assert!(
        matches!(error, Error::NonFiniteSample { index: 1_000 }),
        "got {error:?}"
    );
}

#[test]
fn word_timestamps_are_ordered_and_inside_the_clip() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the word-timestamp test");
        return;
    };

    let mut whistle = build(weights);
    let jfk = clip("jfk");
    let transcript = whistle
        .transcribe_with_options(&jfk, &TranscribeOptions::new().with_word_timestamps(true))
        .expect("the clip transcribes");
    let words = transcript.words();

    if cfg!(target_arch = "x86_64") {
        assert_eq!(words.len(), 22, "one entry per word of {JFK:?}");
    } else {
        assert!(!words.is_empty());
    }

    for pair in words.windows(2) {
        assert!(
            pair[0].start() <= pair[1].start(),
            "{:?} starts after {:?}",
            pair[0],
            pair[1]
        );
    }
    for word in words {
        assert!(!word.word().is_empty());
        assert!(
            0.0 <= word.start() && word.start() <= word.end() && word.end() <= JFK_SECONDS,
            "{word:?} is outside the clip"
        );
        assert!(
            (0.0..=1.0).contains(&word.probability()),
            "{word:?} has no probability"
        );
    }

    let plain = whistle.transcribe(&jfk).expect("the clip transcribes");
    assert!(plain.words().is_empty(), "no timestamps were asked for");
}

/// Keyword biasing turns a name the model cannot spell into the one it was told to expect.
///
/// `names_en.wav` says "please call Siobhan and Krzysztof". Without keywords the pinned engine
/// hears "Please call Chevorne and Crystal."; with `Siobhan` and `Krzysztof` it hears "Please
/// call Shivorn and Krzysztof.", as upstream's Python binding does.
#[test]
fn keywords_bias_the_transcript_towards_them() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the keyword-biasing test");
        return;
    };

    let mut whistle = build(weights);
    let names = clip("names_en");

    let plain = whistle.transcribe(&names).expect("the clip transcribes");
    let biased = whistle
        .transcribe_with_options(
            &names,
            &TranscribeOptions::new().with_keywords(["Siobhan", "Krzysztof"]),
        )
        .expect("the clip transcribes");

    assert!(
        !plain.text().contains("Krzysztof"),
        "got {:?}",
        plain.text()
    );
    assert!(
        biased.text().contains("Krzysztof"),
        "got {:?}",
        biased.text()
    );

    let error = whistle
        .transcribe_with_options(&names, &TranscribeOptions::new().with_keyword("a\nb"))
        .expect_err("a line break would split the keyword");
    assert!(
        matches!(&error, Error::InvalidKeyword { keyword } if keyword == "a\nb"),
        "got {error:?}"
    );

    let error = whistle
        .transcribe_with_options(&names, &TranscribeOptions::new().with_keyword("a\0"))
        .expect_err("a NUL is not a C string");
    assert!(
        matches!(error, Error::InteriorNul { field: "keywords" }),
        "got {error:?}"
    );
}

#[test]
fn an_embedding_has_one_row_per_frame() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the speech embedding test");
        return;
    };

    let mut whistle = build(weights);
    let width = whistle.embedding_width().expect("the model has a width");
    assert_eq!(width, 512);

    let lights = clip("lights_en");
    let first = whistle.embed(&lights).expect("the clip embeds");
    let again = whistle.embed(&lights).expect("the clip embeds");

    assert_eq!(first.len(), frame_count(lights.len()) * width);
    assert_eq!(first, again, "the same clip embeds to the same rows");
    assert!(
        first.iter().any(|value| *value != 0.0),
        "an all-zero embedding is not an embedding"
    );

    let empty = whistle.embed(&[]).expect("an empty clip embeds");
    assert_eq!(empty.len(), width, "an empty clip still gets one frame");
}

#[test]
fn a_second_whistle_finds_the_speech_model_busy() {
    let _engine = lock();
    let (Some(first), Some(second), Some(third)) =
        (whistle_weights(), whistle_weights(), whistle_weights())
    else {
        skipped("the speech-model-busy test");
        return;
    };

    let alive = build(first);
    let error = Whistle::builder(second)
        .build()
        .expect_err("the speech model is taken");
    assert!(matches!(error, Error::EngineBusy), "got {error:?}");

    drop(alive);
    let mut again = Whistle::builder(third)
        .build()
        .expect("the slot is free again");
    assert!(
        again
            .transcribe(&clip("silence"))
            .expect("it transcribes")
            .is_empty()
    );
}

#[test]
fn a_whistle_built_on_one_thread_transcribes_on_another() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the cross-thread test");
        return;
    };

    let lights = clip("lights_en");
    let mut whistle = thread::spawn(move || build(weights))
        .join()
        .expect("the builder thread finishes");
    let here = whistle.transcribe(&lights).expect("the clip transcribes");

    let (whistle, there) = thread::spawn(move || {
        let there = whistle.transcribe(&lights).expect("the clip transcribes");
        (whistle, there)
    })
    .join()
    .expect("the transcribing thread finishes");

    assert_eq!(there.text(), here.text());
    assert_eq!(there.language(), here.language());
    drop(whistle);
}

/// Loads both models from the right archives, so a mismatched archive is then known by its
/// fingerprint and named as the wrong model whichever test loaded what first.
fn load_both(needle_weights: needle::Weights, whistle_weights: whistle::Weights) {
    let mut needle = lights(needle_weights);
    needle.reset();
    let completion = needle
        .complete("turn the kitchen light on")
        .expect("the engine answers");
    assert_eq!(completion.calls()[0].name(), "set_light");

    let mut whistle = build(whistle_weights);
    assert_eq!(
        whistle
            .transcribe(&clip("lights_en"))
            .expect("the clip transcribes")
            .text(),
        "Turn off the kitchen lights."
    );
}

#[test]
fn a_needle_archive_is_not_a_whistle() {
    let _engine = lock();
    let (Some(first), Some(second), Some(whistle_weights)) =
        (needle_weights(), needle_weights(), whistle_weights())
    else {
        skipped_both("the wrong-model test for Whistle");
        return;
    };
    load_both(first, whistle_weights.clone());

    let as_whistle =
        whistle::Weights::from_file(needle_path()).expect("both archives share one magic tag");
    let error = Whistle::builder(as_whistle)
        .build()
        .expect_err("a Needle archive is not a Whistle archive");
    assert!(
        matches!(
            error,
            Error::WrongModel {
                expected: Model::Whistle
            }
        ),
        "got {error:?}"
    );

    // The refusal left both models usable.
    load_both(second, whistle_weights);
}

#[test]
fn a_whistle_archive_is_not_a_needle() {
    let _engine = lock();
    let (Some(needle_weights), Some(first), Some(second)) =
        (needle_weights(), whistle_weights(), whistle_weights())
    else {
        skipped_both("the wrong-model test for Needle");
        return;
    };
    load_both(needle_weights.clone(), first);

    let as_needle =
        needle::Weights::from_file(whistle_path()).expect("both archives share one magic tag");
    let error = Needle::builder(as_needle)
        .build()
        .expect_err("a Whistle archive is not a Needle archive");
    assert!(
        matches!(
            error,
            Error::WrongModel {
                expected: Model::Needle
            }
        ),
        "got {error:?}"
    );

    load_both(needle_weights, second);
}

#[test]
fn needle_and_whistle_take_turns_from_two_threads() {
    let _engine = lock();
    let (Some(needle_weights), Some(whistle_weights)) = (needle_weights(), whistle_weights())
    else {
        skipped_both("the two-model concurrency test");
        return;
    };

    const ROUNDS: usize = 5;
    let mut needle = lights(needle_weights);
    let mut whistle = build(whistle_weights);
    let pcm = clip("lights_en");

    // One round of each on this thread is what every concurrent round must reproduce.
    needle.reset();
    let want_calls = calls(&needle.complete("turn the hall light on").expect("answers"));
    let want_text = whistle
        .transcribe(&pcm)
        .expect("transcribes")
        .text()
        .to_owned();

    let (got_calls, got_texts) = thread::scope(|scope| {
        let needle = &mut needle;
        let whistle = &mut whistle;
        let pcm = &pcm;

        let calls = scope.spawn(move || {
            (0..ROUNDS)
                .map(|_| {
                    needle.reset();
                    calls(&needle.complete("turn the hall light on").expect("answers"))
                })
                .collect::<Vec<_>>()
        });
        let texts = scope.spawn(move || {
            (0..ROUNDS)
                .map(|_| {
                    whistle
                        .transcribe(pcm)
                        .expect("transcribes")
                        .text()
                        .to_owned()
                })
                .collect::<Vec<_>>()
        });

        (
            calls.join().expect("the Needle thread finishes"),
            texts.join().expect("the Whistle thread finishes"),
        )
    });

    assert_eq!(got_calls, vec![want_calls; ROUNDS]);
    assert_eq!(got_texts, vec![want_text; ROUNDS]);
}

#[test]
fn needle_hears_a_spoken_command_and_a_second_turn() {
    let _engine = lock();
    let (Some(needle_weights), Some(whistle_weights)) = (needle_weights(), whistle_weights())
    else {
        skipped_both("the audio completion test");
        return;
    };

    // Building the Whistle is what loads the speech model Needle transcribes with.
    let _whistle = build(whistle_weights);
    let mut needle = lights(needle_weights);
    needle.reset();

    let first = needle
        .complete_audio(&clip("lights_en"))
        .expect("the clip is answered");
    let heard = first.audio().expect("an audio turn says what it heard");
    assert!(heard.text().contains("kitchen"), "got {:?}", heard.text());
    assert_eq!(heard.language(), Some(Language::En));
    assert_eq!(
        calls(&first),
        vec![(
            "set_light".to_owned(),
            json!({ "room": "kitchen", "on": false })
        )]
    );

    let second = needle
        .complete_audio(&clip("hall_on_en"))
        .expect("the clip is answered");
    assert!(
        second
            .audio()
            .expect("an audio turn says what it heard")
            .text()
            .contains("hall")
    );
    assert_eq!(
        calls(&second),
        vec![(
            "set_light".to_owned(),
            json!({ "room": "hall", "on": true })
        )]
    );
}

#[test]
fn needle_hears_german() {
    let _engine = lock();
    let (Some(needle_weights), Some(whistle_weights)) = (needle_weights(), whistle_weights())
    else {
        skipped_both("the German audio completion test");
        return;
    };

    let _whistle = build(whistle_weights);
    let mut needle = lights(needle_weights);
    needle.reset();

    let completion = needle
        .complete_audio(&clip("lights_de"))
        .expect("the clip is answered");
    let heard = completion
        .audio()
        .expect("an audio turn says what it heard");

    // "Küche" comes out as "Kitche", so only the words the model hears reliably are checked.
    assert_eq!(heard.language(), Some(Language::De));
    let text = heard.text().to_lowercase();
    assert!(
        text.contains("licht") && text.contains("aus"),
        "got {text:?}"
    );
}

#[test]
fn audio_options_are_applied_afresh_on_every_turn() {
    let _engine = lock();
    let (Some(needle_weights), Some(whistle_weights)) = (needle_weights(), whistle_weights())
    else {
        skipped_both("the audio-options test");
        return;
    };

    let _whistle = build(whistle_weights);
    let mut needle = lights(needle_weights);
    let pcm = clip("lights_en");

    needle.reset();
    let timed = needle
        .complete_audio_with_options(
            &pcm,
            CompleteOptions::new(),
            &TranscribeOptions::new().with_word_timestamps(true),
        )
        .expect("the clip is answered");
    let words = timed.audio().expect("an audio turn").words();
    assert!(!words.is_empty(), "timestamps were asked for");

    // The engine's setting is sticky; this turn only loses the words if it is set again.
    needle.reset();
    let untimed = needle
        .complete_audio_with_options(&pcm, CompleteOptions::new(), &TranscribeOptions::new())
        .expect("the clip is answered");
    assert!(
        untimed.audio().expect("an audio turn").words().is_empty(),
        "the previous turn's timestamps leaked into this one"
    );
}
