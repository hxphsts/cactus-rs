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
    self, Language, MAX_SAMPLES, PASS_SAMPLES, SAMPLE_RATE, TranscribeOptions, Update, Whistle,
    frame_count,
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
fn a_long_clip_is_transcribed_and_nan_is_refused() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the clip-length test");
        return;
    };

    let mut whistle = build(weights);

    // Thirty seconds was the limit of the older engine; a sample more is now a second window.
    let silence = vec![0.0; MAX_SAMPLES + 1];
    let transcript = whistle
        .transcribe(&silence)
        .expect("more than thirty seconds is accepted");
    assert!(transcript.is_empty());

    // 33 seconds of speech: the same sentence three times, heard three times.
    let jfk = clip("jfk");
    let long: Vec<f32> = jfk.iter().chain(&jfk).chain(&jfk).copied().collect();
    assert!(long.len() > MAX_SAMPLES);
    let options = TranscribeOptions::new().with_word_timestamps(true);
    let transcript = whistle
        .transcribe_with_options(&long, &options)
        .expect("a long clip transcribes");
    let heard = plain(transcript.text());
    assert_eq!(
        heard.matches("ask not what your country").count(),
        3,
        "{heard}"
    );
    // Word times run from the start of the clip, across the window boundary.
    let last = transcript.words().last().expect("words were asked for");
    assert!(last.end() > 30.0 && last.end() <= 33.1, "{last:?}");

    // An embedding still takes at most thirty seconds.
    let error = whistle
        .embed(&silence)
        .expect_err("an embedding is bounded");
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

/// Text reduced to lower-case words: what a live transcription and a one-shot one agree on,
/// since the two punctuate differently.
fn plain(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Streams `pcm` in pieces of `piece` samples and returns every update, the last from `finish`.
fn streamed(whistle: &mut Whistle, pcm: &[f32], piece: usize) -> Vec<Update> {
    let mut stream = whistle.stream().expect("a stream opens");
    let mut updates: Vec<Update> = pcm
        .chunks(piece)
        .map(|chunk| stream.push(chunk).expect("a push"))
        .collect();
    updates.push(stream.finish().expect("the stream finishes"));
    updates
}

/// The committed text of a whole stream.
fn committed(updates: &[Update]) -> String {
    let texts: Vec<&str> = updates
        .iter()
        .map(Update::text)
        .filter(|text| !text.is_empty())
        .collect();
    texts.join(" ")
}

#[test]
fn a_stream_commits_what_a_one_shot_transcription_hears() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the streaming test");
        return;
    };
    let mut whistle = build(weights);
    let jfk = clip("jfk");

    let updates = streamed(&mut whistle, &jfk, PASS_SAMPLES);
    assert_eq!(plain(&committed(&updates)), plain(JFK));

    // Words are committed once, in order, with times inside the clip.
    let words: Vec<_> = updates.iter().flat_map(|update| update.words()).collect();
    assert_eq!(words.len(), plain(JFK).split(' ').count());
    for pair in words.windows(2) {
        assert!(pair[0].end() <= pair[1].end() + 1e-3, "{pair:?}");
    }
    assert!(words.iter().all(|word| word.end() <= JFK_SECONDS));

    // The tail is known before it is committed: after two seconds the model is already
    // hearing more than the one word it has settled on.
    let ahead = updates[..updates.len() - 1]
        .iter()
        .find(|update| update.pending().split_whitespace().count() >= 2)
        .expect("an update whose tail runs ahead of what is committed");
    assert!(ahead.received() < JFK_SECONDS);
    assert_eq!(updates[1].received(), 2.0);
    assert_eq!(updates[1].language(), Some(Language::En));

    let last = updates.last().expect("finish returns an update");
    assert_eq!(last.pending(), "");
    assert!((last.received() - jfk.len() as f32 / SAMPLE_RATE as f32).abs() < 0.01);

    // The size of the pieces does not matter: passes are cut at whole seconds either way.
    for piece in [1_600, 7_001, jfk.len()] {
        let again = streamed(&mut whistle, &jfk, piece);
        assert_eq!(committed(&again), committed(&updates), "pieces of {piece}");
    }

    // The stream is over: the same Whistle transcribes a clip as before.
    let transcript = whistle.transcribe(&jfk).expect("a one-shot transcription");
    assert_eq!(transcript.text(), JFK);
}

#[test]
fn a_push_short_of_a_second_waits_and_a_dropped_stream_leaves_nothing() {
    let _engine = lock();
    let Some(weights) = whistle_weights() else {
        skipped("the stream lifecycle test");
        return;
    };
    let mut whistle = build(weights);
    let jfk = clip("jfk");

    let mut stream = whistle.stream().expect("a stream opens");
    let update = stream.push(&jfk[..10_000]).expect("a short push");
    assert!(update.is_empty() && update.pending().is_empty());
    assert_eq!((update.received(), stream.waiting()), (0.0, 10_000));

    // The rest of the second and half of the next: one pass, with a tail to show for it.
    let update = stream.push(&jfk[10_000..24_000]).expect("a push");
    assert_eq!((update.received(), stream.waiting()), (1.0, 8_000));
    let tail = update.pending().to_owned();
    assert!(!tail.is_empty());

    // A push that runs no pass repeats the tail and commits nothing.
    let update = stream.push(&jfk[24_000..25_000]).expect("a short push");
    assert!(update.is_empty());
    assert_eq!(update.pending(), tail);

    // A NaN is refused before any of the push is taken, and the stream goes on.
    let error = stream
        .push(&[0.0, f32::NAN])
        .expect_err("a NaN is not audio");
    assert!(
        matches!(error, Error::NonFiniteSample { index: 1 }),
        "got {error:?}"
    );
    assert_eq!(stream.waiting(), 9_000);

    // Dropped, not finished: its pending words are gone, and the next stream starts clean.
    drop(stream);
    let lights = clip("lights_en");
    let updates = streamed(&mut whistle, &lights, PASS_SAMPLES);
    let heard = plain(&committed(&updates));
    assert_eq!(heard, "turn off the kitchen lights");
    let first = updates.first().expect("at least one update");
    assert_eq!(first.received(), 1.0);
}

#[test]
fn needle_answers_the_pending_tail_while_the_stream_goes_on() {
    let _engine = lock();
    let (Some(needle_weights), Some(whistle_weights)) = (needle_weights(), whistle_weights())
    else {
        skipped_both("the streaming voice-to-call test");
        return;
    };
    let mut needle = lights(needle_weights);
    let mut whistle = build(whistle_weights);
    let command = clip("lights_en");

    // Without Needle in between, for comparison.
    let alone = committed(&streamed(&mut whistle, &command, PASS_SAMPLES));

    let mut stream = whistle.stream().expect("a stream opens");
    let mut texts = Vec::new();
    let mut early = None;
    for (second, chunk) in command.chunks(PASS_SAMPLES).enumerate() {
        let update = stream.push(chunk).expect("a push");
        if !update.text().is_empty() {
            texts.push(update.text().to_owned());
        }
        // Route what has been heard so far, committed and pending, between pushes.
        let heard = format!("{} {}", texts.join(" "), update.pending());
        if heard.trim().is_empty() {
            continue;
        }
        needle.reset();
        let completion = needle
            .complete(heard.trim())
            .expect("a completion mid-stream");
        if early.is_none() && !calls(&completion).is_empty() {
            early = Some((second + 1, calls(&completion)));
        }
    }
    let last = stream.finish().expect("the stream finishes");
    if !last.text().is_empty() {
        texts.push(last.text().to_owned());
    }

    // Needle running between pushes did not disturb the stream.
    assert_eq!(texts.join(" "), alone);

    // The call the whole sentence gets is the call the tail already got, before the end.
    needle.reset();
    let completion = needle.complete(&alone).expect("a completion");
    let expected = vec![(
        "set_light".to_owned(),
        json!({"room": "kitchen", "on": false}),
    )];
    assert_eq!(calls(&completion), expected);
    let (second, early_calls) = early.expect("a call before the stream ended");
    println!(
        "the call was known after {second} s of a {:.1} s clip",
        command.len() as f32 / 16_000.0
    );
    assert_eq!(early_calls, expected);
    assert!(second * PASS_SAMPLES <= command.len());
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

/// Keyword biasing pulls the decode towards names the model cannot spell on its own.
///
/// `names_en.wav` says "please call Siobhan and Krzysztof". Without keywords the pinned engine
/// hears "Please call Chevorne and Crystal." on every machine measured. With `Siobhan` and
/// `Krzysztof` the decode changes, but which name lands depends on the CPU's kernels, so the
/// test asserts the change rather than a spelling.
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

    // Which spelling wins is a kernel-level tie on this clip, so the test pins the effect and
    // not a spelling. The engine picks its kernels per CPU, and on 2026-10-03 at `c7c415a3` the
    // same keywords turned "Please call Chevorne and Crystal." into "Please call Shivorn and
    // Krzysztof." on one Xeon and into "Please call Shivorn and Crystal." on GitHub's Ubuntu
    // runner. Upstream's Python package agrees with the engine of the machine it runs on.
    for name in ["Siobhan", "Krzysztof"] {
        assert!(
            !plain.text().contains(name),
            "the unbiased decode spelt {name} right: {:?}",
            plain.text()
        );
    }
    assert_ne!(biased.text(), plain.text(), "the keywords changed nothing");

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

/// Loads both models from the right archives and checks each answers, so a wrong-model test
/// starts from, and can return to, two working models whichever test loaded what first.
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
