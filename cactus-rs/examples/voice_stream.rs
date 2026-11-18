//! Turn speech into a tool call while it is still being spoken.
//!
//! Run it with `cargo run --example voice_stream`, which plays the vendored
//! `tests/data/clips/lights_en.wav` ("turn off the kitchen lights") into a live transcription a
//! second at a time, as a microphone would deliver it, or name your own 16 kHz mono 16-bit clip:
//!
//! ```text
//! cargo run --example voice_stream -- lights.wav
//! ```
//!
//! After every second Whistle reports the words it has settled on and the tail it is still
//! hearing. The example hands both to Needle straight away, so the call is known as soon as the
//! tail says enough, before the speaker has finished. It is acted on only when the finished
//! transcript gives the same call: a tail can still change.
//!
//! Compare `voice_lights`, which waits for the whole clip and answers it in one engine call.
//!
//! Both sets of weights are fetched once and cached under `~/.cache/cactus-rs`; set
//! `CACTUS_WHISTLE_WEIGHTS` and `CACTUS_NEEDLE_WEIGHTS` to local archives to skip the downloads.

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cactus_rs::needle::{Completion, Needle, Tool, Weights};
use cactus_rs::whistle::{self, PASS_SAMPLES, SAMPLE_RATE, Whistle};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/clips/lights_en.wav"),
        PathBuf::from,
    );
    let pcm = read_wav(&path)?;
    let seconds = pcm.len() as f32 / SAMPLE_RATE as f32;

    let mut whistle = Whistle::builder(whistle::Weights::fetch()?).build()?;
    let mut needle = Needle::builder(Weights::fetch()?) // cached under ~/.cache/cactus-rs
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
        .build()?;

    let mut stream = whistle.stream()?;
    let mut committed = String::new();
    let mut armed: Option<(f32, Vec<String>)> = None;
    let mut received = 0.0;

    // A microphone callback would push whatever it has; whole seconds make the output readable.
    for chunk in pcm.chunks(PASS_SAMPLES) {
        let started = Instant::now();
        let update = stream.push(chunk)?;
        append(&mut committed, update.text());

        // What has been heard so far: the settled words, then the tail that may still change.
        let heard = format!("{committed} {}", update.pending());
        let heard = heard.trim();
        // A push that did not complete a second ran no pass: there is nothing new to route.
        if heard.is_empty() || update.received() == received {
            continue;
        }
        received = update.received();
        println!(
            "{:>4.1} s  {committed:?} + {:?}",
            update.received(),
            update.pending()
        );

        // Needle runs between pushes: the stream holds the engine only while a pass runs.
        needle.reset();
        let calls = calls(&needle.complete(heard)?);
        println!(
            "          -> {} ({:.0} ms for the pass and the routing)",
            if calls.is_empty() {
                "no call yet".to_owned()
            } else {
                calls.join(", ")
            },
            started.elapsed().as_secs_f64() * 1000.0
        );
        if !calls.is_empty() && armed.as_ref().is_none_or(|(_, known)| known != &calls) {
            armed = Some((update.received(), calls));
        }
    }

    // The speaker has stopped: commit the tail and route the transcript as it now stands.
    let last = stream.finish()?;
    append(&mut committed, last.text());
    needle.reset();
    let confirmed = calls(&needle.complete(&committed)?);
    println!("{seconds:>4.1} s  {committed:?}, final");

    match armed {
        Some((at, early)) if early == confirmed => println!(
            "act: {} (known after {at:.1} s of {seconds:.1} s)",
            confirmed.join(", ")
        ),
        Some((_, early)) => {
            println!("the early call {early:?} did not survive; the final call is {confirmed:?}")
        }
        None => println!("final call: {confirmed:?}"),
    }
    Ok(())
}

/// Appends newly committed words to the transcript so far.
fn append(committed: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !committed.is_empty() {
        committed.push(' ');
    }
    committed.push_str(text);
}

/// Each call of a completion as `name {arguments}`.
fn calls(completion: &Completion) -> Vec<String> {
    completion
        .calls()
        .iter()
        .map(|call| format!("{} {}", call.name(), call.arguments_json()))
        .collect()
}

/// Reads a 16 kHz mono 16-bit PCM WAV file as samples in `[-1, 1]`.
fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(format!(
            "{} must be 16 kHz mono 16-bit PCM, not {} Hz with {} channel(s) at {} bits",
            path.display(),
            spec.sample_rate,
            spec.channels,
            spec.bits_per_sample
        )
        .into());
    }
    reader
        .samples::<i16>()
        .map(|sample| Ok(f32::from(sample?) / 32768.0))
        .collect()
}
