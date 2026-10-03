//! The README Quick Start, spoken: Needle answers a voice command with tool calls.
//!
//! Run it with `cargo run --example voice_lights`, which plays the vendored
//! `tests/data/clips/lights_en.wav` ("turn off the kitchen lights") to the lights Needle, or name
//! your own clip: `cargo run --example voice_lights -- lights.wav`.
//!
//! Clips are 16 kHz mono 16-bit PCM of at most 30 seconds. To record one:
//!
//! ```text
//! arecord -f S16_LE -r 16000 -c 1 -d 3 lights.wav   # Linux, ALSA
//! sox -d -r 16000 -c 1 -b 16 lights.wav             # anywhere SoX runs; Ctrl-C to stop
//! ```
//!
//! Both sets of weights are fetched once and cached under `~/.cache/cactus-rs`; set
//! `CACTUS_WHISTLE_WEIGHTS` and `CACTUS_NEEDLE_WEIGHTS` to a local `whistle.cact` and
//! `needle3.cact` to skip the downloads.

use std::env;
use std::path::{Path, PathBuf};

use cactus_rs::needle::{Needle, Tool, Weights};
use cactus_rs::whistle::{self, SAMPLE_RATE, Whistle};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/clips/lights_en.wav"),
        PathBuf::from,
    );
    let pcm = read_wav(&path)?;

    // Building a Whistle loads the speech model Needle transcribes the clip with.
    let _whistle = Whistle::builder(whistle::Weights::fetch()?).build()?;

    let weights = Weights::fetch()?; // cached under ~/.cache/cactus-rs
    let mut needle = Needle::builder(weights)
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

    let completion = needle.complete_audio(&pcm)?;
    if let Some(heard) = completion.audio() {
        println!("heard {:?} ({})", heard.text(), heard.language_code());
    }
    for call in completion.calls() {
        println!("{} {}", call.name(), call.arguments_json());
    }

    println!("confidence {:.2}", completion.confidence());
    println!("decode {:.1} tok/s", completion.stats().decode_tps());

    Ok(())
}

/// Reads a 16 kHz mono 16-bit WAV file as the `f32` samples the engine takes.
fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let spec = reader.spec();

    if spec.sample_rate != SAMPLE_RATE
        || spec.channels != 1
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(format!(
            "{} is {} Hz, {} channel(s), {}-bit; the engine takes 16 kHz mono 16-bit PCM",
            path.display(),
            spec.sample_rate,
            spec.channels,
            spec.bits_per_sample
        )
        .into());
    }

    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|sample| f32::from(sample) / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(samples)
}
