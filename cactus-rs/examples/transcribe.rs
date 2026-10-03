//! Transcribe a WAV file with Whistle, with its language, timings and words.
//!
//! Run it with `cargo run --example transcribe`, which transcribes the vendored
//! `tests/data/clips/jfk.wav`, or name your own clip:
//!
//! ```text
//! cargo run --example transcribe -- clip.wav --language de --keyword Siobhan --timestamps
//! ```
//!
//! The clip must be 16 kHz mono 16-bit PCM and at most 30 seconds; this example refuses anything
//! else rather than resampling it. The weights are fetched once and cached under
//! `~/.cache/cactus-rs`; set `CACTUS_WHISTLE_WEIGHTS` to a local `whistle.cact` to skip the
//! download.

use std::env;
use std::path::{Path, PathBuf};

use cactus_rs::whistle::{Language, SAMPLE_RATE, TranscribeOptions, Weights, Whistle};

const USAGE: &str =
    "usage: transcribe [path.wav] [--language xx] [--keyword word]... [--timestamps]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/clips/jfk.wav");
    let mut options = TranscribeOptions::new();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--language" => {
                let code = args.next().ok_or(USAGE)?;
                options = options.with_language(code.parse::<Language>()?);
            }
            "--keyword" => options = options.with_keyword(args.next().ok_or(USAGE)?),
            "--timestamps" => options = options.with_word_timestamps(true),
            flag if flag.starts_with("--") => return Err(USAGE.into()),
            _ => path = PathBuf::from(arg),
        }
    }

    let pcm = read_wav(&path)?;

    let mut whistle = Whistle::builder(Weights::fetch()?).build()?; // cached under ~/.cache/cactus-rs
    let transcript = whistle.transcribe_with_options(&pcm, &options)?;

    println!("text      {}", transcript.text());
    println!("language  {}", transcript.language_code());
    println!(
        "ttft {:.1} ms, decode {:.1} tok/s",
        transcript.ttft_ms(),
        transcript.decode_tps()
    );
    for word in transcript.words() {
        println!(
            "{:>6.2}s {:>6.2}s  p={:.3}  {}",
            word.start(),
            word.end(),
            word.probability(),
            word.word()
        );
    }

    Ok(())
}

/// Reads a 16 kHz mono 16-bit WAV file as the `f32` samples Whistle takes.
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
            "{} is {} Hz, {} channel(s), {}-bit; Whistle takes 16 kHz mono 16-bit PCM. \
             Convert it first, e.g. `sox in.wav -r 16000 -c 1 -b 16 out.wav`.",
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
