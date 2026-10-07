<p align="center"><img src="https://github.com/hxphsts/cactus-rs/raw/main/assets/banner.png" alt="cactus-rs: Rust bindings for the Cactus engines" width="100%"></p>

# cactus-rs

Safe Rust bindings for [Cactus Compute](https://cactuscompute.com)'s on-device engines: [Needle 3](https://github.com/cactus-compute/needle) for tool calling, structured extraction and embeddings, and Whistle for speech-to-text.

[![Crates.io](https://img.shields.io/crates/v/cactus-rs.svg)](https://crates.io/crates/cactus-rs) [![Documentation](https://docs.rs/cactus-rs/badge.svg)](https://docs.rs/cactus-rs) [![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/hxphsts/cactus-rs/blob/main/LICENSE-MIT)

## What is Needle?

Needle 3 is a small model that reads a sentence and decides which of your functions to call, with which arguments. It runs on the CPU in about 100 MB of RAM at hundreds of tokens per second. This crate links the engine Cactus Compute publishes for it and gives it a typed Rust API.

## What is Whistle?

Whistle is a 16.9 MB speech-to-text model that runs inside the same engine as Needle, so both live in one process. It transcribes clips of up to 30 seconds of 16 kHz mono audio in seven languages (en, de, fr, es, it, nl, pl), with word timestamps, keyword biasing and speech embeddings. Hand a clip to Needle and it transcribes and answers it in one call, so a spoken request becomes tool calls.

## Quick Start

```toml
[dependencies]
cactus-rs = "0.2"
serde_json = "1.0"
```

```rust
use cactus_rs::needle::{Needle, Tool, Weights};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let completion = needle.complete("turn the kitchen light on")?;
    for call in completion.calls() {
        println!("{} {}", call.name(), call.arguments_json());
    }
    Ok(())
}
```

That prints `set_light {"room":"kitchen","on":true}`. The first build downloads the 1 MB engine library and the first run downloads the 35 MB weights; both are pinned and SHA-256 verified.

### Speech

```rust
use cactus_rs::whistle::{Weights, Whistle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut whistle = Whistle::builder(Weights::fetch()?).build()?; // 16.9 MB, cached
    let pcm: Vec<f32> = vec![0.0; 16_000]; // 16 kHz mono in [-1, 1], from your WAV reader
    let transcript = whistle.transcribe(&pcm)?;
    println!("{} ({})", transcript.text(), transcript.language_code());
    Ok(())
}
```

Once a `Whistle` has loaded the speech model, `needle.complete_audio(&pcm)` turns a clip into tool calls, with the transcript in `completion.audio()`.

## Features

- Tool calls as typed values, with `call.parse::<YourStruct>()`
- Structured extraction into your own types
- Text embeddings, 3072 dimensions
- Speech-to-text in seven languages, with language detection or forcing, keyword biasing and word timestamps
- Speech embeddings, 512 floats per 80 ms frame
- Voice to tool calls: `Needle::complete_audio` transcribes and answers in one engine call
- One model of each kind per process, enforced by the type system; `Needle` and `Whistle` are `Send + Sync` and share one engine lock
- Pinned, checksum-verified engine and weights, with fully offline builds
- `cactus-sys` for the raw C API underneath

## Examples

| Example | Shows |
| --- | --- |
| [`lights`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/lights.rs) | The Quick Start, with confidence and speed |
| [`extract`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/extract.rs) | A sentence parsed into a typed struct |
| [`embed`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/embed.rs) | Embeddings and cosine similarity |
| [`custom_weights`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/custom_weights.rs) | Weights loaded from a path |
| [`desktop`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/desktop.rs) | A twelve-tool agent loop that scores itself, and how to write schemas the model follows |
| [`transcribe`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/transcribe.rs) | A WAV file transcribed, with language, keywords and word timestamps |
| [`voice_lights`](https://github.com/hxphsts/cactus-rs/blob/main/cactus-rs/examples/voice_lights.rs) | A spoken request turned into light-switch calls by Needle |

Run one with `cargo run -p cactus-rs --example lights`.

## Platforms

| Target | Status |
| --- | --- |
| macOS arm64 | Tested |
| Linux x86_64 | Tested on Ubuntu 24.04. Needs LLVM libc++: `sudo apt install libc++-dev libc++abi-dev` |
| Other Linux, Android, iOS, tvOS, watchOS, Windows gnullvm | Mapped to upstream's archives, untested |

Intel macOS and Windows MSVC are not supported, because upstream ships no archive they can link. Offline builds, a local engine archive, a static C++ runtime and the other build options are covered in the [`cactus-sys` documentation](https://docs.rs/cactus-sys).

## Roadmap

- Streaming and microphone capture
- Clips longer than 30 s
- The grounding layer upstream implements in Python
- Tool-index persistence
- Subprocess isolation for several tuned models

Not planned: the general Cactus engine (`cactus_engine.h`). It builds for ARM64 only, ships no prebuilt archive or stable header, is distributed under a source-available licence that permits free use only below a funding and revenue threshold, and reports usage to Cactus Compute at every model load. Needle 2 is not planned either: its archive exports the same symbols as Needle 3, so the two cannot be linked into one binary. A Needle 2 archive handed to this crate is named as such in the error.

## Documentation

See https://docs.rs/cactus-rs

## License

MIT OR Apache-2.0. The vendored `needle.h` and the smart-home test data are Cactus Compute's, under Apache-2.0, with provenance noted beside each. The spoken test clips were synthesised for this repository with Piper TTS from public-domain and CC0 voices, and the JFK clip is in the public domain; see `cactus-rs/tests/data/README.md`. This project is unofficial and not affiliated with Cactus Compute.
