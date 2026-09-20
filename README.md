<p align="center"><img src="https://github.com/hxphsts/cactus-rs/raw/main/assets/banner.png" alt="cactus-rs: Rust bindings for the Cactus engines" width="100%"></p>

# cactus-rs

Safe Rust bindings for [Cactus Compute](https://cactuscompute.com)'s on-device engines, starting with [Needle 3](https://github.com/cactus-compute/needle): tool calling, structured extraction and text embedding.

[![Crates.io](https://img.shields.io/crates/v/cactus-rs.svg)](https://crates.io/crates/cactus-rs) [![Documentation](https://docs.rs/cactus-rs/badge.svg)](https://docs.rs/cactus-rs) [![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/hxphsts/cactus-rs/blob/main/LICENSE-MIT)

## What is Needle?

Needle 3 is a small model that reads a sentence and decides which of your functions to call, with which arguments. It runs on the CPU in about 100 MB of RAM at hundreds of tokens per second. This crate links the engine Cactus Compute publishes for it and gives it a typed Rust API.

## Quick Start

```toml
[dependencies]
cactus-rs = "0.1"
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

## Features

- Tool calls as typed values, with `call.parse::<YourStruct>()`
- Structured extraction into your own types
- Text embeddings, 3072 dimensions
- One engine per process, enforced by the type system; `Needle` is `Send + Sync`
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

Run one with `cargo run -p cactus-rs --example lights`.

## Platforms

| Target | Status |
| --- | --- |
| macOS arm64 | Tested |
| Linux x86_64 | Tested on Ubuntu 24.04. Needs LLVM libc++: `sudo apt install libc++-dev libc++abi-dev` |
| Other Linux, Android, iOS, tvOS, watchOS, Windows gnullvm | Mapped to upstream's archives, untested |

Intel macOS and Windows MSVC are not supported, because upstream ships no archive they can link. Offline builds, a local engine archive, a static C++ runtime and the other build options are covered in the [`cactus-sys` documentation](https://docs.rs/cactus-sys).

## Roadmap

- The general Cactus engine (`cactus_engine.h`: chat, vision, transcription, streaming) as a second `cactus-sys` feature
- The grounding layer upstream implements in Python
- Tool-index persistence
- Subprocess isolation for several tuned models

## Documentation

See https://docs.rs/cactus-rs

## License

MIT OR Apache-2.0. The vendored `needle.h` and the smart-home test data are Cactus Compute's, under Apache-2.0, with provenance noted beside each. This project is unofficial and not affiliated with Cactus Compute.
