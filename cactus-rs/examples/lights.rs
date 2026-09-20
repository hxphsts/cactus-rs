//! The README Quick Start, plus what the engine thought of its own answer.
//!
//! Run it with `cargo run --example lights`. The weights are fetched once and cached under
//! `~/.cache/cactus-rs`; set `CACTUS_NEEDLE_WEIGHTS` to a local `needle3.cact` to skip the
//! download.

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

    println!("confidence {:.2}", completion.confidence());
    println!("decode {:.1} tok/s", completion.stats().decode_tps());

    Ok(())
}
