//! Loading a weight archive from a path instead of the cache.
//!
//! `Weights::fetch()` is the convenient path; this is the one a deployment takes, where the 35 MB
//! archive is shipped alongside the binary and the machine has no network at all. The archive is
//! validated by magic tag before the engine sees it, so a Needle 2 file or a half-finished
//! download is named rather than handed over.
//!
//! Run it with `cargo run --example custom_weights -- /path/to/needle3.cact`.

use std::env;
use std::process;

use cactus_rs::needle::{Needle, Tool, Weights};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: custom_weights <path to needle3.cact>");
        process::exit(2);
    };

    let weights = Weights::from_file(&path)?;
    println!("{path}");
    println!("  generation {}", weights.generation());
    println!("  {} bytes", weights.len());

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

    println!("  prefix {} tokens", needle.prefix_tokens());

    let completion = needle.complete("turn the study light off")?;
    for call in completion.calls() {
        println!("{} {}", call.name(), call.arguments_json());
    }

    Ok(())
}
