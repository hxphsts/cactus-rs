//! Structured extraction: a sentence in, a Rust struct out.
//!
//! A tool whose parameters *are* the record schema turns tool calling into extraction. The
//! engine fills the schema from the text, and [`Call::parse`] hands it back as the type the
//! schema described, so nothing in between has to pick a [`serde_json::Value`] apart.
//!
//! Run it with `cargo run --example extract`. The weights are fetched once and cached under
//! `~/.cache/cactus-rs`; set `CACTUS_NEEDLE_WEIGHTS` to a local `needle3.cact` to skip the
//! download.

use cactus_rs::needle::{Needle, Tool, Weights};
use serde::Deserialize;
use serde_json::json;

/// The record to pull out of the sentence, and the shape the tool declares.
#[derive(Debug, Deserialize)]
struct Contact {
    name: String,
    email: String,
    company: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sentence = "Ada Lovelace, ada@analytical.co, heads research at Analytical Engines.";

    let mut needle = Needle::builder(Weights::fetch()?)
        .system("Record every contact the text describes. Copy values exactly as written.")
        .tool(Tool::new(
            "record_contact",
            "Record one person's name, email address and employer",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "The person's full name." },
                    "email": { "type": "string", "description": "The person's email address." },
                    "company": { "type": "string", "description": "The employer named for them." }
                },
                "required": ["name", "email", "company"]
            }),
        ))
        .build()?;

    let completion = needle.complete(sentence)?;
    println!("{sentence}");

    for call in completion.calls() {
        let contact: Contact = call.parse()?;
        println!("  name    {}", contact.name);
        println!("  email   {}", contact.email);
        println!("  company {}", contact.company);
    }

    if completion.calls().is_empty() {
        println!("  nothing to record");
    }

    println!("confidence {:.2}", completion.confidence());
    Ok(())
}
