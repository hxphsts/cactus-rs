//! # cactus-rs
//!
//! Safe Rust bindings for [Cactus Compute](https://cactuscompute.com)'s Needle engine: tool
//! calling, structured extraction and text embedding, on the CPU, with no server.
//!
//! ## What is Needle?
//!
//! Needle 3 is a 29M to 121M parameter model that does one thing: read a sentence and decide which
//! of your functions to call with which arguments. It ships as a closed-source prebuilt static
//! library of five C functions, with its weights in a separate 35 MB `needle3.cact` archive.
//! [`cactus-sys`](https://docs.rs/cactus-sys) links that library; this crate is the safe API over
//! it.
//!
//! The engine is one process-global, non-thread-safe model. There are no handles, no streaming,
//! and weights cannot be unloaded once they are in. Those are the constraints the API is shaped
//! around, and [`needle::engine`] spells out what each one means for a program.
//!
//! ## Key Features
//!
//! - **Tool Calling**: declare [`Tool`](needle::Tool)s, get back typed
//!   [`Call`](needle::Call)s with their arguments parsed into your own structs
//! - **Grounding**: the engine reports which argument values it could not trace back to the
//!   input, and [`Completion::grounded_calls`](needle::Completion::grounded_calls) acts on it
//! - **Embeddings**: 3072-dimension vectors from the same loaded model
//! - **One Engine, Enforced**: the process-global singleton is a Rust value, so a second
//!   [`Needle`](needle::Needle) is an [`Error::EngineBusy`], not a data race
//! - **Offline**: with `--no-default-features` and `CACTUS_NEEDLE_LIB_DIR` pointing at a local
//!   archive, nothing is downloaded at build time or at run time
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! cactus-rs = "0.1"
//! serde_json = "1.0"
//! ```
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Tool, Weights};
//! use serde_json::json;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let weights = Weights::fetch()?; // cached under ~/.cache/cactus-rs
//!     let mut needle = Needle::builder(weights)
//!         .system("You control the lights.")
//!         .tool(Tool::new(
//!             "set_light",
//!             "Turn a room light on or off",
//!             json!({
//!                 "type": "object",
//!                 "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
//!                 "required": ["room", "on"]
//!             }),
//!         ))
//!         .build()?;
//!
//!     let completion = needle.complete("turn the kitchen light on")?;
//!     for call in completion.calls() {
//!         println!("{} {}", call.name(), call.arguments_json());
//!     }
//!     Ok(())
//! }
//! ```
//!
//! That prints `set_light {"room":"kitchen","on":true}`.
//!
//! ### Typed Arguments
//!
//! A [`Call`](needle::Call) parses into whatever type its schema describes, so the tool's
//! arguments arrive as a struct rather than a [`serde_json::Value`] to pick apart:
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Weights};
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct SetLight {
//!     room: String,
//!     on: bool,
//! }
//!
//! let mut needle = Needle::builder(Weights::fetch()?).build()?;
//! let completion = needle.complete("turn the kitchen light on")?;
//!
//! for call in completion.calls() {
//!     let args: SetLight = call.parse()?;
//!     println!("{} -> {}", args.room, args.on);
//! }
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ### Acting Only on Grounded Calls
//!
//! The engine checks its own work: it reports argument values it invented, and inputs that
//! negated the action. When a wrong call would be expensive, act on
//! [`grounded_calls`](needle::Completion::grounded_calls) instead of
//! [`calls`](needle::Completion::calls):
//!
//! ```no_run
//! # use cactus_rs::needle::{Needle, Weights};
//! # let mut needle = Needle::builder(Weights::fetch()?).build()?;
//! let completion = needle.complete("do not turn the kitchen light on")?;
//!
//! for call in completion.grounded_calls() {
//!     println!("acting on {}", call.name());
//! }
//! if !completion.validation().is_grounded() {
//!     println!("held back: {:?}", completion.validation().ungrounded());
//! }
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ### Embeddings
//!
//! ```no_run
//! # use cactus_rs::needle::{Needle, Weights};
//! let mut needle = Needle::builder(Weights::fetch()?).build()?;
//!
//! let kitchen = needle.embed("kitchen")?;
//! assert_eq!(kitchen.len(), needle.embedding_dimension()?);
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ### A Conversation
//!
//! Each turn is appended to the one before it, until [`reset`](needle::Needle::reset) rewinds to
//! the system prompt and tool declarations:
//!
//! ```no_run
//! # use cactus_rs::needle::{Needle, Weights};
//! # let mut needle = Needle::builder(Weights::fetch()?).build()?;
//! needle.complete("turn the kitchen light on")?;
//! needle.complete("now the hall")?; // the engine still remembers the kitchen
//! needle.reset(); // it no longer does
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ## Performance Characteristics
//!
//! One measurement per machine, through this crate, of the README's one-tool example, on an
//! otherwise quiet system (the engine's threads spin rather than sleep, so a busy machine costs
//! it a lot):
//!
//! | Machine | Prefill | Decode | Peak RAM | `complete` | `embed` | `build` |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | Apple M1 Pro, macOS | 1900 to 1970 tok/s | 790 to 810 tok/s | 91 MB | 54 ms | 1.9 ms | 6 ms |
//! | Ryzen 7 3800X (AVX2, no VNNI), Ubuntu 24.04 | 1050 to 1130 tok/s | 230 to 285 tok/s | 103 MB | 176 ms | 5.9 ms | 27 ms |
//!
//! The engine runs on the CPU only, with no CUDA or Metal path, and picks its kernels from what
//! the CPU offers: on x86 one of AVX2, AVX-512, AVX-512 VNNI or AMX, on ARM NEON. The Ryzen above
//! has only AVX2. Because the kernels differ, a borderline request can come out differently on two
//! architectures, as it does with upstream's own engine; clear requests give the same calls
//! everywhere. Embeddings have 3072 dimensions. Every
//! [`Completion`](needle::Completion) carries its own numbers in [`Stats`](needle::Stats).
//!
//! - **Prefix**: the system prompt and tool declarations are tokenised once, at
//!   [`build`](needle::NeedleBuilder::build); [`prefix_tokens`](needle::Needle::prefix_tokens)
//!   is the standing cost every turn pays
//! - **Buffers**: the output buffer is sized from the token budget and reused across turns, so a
//!   conversation allocates it once
//! - **Weights**: loaded once per process, and copied by the engine, so the 35 MB archive is
//!   dropped as soon as [`build`](needle::NeedleBuilder::build) returns
//! - **Embedding**: the dimension is asked for once and cached, so
//!   [`embed`](needle::Needle::embed) is a single engine call afterwards
//!
//! ## Public Dependencies
//!
//! [`serde_json::Value`] and [`serde_json::Error`] appear in this crate's signatures, as does
//! `schemars::JsonSchema` behind the `schemars` feature, so a major release of either is a major
//! release here. The crate also enables `serde_json`'s `preserve_order` feature, because the
//! model follows tool schemas best in their declared key order. Cargo unifies features across a
//! build, so every `serde_json::Map` in a program that depends on this crate keeps insertion
//! order.
//!
//! ## Safety Guarantees
//!
//! - Every `unsafe` block in the crate is in [`needle::engine`], and each one carries a
//!   `// SAFETY:` comment naming the contract it satisfies
//! - Only one [`Needle`](needle::Needle) exists per process: a second
//!   [`build`](needle::NeedleBuilder::build) returns [`Error::EngineBusy`] rather than calling
//!   into an engine someone else is using
//! - Every method that reaches the engine takes `&mut self`, so its calls cannot overlap.
//!   [`Needle`](needle::Needle) is [`Send`] and [`Sync`]; sharing it still means a lock
//! - Weights are validated by magic tag before the engine sees them, and a second, different
//!   archive returns [`Error::WeightsAlreadyLoaded`] instead of being silently ignored
//! - Strings crossing the FFI boundary are rejected for interior NUL bytes rather than being
//!   silently cut short
//! - The engine truncates its output without saying so; this crate detects it and returns
//!   [`Error::Truncated`] rather than broken JSON
//!
//! ## Affiliation
//!
//! This crate is unofficial and is not affiliated with Cactus Compute. The engine it links is
//! published by them under Apache-2.0.
//!
//! ## Examples
//!
//! - **`lights.rs`**: the Quick Start, with confidence and decode speed
//! - **`extract.rs`**: a sentence parsed into a typed struct
//! - **`embed.rs`**: embeddings and cosine similarity
//! - **`custom_weights.rs`**: weights loaded from a path you give
//! - **`desktop.rs`**: a twelve-tool simulated desktop agent that scores itself against expected calls
//!
//! Run any example with `cargo run -p cactus-rs --example <name>`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod error;
pub mod needle;

pub use error::{Error, Result};
