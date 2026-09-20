//! The Needle 3 engine: tool calling, structured extraction and text embedding.
//!
//! ## Overview
//!
//! Four types carry the whole API:
//!
//! | Type | What it is |
//! | --- | --- |
//! | [`Weights`] | the 35 MB `needle3.cact` archive, validated |
//! | [`Tool`] | one tool declaration: name, description, JSON Schema |
//! | [`Needle`] | the engine, held by exactly one value in the process |
//! | [`Completion`] | one turn's answer, with its [`Call`]s, [`Stats`] and [`Validation`] |
//!
//! [`NeedleBuilder`] collects the conversation prefix and [`CompleteOptions`] sizes a turn.
//! [`Kind`] says whether the engine reached for tools or answered in prose.
//!
//! The engine is one process-global, non-thread-safe model whose weights cannot be unloaded.
//! [`engine`] documents what that means for a program; the short version is that a second
//! [`NeedleBuilder::build`] while a [`Needle`] is alive returns
//! [`Error::EngineBusy`](crate::Error::EngineBusy), and dropping the [`Needle`] frees the slot.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Tool, Weights};
//! use serde_json::json;
//!
//! let mut needle = Needle::builder(Weights::fetch()?)
//!     .system("You control the lights.")
//!     .tool(Tool::new(
//!         "set_light",
//!         "Turn a room light on or off",
//!         json!({
//!             "type": "object",
//!             "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
//!             "required": ["room", "on"]
//!         }),
//!     ))
//!     .build()?;
//!
//! let completion = needle.complete("turn the kitchen light on")?;
//! for call in completion.calls() {
//!     println!("{} {}", call.name(), call.arguments_json());
//! }
//!
//! let vector = needle.embed("kitchen")?;
//! assert_eq!(vector.len(), 3072);
//! # Ok::<(), cactus_rs::Error>(())
//! ```

pub mod completion;
pub mod engine;
pub mod tool;
pub mod weights;

pub use completion::{Call, Completion, Kind, Stats, Validation};
pub use engine::{CompleteOptions, Needle, NeedleBuilder};
pub use tool::Tool;
pub use weights::Weights;
