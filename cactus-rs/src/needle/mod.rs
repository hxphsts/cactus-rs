//! The Needle 3 model: tool calling, structured extraction and text embedding.
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
//! The engine holds one process-global text model, beside one speech model, behind a
//! non-thread-safe runtime whose weights cannot be unloaded. [`engine`] documents what that
//! means for a program; the short version is that a second [`NeedleBuilder::build`] while a
//! [`Needle`] is alive returns [`Error::EngineBusy`](crate::Error::EngineBusy), and dropping the
//! [`Needle`] frees the slot.
//!
//! ## Audio Input
//!
//! [`Needle::complete_audio`] takes a 16 kHz mono clip instead of a sentence: the engine
//! transcribes it with the speech model, answers the transcript as it would a typed turn, and
//! [`Completion::audio`] carries the [`Transcript`](crate::whistle::Transcript) it heard. The
//! speech model comes from [`crate::whistle`]: build a [`Whistle`](crate::whistle::Whistle) once
//! in the process first, or the call returns
//! [`Error::SpeechModelNotLoaded`](crate::Error::SpeechModelNotLoaded).
//! [`Needle::complete_audio_with_options`] takes
//! [`TranscribeOptions`](crate::whistle::TranscribeOptions) for the language, keywords and word
//! timestamps.
//!
//! ## Usage
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Tool, Weights};
//! use serde_json::json;
//!
//! let mut needle = Needle::builder(Weights::from_file("needle3.cact")?)
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
