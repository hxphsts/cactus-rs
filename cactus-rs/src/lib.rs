//! # cactus-rs
//!
//! Safe Rust bindings for [Cactus Compute](https://cactuscompute.com)'s Needle engine and the two
//! models it runs: Needle 3 for tool calling, structured extraction and text embedding, and
//! Whistle for speech-to-text. On the CPU, with no server.
//!
//! ## What is Needle?
//!
//! Needle 3 is a 29M to 121M parameter model that does one thing: read a sentence and decide which
//! of your functions to call with which arguments. It ships as a closed-source prebuilt static
//! library of nine C functions, with its weights in a separate 35 MB `needle3.cact` archive.
//! [`cactus-sys`](https://docs.rs/cactus-sys) links that library; this crate is the safe API over
//! it.
//!
//! The engine holds one process-global model per kind, text and speech, behind one
//! non-thread-safe runtime. There are no handles, no streaming, and weights cannot be unloaded
//! once they are in. Those are the constraints the API is shaped around, and [`needle::engine`]
//! and [`whistle::engine`] spell out what each one means for a program.
//!
//! ## What is Whistle?
//!
//! Whistle is a 16.9 MB speech-to-text model, `whistle.cact`, that runs in the same engine as
//! Needle rather than in a second one. It transcribes clips of up to 30 seconds of 16 kHz mono
//! audio in seven languages (English, German, French, Spanish, Italian, Dutch and Polish), with
//! word timestamps and keyword biasing on request, and turns speech into 512-float embeddings,
//! one per 80 ms frame. Because the engine keeps one model of each kind, a
//! [`Whistle`](whistle::Whistle) and a [`Needle`](needle::Needle) live in one process, and
//! [`Needle::complete_audio`](needle::Needle::complete_audio) turns a spoken request straight
//! into tool calls.
//!
//! ## Key Features
//!
//! - **Tool Calling**: declare [`Tool`](needle::Tool)s, get back typed
//!   [`Call`](needle::Call)s with their arguments parsed into your own structs
//! - **Grounding**: the engine reports which argument values it could not trace back to the
//!   input, and [`Completion::grounded_calls`](needle::Completion::grounded_calls) acts on it
//! - **Embeddings**: 3072-dimension vectors from the same loaded model
//! - **Transcription**: [`Whistle::transcribe`](whistle::Whistle::transcribe) turns a clip into
//!   a [`Transcript`](whistle::Transcript), with the language detected or forced
//! - **Keyword Biasing**: names and jargon passed in
//!   [`TranscribeOptions`](whistle::TranscribeOptions) are favoured when the model hears them
//! - **Word Timestamps**: each [`Word`](whistle::Word) with its start, end and probability
//! - **Speech Embeddings**: one 512-float row per 80 ms of audio, from
//!   [`Whistle::embed`](whistle::Whistle::embed)
//! - **Voice to Tool Calls**: [`Needle::complete_audio`](needle::Needle::complete_audio)
//!   transcribes and answers in one engine call, with the transcript in
//!   [`Completion::audio`](needle::Completion::audio)
//! - **One Model per Kind, Enforced**: each process-global model is a Rust value, so a second
//!   [`Needle`](needle::Needle) or [`Whistle`](whistle::Whistle) is an [`Error::EngineBusy`],
//!   not a data race
//! - **Offline**: with `--no-default-features` and `CACTUS_NEEDLE_LIB_DIR` pointing at a local
//!   archive, nothing is downloaded at build time or at run time. `HF_ENDPOINT` points the
//!   weight downloads at a Hugging Face mirror, and `HF_HUB_OFFLINE=1` makes them fail instead
//!   of touching the network
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! cactus-rs = "0.2"
//! serde_json = "1.0"
//! ```
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Tool, Weights};
//! use serde_json::json;
//!
//! # #[cfg(feature = "download")]
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
//! # #[cfg(not(feature = "download"))]
//! # fn main() {}
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
//! let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
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
//! # let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
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
//! let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
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
//! # let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
//! needle.complete("turn the kitchen light on")?;
//! needle.complete("now the hall")?; // the engine still remembers the kitchen
//! needle.reset(); // it no longer does
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ### Transcribing a Clip
//!
//! Clips are plain `&[f32]` slices of 16 kHz mono samples in `[-1, 1]`, so the output of a WAV
//! reader or a resampler goes in as it is:
//!
//! ```no_run
//! use cactus_rs::whistle::{Language, TranscribeOptions, Weights, Whistle};
//!
//! let mut whistle = Whistle::builder(Weights::from_file("whistle.cact")?).build()?;
//! let pcm = vec![0.0_f32; 16_000]; // one second; read yours from a WAV file
//!
//! let transcript = whistle.transcribe(&pcm)?;
//! println!("{} ({})", transcript.text(), transcript.language_code());
//!
//! let options = TranscribeOptions::new()
//!     .with_language(Language::En)
//!     .with_keyword("Siobhan")
//!     .with_word_timestamps(true);
//! for word in whistle.transcribe_with_options(&pcm, &options)?.words() {
//!     println!("{:>6.2}s {}", word.start(), word.word());
//! }
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! ### Talking to Needle
//!
//! Once a [`Whistle`](whistle::Whistle) has loaded the speech model, a
//! [`Needle`](needle::Needle) can take a clip instead of a sentence. The engine transcribes it,
//! answers the transcript, and reports what it heard beside the calls. The
//! [`Whistle`](whistle::Whistle) may be dropped afterwards, because the speech model stays
//! loaded:
//!
//! ```no_run
//! use cactus_rs::needle::{Needle, Weights};
//! use cactus_rs::whistle::{self, Whistle};
//!
//! let whistle = Whistle::builder(whistle::Weights::from_file("whistle.cact")?).build()?;
//! drop(whistle); // the speech model stays loaded
//!
//! let mut needle = Needle::builder(Weights::from_file("needle3.cact")?).build()?;
//! let pcm = vec![0.0_f32; 16_000]; // "turn off the kitchen lights", say
//!
//! let completion = needle.complete_audio(&pcm)?;
//! if let Some(heard) = completion.audio() {
//!     println!("heard: {}", heard.text());
//! }
//! for call in completion.calls() {
//!     println!("{} {}", call.name(), call.arguments_json());
//! }
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
//! Whistle was measured on one machine so far, a shared 4 vCPU Xeon at 2.1 GHz (x86_64, Linux),
//! so take the numbers as rough. The clip is the 11 second JFK recording in the test data:
//!
//! | Machine | `transcribe` (11 s clip) | Time to first token | Decode | `embed` (1 s) | `build` | Peak RAM |
//! | --- | --- | --- | --- | --- | --- | --- |
//! | Xeon 2.1 GHz, 4 vCPU, Linux | 1.1 to 1.3 s | 950 to 1200 ms | 200 to 245 tok/s | 30 to 40 ms | 28 ms | 42 MB alone, 90 to 118 MB with Needle |
//!
//! macOS has not been measured for Whistle yet.
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
//! - **Speech**: [`Whistle`](whistle::Whistle) reuses one 256 KiB output buffer across clips,
//!   and a speech embedding has [`frame_count`](whistle::frame_count) rows of 512 floats,
//!   `max(1, (samples + 1120) / 1280)`
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
//! - All `unsafe` lives in the crate's private FFI layer, and each block carries a `// SAFETY:`
//!   comment naming the lock it holds and the contract it satisfies
//! - Every engine call, of either model, takes one process-wide lock, so a
//!   [`Needle`](needle::Needle) and a [`Whistle`](whistle::Whistle) on different threads take
//!   turns instead of overlapping. Both are [`Send`] and [`Sync`], and every method that reaches
//!   the engine takes `&mut self`, so one conversation's turns stay in order
//! - One value holds each model: a second [`NeedleBuilder::build`](needle::NeedleBuilder::build)
//!   or [`WhistleBuilder::build`](whistle::WhistleBuilder::build) returns
//!   [`Error::EngineBusy`] rather than calling into a model someone else is using
//! - Weights are validated by magic tag before the engine sees them. Needle and Whistle archives
//!   share one container format, so a Whistle archive handed to Needle (or the reverse) is
//!   caught as [`Error::WrongModel`]: at once when it is already loaded as the other kind,
//!   otherwise when the engine has read it. Loading the same bytes
//!   again is skipped, which keeps a text model's conversation intact, and a second, different
//!   archive of the same kind returns [`Error::WeightsAlreadyLoaded`] instead of being silently
//!   ignored
//! - Audio is checked before it crosses: a clip over 30 seconds is [`Error::AudioTooLong`], a
//!   NaN or an infinity is [`Error::NonFiniteSample`], and a keyword that would split the
//!   engine's newline-separated list is [`Error::InvalidKeyword`]. Transcription settings are
//!   passed on every call, so one call's language or keywords never leak into the next
//! - The engine's error string is read and copied under the same lock as the call that set it,
//!   because the header makes it valid only until the next call
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
//! - **`transcribe.rs`**: a WAV file transcribed, with language, keywords and word timestamps
//! - **`voice_lights.rs`**: a spoken request turned into light-switch calls by Needle
//!
//! Run any example with `cargo run -p cactus-rs --example <name>`.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

mod archive;
#[cfg(feature = "download")]
mod download;
pub mod error;
mod ffi;
mod model;
pub mod needle;
pub mod whistle;

pub use error::{Error, Result};
pub use model::Model;
