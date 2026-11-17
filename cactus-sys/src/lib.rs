//! # cactus-sys
//!
//! Raw FFI bindings to the [Cactus Compute](https://cactuscompute.com) inference engine.
//!
//! ## What is cactus-sys?
//!
//! Cactus Compute ships its engine as a prebuilt static library with a small C API. This crate
//! links it and declares that API, nothing more: every item is `unsafe`, and no Rust
//! type stands between the caller and the engine. For a safe, typed API use
//! [`cactus-rs`](https://docs.rs/cactus-rs) instead.
//!
//! ## Key Features
//!
//! - **One Engine, Three Models**: the thirteen functions of `cactus.h`, which run Needle 3
//!   (text), Whistle (speech) and Mirage (vision), behind the `engine` feature (default)
//! - **Pinned Binaries**: the archive is fetched from one release at one Hugging Face commit and checked
//!   against a SHA-256 recorded in this crate
//! - **Offline Builds**: point `CACTUS_LIB_DIR` at a local archive and nothing is
//!   downloaded; `HF_ENDPOINT` selects a Hugging Face mirror and `HF_HUB_OFFLINE=1` refuses to
//!   download at all
//! - **No Dependencies**: `#![no_std]`, and no runtime crates
//!
//! ## Models
//!
//! | Model | Kind bit | Entry points |
//! | --- | --- | --- |
//! | Needle 3 | [`CACTUS_TEXT`] | [`cactus_init`], [`cactus_complete`], [`cactus_reset`], [`cactus_embed`] |
//! | Whistle | [`CACTUS_SPEECH`] | [`cactus_transcribe`], [`cactus_stream_transcribe_process`], [`cactus_stream_transcribe_stop`], [`cactus_set_audio`], [`cactus_embed`] |
//! | Mirage | [`CACTUS_VISION`] | [`cactus_describe`] |
//!
//! The engine was published as `libneedle.a` with a `needle_*` API until release 4.0, which
//! renamed every symbol to `cactus_*`. Upstream's older general SDK (`cactus_engine.h`) is a
//! different library and is not bound here: ARM64-only, no prebuilt archive, source-available
//! licence.
//!
//! ## Linking
//!
//! The build script picks the first source that applies:
//!
//! | Source | Behaviour |
//! | --- | --- |
//! | `DOCS_RS` | Links nothing, so documentation builds need no archive. |
//! | `CACTUS_LIB_DIR` | Links `libcactus.a` from that directory. |
//! | feature `download-binaries` (default) | Downloads the archive for the target from the pinned commit and verifies its SHA-256. |
//!
//! With `download-binaries` off and `CACTUS_LIB_DIR` unset the build fails and says so.
//!
//! Every upstream archive is built against LLVM libc++, never libstdc++. The build script links
//! `c++` on Apple and Windows (gnullvm) targets, `c++` and `c++abi` on Linux, and `c++_shared`
//! on Android. On Debian
//! and Ubuntu that means installing `libc++-dev libc++abi-dev`, and a native Linux build without
//! libc++ stops early and says so. The archive also needs one function that libc++ 18 and 20 do
//! not export, so on Linux this crate links a weak definition of it (`shim/libcxx_compat.c`);
//! a libc++ that has the real one takes precedence.
//!
//! Set `CACTUS_CXXSTDLIB` to a comma-separated list to replace the default: a bare name links
//! dynamically, `static=name` links the static archive, and an empty string links nothing.
//! `CACTUS_CXXSTDLIB="static=c++,static=c++abi"` was checked on Ubuntu 24.04 and gives a binary
//! with no libc++ dependency (add the directory holding `libc++.a` to the linker search path).
//!
//! Upstream publishes no archive for Intel macOS, and its Windows archives are llvm-mingw builds
//! that link only with the `*-pc-windows-gnullvm` targets, not MSVC.
//!
//! ## What the archive contains
//!
//! The engine is CPU only. Its symbol table names kernels for AVX2, AVX-512 (VNNI, BF16) and AMX
//! on x86 and for NEON on ARM, chosen at startup from what the CPU offers, and it has no CUDA,
//! Vulkan, OpenCL or Metal symbols. `nm` also shows no socket, resolver or HTTP symbols, so the
//! telemetry note in upstream's documentation covers their command-line runner and Python
//! package rather than this library.
//!
//! ## Safety Guarantees
//!
//! None. The engine holds one process-global model per kind, text (Needle 3), speech (Whistle) and
//! vision (Mirage), behind one non-thread-safe runtime with no handles and one global error string.
//! Every call, of any kind, must be serialised by the caller, and nothing here enforces that. Each function
//! documents the contract that was measured against the shipped archive under `# Safety`.
//!
//! ## Affiliation
//!
//! This crate is unofficial and is not affiliated with Cactus Compute. The engine it links is
//! published by them under Apache-2.0.
//!
//! ## Examples
//!
//! ```
//! // The pin the `download-binaries` feature fetches from.
//! assert_eq!(cactus_sys::CACTUS_ENGINE_COMMIT.len(), 40);
//! ```

#![no_std]
#![warn(missing_docs)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(feature = "engine")]
pub mod engine;

#[cfg(feature = "engine")]
pub use engine::{
    CACTUS_ENGINE_COMMIT, CACTUS_ENGINE_VERSION, CACTUS_SPEECH, CACTUS_TEXT, CACTUS_VISION,
    cactus_complete, cactus_describe, cactus_embed, cactus_init, cactus_last_error, cactus_load,
    cactus_models, cactus_reset, cactus_set_audio, cactus_set_threads,
    cactus_stream_transcribe_process, cactus_stream_transcribe_stop, cactus_transcribe,
};
