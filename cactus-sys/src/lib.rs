//! # cactus-sys
//!
//! Raw FFI bindings to [Cactus Compute](https://cactuscompute.com) inference engines.
//!
//! ## What is cactus-sys?
//!
//! Cactus Compute ships its engines as prebuilt static libraries with a small C API. This crate
//! links one of them and declares that API, nothing more: every item is `unsafe`, and no Rust
//! type stands between the caller and the engine. For a safe, typed API use
//! [`cactus-rs`](https://docs.rs/cactus-rs) instead.
//!
//! ## Key Features
//!
//! - **Needle 3**: the five functions of `needle.h`, behind the `needle` feature (default)
//! - **Pinned Binaries**: the archive is fetched from one Hugging Face commit and checked
//!   against a SHA-256 recorded in this crate
//! - **Offline Builds**: point `CACTUS_NEEDLE_LIB_DIR` at a local archive and nothing is
//!   downloaded
//! - **No Dependencies**: `#![no_std]`, and no runtime crates
//!
//! ## Engines
//!
//! | Engine | Header | Feature | Status |
//! | --- | --- | --- | --- |
//! | Needle 3 | `include/needle.h` | `needle` | supported |
//! | Cactus | `cactus_engine.h` | none yet | planned |
//!
//! ## Linking
//!
//! The build script picks the first source that applies:
//!
//! | Source | Behaviour |
//! | --- | --- |
//! | `DOCS_RS` | Links nothing, so documentation builds need no archive. |
//! | `CACTUS_NEEDLE_LIB_DIR` | Links `libneedle.a` from that directory. |
//! | feature `download-binaries` (default) | Downloads the archive for the target from the pinned commit and verifies its SHA-256. |
//!
//! With `download-binaries` off and `CACTUS_NEEDLE_LIB_DIR` unset the build fails and says so.
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
//! The engine is CPU only. Its symbol table names kernels for AVX2, AVX-512, AVX-512 VNNI and AMX
//! on x86 and for NEON on ARM, chosen at startup from what the CPU offers, and it has no CUDA,
//! Vulkan, OpenCL or Metal symbols. `nm` also shows no socket, resolver or HTTP symbols, so the
//! telemetry note in upstream's documentation covers their command-line runner and Python
//! package rather than this library.
//!
//! ## Safety Guarantees
//!
//! None. The engine is one process-global, non-thread-safe model with no handles, so every call
//! must be serialised by the caller, and nothing here enforces that. Each function documents the
//! contract that was measured against the shipped archive under `# Safety`.
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
//! assert_eq!(cactus_sys::NEEDLE_ENGINE_COMMIT.len(), 40);
//! ```

#![no_std]

#[cfg(feature = "needle")]
pub mod needle;

#[cfg(feature = "needle")]
pub use needle::{
    NEEDLE_ENGINE_COMMIT, needle_complete, needle_embed, needle_init, needle_load, needle_reset,
};
