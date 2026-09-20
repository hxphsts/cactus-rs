# Changelog

## [0.1.0] - 2026-09-20

### Added

- `cactus-rs`: safe API for Cactus Compute's Needle 3 engine: tool calling, structured extraction and 3072-dimension text embeddings (`Needle`, `Tool`, `Weights`, `Completion`)
- `cactus-sys`: raw FFI for the five functions of `needle.h`, `#![no_std]`
- One `Needle` per process, enforced by the type system: a second `build()` returns `Error::EngineBusy`, and `Needle` is `Send + Sync`
- Engine archive fetched at build time from a pinned Hugging Face commit and SHA-256 verified; `CACTUS_NEEDLE_LIB_DIR` links a local archive for offline builds
- `Weights::fetch` downloads, verifies and caches the weights under `~/.cache/cactus-rs`
- Tested on macOS arm64 and Ubuntu 24.04 x86_64, with a conformance suite that asserts parity with upstream's own engine
- Examples: `lights`, `extract`, `embed`, `custom_weights`, `desktop`
