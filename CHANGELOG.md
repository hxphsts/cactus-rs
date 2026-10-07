# Changelog

## [0.2.1] - 2026-10-06

### Added

- `Error::LoadFailed`, `Error::InitFailed` and `Error::EmbedFailed`, carrying the engine's own reason from `needle_last_error` (for `needle_init`, the measured token count when the prefix overflows the context), and `Error::detail()` to read the engine's words from any engine-call error

### Changed

- Load, init and text-embedding failures return the new variants; `Error::Load`, `Error::Init` and `Error::Embed` are no longer returned, so match the new variants instead
- The output buffer is capped at 4 MiB however large `max_new_tokens` is; a budget of `u32::MAX` used to allocate about 2 GiB and keep it
- Weight and engine downloads time out and retry transient failures with backoff (`CACTUS_DOWNLOAD_TIMEOUT`, `CACTUS_DOWNLOAD_RETRIES`); interrupted downloads older than an hour are removed from the cache
- `HF_ENDPOINT` selects a Hugging Face mirror for weight and engine downloads; `HF_HUB_OFFLINE=1` never touches the network
- A wrong-kind archive (`whistle::Weights` holding `needle3.cact`, or the reverse) is refused by reading its tensor directory, before anything reaches the engine; four probe tests in `cactus-sys/tests/` record that the engine replaces a loaded model on any successful load and drops the text init state on every text load
- The conformance harness folds case and integral floats before comparing calls, as upstream's harness does
- CI runs clippy and docs on Linux as well as macOS, checks the MSRV with all features, runs `cargo semver-checks` and `cargo deny`, cross-checks aarch64 Linux, Android and Windows gnullvm, verifies that the vendored header, pins and versions agree, and checks upstream weekly for new engine or weight revisions; releases verify both crate versions and the changelog before publishing
- README: the general Cactus engine and Needle 2 are documented as not planned, with the reasons

## [0.2.0] - 2026-10-03

### Added

- `cactus-rs`: `whistle` module for Cactus Compute's Whistle speech-to-text model, loaded into the same engine as Needle (`Whistle`, `WhistleBuilder`, `Weights`, `TranscribeOptions`, `Transcript`, `Word`, `Language`)
- Transcription of 16 kHz mono `f32` clips up to 30 s in seven languages (en, de, fr, es, it, nl, pl), detected or forced, with keyword biasing and word timestamps
- Speech embeddings: `Whistle::embed` returns one 512-float row per 80 ms frame; `frame_count`, `SAMPLE_RATE`, `MAX_SECONDS`, `MAX_SAMPLES`, `FRAME_MS`, `FRAME_SAMPLES`
- `whistle::Weights::fetch` downloads, verifies and caches `whistle.cact` (16.9 MB, revision `b358ddad`); `CACTUS_WHISTLE_WEIGHTS` points at a local copy
- `Needle::complete_audio` and `Needle::complete_audio_with_options`: a spoken request answered with tool calls in one engine call, once a `Whistle` has loaded the speech model
- `Completion::audio`: the transcript behind an audio turn
- `Model`, naming the Needle or Whistle archive an error is about
- Error variants `WrongModel`, `SpeechModelNotLoaded`, `AudioTooLong`, `NonFiniteSample`, `Transcribe`, `EmbedAudio`, `UnsupportedLanguage` and `InvalidKeyword`
- `cactus-sys`: declarations for `needle_models`, `needle_last_error`, `needle_set_audio` and `needle_transcribe`, and the constants `NEEDLE_TEXT` and `NEEDLE_SPEECH`
- Examples: `transcribe`, `voice_lights`

### Changed

- Engine pinned to `Cactus-Compute/needle3` commit `c7c415a3` (engine 3.1.0), which holds one text and one speech model at once
- **Breaking for `cactus-sys`**: `needle_complete` and `needle_embed` take `pcm` and `samples` parameters, following the new C API
- `cactus-rs`: the public API is source compatible with 0.1; every change is an addition (`Error` is `#[non_exhaustive]`)
- One process-wide lock serialises every engine call, so a `Needle` and a `Whistle` on different threads never overlap
- Error messages reworded to cover both models (`EngineBusy`, `WeightsAlreadyLoaded`, `UnsupportedWeights`, `Truncated`, `Download`, `Load`)
- The `needle3.cact` cache directory moves with the engine pin, so `Needle` weights are downloaded once more (35 MB)
- The conformance harness mirrors upstream's date prefix, and its baselines were re-measured against the new engine
- Examples declare `required-features = ["download"]`

## [0.1.0] - 2026-09-20

### Added

- `cactus-rs`: safe API for Cactus Compute's Needle 3 engine: tool calling, structured extraction and 3072-dimension text embeddings (`Needle`, `Tool`, `Weights`, `Completion`)
- `cactus-sys`: raw FFI for the five functions of `needle.h`, `#![no_std]`
- One `Needle` per process, enforced by the type system: a second `build()` returns `Error::EngineBusy`, and `Needle` is `Send + Sync`
- Engine archive fetched at build time from a pinned Hugging Face commit and SHA-256 verified; `CACTUS_NEEDLE_LIB_DIR` links a local archive for offline builds
- `Weights::fetch` downloads, verifies and caches the weights under `~/.cache/cactus-rs`
- Tested on macOS arm64 and Ubuntu 24.04 x86_64, with a conformance suite that asserts parity with upstream's own engine
- Examples: `lights`, `extract`, `embed`, `custom_weights`, `desktop`
