# Changelog

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
