# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-17

### Changed

- `Whistle::transcribe` takes a clip of any length; the engine transcribes one over 30 seconds window by window. `Error::AudioTooLong` now concerns speech embeddings and audio completions, which keep the 30 second limit, and `MAX_SAMPLES` and `MAX_SECONDS` are documented as theirs
- A transcript larger than the output buffer is no longer an error: the engine states the size it needs and the call is repeated with it
- **Breaking:** the engine is Cactus engine 4.1.1 (`libcactus.a`, `cactus.h`), pinned at `Cactus-Compute/cactus` commit `d2c5cd70`. Upstream renamed the library and every symbol with release 4.0 and no longer publishes `libneedle.a`; one engine now runs Needle 3, Whistle and Mirage. Needle and Whistle weights are unchanged, and nothing is downloaded again
- **Breaking, `cactus-sys`:** every function and constant moves from `needle_*` / `NEEDLE_*` to `cactus_*` / `CACTUS_*`, the module from `cactus_sys::needle` to `cactus_sys::engine`, and the `needle` feature to `engine`. Signatures are unchanged
- `CACTUS_NEEDLE_LIB_DIR` is now `CACTUS_LIB_DIR` and must hold `libcactus.a`. The old name is still read, with a build warning
- `needle::weights::WEIGHTS_REVISION` pins the Needle weights on their own; they used to be fetched at the engine's commit
- The weekly upstream check watches for a new engine release and for a changed weight file, instead of any commit to the upstream repositories
- Every GitHub Action is pinned to a commit SHA and kept current by Dependabot; workflows have concurrency groups and timeouts, and the weekly pin check retries transient Hugging Face errors and can no longer file an issue twice
- `Cargo.lock` is committed, so CI, `cargo deny` and local builds resolve the same dependency versions
- A test in `cactus-sys` compares the hand-written `extern "C"` declarations with the vendored header, function by function
- With `CACTUS_LIB_DIR` set, replacing the archive in that directory now triggers a relink
- The published `cactus-sys` crate no longer ships probe tests that read files outside its own package
- `cargo deny` rejects wildcard dependency versions

### Added

- `Whistle::stream` and `Whistle::stream_with_options`: live transcription. A `TranscriptStream` takes audio in pieces of any size, runs one pass per second, and returns an `Update` with the words settled on and the tail still being heard; `finish` commits the tail (`TranscriptStream`, `Update`, `PASS_SAMPLES`)
- `voice_stream` example: a spoken command routed to a tool call from the pending tail, before the speaker has finished
- `cactus-sys/tests/probe_stream.rs` records what the two streaming calls and `cactus_set_threads` do
- `cactus-rs`: `mirage` module for Cactus Compute's Mirage vision model, loaded into the same engine as Needle and Whistle (`Mirage`, `MirageBuilder`, `Weights`, `Image`, `DescribeOptions`, `Section`, `Field`, `MaskFormat`, `Description`, `Object`, `Region`, `TextLine`, `Relation`, `BBox`, `Mask`, `Bitmap`)
- `Mirage::describe`: the scene, objects with boxes, scores and masks, the text that can be read, relations and the model's reasoning, in one pass; names and descriptions in seven languages
- `Mirage::extract` with `Form`: choices, counts, short text and boxes answered from the picture, each with its probability (`Answers`, `Answer`, `AnswerValue`, `ABSTAIN`)
- Feature `image`: `Image::open` and `Image::decode` read PNG and JPEG files and apply their EXIF orientation
- `mirage::Weights::fetch` downloads, verifies and caches `mirage.cact` (21.6 MB, revision `3d03cee8`); `CACTUS_MIRAGE_WEIGHTS` points at a local copy
- `Model::Mirage`, `Error::InvalidImage`, `Error::ImageDecode` and `Error::Describe`
- `cactus-sys/tests/probe_vision.rs` records what `cactus_describe` accepts, writes and refuses, which upstream does not document
- `cactus-sys`: `CACTUS_VISION`, `CACTUS_ENGINE_VERSION`, and declarations for `cactus_describe`, `cactus_stream_transcribe_process`, `cactus_stream_transcribe_stop` and `cactus_set_threads`
- `SECURITY.md`: how to report a vulnerability privately

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
