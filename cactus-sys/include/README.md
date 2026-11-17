# Vendored header

`cactus.h` is an unmodified copy of the header Cactus Compute publishes next to every engine archive
at [Cactus-Compute/cactus](https://huggingface.co/Cactus-Compute/cactus), taken at commit
`d2c5cd70432ed78345f3ba6b620bff2feeec6c86` (`releases/4.1.1/macos-arm64/cactus.h`, identical on
every platform). Copyright Cactus Compute, licensed under Apache-2.0.

It is kept for reference only. Nothing compiles it: the declarations in `src/engine.rs` are written by
hand to match it, and `tests/abi.rs` checks that they do.

The header carries no comments since engine 4.0. The contract of each function is recorded in the
doc comments of `src/engine.rs`, as measured against the archive by the `tests/probe_*.rs` tests.
