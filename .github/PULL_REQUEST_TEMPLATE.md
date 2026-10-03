## Description

Brief description of changes.

## Checklist

- [ ] Tests pass (`cargo test --all --all-features`), with `CACTUS_NEEDLE_WEIGHTS` and `CACTUS_WHISTLE_WEIGHTS` set so the engine tests run instead of skipping
- [ ] Clippy passes (`cargo clippy --all-targets --all-features -- -D warnings`)
- [ ] Formatting checked (`cargo fmt --all -- --check`)
- [ ] Docs build (`DOCS_RS=1 RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`)
- [ ] MSRV holds (`cargo +1.85 check --all`)
- [ ] Offline build works (`DOCS_RS=1 cargo check -p cactus-rs --no-default-features`)
- [ ] Crates package (`cargo package --workspace`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

## Testing

Describe tests run and how to reproduce.
