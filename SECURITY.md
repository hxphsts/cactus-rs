# Security policy

## Supported versions

Fixes land on the latest released minor version of `cactus-rs` and `cactus-sys`. Older versions
are not patched; upgrade to the latest release.

## Reporting a vulnerability

Report it privately through GitHub:
[open a private advisory](https://github.com/hxphsts/cactus-rs/security/advisories/new).
Do not open a public issue for a vulnerability.

Include the crate version, the target platform, and the smallest input or program that shows the
problem. You should get a first answer within a week.

## Scope

In scope:

- memory safety or undefined behaviour reachable through the safe `cactus-rs` API
- the build script and the weight downloader: integrity checks, the cache, anything that makes
  them fetch or accept a file other than the pinned one
- the release workflow and the published crates

Out of scope here, report it upstream to [Cactus Compute](https://github.com/cactus-compute):

- defects inside the prebuilt engine archive that are not caused by how this crate calls it
- model behaviour: a wrong tool call or a wrong transcript is a quality issue, not a
  vulnerability
