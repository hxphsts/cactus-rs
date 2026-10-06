#!/usr/bin/env bash
# Checks that every copy of a pinned value agrees with its single source of truth.
#
# Sources of truth:
#   cactus-sys/prebuilt.toml          engine commit, per-platform libneedle.a digests, needle.h digest
#   cactus-rs/src/needle/weights.rs   WEIGHTS_SHA256 (needle3.cact; its revision is the engine commit)
#   cactus-rs/src/whistle/weights.rs  WEIGHTS_REVISION, WEIGHTS_SHA256 (whistle.cact)
#   Cargo.toml [workspace.package]    version
#
# Run from anywhere inside the repository; works with GNU or BSD tools (Linux and macOS).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PREBUILT=cactus-sys/prebuilt.toml
HEADER=cactus-sys/include/needle.h
HEADER_README=cactus-sys/include/README.md
NEEDLE_WEIGHTS=cactus-rs/src/needle/weights.rs
WHISTLE_WEIGHTS=cactus-rs/src/whistle/weights.rs

failures=0
fail() {
  echo "pin mismatch: $*" >&2
  failures=$((failures + 1))
}

sha256() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# First `key = "value"` in a TOML-ish file, comments excluded.
toml_value() {
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$2" | head -n1 | sed -E 's/^[^=]*=[[:space:]]*"?([^"]*)"?.*$/\1/'
}

# Value of `pub const NAME: &str = "...";` in a Rust file.
rust_const() {
  grep -E "const $1: &str = \"" "$2" | head -n1 | sed -E 's/.*= "([^"]*)".*/\1/'
}

# Hex runs of exactly N digits in stdin, one per line (portable stand-in for `grep -o '\b...\b'`).
hex_runs() {
  tr -c '0-9a-f' '\n' | grep -xE "[0-9a-f]{$1}" || true
}

require() {
  [ -n "$2" ] || { echo "check-pins: could not read $1" >&2; exit 2; }
}

engine_commit=$(toml_value commit "$PREBUILT")
needle_sha=$(rust_const WEIGHTS_SHA256 "$NEEDLE_WEIGHTS")
whistle_rev=$(rust_const WEIGHTS_REVISION "$WHISTLE_WEIGHTS")
whistle_sha=$(rust_const WEIGHTS_SHA256 "$WHISTLE_WEIGHTS")
header_sha=$(grep -E '^#.*sha256 [0-9a-f]{64}' "$PREBUILT" | head -n1 | sed -E 's/.*sha256 ([0-9a-f]{64}).*/\1/' || true)
require "commit from $PREBUILT" "$engine_commit"
require "WEIGHTS_SHA256 from $NEEDLE_WEIGHTS" "$needle_sha"
require "WEIGHTS_REVISION from $WHISTLE_WEIGHTS" "$whistle_rev"
require "WEIGHTS_SHA256 from $WHISTLE_WEIGHTS" "$whistle_sha"
require "the needle.h digest in the comments of $PREBUILT" "$header_sha"
# Every libneedle.a digest in the [sha256] table.
archive_shas=$(sed -n '/^\[sha256\]/,$p' "$PREBUILT" | grep -oE '"[0-9a-f]{64}"' | tr -d '"')
require "the [sha256] table of $PREBUILT" "$archive_shas"

# (a) The vendored header is the one the pin describes.
actual_header=$(sha256 "$HEADER")
[ "$actual_header" = "$header_sha" ] ||
  fail "$HEADER hashes to $actual_header, but $PREBUILT says $header_sha"

# (b) The header README names the pinned commit, and no other.
readme_commits=$(hex_runs 40 < "$HEADER_README" | sort -u)
[ -n "$readme_commits" ] || fail "$HEADER_README names no commit (expected $engine_commit)"
for c in $readme_commits; do
  [ "$c" = "$engine_commit" ] || fail "$HEADER_README names commit $c, but $PREBUILT pins $engine_commit"
done

# (c) Duplicates elsewhere in the tree. CHANGELOG.md is history and is skipped; the sources of
# truth are checked above or are the values themselves.
is_prefix() { # is_prefix SHORT FULL...
  local short=$1 full
  shift
  for full in "$@"; do
    case "$full" in "$short"*) return 0 ;; esac
  done
  return 1
}

# Only lines holding a run of seven or more hex digits can hold a pin, so let git find them.
while IFS= read -r match; do
  file=${match%%:*}
  rest=${match#*:}
  where="$file:${rest%%:*}"
  line=${rest#*:}
  case "$file" in
    "$PREBUILT" | "$NEEDLE_WEIGHTS" | "$WHISTLE_WEIGHTS" | "$HEADER_README" | CHANGELOG.md | Cargo.lock) continue ;;
  esac

  # Full commits: any 40-hex token is one of the two pins, and the right one for its repo.
  for token in $(printf '%s\n' "$line" | hex_runs 40); do
    case "$line" in
      *needle3*) [ "$token" = "$engine_commit" ] || fail "$where: needle3 commit $token, pinned $engine_commit" ;;
      *whistle*) [ "$token" = "$whistle_rev" ] || fail "$where: whistle revision $token, pinned $whistle_rev" ;;
      *) [ "$token" = "$engine_commit" ] || [ "$token" = "$whistle_rev" ] ||
        fail "$where: commit $token is neither the engine pin $engine_commit nor the whistle pin $whistle_rev" ;;
    esac
  done

  # Abbreviated commits, where the wording says they name a pin: quoted strings (cache
  # directories), "commit `x`", "measured at x", "at `x`". A capitalised "At `x`" is left alone,
  # so prose can still cite an older engine. Upstream's GitHub source repository has its own
  # history, which tests/data cites by path.
  case "$line" in *environments/*) ;; *)
    # shellcheck disable=SC2016 # the backticks are literal Markdown
    for token in $(printf '%s\n' "$line" |
      grep -oE '("[0-9a-f]{8,39}"|[Cc]ommit `[0-9a-f]{7,39}`|[Mm]easured at `?[0-9a-f]{7,39}|[[:space:]]at `[0-9a-f]{7,39}`)' |
      grep -oE '[0-9a-f]{7,39}[`"]?$' | tr -d '`"' || true); do
      is_prefix "$token" "$engine_commit" "$whistle_rev" ||
        fail "$where: \`$token\` names neither the engine pin $engine_commit nor the whistle pin $whistle_rev"
    done
    ;;
  esac

  # Digests of pinned files: a 64-hex token next to the file name must be the pinned digest.
  for token in $(printf '%s\n' "$line" | hex_runs 64); do
    case "$line" in
      *needle3.cact* | *NEEDLE_WEIGHTS_SHA256*)
        [ "$token" = "$needle_sha" ] || fail "$where: needle3.cact digest $token, pinned $needle_sha" ;;
      *whistle.cact* | *WHISTLE_WEIGHTS_SHA256*)
        [ "$token" = "$whistle_sha" ] || fail "$where: whistle.cact digest $token, pinned $whistle_sha" ;;
      *libneedle.a* | *ENGINE_SHA256*)
        printf '%s\n' "$archive_shas" | grep -qx "$token" || fail "$where: libneedle.a digest $token is not in $PREBUILT" ;;
      *needle.h*)
        [ "$token" = "$header_sha" ] || fail "$where: needle.h digest $token, pinned $header_sha" ;;
    esac
  done
done < <(git grep -nIE '[0-9a-f]{7,}' -- . || true)

# (d) One version everywhere a release names it.
workspace_version=$(sed -n '/^\[workspace.package\]/,/^\[/p' Cargo.toml | toml_value version /dev/stdin)
sys_requirement=$(grep -E '^cactus-sys[[:space:]]*=' cactus-rs/Cargo.toml | grep -oE 'version[[:space:]]*=[[:space:]]*"[^"]*"' | sed -E 's/.*"[=^~]?([^"]*)"/\1/' || true)
changelog_version=$(grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+[^]]*\]' CHANGELOG.md | head -n1 | sed -E 's/^## \[(.*)\]/\1/' || true)
require "the [workspace.package] version from Cargo.toml" "$workspace_version"
require "the cactus-sys version requirement from cactus-rs/Cargo.toml" "$sys_requirement"
require "a ## [x.y.z] heading from CHANGELOG.md" "$changelog_version"
[ "$sys_requirement" = "$workspace_version" ] ||
  fail "cactus-rs/Cargo.toml requires cactus-sys $sys_requirement, but the workspace version is $workspace_version"
[ "$changelog_version" = "$workspace_version" ] ||
  fail "the latest CHANGELOG.md section is [$changelog_version], but the workspace version is $workspace_version"

if [ "$failures" -gt 0 ]; then
  echo "check-pins: $failures mismatch(es); update the copies above to match the source of truth." >&2
  exit 1
fi
echo "pins consistent (engine ${engine_commit:0:12}, whistle ${whistle_rev:0:12}, version $workspace_version)"
