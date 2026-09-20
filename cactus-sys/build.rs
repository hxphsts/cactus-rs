//! Build script for `cactus-sys`.
//!
//! It resolves exactly one `libneedle.a` for the target and emits the link directives for it.
//! Resolution order, first match wins:
//!
//! 1. feature `needle` off: nothing to link.
//! 2. `DOCS_RS` set: documentation builds do not link the engine.
//! 3. `CACTUS_NEEDLE_LIB_DIR`: link the archive the caller points at.
//! 4. feature `download-binaries`: fetch the pinned archive for the target and verify its
//!    SHA-256 against `prebuilt.toml`.
//! 5. otherwise: fail, naming both options.

use std::env;
use std::path::PathBuf;
use std::process;

/// The pinned upstream release table. Parsed by [`prebuilt`].
const PREBUILT: &str = include_str!("prebuilt.toml");

/// File name of the engine archive on every upstream platform, Windows included.
const ARCHIVE: &str = "libneedle.a";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=prebuilt.toml");
    println!("cargo::rerun-if-env-changed=CACTUS_NEEDLE_LIB_DIR");
    println!("cargo::rerun-if-env-changed=CACTUS_CXXSTDLIB");
    println!("cargo::rerun-if-env-changed=DOCS_RS");

    // Single source of truth for the pin: `cactus_sys::NEEDLE_ENGINE_COMMIT` reads it back.
    println!(
        "cargo::rustc-env=NEEDLE_ENGINE_COMMIT={}",
        prebuilt("commit")
    );

    if !feature("NEEDLE") {
        return;
    }
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let lib_dir = match env::var_os("CACTUS_NEEDLE_LIB_DIR") {
        Some(dir) => lib_dir_from_env(PathBuf::from(dir)),
        None => downloaded_lib_dir(),
    };

    println!("cargo::rustc-link-search=native={}", lib_dir.display());
    println!("cargo::rustc-link-lib=static=needle");
    // After the archive and before the C++ runtime, so the linker reaches the shim while the
    // engine's reference is still open.
    if cfg("CARGO_CFG_TARGET_OS") == "linux" {
        libcxx_compat();
    }
    for lib in cxx_runtime() {
        println!("cargo::rustc-link-lib={lib}");
    }
}

/// Reports a fatal build failure and stops, one `cargo::error` line per line of `message`.
///
/// House style: the last line is a `Help:` line the reader can act on.
fn fail(message: &str) -> ! {
    for line in message.lines() {
        println!("cargo::error={line}");
    }
    process::exit(1);
}

/// Returns whether Cargo enabled the feature, e.g. `feature("DOWNLOAD_BINARIES")`.
fn feature(name: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

/// Returns a `CARGO_CFG_*` value, empty when the target leaves it unset.
fn cfg(name: &str) -> String {
    env::var(name).unwrap_or_default()
}

/// Looks up `key = "value"` in [`PREBUILT`], skipping comments, blanks and table headers.
///
/// The file is a hand-written subset of TOML precisely so that this stays ten lines and the
/// crate needs no TOML parser as a build dependency.
fn prebuilt(key: &str) -> &'static str {
    let found = PREBUILT.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        let (name, value) = line.split_once('=')?;
        (name.trim() == key).then(|| value.trim().trim_matches('"'))
    });

    found.unwrap_or_else(|| {
        fail(&format!(
            "cactus-sys: prebuilt.toml has no entry for `{key}`.\n\
             The pinned release table is incomplete or was edited by hand.\n\
             Help: restore cactus-sys/prebuilt.toml from the repository."
        ))
    })
}

/// Verifies that `CACTUS_NEEDLE_LIB_DIR` really holds an engine archive, and returns it.
fn lib_dir_from_env(dir: PathBuf) -> PathBuf {
    if dir.join(ARCHIVE).is_file() {
        return dir;
    }

    fail(&format!(
        "cactus-sys: CACTUS_NEEDLE_LIB_DIR is set to `{}`, but `{ARCHIVE}` is not in it.\n\
         The variable must name the directory that contains the archive, not the archive itself.\n\
         Help: point it at a directory holding {ARCHIVE}, or unset it and build with the\n\
         `download-binaries` feature to fetch the pinned archive automatically.",
        dir.display()
    ))
}

/// Explains how to supply an archive when no feature can fetch one.
#[cfg(not(feature = "download-binaries"))]
fn downloaded_lib_dir() -> PathBuf {
    fail(
        "cactus-sys: no engine archive available.\n\
         The `needle` feature is on, but `download-binaries` is off and CACTUS_NEEDLE_LIB_DIR\n\
         is unset, so there is nothing to link against. Cactus Compute ships Needle as a\n\
         closed-source static library; this crate cannot build one from source.\n\
         Help: either set CACTUS_NEEDLE_LIB_DIR to a directory containing libneedle.a for your\n\
         target, or enable the `download-binaries` feature to fetch the pinned, SHA-256 verified\n\
         archive from Hugging Face.",
    )
}

/// Fetches the pinned archive for the target into `OUT_DIR` and returns its directory.
///
/// A previously downloaded archive with the expected digest is reused, so a second build of
/// the same target does not hit the network.
#[cfg(feature = "download-binaries")]
fn downloaded_lib_dir() -> PathBuf {
    let platform = platform();
    let expected = prebuilt(&platform);

    let out_dir = match env::var_os("OUT_DIR") {
        Some(dir) => PathBuf::from(dir).join(&platform),
        None => fail(
            "cactus-sys: OUT_DIR is unset.\n\
             Build scripts are always given one, so this build was not started by Cargo.\n\
             Help: build the crate with `cargo build`.",
        ),
    };
    let archive = out_dir.join(ARCHIVE);
    if download::digest(&archive).as_deref() == Some(expected) {
        return out_dir;
    }

    let url = format!(
        "https://huggingface.co/{repo}/resolve/{commit}/{platform}/{ARCHIVE}",
        repo = prebuilt("repo"),
        commit = prebuilt("commit"),
    );
    download::fetch(&url, &out_dir, &archive, expected);
    out_dir
}

/// Maps the target to an upstream platform directory, or explains why there is none.
#[cfg(feature = "download-binaries")]
fn platform() -> String {
    let os = cfg("CARGO_CFG_TARGET_OS");
    let arch = cfg("CARGO_CFG_TARGET_ARCH");
    let abi = cfg("CARGO_CFG_TARGET_ABI");
    let target_env = cfg("CARGO_CFG_TARGET_ENV");
    let endian = cfg("CARGO_CFG_TARGET_ENDIAN");

    let platform = match (os.as_str(), arch.as_str()) {
        ("macos", "aarch64") => "macos-arm64",
        ("macos", "x86_64") => fail(
            "cactus-sys: Intel macOS is not supported.\n\
             Cactus Compute publishes no libneedle.a for x86_64-apple-darwin; the only macOS\n\
             archive upstream ships is arm64.\n\
             Help: build for aarch64-apple-darwin, or set CACTUS_NEEDLE_LIB_DIR to a directory\n\
             holding an x86_64 libneedle.a you built or obtained yourself.",
        ),

        ("linux", _) if target_env == "gnu" || target_env == "musl" => {
            match (arch.as_str(), endian.as_str()) {
                ("x86_64", _) => "linux-x86_64",
                ("aarch64", _) => "linux-arm64",
                ("arm", _) => "linux-armv7",
                ("riscv64", _) => "linux-riscv64",
                ("mips", "little") => "linux-mipsel",
                _ => fail(&unsupported(&os, &arch)),
            }
        }

        ("android", "aarch64") => "android-arm64",
        ("android", "arm") => "android-armv7",
        ("android", "riscv64") => "android-riscv64",

        ("ios", "aarch64") if abi == "sim" => "ios-sim-arm64",
        ("ios", "aarch64") if abi.is_empty() => "ios-arm64",
        ("tvos", "aarch64") => "tvos-arm64",
        ("watchos", "aarch64") => "watchos-arm64",

        ("windows", _) if target_env == "msvc" => fail(
            "cactus-sys: MSVC Windows targets are not supported.\n\
             The upstream Windows archives are llvm-mingw builds: they use the Itanium C++ ABI\n\
             and LLVM libc++, which the MSVC toolchain cannot link.\n\
             Help: build for x86_64-pc-windows-gnullvm or aarch64-pc-windows-gnullvm.",
        ),
        ("windows", _) if abi != "llvm" => fail(
            "cactus-sys: GNU (mingw-w64) Windows targets are not supported.\n\
             The upstream Windows archives are llvm-mingw builds against LLVM libc++, not the\n\
             libstdc++ that the *-pc-windows-gnu targets link.\n\
             Help: build for x86_64-pc-windows-gnullvm or aarch64-pc-windows-gnullvm.",
        ),
        ("windows", "x86_64") => "windows-x86_64",
        ("windows", "aarch64") => "windows-arm64",

        _ => fail(&unsupported(&os, &arch)),
    };

    platform.to_owned()
}

/// The message for a target the pinned release has no archive for.
#[cfg(feature = "download-binaries")]
fn unsupported(os: &str, arch: &str) -> String {
    format!(
        "cactus-sys: no prebuilt Needle engine for this target (os `{os}`, arch `{arch}`).\n\
         Cactus Compute publishes archives for macOS arm64, Linux (x86_64, arm64, armv7,\n\
         riscv64, mipsel), Android (arm64, armv7, riscv64), iOS, iOS Simulator, tvOS, watchOS\n\
         and llvm-mingw Windows (x86_64, arm64).\n\
         Help: build for one of those targets, or set CACTUS_NEEDLE_LIB_DIR to a directory\n\
         holding a libneedle.a for this one."
    )
}

/// Builds `shim/libcxx_compat.c`: the one libc++ function the archive needs that libc++ 18 and
/// 20 do not export.
///
/// The symbol is weak, so linking it everywhere on Linux is harmless where libc++ is new enough.
fn libcxx_compat() {
    println!("cargo::rerun-if-changed=shim/libcxx_compat.c");
    cc::Build::new()
        .file("shim/libcxx_compat.c")
        .warnings(true)
        .compile("cactus_libcxx_compat");
}

/// The C++ runtime libraries the engine archive needs, in link order, as `kind=name` pairs.
///
/// Every upstream archive is built against LLVM libc++ (`std::__1`, or `std::__ndk1` on
/// Android); none of them reference libstdc++. Nothing else is emitted: Rust's own std
/// already links libm, libdl and pthreads on every platform this crate supports.
///
/// `CACTUS_CXXSTDLIB` replaces the default with a comma-separated list. An entry is a library
/// name, linked dynamically, or `static=name` to link the static archive instead. An empty
/// value links nothing, for callers who supply the runtime themselves.
fn cxx_runtime() -> Vec<String> {
    if let Ok(list) = env::var("CACTUS_CXXSTDLIB") {
        return list
            .split(',')
            .map(str::trim)
            .filter(|lib| !lib.is_empty())
            .map(|lib| match lib.split_once('=') {
                Some(_) => lib.to_owned(),
                None => format!("dylib={lib}"),
            })
            .collect();
    }

    let libs: &[&str] = match cfg("CARGO_CFG_TARGET_OS").as_str() {
        "macos" | "ios" | "tvos" | "watchos" | "visionos" => &["c++"],
        "linux" => {
            require_libcxx();
            &["c++", "c++abi"]
        }
        "android" => &["c++_shared"],
        "windows" => &["c++"],
        _ => &[],
    };

    libs.iter().map(|lib| format!("dylib={lib}")).collect()
}

/// Fails early, in words, when a native Linux build has no libc++ to link.
///
/// Most distributions ship GNU libstdc++ only, and without this check the first sign of trouble
/// is a page of linker arguments ending in `unable to find library -lc++`. The C compiler is
/// asked where it would find the library; a bare file name back means it would not. Cross builds
/// are skipped, because the host compiler says nothing about the target's sysroot.
fn require_libcxx() {
    if env::var("HOST").ok() != env::var("TARGET").ok() {
        return;
    }

    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let found = |library: &str| {
        let output = process::Command::new(&compiler)
            .arg(format!("-print-file-name={library}"))
            .output()
            .ok()?;
        Some(String::from_utf8_lossy(&output.stdout).trim().contains('/'))
    };

    // An unusable compiler is not evidence either way; let the link step speak for itself.
    let (Some(shared), Some(archive)) = (found("libc++.so"), found("libc++.a")) else {
        return;
    };
    if shared || archive {
        return;
    }

    fail(
        "cactus-sys: LLVM libc++ was not found.\n\
         The Needle engine archive is built against libc++, and this system appears to have only\n\
         GNU libstdc++, which cannot stand in for it.\n\
         Help: install libc++ (Debian and Ubuntu: `sudo apt install libc++-dev libc++abi-dev`;\n\
         Fedora: `sudo dnf install libcxx-devel libcxxabi-devel`), or set CACTUS_CXXSTDLIB to the\n\
         libraries to link instead.",
    )
}

/// Network and digest handling, pulled in only by the `download-binaries` feature.
#[cfg(feature = "download-binaries")]
mod download {
    use std::fmt::Write as _;
    use std::fs;
    use std::io::{self, Read as _};
    use std::path::Path;

    use sha2::{Digest as _, Sha256};

    /// Returns the SHA-256 of `path` in lowercase hex, or `None` when it cannot be read.
    pub fn digest(path: &Path) -> Option<String> {
        let mut file = fs::File::open(path).ok()?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let read = file.read(&mut buffer).ok()?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        let mut hex = String::with_capacity(64);
        for byte in hasher.finalize() {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Some(hex)
    }

    /// Downloads `url` to `archive`, refusing to install it unless it hashes to `expected`.
    ///
    /// The body lands on a temporary name first, so an interrupted build never leaves a
    /// half-written archive that a later build would mistake for a complete one.
    pub fn fetch(url: &str, out_dir: &Path, archive: &Path, expected: &str) {
        if let Err(error) = fs::create_dir_all(out_dir) {
            super::fail(&format!(
                "cactus-sys: could not create `{}`: {error}\n\
                 Help: check the permissions on the Cargo target directory.",
                out_dir.display()
            ));
        }

        let partial = archive.with_extension("a.partial");
        get(url, &partial);

        let actual = digest(&partial).unwrap_or_default();
        if actual != expected {
            let _ = fs::remove_file(&partial);
            super::fail(&format!(
                "cactus-sys: checksum mismatch for the downloaded engine archive.\n\
                 url:      {url}\n\
                 expected: {expected}\n\
                 actual:   {actual}\n\
                 The download was discarded. Either the transfer was corrupted, a proxy served\n\
                 something else, or prebuilt.toml no longer matches the pinned commit.\n\
                 Help: retry the build; if it keeps failing, verify cactus-sys/prebuilt.toml\n\
                 against the upstream repository before trusting the archive."
            ));
        }

        if let Err(error) = fs::rename(&partial, archive) {
            let _ = fs::remove_file(&partial);
            super::fail(&format!(
                "cactus-sys: could not install `{}`: {error}\n\
                 Help: remove the file and rebuild.",
                archive.display()
            ));
        }
    }

    /// Streams a GET into `dest`, identifying only as `cactus-sys/<version>`.
    ///
    /// The body is capped well above any archive upstream has ever shipped (the largest at the
    /// pinned commit is 3.5 MiB), so a misdirected response cannot fill the disk.
    fn get(url: &str, dest: &Path) {
        const MAX_BODY: u64 = 64 * 1024 * 1024;

        let config = ureq::Agent::config_builder()
            .user_agent(concat!("cactus-sys/", env!("CARGO_PKG_VERSION")))
            .build();
        let agent = ureq::Agent::new_with_config(config);

        let mut response = match agent.get(url).call() {
            Ok(response) => response,
            Err(error) => super::fail(&format!(
                "cactus-sys: could not download the Needle engine archive.\n\
                 url:   {url}\n\
                 error: {error}\n\
                 Help: check network access to huggingface.co, or build offline by setting\n\
                 CACTUS_NEEDLE_LIB_DIR to a directory holding libneedle.a for your target."
            )),
        };

        let mut file = match fs::File::create(dest) {
            Ok(file) => file,
            Err(error) => super::fail(&format!(
                "cactus-sys: could not write `{}`: {error}\n\
                 Help: check the free space and permissions on the Cargo target directory.",
                dest.display()
            )),
        };

        let mut body = response.body_mut().with_config().limit(MAX_BODY).reader();
        if let Err(error) = io::copy(&mut body, &mut file) {
            let _ = fs::remove_file(dest);
            super::fail(&format!(
                "cactus-sys: the engine archive download failed part way through.\n\
                 url:   {url}\n\
                 error: {error}\n\
                 Help: retry the build."
            ));
        }
    }
}
