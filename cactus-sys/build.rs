//! Build script for `cactus-sys`.
//!
//! It resolves exactly one `libneedle.a` for the target and emits the link directives for it.
//! Resolution order, first match wins:
//!
//! 1. feature `needle` off: nothing to link.
//! 2. `DOCS_RS` set: documentation builds do not link the engine.
//! 3. `CACTUS_NEEDLE_LIB_DIR`: link the archive the caller points at.
//! 4. feature `download-binaries`: fetch the pinned archive for the target from `HF_ENDPOINT`
//!    (default `https://huggingface.co`) and verify its SHA-256 against `prebuilt.toml`. With
//!    `HF_HUB_OFFLINE` set (to anything but `0`) only an archive already in `OUT_DIR` is used.
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
    println!("cargo::rerun-if-env-changed=HF_ENDPOINT");
    println!("cargo::rerun-if-env-changed=HF_HUB_OFFLINE");

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

    if download::offline(env::var("HF_HUB_OFFLINE").ok().as_deref()) {
        fail(&format!(
            "cactus-sys: offline mode is on (HF_HUB_OFFLINE is set), and the Needle engine\n\
             archive for `{platform}` has not been downloaded into this target directory yet.\n\
             Help: set CACTUS_NEEDLE_LIB_DIR to a directory holding libneedle.a for your target,\n\
             or unset HF_HUB_OFFLINE (or set it to 0) to allow the download."
        ));
    }

    let endpoint = download::endpoint(env::var("HF_ENDPOINT").ok().as_deref());
    let url = download::url_with(
        &endpoint,
        prebuilt("repo"),
        prebuilt("commit"),
        &format!("{platform}/{ARCHIVE}"),
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
///
/// The timeout, retry and cleanup rules are the ones `cactus-rs` applies to weight downloads in
/// `cactus-rs/src/download.rs`, with helpers of the same names; that file holds their tests.
#[cfg(feature = "download-binaries")]
mod download {
    use std::env;
    use std::fmt::Write as _;
    use std::fs;
    use std::io::{self, Read as _};
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use sha2::{Digest as _, Sha256};

    /// Ceiling on the body, well above any archive upstream has ever shipped (the largest at the
    /// pinned commit is 3.5 MiB), so a misdirected response cannot fill the disk.
    const MAX_BODY: u64 = 64 * 1024 * 1024;

    /// How long establishing the connection, TLS handshake included, may take.
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

    /// How long the server may take to send the response headers once asked.
    const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

    /// How long the whole body may take, unless `CACTUS_DOWNLOAD_TIMEOUT` says otherwise.
    const BODY_TIMEOUT: Duration = Duration::from_secs(900);

    /// Attempts after the first, unless `CACTUS_DOWNLOAD_RETRIES` says otherwise.
    const RETRIES: u32 = 3;

    /// The longest `Retry-After` honoured.
    const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

    /// Age past which a `.partial` file is taken to belong to a build that died.
    const STALE_AFTER: Duration = Duration::from_secs(60 * 60);

    /// Where downloads come from unless `HF_ENDPOINT` names a mirror.
    const HF_DEFAULT_ENDPOINT: &str = "https://huggingface.co";

    /// The download URL of `file` at `revision` of `repo` on the hub at `endpoint`.
    pub fn url_with(endpoint: &str, repo: &str, revision: &str, file: &str) -> String {
        format!(
            "{}/{repo}/resolve/{revision}/{file}",
            endpoint.trim_end_matches('/')
        )
    }

    /// The hub to download from: `HF_ENDPOINT` without a trailing `/`, or Hugging Face itself.
    pub fn endpoint(value: Option<&str>) -> String {
        match value.map(|value| value.trim().trim_end_matches('/')) {
            Some(value) if !value.is_empty() => value.to_owned(),
            _ => HF_DEFAULT_ENDPOINT.to_owned(),
        }
    }

    /// Whether `HF_HUB_OFFLINE` forbids the network: any value but empty or `0`.
    pub fn offline(value: Option<&str>) -> bool {
        value.is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0"
        })
    }

    /// Why one download attempt failed, and whether another could succeed.
    enum Failure {
        /// A timeout, a dropped connection, a busy or failing server, or a corrupted transfer.
        Transient {
            detail: String,
            retry_after: Option<Duration>,
        },
        /// Trying again would fail the same way; the whole message for [`super::fail`].
        Fatal(String),
    }

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
    /// half-written archive that a later build would mistake for a complete one; temporary
    /// files more than an hour old, left by builds that died, are removed first. Transient
    /// failures and one checksum mismatch are retried, as in `cactus-rs`.
    pub fn fetch(url: &str, out_dir: &Path, archive: &Path, expected: &str) {
        if let Err(error) = fs::create_dir_all(out_dir) {
            super::fail(&format!(
                "cactus-sys: could not create `{}`: {error}\n\
                 Help: check the permissions on the Cargo target directory.",
                out_dir.display()
            ));
        }
        remove_stale_partials(out_dir, super::ARCHIVE, SystemTime::now());

        let agent = agent(body_timeout(
            env::var("CACTUS_DOWNLOAD_TIMEOUT").ok().as_deref(),
        ));
        let attempts = attempts(env::var("CACTUS_DOWNLOAD_RETRIES").ok().as_deref());
        let mut mismatches = 0;

        let result = retrying(attempts, thread::sleep, |_| {
            let partial = partial_path(archive);
            if let Err(failure) = get(&agent, url, &partial) {
                let _ = fs::remove_file(&partial);
                return Err(failure);
            }

            let actual = digest(&partial).unwrap_or_default();
            if actual == expected {
                return Ok(partial);
            }
            let _ = fs::remove_file(&partial);
            mismatches += 1;
            if mismatches == 1 {
                return Err(Failure::Transient {
                    detail: format!("checksum mismatch: expected {expected}, got {actual}"),
                    retry_after: None,
                });
            }
            Err(Failure::Fatal(format!(
                "cactus-sys: checksum mismatch for the downloaded engine archive, twice.\n\
                 url:      {url}\n\
                 expected: {expected}\n\
                 actual:   {actual}\n\
                 The download was discarded. Either a proxy served something else, or\n\
                 prebuilt.toml no longer matches the pinned commit.\n\
                 Help: retry the build; if it keeps failing, verify cactus-sys/prebuilt.toml\n\
                 against the upstream repository before trusting the archive."
            )))
        });

        let partial = match result {
            Ok(partial) => partial,
            Err(Failure::Fatal(message)) => super::fail(&message),
            Err(Failure::Transient { detail, .. }) => super::fail(&format!(
                "cactus-sys: could not download the Needle engine archive.\n\
                 url:   {url}\n\
                 error: gave up after {attempts} attempts; the last failed with: {detail}\n\
                 Help: check network access to the hub (HF_ENDPOINT, default huggingface.co),\n\
                 raise CACTUS_DOWNLOAD_TIMEOUT or CACTUS_DOWNLOAD_RETRIES, or build offline by\n\
                 setting CACTUS_NEEDLE_LIB_DIR to a directory holding libneedle.a for your target."
            )),
        };

        if let Err(error) = fs::rename(&partial, archive) {
            let _ = fs::remove_file(&partial);
            super::fail(&format!(
                "cactus-sys: could not install `{}`: {error}\n\
                 Help: remove the file and rebuild.",
                archive.display()
            ));
        }
    }

    /// `<archive>.<pid>.<nanos>.partial`: unique to this process and moment.
    fn partial_path(archive: &Path) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let name = archive
            .file_name()
            .map_or_else(|| "archive".into(), |name| name.to_string_lossy());
        archive.with_file_name(format!("{name}.{}.{nanos}.partial", std::process::id()))
    }

    /// Calls `attempt` with `0, 1, ...` until it succeeds, fails fatally, or `attempts` calls
    /// have been made, sleeping between transient failures for `Retry-After` or [`backoff`].
    fn retrying<T>(
        attempts: u32,
        mut sleep: impl FnMut(Duration),
        mut attempt: impl FnMut(u32) -> Result<T, Failure>,
    ) -> Result<T, Failure> {
        let mut number = 0;
        loop {
            match attempt(number) {
                Err(Failure::Transient { retry_after, .. }) if number + 1 < attempts => {
                    sleep(retry_after.unwrap_or_else(|| backoff(number, clock_nanos())));
                    number += 1;
                }
                done => return done,
            }
        }
    }

    /// 1 s, 2 s, 4 s, ... after failed attempt `attempt`, moved by up to 25 % by `nanos`.
    fn backoff(attempt: u32, nanos: u32) -> Duration {
        let base = 1000_u64 << attempt.min(6);
        let millis = base * 3 / 4 + u64::from(nanos) % (base / 2 + 1);
        Duration::from_millis(millis)
    }

    /// The sub-second part of the clock, the jitter source for [`backoff`].
    fn clock_nanos() -> u32 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.subsec_nanos())
    }

    /// Whether a failure with this HTTP status, or with none (transport), may go away.
    fn should_retry(status: Option<u16>) -> bool {
        match status {
            None => true,
            Some(code) => code == 408 || code == 429 || (500..600).contains(&code),
        }
    }

    /// A `Retry-After` header in seconds, capped at [`MAX_RETRY_AFTER`].
    fn retry_after(value: Option<&str>) -> Option<Duration> {
        let seconds = value?.trim().parse::<u64>().ok()?;
        Some(Duration::from_secs(seconds).min(MAX_RETRY_AFTER))
    }

    /// `CACTUS_DOWNLOAD_RETRIES` as a total attempt count.
    fn attempts(value: Option<&str>) -> u32 {
        let retries = value
            .and_then(|value| value.trim().parse::<u32>().ok())
            .unwrap_or(RETRIES);
        retries.saturating_add(1)
    }

    /// `CACTUS_DOWNLOAD_TIMEOUT` as the body timeout, [`BODY_TIMEOUT`] unless positive.
    fn body_timeout(value: Option<&str>) -> Duration {
        value
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|&seconds| seconds > 0)
            .map_or(BODY_TIMEOUT, Duration::from_secs)
    }

    /// Whether `name` is a temporary file for `file` left more than an hour before `now`.
    fn is_stale_partial(name: &str, file: &str, modified: SystemTime, now: SystemTime) -> bool {
        let ours = name
            .strip_prefix(file)
            .and_then(|rest| rest.strip_prefix('.'))
            .is_some_and(|rest| rest == "partial" || rest.ends_with(".partial"));
        ours && now
            .duration_since(modified)
            .is_ok_and(|age| age > STALE_AFTER)
    }

    /// Removes the temporary files of downloads of `file` into `dir` that died; errors ignored.
    fn remove_stale_partials(dir: &Path, file: &str, now: SystemTime) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) else {
                continue;
            };
            if is_stale_partial(name, file, modified, now) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// The HTTP client: identifies only as `cactus-sys/<version>` and times out.
    fn agent(body_timeout: Duration) -> ureq::Agent {
        let config = ureq::Agent::config_builder()
            .user_agent(concat!("cactus-sys/", env!("CARGO_PKG_VERSION")))
            .http_status_as_error(false)
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_recv_response(Some(RESPONSE_TIMEOUT))
            .timeout_recv_body(Some(body_timeout))
            .build();
        ureq::Agent::new_with_config(config)
    }

    /// One GET of `url` streamed into `dest`, sorting any failure into worth retrying or not.
    fn get(agent: &ureq::Agent, url: &str, dest: &Path) -> Result<(), Failure> {
        let mut response = agent
            .get(url)
            .call()
            .map_err(|error| transport(error, url))?;

        let status = response.status().as_u16();
        if status == 404 {
            return Err(Failure::Fatal(format!(
                "cactus-sys: the Needle engine archive is not where the pin says (HTTP 404).\n\
                 url: {url}\n\
                 The pinned revision or the file is missing from the repository, so the pin in\n\
                 prebuilt.toml is wrong.\n\
                 Help: restore cactus-sys/prebuilt.toml from the repository, or set\n\
                 CACTUS_NEEDLE_LIB_DIR to a directory holding libneedle.a for your target."
            )));
        }
        if !(200..300).contains(&status) {
            if !should_retry(Some(status)) {
                return Err(Failure::Fatal(format!(
                    "cactus-sys: could not download the Needle engine archive.\n\
                     url:   {url}\n\
                     error: HTTP {status}\n\
                     Help: check access to the hub (HF_ENDPOINT, default huggingface.co), or\n\
                     build offline by setting CACTUS_NEEDLE_LIB_DIR to a directory holding\n\
                     libneedle.a for your target."
                )));
            }
            let wait = response
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok());
            return Err(Failure::Transient {
                detail: format!("HTTP {status}"),
                retry_after: retry_after(wait),
            });
        }

        let mut file = match fs::File::create(dest) {
            Ok(file) => file,
            Err(error) => {
                return Err(Failure::Fatal(format!(
                    "cactus-sys: could not write `{}`: {error}\n\
                     Help: check the free space and permissions on the Cargo target directory.",
                    dest.display()
                )));
            }
        };

        let mut body = response.body_mut().with_config().limit(MAX_BODY).reader();
        match io::copy(&mut body, &mut file) {
            Ok(_) => Ok(()),
            // Errors from the body reader carry the client's error; those from the file do not.
            Err(error)
                if error
                    .get_ref()
                    .is_some_and(|inner| inner.is::<ureq::Error>()) =>
            {
                Err(transport(ureq::Error::from(error), url))
            }
            Err(error) => Err(Failure::Fatal(format!(
                "cactus-sys: could not write `{}`: {error}\n\
                 Help: check the free space and permissions on the Cargo target directory.",
                dest.display()
            ))),
        }
    }

    /// Sorts a client error: a request that cannot be made, or a body over the cap, is fatal;
    /// anything that went wrong on the wire is worth another attempt.
    fn transport(error: ureq::Error, url: &str) -> Failure {
        let fatal = matches!(
            error,
            ureq::Error::BodyExceedsLimit(_)
                | ureq::Error::BadUri(_)
                | ureq::Error::Http(_)
                | ureq::Error::InvalidProxyUrl
                | ureq::Error::RequireHttpsOnly(_)
        );
        if fatal {
            Failure::Fatal(format!(
                "cactus-sys: could not download the Needle engine archive.\n\
                 url:   {url}\n\
                 error: {error}\n\
                 Help: check network access to the hub (HF_ENDPOINT, default huggingface.co), or\n\
                 build offline by setting CACTUS_NEEDLE_LIB_DIR to a directory holding\n\
                 libneedle.a for your target."
            ))
        } else {
            Failure::Transient {
                detail: error.to_string(),
                retry_after: None,
            }
        }
    }
}
