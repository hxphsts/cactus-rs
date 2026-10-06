//! Finding a pinned weight archive: environment variable, then cache, then the network.
//!
//! Each model's `Weights::fetch` describes one [`Pin`] and hands it to [`fetch`], which returns
//! the archive bytes. Checking the magic tag stays with the caller, which knows which `Weights`
//! type it is building.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest as _, Sha256};

use crate::error::{Error, Result};

/// Ceiling on the download body: the archives are 17 to 35 MB, so this leaves generous room
/// while keeping a misdirected response from filling the disk.
const MAX_BODY: u64 = 256 * 1024 * 1024;

/// How long establishing the connection, TLS handshake included, may take.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the server may take to send the response headers once asked.
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the whole body may take, unless `CACTUS_DOWNLOAD_TIMEOUT` says otherwise: 35 MB in
/// fifteen minutes is about 40 KB/s, slow enough for a poor link, short enough to end a hang.
const BODY_TIMEOUT: Duration = Duration::from_secs(900);

/// Attempts after the first, unless `CACTUS_DOWNLOAD_RETRIES` says otherwise.
const RETRIES: u32 = 3;

/// The longest `Retry-After` honoured; a server asking for more gets this much.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

/// Age past which a `.partial` file is taken to belong to a process that died.
const STALE_AFTER: Duration = Duration::from_secs(60 * 60);

/// Where downloads come from unless `HF_ENDPOINT` names a mirror.
const HF_DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Where one archive lives upstream, what it must hash to, and where it is cached.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pin {
    /// Hugging Face repository, `owner/name`.
    pub(crate) repo: &'static str,
    /// Full commit hash the download is pinned to; its first twelve characters name the cache
    /// directory.
    pub(crate) revision: &'static str,
    /// File name of the archive, upstream's and the cache's alike.
    pub(crate) file: &'static str,
    /// SHA-256 of the archive at that revision, lowercase hex.
    pub(crate) sha256: &'static str,
    /// Environment variable naming a local archive, checked before the cache and the network.
    pub(crate) env: &'static str,
    /// Directory under `cactus-rs/` in the cache, one per model.
    pub(crate) cache_dir: &'static str,
}

impl Pin {
    /// The download URL at the pinned revision, from `HF_ENDPOINT` or Hugging Face itself.
    fn url(&self) -> String {
        self.url_with(&endpoint(env::var("HF_ENDPOINT").ok().as_deref()))
    }

    /// The download URL at the pinned revision on the hub at `endpoint`.
    fn url_with(&self, endpoint: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            endpoint.trim_end_matches('/'),
            self.repo,
            self.revision,
            self.file
        )
    }

    /// `<base>/cactus-rs/<cache_dir>/<revision[..12]>/<file>`.
    ///
    /// The revision belongs in the path: a different commit is a different archive.
    fn cache_path(&self) -> PathBuf {
        cache_base()
            .join("cactus-rs")
            .join(self.cache_dir)
            .join(self.revision.get(..12).unwrap_or(self.revision))
            .join(self.file)
    }
}

/// Finds the archive `pin` describes and returns its bytes.
///
/// 1. The pin's environment variable, if set, is the path to an archive and nothing else is
///    tried. Its bytes are not checked against the pin: pointing at a file is how a different
///    archive is used on purpose.
/// 2. The cache file, used only if it still hashes to the pin.
/// 3. A download from the pinned revision on `HF_ENDPOINT` (default `https://huggingface.co`),
///    verified against the pin and installed under a temporary name before being renamed into
///    place, so an interrupted fetch never leaves a half-written cache entry. A cache that
///    cannot be written is a slow next run, not a failed this one. Temporary files more than an
///    hour old, left by fetches that died, are removed first. With `HF_HUB_OFFLINE` set to
///    anything but `0`, nothing is downloaded and a cache miss is an error.
///
/// The download gives up on a connection after 30 s, on the response headers after 60 s and on
/// the body after 900 s (`CACTUS_DOWNLOAD_TIMEOUT`, in seconds). Timeouts, dropped connections,
/// HTTP 408, 429 and 5xx, and a first checksum mismatch are retried up to three more times
/// (`CACTUS_DOWNLOAD_RETRIES`) after 1 s, 2 s and 4 s, give or take a quarter, or after the
/// server's `Retry-After` up to 60 s. Any other HTTP error fails at once.
///
/// # Errors
///
/// Returns [`Error::Io`] when the named file cannot be read, [`Error::Download`] when the
/// transfer fails (after the last attempt, naming how many were made), and
/// [`Error::ChecksumMismatch`] when the bytes that arrive twice are not the pinned archive.
pub(crate) fn fetch(pin: &Pin) -> Result<Vec<u8>> {
    if let Some(path) = env::var_os(pin.env) {
        return Ok(fs::read(path)?);
    }

    // A cached archive is trusted only if it still hashes to the pin. Anything else, such as a
    // file cut short by a crash, is replaced by the download below.
    let cached = pin.cache_path();
    if let Ok(bytes) = fs::read(&cached) {
        if digest(&bytes) == pin.sha256 {
            return Ok(bytes);
        }
    }

    let url = pin.url();
    if offline(env::var("HF_HUB_OFFLINE").ok().as_deref()) {
        return Err(Error::Download {
            url,
            detail: format!(
                "offline mode is on (HF_HUB_OFFLINE is set), and {} is not in the cache at {}; \
                 set {} to a local copy of the archive, copy it to that cache path, or unset \
                 HF_HUB_OFFLINE",
                pin.file,
                cached.display(),
                pin.env
            ),
        });
    }

    if let Some(dir) = cached.parent() {
        remove_stale_partials(dir, pin.file, SystemTime::now());
    }

    let agent = agent(body_timeout(
        env::var("CACTUS_DOWNLOAD_TIMEOUT").ok().as_deref(),
    ));
    let attempts = attempts(env::var("CACTUS_DOWNLOAD_RETRIES").ok().as_deref());
    let mut mismatches = 0;

    let result = retrying(attempts, thread::sleep, |_| {
        let bytes = download(&agent, &url, pin)?;
        let actual = digest(&bytes);
        if actual == pin.sha256 {
            return Ok(bytes);
        }
        // One mismatch may be a corrupted transfer and earns a second download; two in a row
        // mean the pin and the server disagree, and downloading again will not change that.
        mismatches += 1;
        let error = Error::ChecksumMismatch {
            expected: pin.sha256.to_owned(),
            actual,
        };
        if mismatches > 1 {
            Err(Failure::Fatal(error))
        } else {
            Err(Failure::Transient {
                detail: error.to_string(),
                retry_after: None,
            })
        }
    });

    let bytes = match result {
        Ok(bytes) => bytes,
        Err(Failure::Fatal(error)) => return Err(error),
        Err(Failure::Transient { detail, .. }) => {
            return Err(Error::Download {
                url,
                detail: format!(
                    "gave up after {attempts} attempts; the last failed with: {detail}"
                ),
            });
        }
    };

    let _ = install(&cached, &bytes);
    Ok(bytes)
}

/// Whether `HF_HUB_OFFLINE` forbids the network: any value but empty or `0`, as in Hugging
/// Face's own tools.
fn offline(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0"
    })
}

/// The hub to download from: `HF_ENDPOINT` without a trailing `/`, or Hugging Face itself.
fn endpoint(value: Option<&str>) -> String {
    match value.map(|value| value.trim().trim_end_matches('/')) {
        Some(value) if !value.is_empty() => value.to_owned(),
        _ => HF_DEFAULT_ENDPOINT.to_owned(),
    }
}

/// Why one download attempt failed, and whether another could succeed.
#[derive(Debug)]
enum Failure {
    /// A timeout, a dropped connection, a busy or failing server, or a corrupted transfer.
    Transient {
        /// What went wrong, for the final error message.
        detail: String,
        /// How long the server asked us to wait before trying again, if it said.
        retry_after: Option<Duration>,
    },
    /// Trying again would fail the same way.
    Fatal(Error),
}

/// Calls `attempt` with `0, 1, ...` until it succeeds, fails fatally, or `attempts` calls have
/// been made, sleeping between transient failures for the server's `Retry-After` or else
/// [`backoff`].
///
/// Returns the first success or the last failure.
fn retrying<T>(
    attempts: u32,
    mut sleep: impl FnMut(Duration),
    mut attempt: impl FnMut(u32) -> std::result::Result<T, Failure>,
) -> std::result::Result<T, Failure> {
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

/// The pause after failed attempt `attempt` (counting from zero): 1 s, 2 s, 4 s, ..., each
/// moved by up to 25 % either way by `nanos`, so that many clients failing together do not
/// retry together.
fn backoff(attempt: u32, nanos: u32) -> Duration {
    let base = 1000_u64 << attempt.min(6);
    // `nanos % (base / 2 + 1)` spans 0..=base/2, so the result spans 0.75·base..=1.25·base.
    let millis = base * 3 / 4 + u64::from(nanos) % (base / 2 + 1);
    Duration::from_millis(millis)
}

/// The sub-second part of the clock, the jitter source for [`backoff`].
fn clock_nanos() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos())
}

/// Whether a failure with this HTTP status, or with none (`None`: the request never got an
/// answer), may go away on its own.
fn should_retry(status: Option<u16>) -> bool {
    match status {
        None => true,
        Some(code) => code == 408 || code == 429 || (500..600).contains(&code),
    }
}

/// A `Retry-After` header in seconds, capped at [`MAX_RETRY_AFTER`]. The HTTP-date form is
/// ignored: backoff stands in for it.
fn retry_after(value: Option<&str>) -> Option<Duration> {
    let seconds = value?.trim().parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds).min(MAX_RETRY_AFTER))
}

/// `CACTUS_DOWNLOAD_RETRIES` as a total attempt count: one plus the extra attempts it names, or
/// plus [`RETRIES`] when it is unset or not a number.
fn attempts(value: Option<&str>) -> u32 {
    let retries = value
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(RETRIES);
    retries.saturating_add(1)
}

/// `CACTUS_DOWNLOAD_TIMEOUT` as the body timeout: a positive number of seconds, or
/// [`BODY_TIMEOUT`] when it is unset, zero or not a number.
fn body_timeout(value: Option<&str>) -> Duration {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
        .map_or(BODY_TIMEOUT, Duration::from_secs)
}

/// Whether `name`, in the directory that holds `file`, is a temporary file [`install`] left
/// behind more than an hour before `now`.
///
/// A partial file is `<file>.<pid>.<nanos>.partial`; a file dated after `now` is not stale.
fn is_stale_partial(name: &str, file: &str, modified: SystemTime, now: SystemTime) -> bool {
    let ours = name
        .strip_prefix(file)
        .and_then(|rest| rest.strip_prefix('.'))
        .is_some_and(|rest| rest == "partial" || rest.ends_with(".partial"));
    ours && now
        .duration_since(modified)
        .is_ok_and(|age| age > STALE_AFTER)
}

/// Removes the temporary files of fetches of `file` into `dir` that died part way through.
///
/// Recent ones are left alone: they may belong to a fetch still running. Every error is
/// ignored, since a leftover file costs only disk space.
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

/// The platform cache directory.
///
/// The system temporary directory stands in when the platform names no cache directory, so a
/// process without `HOME` still downloads once rather than on every run.
fn cache_base() -> PathBuf {
    if cfg!(windows) {
        env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }
    .unwrap_or_else(env::temp_dir)
}

/// Writes `bytes` to `path` through a temporary name, so readers never see a partial file.
///
/// The temporary name is unique to this process and moment, so two programs fetching at once
/// cannot write into each other's file; whichever renames last wins, with whole bytes.
fn install(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let name = path
        .file_name()
        .map_or_else(|| "archive".into(), |name| name.to_string_lossy());
    let partial = path.with_file_name(format!("{name}.{}.{nanos}.partial", std::process::id()));

    let written = fs::write(&partial, bytes).and_then(|()| fs::rename(&partial, path));
    if written.is_err() {
        let _ = fs::remove_file(&partial);
    }
    Ok(written?)
}

/// The HTTP client: identifies only as `cactus-rs/<version>`, and gives up on a stalled
/// connection, response or body instead of waiting forever.
///
/// HTTP errors are left to [`download`], which needs their headers to decide what to do.
fn agent(body_timeout: Duration) -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .user_agent(concat!("cactus-rs/", env!("CARGO_PKG_VERSION")))
        .http_status_as_error(false)
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(body_timeout))
        .build();
    ureq::Agent::new_with_config(config)
}

/// One GET of `url` into memory, sorting any failure into worth retrying or not.
fn download(agent: &ureq::Agent, url: &str, pin: &Pin) -> std::result::Result<Vec<u8>, Failure> {
    let fatal = |detail: String| {
        Failure::Fatal(Error::Download {
            url: url.to_owned(),
            detail,
        })
    };

    let mut response = agent
        .get(url)
        .call()
        .map_err(|error| transport(error, url))?;

    let status = response.status().as_u16();
    if status == 404 {
        return Err(fatal(format!(
            "HTTP 404: the pinned revision {} or the file {} is missing from the repository {}, \
             so the pin is wrong",
            pin.revision, pin.file, pin.repo
        )));
    }
    if !(200..300).contains(&status) {
        let detail = format!("HTTP {status}");
        if !should_retry(Some(status)) {
            return Err(fatal(detail));
        }
        let wait = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok());
        return Err(Failure::Transient {
            detail,
            retry_after: retry_after(wait),
        });
    }

    response
        .body_mut()
        .with_config()
        .limit(MAX_BODY)
        .read_to_vec()
        .map_err(|error| transport(error, url))
}

/// Sorts an error from the HTTP client: a request that cannot be made, or a body over the cap,
/// is fatal; anything that went wrong on the wire is worth another attempt.
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
        Failure::Fatal(Error::Download {
            url: url.to_owned(),
            detail: error.to_string(),
        })
    } else {
        Failure::Transient {
            detail: error.to_string(),
            retry_after: None,
        }
    }
}

/// The SHA-256 of `bytes` in lowercase hex.
fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);

    let mut hex = String::with_capacity(64);
    for byte in hasher.finalize() {
        // Writing to a String is infallible; the Result exists only to satisfy the trait.
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin `needle::Weights::fetch` uses.
    const NEEDLE: Pin = crate::needle::weights::PIN;

    /// The pin `whistle::Weights::fetch` uses.
    const WHISTLE: Pin = crate::whistle::weights::PIN;

    #[test]
    fn digest_matches_a_known_vector() {
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn cache_path_carries_the_pin() {
        let needle = NEEDLE.cache_path();
        let text = needle.display().to_string();
        assert!(text.contains("cactus-rs"));
        assert!(text.contains("needle3"));
        assert!(text.contains(&cactus_sys::NEEDLE_ENGINE_COMMIT[..12]));
        assert!(text.ends_with("needle3.cact"));

        let whistle = WHISTLE.cache_path();
        let expected = Path::new("cactus-rs")
            .join("whistle")
            .join("b358ddadd89b")
            .join("whistle.cact");
        assert!(whistle.ends_with(&expected), "{}", whistle.display());

        // Both models share one cache root, one directory each.
        assert_eq!(
            needle.ancestors().nth(3),
            whistle.ancestors().nth(3),
            "different cache roots"
        );
    }

    #[test]
    fn urls_point_at_the_pinned_revision() {
        assert_eq!(
            NEEDLE.url_with(HF_DEFAULT_ENDPOINT),
            format!(
                "https://huggingface.co/Cactus-Compute/needle3/resolve/{}/needle3.cact",
                cactus_sys::NEEDLE_ENGINE_COMMIT
            )
        );
        assert_eq!(
            WHISTLE.url_with(HF_DEFAULT_ENDPOINT),
            "https://huggingface.co/Cactus-Compute/whistle/resolve/\
             b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact"
        );
    }

    #[test]
    fn mirrors_replace_the_host_only() {
        let expected = "https://hf-mirror.example/Cactus-Compute/whistle/resolve/\
                        b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact";
        assert_eq!(WHISTLE.url_with("https://hf-mirror.example"), expected);
        assert_eq!(WHISTLE.url_with("https://hf-mirror.example/"), expected);
        assert_eq!(
            WHISTLE.url_with("http://localhost:8080/hub/"),
            "http://localhost:8080/hub/Cactus-Compute/whistle/resolve/\
             b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact"
        );

        assert_eq!(endpoint(None), HF_DEFAULT_ENDPOINT);
        assert_eq!(endpoint(Some("")), HF_DEFAULT_ENDPOINT);
        assert_eq!(endpoint(Some("/")), HF_DEFAULT_ENDPOINT);
        assert_eq!(endpoint(Some("https://m.example//")), "https://m.example");
    }

    #[test]
    fn offline_is_any_value_but_empty_or_zero() {
        assert!(offline(Some("1")));
        assert!(offline(Some("true")));
        assert!(offline(Some("YES")));
        assert!(!offline(Some("0")));
        assert!(!offline(Some("")));
        assert!(!offline(Some(" ")));
        assert!(!offline(None));
    }

    #[test]
    fn a_partial_install_keeps_the_archive_name() {
        let dir = env::temp_dir().join(format!("cactus-rs-install-{}", std::process::id()));
        let path = dir.join("whistle.cact");
        install(&path, b"bytes").expect("the temporary directory is writable");
        assert_eq!(fs::read(&path).expect("installed"), b"bytes");
        // Nothing but the archive is left behind.
        let left: Vec<_> = fs::read_dir(&dir)
            .expect("the directory exists")
            .map(|entry| entry.expect("an entry").file_name())
            .collect();
        assert_eq!(left, ["whistle.cact"]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A temporary directory of its own for one test.
    fn scratch(name: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("cactus-rs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("the temporary directory is writable");
        dir
    }

    #[test]
    fn backoff_doubles_within_a_quarter() {
        for attempt in 0..3 {
            let base = 1000_u64 << attempt;
            for nanos in [0, 1, 499, 500, 777_777, 999_999_999, u32::MAX] {
                let millis = u64::try_from(backoff(attempt, nanos).as_millis()).unwrap();
                assert!(millis * 4 >= base * 3, "{attempt} {nanos}: {millis}");
                assert!(millis * 4 <= base * 5, "{attempt} {nanos}: {millis}");
            }
        }
        // Even the unluckiest jitter keeps each pause longer than the one before it.
        assert!(backoff(1, 0) > backoff(0, 500));
        assert!(backoff(2, 0) > backoff(1, 1000));
        assert_eq!(backoff(0, 0), Duration::from_millis(750));
        assert_eq!(backoff(2, 2000), Duration::from_millis(5000));
        // Absurd attempt numbers do not overflow.
        assert!(backoff(u32::MAX, u32::MAX) <= Duration::from_secs(80));
    }

    #[test]
    fn only_transient_statuses_are_retried() {
        for status in [408, 429, 500, 502, 503, 504] {
            assert!(should_retry(Some(status)), "{status}");
        }
        for status in [400, 401, 403, 404, 410, 451] {
            assert!(!should_retry(Some(status)), "{status}");
        }
        assert!(should_retry(None), "transport errors are retried");
    }

    #[test]
    fn retry_after_is_seconds_capped_at_a_minute() {
        assert_eq!(retry_after(Some("7")), Some(Duration::from_secs(7)));
        assert_eq!(retry_after(Some(" 0 ")), Some(Duration::ZERO));
        assert_eq!(retry_after(Some("3600")), Some(MAX_RETRY_AFTER));
        assert_eq!(retry_after(Some("Wed, 21 Oct 2015 07:28:00 GMT")), None);
        assert_eq!(retry_after(None), None);
    }

    #[test]
    fn settings_fall_back_to_their_defaults() {
        assert_eq!(attempts(None), 4);
        assert_eq!(attempts(Some("0")), 1);
        assert_eq!(attempts(Some("5")), 6);
        assert_eq!(attempts(Some("lots")), 4);
        assert_eq!(attempts(Some("4294967295")), u32::MAX);
        assert_eq!(body_timeout(None), BODY_TIMEOUT);
        assert_eq!(body_timeout(Some("120")), Duration::from_secs(120));
        assert_eq!(body_timeout(Some("0")), BODY_TIMEOUT);
        assert_eq!(body_timeout(Some("-1")), BODY_TIMEOUT);
    }

    #[test]
    fn retrying_stops_at_success_fatal_or_the_last_attempt() {
        let transient = || Failure::Transient {
            detail: "reset".to_owned(),
            retry_after: None,
        };

        // Succeeds on the third attempt after two backoffs.
        let mut slept = Vec::new();
        let result = retrying(
            4,
            |pause| slept.push(pause),
            |attempt| {
                if attempt < 2 {
                    Err(transient())
                } else {
                    Ok(attempt)
                }
            },
        );
        assert_eq!(result.ok(), Some(2));
        assert_eq!(slept.len(), 2);
        assert!(slept[0] >= Duration::from_millis(750) && slept[0] <= Duration::from_millis(1250));

        // Gives up after the last attempt without sleeping after it.
        let mut calls = 0;
        let mut sleeps = 0;
        let result: std::result::Result<(), _> = retrying(
            4,
            |_| sleeps += 1,
            |_| {
                calls += 1;
                Err(transient())
            },
        );
        assert!(matches!(result, Err(Failure::Transient { .. })));
        assert_eq!((calls, sleeps), (4, 3));

        // A fatal failure ends it at once.
        let mut calls = 0;
        let result: std::result::Result<(), _> = retrying(
            4,
            |_| {},
            |_| {
                calls += 1;
                Err(Failure::Fatal(Error::Download {
                    url: "u".to_owned(),
                    detail: "HTTP 403".to_owned(),
                }))
            },
        );
        assert!(matches!(result, Err(Failure::Fatal(_))));
        assert_eq!(calls, 1);

        // The server's Retry-After replaces the backoff.
        let mut slept = Vec::new();
        let _ = retrying(
            2,
            |pause| slept.push(pause),
            |attempt| {
                if attempt == 0 {
                    Err(Failure::Transient {
                        detail: "HTTP 429".to_owned(),
                        retry_after: Some(Duration::from_secs(9)),
                    })
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(slept, [Duration::from_secs(9)]);
    }

    #[test]
    fn stale_partials_are_ours_and_old() {
        let now = SystemTime::now();
        let old = now - Duration::from_secs(2 * 60 * 60);
        let fresh = now - Duration::from_secs(60);

        assert!(is_stale_partial(
            "needle3.cact.123.456.partial",
            "needle3.cact",
            old,
            now
        ));
        assert!(!is_stale_partial(
            "needle3.cact.123.456.partial",
            "needle3.cact",
            fresh,
            now
        ));
        assert!(!is_stale_partial(
            "other.cact.partial",
            "needle3.cact",
            old,
            now
        ));
        assert!(!is_stale_partial("needle3.cact", "needle3.cact", old, now));
        assert!(!is_stale_partial(
            "needle3.cactus.1.2.partial",
            "needle3.cact",
            old,
            now
        ));
        assert!(!is_stale_partial(
            "needle3.cact.1.2.partial.keep",
            "needle3.cact",
            old,
            now
        ));
        // A clock that went backwards never makes a file stale.
        let future = now + Duration::from_secs(60 * 60 * 3);
        assert!(!is_stale_partial(
            "needle3.cact.1.2.partial",
            "needle3.cact",
            future,
            now
        ));
    }

    #[test]
    fn stale_partials_are_removed_and_nothing_else() {
        let dir = scratch("stale");
        let old = SystemTime::now() - Duration::from_secs(2 * 60 * 60);
        let names = [
            "whistle.cact",
            "whistle.cact.1.2.partial",
            "whistle.cact.3.4.partial",
            "needle3.cact.5.6.partial",
        ];
        for name in names {
            fs::write(dir.join(name), b"x").expect("writable");
        }
        // Old: the archive itself, one of ours and someone else's. Fresh: the other of ours.
        for name in [
            "whistle.cact",
            "whistle.cact.1.2.partial",
            "needle3.cact.5.6.partial",
        ] {
            fs::File::options()
                .write(true)
                .open(dir.join(name))
                .and_then(|file| file.set_modified(old))
                .expect("settable mtime");
        }

        remove_stale_partials(&dir, "whistle.cact", SystemTime::now());

        let mut left: Vec<_> = fs::read_dir(&dir)
            .expect("the directory exists")
            .map(|entry| entry.expect("an entry").file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(
            left,
            [
                "needle3.cact.5.6.partial",
                "whistle.cact",
                "whistle.cact.3.4.partial"
            ]
        );
        // A missing directory is not an error.
        remove_stale_partials(&dir.join("missing"), "whistle.cact", SystemTime::now());
        let _ = fs::remove_dir_all(&dir);
    }
}
