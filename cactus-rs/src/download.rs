//! Finding a pinned weight archive: environment variable, then cache, then the network.
//!
//! Each model's `Weights::fetch` describes one [`Pin`] and hands it to [`fetch`], which returns
//! the archive bytes. Checking the magic tag stays with the caller, which knows which `Weights`
//! type it is building.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};

use sha2::{Digest as _, Sha256};

use crate::error::{Error, Result};

/// Ceiling on the download body: the archives are 17 to 35 MB, so this leaves generous room
/// while keeping a misdirected response from filling the disk.
const MAX_BODY: u64 = 256 * 1024 * 1024;

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
    /// The download URL at the pinned revision.
    fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.file
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
/// 3. A download from the pinned revision, verified against the pin and installed under a
///    temporary name before being renamed into place, so an interrupted fetch never leaves a
///    half-written cache entry. A cache that cannot be written is a slow next run, not a failed
///    this one.
///
/// # Errors
///
/// Returns [`Error::Io`] when the named file cannot be read, [`Error::Download`] when the
/// transfer fails, and [`Error::ChecksumMismatch`] when the bytes that arrive are not the pinned
/// archive.
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

    let bytes = download(&pin.url())?;

    let actual = digest(&bytes);
    if actual != pin.sha256 {
        return Err(Error::ChecksumMismatch {
            expected: pin.sha256.to_owned(),
            actual,
        });
    }

    let _ = install(&cached, &bytes);
    Ok(bytes)
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

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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

/// Streams a GET into memory, identifying only as `cactus-rs/<version>`.
fn download(url: &str) -> Result<Vec<u8>> {
    let config = ureq::Agent::config_builder()
        .user_agent(concat!("cactus-rs/", env!("CARGO_PKG_VERSION")))
        .build();
    let agent = ureq::Agent::new_with_config(config);

    let mut response = agent.get(url).call().map_err(|error| Error::Download {
        url: url.to_owned(),
        detail: error.to_string(),
    })?;

    let mut bytes = Vec::new();
    response
        .body_mut()
        .with_config()
        .limit(MAX_BODY)
        .reader()
        .read_to_end(&mut bytes)
        .map_err(|error| Error::Download {
            url: url.to_owned(),
            detail: error.to_string(),
        })?;

    Ok(bytes)
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
            NEEDLE.url(),
            format!(
                "https://huggingface.co/Cactus-Compute/needle3/resolve/{}/needle3.cact",
                cactus_sys::NEEDLE_ENGINE_COMMIT
            )
        );
        assert_eq!(
            WHISTLE.url(),
            "https://huggingface.co/Cactus-Compute/whistle/resolve/\
             b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact"
        );
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
}
