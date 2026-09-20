//! The Needle 3 weight archive, and the three ways to get hold of one.
//!
//! ## Overview
//!
//! The engine ships as code only. Its weights are a separate 35 MB `needle3.cact` archive, and
//! [`Weights`] is that archive in memory with its magic tag checked. The check is cheap and it is
//! worth doing early: `needle_load` answers a bad archive with the same `-1` it gives a truncated
//! download, so without it a Needle 2 file and a half-finished transfer look alike.
//!
//! A `Weights` value is the bytes and nothing else. The engine copies what it needs during the
//! load, so the value is dropped as soon as [`Needle`](crate::needle::Needle) has handed it over.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::needle::Weights;
//!
//! // A real archive starts with the little-endian tag 0x05E12A84.
//! let mut archive = vec![0x84, 0x2A, 0xE1, 0x05];
//! archive.extend_from_slice(b"...the rest of needle3.cact...");
//!
//! let weights = Weights::from_bytes(archive)?;
//! assert_eq!(weights.generation(), 3);
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! With the `download` feature, [`Weights::fetch`] finds the archive without any of that:
//!
//! ```no_run
//! # #[cfg(feature = "download")]
//! # fn main() -> Result<(), cactus_rs::Error> {
//! use cactus_rs::needle::Weights;
//!
//! let weights = Weights::fetch()?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "download"))]
//! # fn main() {}
//! ```

use std::fmt;
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};

/// Little-endian magic tag of a Needle 3 archive, the only generation the linked engine reads.
///
/// A Needle 2 archive carries `0x05E12A83` instead and is rejected along with everything else.
const MAGIC_GENERATION_3: u32 = 0x05E1_2A84;

/// A validated Needle 3 weight archive.
///
/// Build one with [`Weights::from_bytes`], [`Weights::from_file`] or, behind the `download`
/// feature, [`Weights::fetch`]. All three check the magic tag before returning.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Weights;
///
/// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0x00])?;
/// assert_eq!(weights.len(), 5);
/// assert!(!weights.is_empty());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Weights {
    bytes: Vec<u8>,
    generation: u8,
}

impl Weights {
    /// Validates an archive already in memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Weights;
    ///
    /// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05])?;
    /// assert_eq!(weights.generation(), 3);
    ///
    /// // A Needle 2 archive is named in the error rather than handed to the engine.
    /// assert!(Weights::from_bytes(vec![0x83, 0x2A, 0xE1, 0x05]).is_err());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedWeights`] when the archive is shorter than its four-byte
    /// header or does not start with the Needle 3 magic tag.
    pub fn from_bytes<B>(bytes: B) -> Result<Self>
    where
        B: Into<Vec<u8>>,
    {
        let bytes = bytes.into();

        // Too short to hold a tag at all: report tag 0, which no generation uses.
        let Some(header) = bytes.get(..4) else {
            return Err(Error::UnsupportedWeights { tag: 0 });
        };
        let tag = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);

        if tag == MAGIC_GENERATION_3 {
            Ok(Weights {
                bytes,
                generation: 3,
            })
        } else {
            // A Needle 2 archive lands here too; the error message names both tags.
            Err(Error::UnsupportedWeights { tag })
        }
    }

    /// Reads and validates an archive from disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::needle::Weights;
    ///
    /// let weights = Weights::from_file("/opt/models/needle3.cact")?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be read, or [`Error::UnsupportedWeights`] when
    /// its contents are not a Needle 3 archive.
    pub fn from_file<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::from_bytes(fs::read(path.as_ref())?)
    }

    /// The size of the archive in bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Weights;
    ///
    /// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05])?;
    /// assert_eq!(weights.len(), 4);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the archive holds no bytes.
    ///
    /// It never does: validation rejects anything shorter than the four-byte header. The method
    /// exists because [`Weights::len`] does.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Weights;
    ///
    /// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05])?;
    /// assert!(!weights.is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The Needle generation of the archive, always `3` for a value that exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Weights;
    ///
    /// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05])?;
    /// assert_eq!(weights.generation(), 3);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn generation(&self) -> u8 {
        self.generation
    }

    /// The archive bytes, for the one `needle_load` call in this crate.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Prints the shape of the archive, never its contents.
impl fmt::Debug for Weights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weights")
            .field("generation", &self.generation)
            .field("len", &self.bytes.len())
            .finish()
    }
}

#[cfg(feature = "download")]
#[cfg_attr(docsrs, doc(cfg(feature = "download")))]
pub use fetch::WEIGHTS_SHA256;

#[cfg(feature = "download")]
mod fetch {
    use std::env;
    use std::fmt::Write as _;
    use std::fs;
    use std::io::Read as _;
    use std::path::PathBuf;

    use sha2::{Digest as _, Sha256};

    use super::Weights;
    use crate::error::{Error, Result};

    /// File name of the archive, upstream's and this crate's cache alike.
    const ARCHIVE: &str = "needle3.cact";

    /// Environment variable naming a local archive, checked before the cache and the network.
    const WEIGHTS_ENV: &str = "CACTUS_NEEDLE_WEIGHTS";

    /// SHA-256 of `needle3.cact` at the pinned engine commit, lowercase hex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(cactus_rs::needle::weights::WEIGHTS_SHA256.len(), 64);
    /// ```
    pub const WEIGHTS_SHA256: &str =
        "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38";

    /// Ceiling on the download body: the archive is 35 MB, so this leaves generous room while
    /// keeping a misdirected response from filling the disk.
    const MAX_BODY: u64 = 256 * 1024 * 1024;

    impl Weights {
        /// Finds the weight archive: environment variable, then cache, then the network.
        ///
        /// In order:
        ///
        /// 1. `CACTUS_NEEDLE_WEIGHTS`, if set, is the path to an archive and nothing else is
        ///    tried.
        /// 2. The cache file, under `$XDG_CACHE_HOME/cactus-rs/needle3/<commit>/needle3.cact`
        ///    (`$HOME/.cache/...` when `XDG_CACHE_HOME` is unset, `%LOCALAPPDATA%\...` on
        ///    Windows, the system temporary directory when none of those is set), where
        ///    `<commit>` is the first twelve characters of the pinned engine commit. It is used
        ///    only if it still matches [`WEIGHTS_SHA256`].
        /// 3. A download from the pinned Hugging Face commit, verified against
        ///    [`WEIGHTS_SHA256`] and installed under a temporary name before being renamed into
        ///    place, so an interrupted fetch never leaves a half-written cache entry.
        ///
        /// The download is 35 MB and happens once per machine. It identifies itself as
        /// `cactus-rs/<version>` and sends nothing else.
        ///
        /// # Examples
        ///
        /// ```no_run
        /// use cactus_rs::needle::Weights;
        ///
        /// let weights = Weights::fetch()?;
        /// assert_eq!(weights.generation(), 3);
        /// # Ok::<(), cactus_rs::Error>(())
        /// ```
        ///
        /// # Errors
        ///
        /// Returns [`Error::Io`] when a named or cached file cannot be read and no download can
        /// replace it, [`Error::Download`] when the transfer fails,
        /// [`Error::ChecksumMismatch`] when the bytes that arrive are not the pinned archive, and
        /// [`Error::UnsupportedWeights`] when the file found is not a Needle 3 archive.
        #[cfg_attr(docsrs, doc(cfg(feature = "download")))]
        pub fn fetch() -> Result<Self> {
            if let Some(path) = env::var_os(WEIGHTS_ENV) {
                return Weights::from_file(path);
            }

            // A cached archive is trusted only if it still hashes to the pin. Anything else,
            // such as a file cut short by a crash, is replaced by the download below.
            let cached = cache_path();
            if let Ok(bytes) = fs::read(&cached) {
                if digest(&bytes) == WEIGHTS_SHA256 {
                    return Weights::from_bytes(bytes);
                }
            }

            let url = format!(
                "https://huggingface.co/Cactus-Compute/needle3/resolve/{}/{ARCHIVE}",
                cactus_sys::NEEDLE_ENGINE_COMMIT
            );
            let bytes = download(&url)?;

            let actual = digest(&bytes);
            if actual != WEIGHTS_SHA256 {
                return Err(Error::ChecksumMismatch {
                    expected: WEIGHTS_SHA256.to_owned(),
                    actual,
                });
            }

            // A cache that cannot be written is a slow next run, not a failed this one.
            let _ = install(&cached, &bytes);

            Weights::from_bytes(bytes)
        }
    }

    /// Where the archive is cached.
    ///
    /// The system temporary directory stands in when the platform names no cache directory, so
    /// a process without `HOME` still downloads once rather than on every run.
    fn cache_path() -> PathBuf {
        let base = if cfg!(windows) {
            env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        }
        .unwrap_or_else(env::temp_dir);

        // The pin belongs in the path: a different engine commit is a different archive.
        let commit = cactus_sys::NEEDLE_ENGINE_COMMIT;
        // `cactus-rs/<engine>/...`: the next engine gets a sibling directory, not a new root.
        base.join("cactus-rs")
            .join("needle3")
            .join(commit.get(..12).unwrap_or(commit))
            .join(ARCHIVE)
    }

    /// Writes `bytes` to `path` through a temporary name, so readers never see a partial file.
    ///
    /// The temporary name is unique to this process and moment, so two programs fetching at
    /// once cannot write into each other's file; whichever renames last wins, with whole bytes.
    fn install(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let partial = path.with_extension(format!("cact.{}.{nanos}.partial", std::process::id()));

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

        #[test]
        fn digest_matches_a_known_vector() {
            assert_eq!(
                digest(b"abc"),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
        }

        #[test]
        fn cache_path_carries_the_pin() {
            let text = cache_path().display().to_string();
            assert!(text.contains("cactus-rs"));
            assert!(text.contains("needle3"));
            assert!(text.contains(&cactus_sys::NEEDLE_ENGINE_COMMIT[..12]));
            assert!(text.ends_with(ARCHIVE));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_three_is_accepted() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0x45]).expect("gen 3 tag");
        assert_eq!(weights.generation(), 3);
        assert_eq!(weights.len(), 5);
    }

    /// Little-endian magic tag of a Needle 2 archive.
    const MAGIC_GENERATION_2: u32 = 0x05E1_2A83;

    #[test]
    fn generation_two_is_rejected_by_tag() {
        let error = Weights::from_bytes(vec![0x83, 0x2A, 0xE1, 0x05]).expect_err("gen 2 tag");
        assert!(matches!(
            error,
            Error::UnsupportedWeights {
                tag: MAGIC_GENERATION_2
            }
        ));
    }

    #[test]
    fn a_short_file_is_not_an_archive() {
        let error = Weights::from_bytes(vec![0x84, 0x2A]).expect_err("truncated header");
        assert!(matches!(error, Error::UnsupportedWeights { tag: 0 }));
    }

    #[test]
    fn debug_does_not_dump_the_bytes() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0xAB]).expect("gen 3 tag");
        let text = format!("{weights:?}");
        assert_eq!(text, "Weights { generation: 3, len: 5 }");
    }
}
