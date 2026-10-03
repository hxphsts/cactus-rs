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

use crate::archive;
use crate::error::Result;

/// A validated Needle 3 weight archive.
///
/// Build one with [`Weights::from_bytes`], [`Weights::from_file`] or, behind the `download`
/// feature, [`Weights::fetch`]. All three check the magic tag before returning.
///
/// A Whistle archive carries the same magic tag, so it passes this check: the tag says "an
/// archive this engine reads", not which model is inside. Only the engine can tell, and
/// [`NeedleBuilder::build`](crate::needle::NeedleBuilder::build) reports a Whistle archive as
/// [`Error::WrongModel`](crate::Error::WrongModel).
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
    /// Returns [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when the archive is shorter than its four-byte
    /// header or does not start with the Needle 3 magic tag.
    pub fn from_bytes<B>(bytes: B) -> Result<Self>
    where
        B: Into<Vec<u8>>,
    {
        let bytes = bytes.into();
        archive::check_magic(&bytes)?;
        Ok(Weights {
            bytes,
            generation: 3,
        })
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
    /// Returns [`Error::Io`](crate::Error::Io) when the file cannot be read, or [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when
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

/// SHA-256 of `needle3.cact` at the pinned engine commit, lowercase hex.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::needle::weights::WEIGHTS_SHA256.len(), 64);
/// ```
#[cfg(feature = "download")]
#[cfg_attr(docsrs, doc(cfg(feature = "download")))]
pub const WEIGHTS_SHA256: &str = "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38";

/// Where `needle3.cact` is downloaded from and cached: the engine's own Hugging Face commit.
#[cfg(feature = "download")]
pub(crate) const PIN: crate::download::Pin = crate::download::Pin {
    repo: "Cactus-Compute/needle3",
    revision: cactus_sys::NEEDLE_ENGINE_COMMIT,
    file: "needle3.cact",
    sha256: WEIGHTS_SHA256,
    env: "CACTUS_NEEDLE_WEIGHTS",
    cache_dir: "needle3",
};

#[cfg(feature = "download")]
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
    /// Returns [`Error::Io`](crate::Error::Io) when a named or cached file cannot be read and no download can
    /// replace it, [`Error::Download`](crate::Error::Download) when the transfer fails,
    /// [`Error::ChecksumMismatch`](crate::Error::ChecksumMismatch) when the bytes that arrive are not the pinned archive, and
    /// [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when the file found is not a Needle 3 archive.
    #[cfg_attr(docsrs, doc(cfg(feature = "download")))]
    pub fn fetch() -> Result<Self> {
        Weights::from_bytes(crate::download::fetch(&PIN)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

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
