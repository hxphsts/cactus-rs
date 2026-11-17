//! The Mirage weight archive, and the three ways to get hold of one.
//!
//! ## Overview
//!
//! Mirage's weights are a 21.6 MB `mirage.cact` archive, and [`Weights`] is that archive in
//! memory with its magic tag checked. The check rejects files that are no archive at all; it
//! does not reject a Needle or Whistle archive, which carry the same tag. The builder reads the tensor
//! directory before loading, so [`MirageBuilder::build`](super::MirageBuilder::build) reports
//! an archive of another model as [`Error::WrongModel`](crate::Error::WrongModel) without it reaching the
//! engine.
//!
//! A `Weights` value is the bytes and nothing else. The engine copies what it needs during the
//! load, so the value is dropped as soon as [`Mirage`](super::Mirage) has handed it over.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::mirage::Weights;
//!
//! // A real archive starts with the little-endian tag 0x05E12A84.
//! let mut archive = vec![0x84, 0x2A, 0xE1, 0x05];
//! archive.extend_from_slice(b"...the rest of mirage.cact...");
//!
//! let weights = Weights::from_bytes(archive)?;
//! assert!(!weights.is_empty());
//! # Ok::<(), cactus_rs::Error>(())
//! ```
//!
//! With the `download` feature, [`Weights::fetch`] finds the archive without any of that:
//!
//! ```no_run
//! # #[cfg(feature = "download")]
//! # fn main() -> Result<(), cactus_rs::Error> {
//! use cactus_rs::mirage::Weights;
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

/// A validated Mirage weight archive.
///
/// Build one with [`Weights::from_bytes`], [`Weights::from_file`] or, behind the `download`
/// feature, [`Weights::fetch`]. All three check the magic tag before returning.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Weights;
///
/// let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0x00])?;
/// assert_eq!(weights.len(), 5);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Weights {
    bytes: Vec<u8>,
}

impl Weights {
    /// Validates an archive already in memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Weights;
    ///
    /// assert!(Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05]).is_ok());
    /// assert!(Weights::from_bytes(b"\x89PNG....".to_vec()).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when the archive
    /// is shorter than its four-byte header or does not start with the magic tag.
    pub fn from_bytes<B>(bytes: B) -> Result<Self>
    where
        B: Into<Vec<u8>>,
    {
        let bytes = bytes.into();
        archive::check_magic(&bytes)?;
        Ok(Weights { bytes })
    }

    /// Reads and validates an archive from disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::Weights;
    ///
    /// let weights = Weights::from_file("/opt/models/mirage.cact")?;
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`](crate::Error::Io) when the file cannot be read, or
    /// [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when its contents are not
    /// an archive.
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
    /// use cactus_rs::mirage::Weights;
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
    /// use cactus_rs::mirage::Weights;
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

    /// The archive bytes, for the one load in
    /// [`MirageBuilder::build`](super::MirageBuilder::build).
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Prints the size of the archive, never its contents.
impl fmt::Debug for Weights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weights")
            .field("len", &self.bytes.len())
            .finish()
    }
}

/// Commit of the `Cactus-Compute/mirage` Hugging Face repository the download is pinned to.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::mirage::weights::WEIGHTS_REVISION.len(), 40);
/// ```
#[cfg(feature = "download")]
#[cfg_attr(docsrs, doc(cfg(feature = "download")))]
pub const WEIGHTS_REVISION: &str = "3d03cee88c0acbe4e71b94b567ab14e6e55f40c9";

/// SHA-256 of `mirage.cact` at [`WEIGHTS_REVISION`], lowercase hex.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::mirage::weights::WEIGHTS_SHA256.len(), 64);
/// ```
#[cfg(feature = "download")]
#[cfg_attr(docsrs, doc(cfg(feature = "download")))]
pub const WEIGHTS_SHA256: &str = "4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237";

/// Where `mirage.cact` is downloaded from and cached.
#[cfg(feature = "download")]
pub(crate) const PIN: crate::download::Pin = crate::download::Pin {
    repo: "Cactus-Compute/mirage",
    revision: WEIGHTS_REVISION,
    file: "mirage.cact",
    sha256: WEIGHTS_SHA256,
    env: "CACTUS_MIRAGE_WEIGHTS",
    cache_dir: "mirage",
};

#[cfg(feature = "download")]
impl Weights {
    /// Finds the weight archive: environment variable, then cache, then the network.
    ///
    /// In order:
    ///
    /// 1. `CACTUS_MIRAGE_WEIGHTS`, if set, is the path to an archive and nothing else is
    ///    tried.
    /// 2. The cache file, under `$XDG_CACHE_HOME/cactus-rs/mirage/<revision>/mirage.cact`
    ///    (`$HOME/.cache/...` when `XDG_CACHE_HOME` is unset, `%LOCALAPPDATA%\...` on
    ///    Windows, the system temporary directory when none of those is set), where
    ///    `<revision>` is the first twelve characters of [`WEIGHTS_REVISION`]. It is used only
    ///    if it still matches [`WEIGHTS_SHA256`].
    /// 3. A download from the pinned Hugging Face commit, verified against
    ///    [`WEIGHTS_SHA256`] and installed under a temporary name before being renamed into
    ///    place, so an interrupted fetch never leaves a half-written cache entry.
    ///
    /// The download is 21.6 MB and happens once per machine. It identifies itself as
    /// `cactus-rs/<version>` and sends nothing else.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cactus_rs::mirage::Weights;
    ///
    /// let weights = Weights::fetch()?;
    /// assert!(!weights.is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`](crate::Error::Io) when a named or cached file cannot be read and
    /// no download can replace it, [`Error::Download`](crate::Error::Download) when the transfer
    /// fails, [`Error::ChecksumMismatch`](crate::Error::ChecksumMismatch) when the bytes that
    /// arrive are not the pinned archive, and
    /// [`Error::UnsupportedWeights`](crate::Error::UnsupportedWeights) when the file found is
    /// not an archive.
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
    fn the_magic_tag_is_accepted() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0x45]).expect("the tag");
        assert_eq!(weights.len(), 5);
        assert!(!weights.is_empty());
    }

    #[test]
    fn an_image_is_not_an_archive() {
        let error = Weights::from_bytes(b"\x89PNG\r\n\x1a\n".to_vec()).expect_err("a PNG header");
        assert!(matches!(
            error,
            Error::UnsupportedWeights { tag: 0x474E_5089 }
        ));
    }

    #[test]
    fn a_short_file_is_not_an_archive() {
        let error = Weights::from_bytes(vec![0x84, 0x2A]).expect_err("truncated header");
        assert!(matches!(error, Error::UnsupportedWeights { tag: 0 }));
    }

    #[test]
    fn debug_does_not_dump_the_bytes() {
        let weights = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0xAB]).expect("the tag");
        assert_eq!(format!("{weights:?}"), "Weights { len: 5 }");
    }

    #[cfg(feature = "download")]
    #[test]
    fn the_pin_names_the_mirage_archive() {
        assert_eq!(PIN.file, crate::Model::Mirage.archive());
        assert_eq!(PIN.revision.len(), 40);
        assert_eq!(PIN.sha256.len(), 64);
        for other in [crate::needle::weights::PIN, crate::whistle::weights::PIN] {
            assert_ne!(PIN.env, other.env);
            assert_ne!(PIN.cache_dir, other.cache_dir);
        }
    }
}
