//! The one check every `.cact` archive gets before the engine sees it.
//!
//! Needle 3 and Whistle archives start with the same little-endian magic tag; the engine tells
//! them apart by what is inside, which only `needle_load` reads. So this check rejects files that
//! are no archive at all, or an archive of a generation the linked engine cannot read, and leaves
//! "which model is it" to the engine slot that loads it.

use crate::error::{Error, Result};

/// Little-endian magic tag of a Needle 3 or Whistle archive, the only one the linked engine reads.
///
/// A Needle 2 archive carries `0x05E12A83` instead and is rejected along with everything else.
pub(crate) const MAGIC: u32 = 0x05E1_2A84;

/// Checks that `bytes` start with [`MAGIC`].
///
/// # Errors
///
/// Returns [`Error::UnsupportedWeights`] with the tag that was found, or tag `0` when the bytes
/// are too short to hold one (no generation uses `0`).
pub(crate) fn check_magic(bytes: &[u8]) -> Result<()> {
    let Some(&[a, b, c, d]) = bytes.get(..4) else {
        return Err(Error::UnsupportedWeights { tag: 0 });
    };
    let tag = u32::from_le_bytes([a, b, c, d]);

    if tag == MAGIC {
        Ok(())
    } else {
        // A Needle 2 archive lands here too; the error message names both tags.
        Err(Error::UnsupportedWeights { tag })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_magic_tag_is_accepted() {
        assert!(check_magic(&[0x84, 0x2A, 0xE1, 0x05]).is_ok());
        assert!(check_magic(&[0x84, 0x2A, 0xE1, 0x05, 0xFF]).is_ok());
    }

    #[test]
    fn another_tag_is_reported() {
        let error = check_magic(&[0x83, 0x2A, 0xE1, 0x05]).expect_err("a Needle 2 tag");
        assert!(matches!(
            error,
            Error::UnsupportedWeights { tag: 0x05E1_2A83 }
        ));
    }

    #[test]
    fn a_short_header_reports_tag_zero() {
        let error = check_magic(&[0x84, 0x2A, 0xE1]).expect_err("three bytes");
        assert!(matches!(error, Error::UnsupportedWeights { tag: 0 }));
    }
}
