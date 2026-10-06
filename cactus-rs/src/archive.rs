//! The checks every `.cact` archive gets before the engine sees it.
//!
//! Needle 3 and Whistle archives start with the same little-endian magic tag. [`check_magic`]
//! rejects files that are no archive at all, or an archive of a generation the linked engine
//! cannot read. [`kind_of`] then tells the two models apart from the tensor directory, so an
//! archive of the wrong kind is refused before `needle_load` could swap out a loaded model.

use crate::error::{Error, Result};
use crate::model::Model;

/// Bytes before the codebook: 48 `u32` fields and one `f32`.
const HEADER: usize = 196;

/// Bytes per tensor directory record, laid out as `<BBHIIIIQQII` little-endian: dtype, ndim,
/// padding, four dims, absolute data offset, data size, and two `u32`s this crate does not read.
const RECORD: usize = 44;

/// The dtype code of a 32-bit float tensor.
const FP32: u8 = 2;

/// Values in Whistle's audio manifest, a one-dimensional FP32 tensor only Whistle carries.
const MANIFEST_VALUES: u32 = 13;

/// The manifest value that identifies it: the 16 kHz sample rate, at index 9.
const MANIFEST_RATE: (usize, f32) = (9, 16_000.0);

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

/// Which model an archive holds, read from its tensor directory without the engine.
///
/// Whistle archives carry an audio manifest (an FP32 tensor of 13 values whose tenth is the
/// 16 000 Hz sample rate) and Needle 3 archives do not. Every read is bounds-checked.
///
/// Returns `None` when the magic tag is wrong or the directory, or the manifest's data, lies
/// outside `bytes`.
pub(crate) fn kind_of(bytes: &[u8]) -> Option<Model> {
    if read_u32(bytes, 0)? != MAGIC {
        return None;
    }
    let count = usize::try_from(read_u32(bytes, 4)?).ok()?;
    let codebook = usize::try_from(read_u32(bytes, 8)?).ok()?;

    let start = codebook.checked_mul(4)?.checked_add(HEADER)?;
    let end = count.checked_mul(RECORD)?.checked_add(start)?;
    let directory = bytes.get(start..end)?;

    for record in directory.chunks_exact(RECORD) {
        if is_manifest(bytes, record)? {
            return Some(Model::Whistle);
        }
    }
    Some(Model::Needle)
}

/// Whether one directory record is Whistle's audio manifest, or `None` when a record shaped
/// like one points outside the archive.
fn is_manifest(bytes: &[u8], record: &[u8]) -> Option<bool> {
    let shaped = record.first() == Some(&FP32)
        && record.get(1) == Some(&1)
        && read_u32(record, 4)? == MANIFEST_VALUES
        && read_u64(record, 28)? == u64::from(MANIFEST_VALUES) * 4;
    if !shaped {
        return Some(false);
    }

    let offset = usize::try_from(read_u64(record, 20)?).ok()?;
    let (index, rate) = MANIFEST_RATE;
    let at = index.checked_mul(4)?.checked_add(offset)?;
    // The whole tensor must lie inside the archive, not only the value read.
    bytes.get(offset..offset.checked_add(MANIFEST_VALUES as usize * 4)?)?;
    let value = f32::from_bits(read_u32(bytes, at)?);
    Some(value.to_bits() == rate.to_bits())
}

/// The little-endian `u32` at `at`, if four bytes are there.
fn read_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let word = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes(word.try_into().ok()?))
}

/// The little-endian `u64` at `at`, if eight bytes are there.
fn read_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let word = bytes.get(at..at.checked_add(8)?)?;
    Some(u64::from_le_bytes(word.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header with `count` records and no codebook, then `records`, then `data`.
    fn archive(count: u32, records: &[[u8; RECORD]], data: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0_u8; HEADER];
        bytes[..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&count.to_le_bytes());
        for record in records {
            bytes.extend_from_slice(record);
        }
        bytes.extend_from_slice(data);
        bytes
    }

    /// One directory record.
    fn record(dtype: u8, ndim: u8, dim0: u32, offset: u64, size: u64) -> [u8; RECORD] {
        let mut record = [0_u8; RECORD];
        record[0] = dtype;
        record[1] = ndim;
        record[4..8].copy_from_slice(&dim0.to_le_bytes());
        record[20..28].copy_from_slice(&offset.to_le_bytes());
        record[28..36].copy_from_slice(&size.to_le_bytes());
        record
    }

    /// Thirteen FP32 values with `rate` at index 9.
    fn manifest(rate: f32) -> Vec<u8> {
        (0..13)
            .map(|i| if i == 9 { rate } else { i as f32 })
            .flat_map(f32::to_le_bytes)
            .collect()
    }

    /// Where data starts after `count` records.
    fn data_at(count: usize) -> u64 {
        (HEADER + count * RECORD) as u64
    }

    #[test]
    fn an_audio_manifest_makes_a_whistle() {
        let records = [
            record(FP32, 2, 64, 0, 0),
            record(FP32, 1, 13, data_at(2), 52),
        ];
        let bytes = archive(2, &records, &manifest(16_000.0));
        assert_eq!(kind_of(&bytes), Some(Model::Whistle));
    }

    #[test]
    fn no_manifest_makes_a_needle() {
        let records = [record(FP32, 2, 64, 0, 0)];
        assert_eq!(kind_of(&archive(1, &records, &[])), Some(Model::Needle));
        assert_eq!(kind_of(&archive(0, &[], &[])), Some(Model::Needle));

        // The right shape with another rate is not the manifest.
        let records = [record(FP32, 1, 13, data_at(1), 52)];
        let bytes = archive(1, &records, &manifest(22_050.0));
        assert_eq!(kind_of(&bytes), Some(Model::Needle));
    }

    #[test]
    fn a_truncated_directory_is_no_archive() {
        let records = [record(FP32, 2, 64, 0, 0)];
        assert_eq!(kind_of(&archive(2, &records, &[])), None);
        assert_eq!(kind_of(&MAGIC.to_le_bytes()), None);
        assert_eq!(kind_of(&[]), None);
    }

    #[test]
    fn a_manifest_pointing_outside_the_archive_is_no_archive() {
        let records = [record(FP32, 1, 13, data_at(1), 52)];
        let mut bytes = archive(1, &records, &manifest(16_000.0));
        bytes.truncate(bytes.len() - 1);
        assert_eq!(kind_of(&bytes), None);

        let records = [record(FP32, 1, 13, u64::MAX - 4, 52)];
        assert_eq!(kind_of(&archive(1, &records, &[])), None);
    }

    #[test]
    fn a_huge_count_is_refused_without_a_panic() {
        assert_eq!(kind_of(&archive(u32::MAX, &[], &[])), None);

        let mut bytes = archive(0, &[], &[]);
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(kind_of(&bytes), None);
    }

    #[test]
    fn the_wrong_tag_is_no_archive() {
        let mut bytes = archive(0, &[], &[]);
        bytes[0] = 0x83;
        assert_eq!(kind_of(&bytes), None);
    }

    /// The pinned archives, when this machine has them.
    #[test]
    fn the_pinned_archives_are_told_apart() {
        for (variable, model) in [
            ("CACTUS_NEEDLE_WEIGHTS", Model::Needle),
            ("CACTUS_WHISTLE_WEIGHTS", Model::Whistle),
        ] {
            let Some(bytes) = std::env::var_os(variable).and_then(|path| std::fs::read(path).ok())
            else {
                println!("skipping the {model} archive kind test: set {variable} to run it");
                continue;
            };
            assert_eq!(kind_of(&bytes), Some(model), "{variable}");
        }
    }

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
