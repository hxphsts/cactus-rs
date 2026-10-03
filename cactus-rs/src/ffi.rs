//! The crate's only door into the C API, and the only module with `unsafe` in it.
//!
//! The engine is one process-global, non-thread-safe runtime that holds one model per kind (text
//! and speech) and one error string for both. Three things make that safe to drive from Rust:
//!
//! - **One lock.** [`ENGINE`] guards every C call. [`lock`] returns a [`Guard`], and only a
//!   `Guard` has methods that reach the engine, so two calls never overlap, whichever model or
//!   thread they come from. [`Guard::last_error`] copies the engine's error string before the
//!   guard can be released, as the header requires.
//! - **One slot per kind.** [`Slot::take`] claims the text or the speech model for one value
//!   (a `Needle`, later a `Whistle`), and its `Drop` gives the claim back.
//! - **Types for the pointer rules.** [`Input`] makes "exactly one of `input` and `pcm` is
//!   non-null" unrepresentable to get wrong, slices carry their own lengths, and every length
//!   that crosses into an `int` is clamped to [`c_int::MAX`], which only ever under-reports an
//!   allocation.
//!
//! ## Lock ordering
//!
//! [`Slot::take`] and `Slot`'s `Drop` lock [`ENGINE`] themselves, and the mutex is not
//! re-entrant. A thread holding a [`Guard`] must therefore never take or drop a [`Slot`]. Where
//! both live in one scope, declare the slot **before** the guard: locals drop in reverse order,
//! so the guard is released before the slot's `Drop` locks again.
//!
//! A poisoned lock is recovered with [`PoisonError::into_inner`]: the state it protects is a few
//! flags and fingerprints that every writer leaves consistent, and the engine itself has no Rust
//! invariants a panic could break.

use std::ffi::{CStr, CString, c_char, c_int, c_ulonglong};
use std::ptr;
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};
use crate::model::Model;

/// The most samples one call may carry: 30 seconds at 16 kHz, as the header requires.
pub(crate) const MAX_SAMPLES: usize = 480_000;

/// Everything this crate knows about the engine, behind the one lock that serialises it.
static ENGINE: Mutex<State> = Mutex::new(State {
    taken: [false; 2],
    loaded: [None; 2],
});

/// What [`ENGINE`] protects.
struct State {
    /// Whether a value holds the slot of each kind, indexed by [`Kind::index`].
    taken: [bool; 2],
    /// Fingerprint of the archive loaded as each kind, set once and never cleared, because the
    /// engine cannot unload weights.
    loaded: [Option<u64>; 2],
}

/// The two kinds of model the engine holds at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The text model, Needle.
    Text,
    /// The speech model, Whistle.
    Speech,
}

impl Kind {
    /// This kind's bit in the [`Guard::models`] mask.
    pub(crate) const fn bit(self) -> c_int {
        match self {
            Kind::Text => cactus_sys::NEEDLE_TEXT,
            Kind::Speech => cactus_sys::NEEDLE_SPEECH,
        }
    }

    /// The other kind.
    const fn other(self) -> Kind {
        match self {
            Kind::Text => Kind::Speech,
            Kind::Speech => Kind::Text,
        }
    }

    /// Position in [`State`]'s arrays.
    const fn index(self) -> usize {
        match self {
            Kind::Text => 0,
            Kind::Speech => 1,
        }
    }

    /// The model this crate builds for the kind.
    const fn model(self) -> Model {
        match self {
            Kind::Text => Model::Needle,
            Kind::Speech => Model::Whistle,
        }
    }
}

/// What one completion or embedding is about: exactly one of text and a clip.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Input<'a> {
    /// A NUL-terminated string for the text model.
    Text(&'a CStr),
    /// 16 kHz mono PCM for the speech model.
    #[allow(dead_code, reason = "used by audio completion and Whistle")]
    Audio(&'a [f32]),
}

impl Input<'_> {
    /// The `(input, pcm, samples)` triple the C API takes, with exactly one pointer non-null.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`]: the header makes that a
    /// precondition, so it is checked here rather than trusted to the caller.
    fn raw(self) -> Result<(*const c_char, *const f32, c_int)> {
        match self {
            Input::Text(text) => Ok((text.as_ptr(), ptr::null(), 0)),
            // A slice pointer is never null, even for an empty clip, so `pcm` is the non-null one.
            Input::Audio(pcm) => Ok((ptr::null(), pcm.as_ptr(), samples(pcm)?)),
        }
    }
}

/// A clip's length as the C API's `int`, refusing clips over [`MAX_SAMPLES`].
fn samples(pcm: &[f32]) -> Result<c_int> {
    if pcm.len() > MAX_SAMPLES {
        return Err(Error::AudioTooLong { samples: pcm.len() });
    }
    // At most 480 000, so this always fits.
    Ok(clamp(pcm.len()))
}

/// A buffer length as the C API's `int`. Clamping only ever under-reports the allocation.
fn clamp(len: usize) -> c_int {
    c_int::try_from(len).unwrap_or(c_int::MAX)
}

/// The transcription settings, already in the shape the C API takes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct EncodedAudio {
    /// A language code, or `None` to detect it.
    language: Option<CString>,
    /// Newline-separated keywords, or `None` for none.
    keywords: Option<CString>,
    /// `1` to report word timestamps, `0` not to.
    word_timestamps: c_int,
}

#[allow(dead_code, reason = "built by Whistle's TranscribeOptions")]
impl EncodedAudio {
    /// Settings from their encoded parts.
    pub(crate) fn new(
        language: Option<CString>,
        keywords: Option<CString>,
        word_timestamps: bool,
    ) -> Self {
        EncodedAudio {
            language,
            keywords,
            word_timestamps: c_int::from(word_timestamps),
        }
    }

    /// The `(language, keywords, word_timestamps)` triple, each pointer null or a C string that
    /// lives as long as `self`.
    fn raw(&self) -> (*const c_char, *const c_char, c_int) {
        (
            self.language.as_deref().map_or(ptr::null(), CStr::as_ptr),
            self.keywords.as_deref().map_or(ptr::null(), CStr::as_ptr),
            self.word_timestamps,
        )
    }
}

/// Exclusive use of the engine for as long as it lives. Every C call goes through one.
pub(crate) struct Guard(MutexGuard<'static, State>);

/// Takes the engine lock, waiting for whoever has it.
///
/// Must not be called by a thread that already holds a [`Guard`]; see the module docs.
pub(crate) fn lock() -> Guard {
    Guard(ENGINE.lock().unwrap_or_else(PoisonError::into_inner))
}

impl Guard {
    /// The bitmask of loaded kinds; see [`Kind::bit`].
    pub(crate) fn models(&mut self) -> c_int {
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight; the function
        // takes no pointers.
        unsafe { cactus_sys::needle_models() }
    }

    /// The engine's reason for the last failure, copied out; empty when it gave none.
    pub(crate) fn last_error(&mut self) -> String {
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight, and nothing is
        // called between this and the copy below, so the runtime-owned string is still valid.
        let reason = unsafe { cactus_sys::needle_last_error() };
        if reason.is_null() {
            return String::new();
        }
        // SAFETY: non-null, NUL-terminated per the header, and valid until the next engine
        // call, which cannot happen while this guard is held; it is copied before returning.
        unsafe { CStr::from_ptr(reason) }
            .to_string_lossy()
            .into_owned()
    }

    /// Loads an archive. Only [`Slot::load`] calls this, after its fingerprint checks.
    fn load(&mut self, bytes: &[u8]) -> c_int {
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. `bytes` is valid
        // for reads of `bytes.len()` bytes for the whole call, and that is the length passed
        // (a `usize` always fits a `c_ulonglong`). The engine copies what it needs before
        // returning.
        unsafe { cactus_sys::needle_load(bytes.as_ptr(), bytes.len() as c_ulonglong) }
    }

    /// Builds the text model's conversation prefix and returns its token count.
    pub(crate) fn init(
        &mut self,
        system: Option<&CStr>,
        tools: &CStr,
        tool_index: Option<&CStr>,
    ) -> c_int {
        let system = system.map_or(ptr::null(), CStr::as_ptr);
        let tool_index = tool_index.map_or(ptr::null(), CStr::as_ptr);
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. Each pointer is
        // null or a NUL-terminated string borrowed for the whole call. A missing text model is a
        // failure the engine reports, not undefined behaviour.
        unsafe { cactus_sys::needle_init(system, tools.as_ptr(), tool_index) }
    }

    /// Runs one completion, writing the NUL-terminated envelope into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`]; the engine's own
    /// failures come back as a negative count, as in C.
    pub(crate) fn complete(
        &mut self,
        input: Input<'_>,
        max_new_tokens: c_int,
        out: &mut [u8],
    ) -> Result<c_int> {
        let (text, pcm, samples) = input.raw()?;
        let capacity = clamp(out.len());
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. `Input::raw`
        // gives exactly one non-null pointer: a NUL-terminated string, or `samples` (at most
        // MAX_SAMPLES) readable floats, both borrowed for the call. `out` is valid for writes of
        // `out.len()` bytes and `capacity` is no larger.
        let rc = unsafe {
            cactus_sys::needle_complete(
                text,
                pcm,
                samples,
                max_new_tokens,
                out.as_mut_ptr().cast::<c_char>(),
                capacity,
            )
        };
        Ok(rc)
    }

    /// Configures the transcription an audio completion runs for itself.
    #[allow(dead_code, reason = "used by audio completion")]
    pub(crate) fn set_audio(&mut self, audio: &EncodedAudio) {
        let (language, keywords, word_timestamps) = audio.raw();
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. Both pointers
        // are null or NUL-terminated strings owned by `audio`, borrowed for the call; the engine
        // copies them.
        unsafe { cactus_sys::needle_set_audio(language, keywords, word_timestamps) };
    }

    /// Transcribes one clip, writing the NUL-terminated envelope into `out`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`]; the engine's own
    /// failures come back as a negative count, as in C.
    #[allow(dead_code, reason = "used by Whistle")]
    pub(crate) fn transcribe(
        &mut self,
        pcm: &[f32],
        audio: &EncodedAudio,
        out: &mut [u8],
    ) -> Result<c_int> {
        let samples = samples(pcm)?;
        let (language, keywords, word_timestamps) = audio.raw();
        let capacity = clamp(out.len());
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. `pcm` is
        // readable for `samples` floats (at most MAX_SAMPLES), the two strings are null or
        // NUL-terminated and owned by `audio`, and `out` is valid for writes of `out.len()`
        // bytes with `capacity` no larger; all are borrowed for the call.
        let rc = unsafe {
            cactus_sys::needle_transcribe(
                pcm.as_ptr(),
                samples,
                language,
                keywords,
                word_timestamps,
                out.as_mut_ptr().cast::<c_char>(),
                capacity,
            )
        };
        Ok(rc)
    }

    /// How many floats embedding `input` takes, computing nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`]; the engine's own
    /// failures come back as a negative count, as in C.
    pub(crate) fn embed_len(&mut self, input: Input<'_>) -> Result<c_int> {
        let (text, pcm, samples) = input.raw()?;
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. `Input::raw`
        // gives exactly one non-null input pointer, valid for the call, and a null `out` with
        // capacity 0 is the documented size query.
        let rc = unsafe { cactus_sys::needle_embed(text, pcm, samples, ptr::null_mut(), 0) };
        Ok(rc)
    }

    /// Embeds `input` into `out`, returning the float count written.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AudioTooLong`] for a clip over [`MAX_SAMPLES`]; the engine's own
    /// failures come back as a negative count, as in C.
    pub(crate) fn embed(&mut self, input: Input<'_>, out: &mut [f32]) -> Result<c_int> {
        let (text, pcm, samples) = input.raw()?;
        let capacity = clamp(out.len());
        // SAFETY: the guard holds ENGINE, so no other engine call is in flight. `Input::raw`
        // gives exactly one non-null input pointer, valid for the call; `out` is valid for
        // writes of `out.len()` floats and `capacity` is no larger.
        let rc =
            unsafe { cactus_sys::needle_embed(text, pcm, samples, out.as_mut_ptr(), capacity) };
        Ok(rc)
    }

    /// Rewinds the text model's conversation to its prefix.
    pub(crate) fn reset(&mut self) {
        // SAFETY: the guard holds ENGINE, so no other engine call, of either kind, is in flight;
        // the function takes no pointers.
        unsafe { cactus_sys::needle_reset() };
    }
}

/// One value's claim on one kind of model, released on drop.
///
/// It exists so that an error between taking the slot and building the value that keeps it still
/// frees it: the slot is dropped on the way out whichever path is taken.
#[derive(Debug)]
pub(crate) struct Slot {
    kind: Kind,
}

impl Slot {
    /// Claims `kind`, or reports that another value has it.
    ///
    /// Locks [`ENGINE`], so the caller must not hold a [`Guard`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EngineBusy`] while another slot of the same kind is alive.
    pub(crate) fn take(kind: Kind) -> Result<Self> {
        let mut guard = lock();
        let taken = &mut guard.0.taken[kind.index()];
        if *taken {
            return Err(Error::EngineBusy);
        }
        *taken = true;
        Ok(Slot { kind })
    }

    /// Loads `bytes` as this slot's kind, unless that archive is already loaded.
    ///
    /// Needle 3 and Whistle archives share a magic tag, so the kind is learned from the engine:
    /// the load must add this slot's bit to [`Guard::models`]. If it adds the other bit
    /// instead, the archive stays loaded as the other kind (nothing unloads weights), its
    /// fingerprint is recorded there, and the call returns [`Error::WrongModel`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::WeightsAlreadyLoaded`] when a different archive of this kind is loaded,
    /// [`Error::WrongModel`] when the archive is, or turns out to be, of the other kind, and
    /// [`Error::Load`] when the engine rejects it.
    pub(crate) fn load(&self, guard: &mut Guard, bytes: &[u8]) -> Result<()> {
        let mine = self.kind;
        let other = mine.other();
        let wanted = fingerprint(bytes);
        let expected = Error::WrongModel {
            expected: mine.model(),
        };

        if let Some(loaded) = guard.0.loaded[mine.index()] {
            return if loaded == wanted {
                Ok(())
            } else {
                Err(Error::WeightsAlreadyLoaded)
            };
        }
        // Already loaded, as the other kind: the engine need not see it again.
        if guard.0.loaded[other.index()] == Some(wanted) {
            return Err(expected);
        }

        let before = guard.models();
        if guard.load(bytes) < 0 {
            return Err(Error::Load);
        }
        let after = guard.models();
        let gained = after & !before;

        // The usual case: a bit appeared, and it says which kind the archive is.
        let is_mine = if gained & mine.bit() != 0 {
            true
        } else if gained & other.bit() != 0 {
            false
        } else {
            // No new bit: the engine already held the kind this archive is. That happens when
            // something outside this crate loaded weights through cactus-sys, or when this is a
            // second, different archive of the other kind.
            // TODO(probe f): confirm whether such a load replaces the other kind's model or is
            // ignored; today it is reported as the wrong model either way.
            after & mine.bit() != 0 && after & other.bit() == 0
        };

        if is_mine {
            guard.0.loaded[mine.index()] = Some(wanted);
            Ok(())
        } else {
            // Recorded only if the other kind had nothing recorded: a fingerprint already there
            // belongs to the archive its slot holder loaded.
            guard.0.loaded[other.index()].get_or_insert(wanted);
            Err(expected)
        }
    }
}

/// Gives the claim back. Locks [`ENGINE`]: never drop a slot while holding a [`Guard`].
impl Drop for Slot {
    fn drop(&mut self) {
        lock().0.taken[self.kind.index()] = false;
    }
}

/// Word-wise FNV-1a over the archive, enough to tell two weight files apart.
///
/// A 64-bit non-cryptographic hash, so this is not a security check (the archive has already
/// been verified by whoever produced it),
/// only a way to answer "are these the bytes already loaded?" without keeping 35 MB alive or
/// depending on a hash crate the `download` feature happens to pull in.
fn fingerprint(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    // Eight bytes per round rather than one: every `build()` pays for this pass over a 35 MB
    // archive, and byte-at-a-time FNV made it the slowest step of building a `Needle`.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for word in &mut words {
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(word);
        hash = (hash ^ u64::from_le_bytes(buffer)).wrapping_mul(PRIME);
    }
    for byte in words.remainder() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_archives_fingerprint_differently() {
        assert_ne!(fingerprint(b"needle-a"), fingerprint(b"needle-b"));
        assert_ne!(fingerprint(b"needle"), fingerprint(b"needle\0"));
        assert_eq!(fingerprint(b"needle"), fingerprint(b"needle"));
    }

    // Only the speech slot: no other unit test in this process takes it, while a text slot could
    // one day be taken by a Needle test running in parallel.
    #[test]
    fn a_slot_is_exclusive_until_dropped() {
        let slot = Slot::take(Kind::Speech).expect("nothing else takes the speech slot");
        assert!(matches!(Slot::take(Kind::Speech), Err(Error::EngineBusy)));
        drop(slot);
        drop(Slot::take(Kind::Speech).expect("the slot was freed"));
    }

    #[test]
    fn kinds_map_to_distinct_bits_and_models() {
        assert_eq!(Kind::Text.bit() & Kind::Speech.bit(), 0);
        assert_eq!(Kind::Text.other(), Kind::Speech);
        assert_eq!(Kind::Speech.model(), Model::Whistle);
        assert_eq!(Kind::Text.model(), Model::Needle);
    }

    #[test]
    fn input_puts_exactly_one_pointer_in_play() {
        let (text, pcm, samples) = Input::Text(c"kitchen")
            .raw()
            .expect("text is never too long");
        assert!(!text.is_null() && pcm.is_null() && samples == 0);

        let (text, pcm, samples) = Input::Audio(&[]).raw().expect("an empty clip is short");
        assert!(text.is_null() && !pcm.is_null() && samples == 0);
    }

    #[test]
    fn a_clip_over_thirty_seconds_never_reaches_the_engine() {
        let clip = vec![0.0_f32; MAX_SAMPLES + 1];
        let error = Input::Audio(&clip).raw().expect_err("too long");
        assert!(matches!(error, Error::AudioTooLong { samples } if samples == MAX_SAMPLES + 1));
        assert!(Input::Audio(&clip[..MAX_SAMPLES]).raw().is_ok());
    }

    #[test]
    fn lengths_clamp_to_an_int() {
        assert_eq!(clamp(usize::MAX), c_int::MAX);
        assert_eq!(clamp(7), 7);
    }

    #[test]
    fn encoded_audio_passes_nulls_for_unset_fields() {
        let (language, keywords, words) = EncodedAudio::default().raw();
        assert!(language.is_null() && keywords.is_null() && words == 0);

        let audio = EncodedAudio::new(Some(c"de".to_owned()), None, true);
        let (language, keywords, words) = audio.raw();
        assert!(!language.is_null() && keywords.is_null() && words == 1);
    }
}
