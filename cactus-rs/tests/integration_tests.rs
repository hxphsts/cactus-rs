//! The safe API against the live engine.
//!
//! These tests are about the rules the C API imposes and this crate enforces: one engine per
//! process, weights that never unload, strings that must survive a trip through C, and a
//! conversation that only forgets when it is told to.
//!
//! The engine is one process-global, non-thread-safe model, and `cargo test` runs a file's tests
//! on parallel threads, so every test that touches it holds [`ENGINE`] for its whole life and lets
//! its [`Needle`] drop before the guard does. Files run in separate processes, so this lock only
//! has to cover this one.
//!
//! Tests that need the engine skip when `CACTUS_NEEDLE_WEIGHTS` names no readable archive; the
//! rest run everywhere. Nothing here downloads anything.

use std::env;
use std::sync::{Mutex, MutexGuard, PoisonError};

use cactus_rs::Error;
use cactus_rs::needle::{CompleteOptions, Needle, Tool, Weights};
use serde_json::json;

/// Serialises every engine-touching test in this binary.
static ENGINE: Mutex<()> = Mutex::new(());

/// Takes the engine lock, ignoring a poisoning left by an earlier failed test.
///
/// A panicking test says nothing about the engine's state that the next test's `reset` does not
/// fix, so a poisoned lock is recovered rather than propagated.
fn lock() -> MutexGuard<'static, ()> {
    ENGINE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The weight archive, or `None` when this machine has none to hand.
///
/// Only `CACTUS_NEEDLE_WEIGHTS` is consulted: `Weights::fetch()` would reach for the network, and
/// a 35 MB download is not something a test suite should start.
fn weights() -> Option<Weights> {
    let path = env::var_os("CACTUS_NEEDLE_WEIGHTS")?;
    Weights::from_file(path).ok()
}

/// The line a skipped test prints, so a green run with no engine is not mistaken for a real one.
fn skipped(what: &str) {
    println!("skipping {what}: set CACTUS_NEEDLE_WEIGHTS to a needle3.cact to run it");
}

/// The one tool these tests declare: enough for the engine to have something to reach for.
fn set_light() -> Tool {
    Tool::new(
        "set_light",
        "Turn a room light on or off",
        json!({
            "type": "object",
            "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
            "required": ["room", "on"]
        }),
    )
}

#[test]
fn a_second_needle_finds_the_engine_busy() {
    let _engine = lock();
    let (Some(first), Some(second)) = (weights(), weights()) else {
        skipped("the engine-busy test");
        return;
    };

    let alive = Needle::builder(first).build().expect("the engine is free");
    let error = Needle::builder(second)
        .build()
        .expect_err("the engine is taken");

    assert!(matches!(error, Error::EngineBusy), "got {error:?}");
    assert!(alive.prefix_tokens() > 0);
}

#[test]
fn dropping_a_needle_frees_the_engine() {
    let _engine = lock();
    let (Some(first), Some(second)) = (weights(), weights()) else {
        skipped("the drop-frees-the-slot test");
        return;
    };

    let needle = Needle::builder(first).build().expect("the engine is free");
    drop(needle);

    let again = Needle::builder(second)
        .build()
        .expect("the slot is free again");
    assert!(again.prefix_tokens() > 0);
}

#[test]
fn an_interior_nul_in_the_system_prompt_names_its_field() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the system-prompt NUL test");
        return;
    };

    let error = Needle::builder(weights)
        .system("You control\0the lights.")
        .build()
        .expect_err("a NUL is not a C string");

    assert!(
        matches!(
            error,
            Error::InteriorNul {
                field: "system prompt"
            }
        ),
        "got {error:?}"
    );
}

#[test]
fn an_interior_nul_in_the_input_names_its_field() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the input NUL test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .build()
        .expect("the engine is free");
    let error = needle
        .complete("turn the kitchen\0light on")
        .expect_err("a NUL is not a C string");

    assert!(
        matches!(error, Error::InteriorNul { field: "input" }),
        "got {error:?}"
    );
}

#[test]
fn an_embedding_is_as_wide_as_the_model_says() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the embedding width test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .build()
        .expect("the engine is free");
    let dimension = needle.embedding_dimension().expect("the model has a width");
    let vector = needle.embed("kitchen").expect("the model embeds");

    assert_eq!(vector.len(), dimension);
    assert!(dimension > 0);
    assert!(
        vector.iter().any(|value| *value != 0.0),
        "an all-zero embedding is not an embedding"
    );
}

#[test]
fn embedding_is_deterministic_and_input_dependent() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the embedding determinism test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .build()
        .expect("the engine is free");

    let kitchen = needle.embed("kitchen").expect("the model embeds");
    let again = needle.embed("kitchen").expect("the model embeds");
    let hallway = needle
        .embed("a treatise on marine insurance")
        .expect("embeds");

    assert_eq!(kitchen, again, "the same input embeds to the same vector");
    assert_ne!(kitchen, hallway, "different inputs embed differently");
}

#[test]
fn reset_makes_a_repeated_query_answer_the_same_way() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the reset test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .system("You control the lights.")
        .tool(set_light())
        .build()
        .expect("the engine is free");

    needle.reset();
    let first = needle
        .complete("turn the kitchen light on")
        .expect("the engine answers");

    needle.reset();
    let second = needle
        .complete("turn the kitchen light on")
        .expect("the engine answers");

    assert_eq!(
        first.calls(),
        second.calls(),
        "a rewound conversation answers the same query the same way"
    );
}

#[test]
fn a_zero_token_budget_leaves_no_room_for_a_call() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the zero-budget test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .system("You control the lights.")
        .tool(set_light())
        .build()
        .expect("the engine is free");

    let options = CompleteOptions::new().with_max_new_tokens(0);
    let completion = needle
        .complete_with_options("turn the kitchen light on", options)
        .expect("a budget of zero is not a failure");

    // The engine answers an empty turn with `type: "call"` and no calls at all, so the tag says
    // nothing useful here; the calls do.
    assert!(
        completion.calls().is_empty(),
        "got {:?}",
        completion.calls()
    );
}

#[test]
fn a_query_no_tool_covers_produces_no_calls() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the no-matching-tool test");
        return;
    };

    let mut needle = Needle::builder(weights)
        .system("You control the lights.")
        .tool(set_light())
        .build()
        .expect("the engine is free");

    let completion = needle
        .complete("recite the first stanza of the Aeneid")
        .expect("the engine answers");

    // Verified quirk: this can come back tagged `call` with an empty list, so the assertion is on
    // the calls rather than on `kind()`.
    assert!(
        completion.calls().is_empty(),
        "got {:?}",
        completion.calls()
    );
}

#[test]
fn a_different_archive_cannot_replace_the_loaded_one() {
    let _engine = lock();
    let Some(weights) = weights() else {
        skipped("the weights-already-loaded test");
        return;
    };

    // The real archive must be in the engine before a different one can be refused, so this test
    // loads it itself rather than depending on which test ran first.
    let loaded = Needle::builder(weights.clone())
        .build()
        .expect("the engine is free");
    drop(loaded);

    // A valid Needle 3 header over bytes that are not the loaded archive: it gets past the magic
    // check and is stopped by the fingerprint, so the engine never sees it.
    let other = Weights::from_bytes(vec![0x84, 0x2A, 0xE1, 0x05, 0xDE, 0xAD, 0xBE, 0xEF])
        .expect("the tag is a Needle 3 tag");
    let error = Needle::builder(other)
        .build()
        .expect_err("a second archive is refused");

    assert!(
        matches!(error, Error::WeightsAlreadyLoaded),
        "got {error:?}"
    );

    // And the refusal left the engine usable with the archive it already has.
    let again = Needle::builder(weights)
        .build()
        .expect("the loaded archive is still loadable");
    assert!(again.prefix_tokens() > 0);
}

#[test]
fn junk_is_not_a_weight_archive() {
    let error = Weights::from_bytes(b"nope\x01".to_vec()).expect_err("junk is not an archive");

    let Error::UnsupportedWeights { tag } = error else {
        panic!("expected Error::UnsupportedWeights, got {error:?}");
    };
    assert_eq!(tag, u32::from_le_bytes(*b"nope"));
}

#[test]
fn a_needle_two_archive_is_named_rather_than_loaded() {
    let error = Weights::from_bytes(vec![0x83, 0x2A, 0xE1, 0x05, 0x00])
        .expect_err("generation 2 is not generation 3");

    let Error::UnsupportedWeights { tag } = error else {
        panic!("expected Error::UnsupportedWeights, got {error:?}");
    };
    assert_eq!(tag, 0x05E1_2A83);
    assert!(error.to_string().contains("Needle 2"));
}
