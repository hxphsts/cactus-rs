//! Property-based tests for the parts of the crate that never touch the engine.
//!
//! Three guarantees are worth holding across generated input rather than a handful of examples:
//! a [`Tool`] survives the trip to the wire and back whatever it is named, a [`Completion`]
//! parses whatever JSON object the engine hands it without panicking, and a [`Call`]'s arguments
//! mean the same thing after [`Call::arguments_json`] as before it.
//!
//! Nothing here loads weights or builds a [`Needle`], so this file runs on every machine.

use std::fmt::Debug;

use cactus_rs::needle::{Call, Completion, Tool};
use proptest::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// Characters that have broken naive JSON handling before: quotes, escapes, the delimiter this
/// crate's sibling format uses, whitespace that a pretty-printer would introduce, and text from
/// three scripts.
const TRICKY: &[char] = &[
    'a', 'Z', '0', '_', '-', ' ', '"', '\'', '\\', '|', ':', ',', '{', '}', '\n', '\t', 'é', '中',
    '😀',
];

/// Serialises a value and reads it back, reporting what went wrong rather than only that it did.
fn roundtrip<T>(value: &T) -> bool
where
    T: Serialize + DeserializeOwned + PartialEq + Debug,
{
    match serde_json::to_string(value) {
        Ok(wire) => match serde_json::from_str::<T>(&wire) {
            Ok(back) => *value == back,
            Err(error) => {
                eprintln!("deserialize failed: {error}");
                eprintln!("wire was: {wire}");
                false
            }
        },
        Err(error) => {
            eprintln!("serialize failed: {error}");
            false
        }
    }
}

/// Text drawn from [`TRICKY`], up to `max` characters.
fn text(max: usize) -> impl Strategy<Value = String> {
    prop::collection::vec(prop::sample::select(TRICKY), 0..max)
        .prop_map(|chars| chars.into_iter().collect())
}

/// An arbitrary JSON value: nulls, booleans, integers, tricky strings, and small nestings of
/// both containers. Floats are left out because a non-finite one is not JSON at all.
fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|number| json!(number)),
        text(8).prop_map(Value::String),
    ];

    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::vec((text(6), inner), 0..4)
                .prop_map(|pairs| Value::Object(pairs.into_iter().collect())),
        ]
    })
}

/// An arbitrary JSON object, which is the shape the engine always answers with.
fn json_object() -> impl Strategy<Value = Value> {
    prop::collection::vec((text(8), json_value()), 0..6)
        .prop_map(|pairs| Value::Object(pairs.into_iter().collect()))
}

/// The field names [`Completion`] reads, which an "unknown field" must not collide with.
const KNOWN: &[&str] = &[
    "type",
    "success",
    "error",
    "error_code",
    "function_calls",
    "suppressed_calls",
    "reasoning",
    "reason",
    "confidence",
    "prefill_tps",
    "decode_tps",
    "peak_ram_mb",
    "validation",
];

proptest! {
    /// A declaration means the same thing after a trip through the wire format.
    #[test]
    fn a_tool_round_trips(
        name in text(24),
        description in text(48),
        triggers in prop::collection::vec(text(12), 0..4),
    ) {
        let mut tool = Tool::new(name, description, json!({ "type": "object" }));
        for trigger in triggers {
            tool = tool.with_trigger(trigger);
        }

        prop_assert!(roundtrip(&tool));
    }

    /// The wire form is the compact one: re-serialising the parsed text reproduces it byte for
    /// byte, which no indented or spaced encoding would.
    #[test]
    fn the_wire_form_is_compact(name in text(24), description in text(48)) {
        let tool = Tool::new(name, description, json!({ "type": "object" }));
        let wire = serde_json::to_string(&tool).expect("a Tool always serialises");

        let parsed: Value = serde_json::from_str(&wire).expect("its own output is JSON");
        prop_assert_eq!(serde_json::to_string(&parsed).expect("a Value serialises"), wire);
    }

    /// `triggers` appears exactly when there are some.
    #[test]
    fn triggers_are_serialised_only_when_present(
        name in text(16),
        triggers in prop::collection::vec(text(12), 0..4),
    ) {
        let wanted = triggers.len();
        let mut tool = Tool::new(name, "", json!({}));
        for trigger in triggers {
            tool = tool.with_trigger(trigger);
        }

        let wire: Value = serde_json::to_value(&tool).expect("a Tool always serialises");
        let object = wire.as_object().expect("a Tool is an object");

        prop_assert_eq!(object.contains_key("triggers"), wanted > 0);
        prop_assert!(object.contains_key("name"));
        prop_assert!(object.contains_key("description"));
        prop_assert!(object.contains_key("parameters"));
    }

    /// Whatever JSON object arrives, parsing it answers rather than panicking.
    #[test]
    fn parsing_an_arbitrary_envelope_never_panics(envelope in json_object()) {
        let text = serde_json::to_string(&envelope).expect("a Value serialises");
        let _ = text.parse::<Completion>();
    }

    /// Fields this crate does not know are ignored, and the ones it does survive them.
    #[test]
    fn unknown_fields_are_ignored(extra in json_object()) {
        let mut envelope = json!({
            "type": "call",
            "success": true,
            "function_calls": [{ "name": "set_light", "arguments": { "room": "kitchen" } }],
            "confidence": 0.5,
        });

        let object = envelope.as_object_mut().expect("the envelope is an object");
        for (key, value) in extra.as_object().expect("the extra fields are an object") {
            if !KNOWN.contains(&key.as_str()) {
                object.insert(key.clone(), value.clone());
            }
        }

        let text = serde_json::to_string(&envelope).expect("a Value serialises");
        let completion: Completion = text.parse().expect("unknown fields are not a failure");

        prop_assert!(completion.kind().is_call());
        prop_assert_eq!(completion.calls().len(), 1);
        prop_assert_eq!(completion.calls()[0].name(), "set_light");
        prop_assert_eq!(completion.confidence(), 0.5);
    }

    /// `arguments_json` is the arguments, not a rendering of them.
    #[test]
    fn arguments_json_round_trips(name in text(16), arguments in json_value()) {
        let call: Call = serde_json::from_value(json!({
            "name": name,
            "arguments": arguments,
        }))
        .expect("a call is name and arguments");

        let back: Value =
            serde_json::from_str(&call.arguments_json()).expect("arguments_json is JSON");
        prop_assert_eq!(&back, call.arguments());
        prop_assert_eq!(back, arguments);
    }

    /// A call itself round-trips, so an envelope can be logged and replayed.
    #[test]
    fn a_call_round_trips(name in text(16), arguments in json_value()) {
        let call: Call = serde_json::from_value(json!({
            "name": name,
            "arguments": arguments,
        }))
        .expect("a call is name and arguments");

        prop_assert!(roundtrip(&call));
    }
}
