//! Tool declarations, in the flat shape the engine expects.
//!
//! ## Overview
//!
//! A [`Tool`] is a name, a sentence of description and a JSON Schema for its arguments. The
//! engine reads the declarations once, when the conversation prefix is built, and every
//! completion afterwards may answer with calls to them.
//!
//! The wire format is flat, `{"name": ..., "description": ..., "parameters": ...}`, not the
//! OpenAI-style `{"type": "function", "function": {...}}` nesting. A declaration in the nested
//! shape is not rejected by the engine; it is simply ignored, and the model answers as if it had
//! no tools at all. Building declarations through this type is what keeps that from happening.
//!
//! Triggers are optional case-insensitive regular expressions. A request that matches one makes
//! the engine call that tool, bypassing its confidence floor. They are serialised only when
//! there are some.
//!
//! ## Writing schemas the model follows
//!
//! Needle copies surface words from the request into the call, so the schema decides accuracy
//! more than the prompt does. Four rules, each measured in `examples/desktop.rs`:
//!
//! - Spell closed sets the way people speak: `"left half"` rather than `"left_half"`, and one
//!   enum `["cmd s", "enter"]` rather than a key plus an array of modifiers.
//! - Give each intent one tool and say so in its description. Avoid optional arguments the
//!   model may feel obliged to fill.
//! - Pin stubborn phrasings with [`Tool::with_trigger`].
//! - Keep conversations short: call [`Needle::reset`](crate::needle::Needle::reset) between
//!   unrelated requests.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::needle::Tool;
//! use serde_json::json;
//!
//! let tool = Tool::new(
//!     "set_light",
//!     "Turn a room light on or off",
//!     json!({
//!         "type": "object",
//!         "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
//!         "required": ["room", "on"]
//!     }),
//! )
//! .with_trigger("light");
//!
//! assert_eq!(tool.name(), "set_light");
//! assert_eq!(tool.triggers(), ["light"]);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One tool the engine may call.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Tool;
/// use serde_json::json;
///
/// let tool = Tool::new("ping", "Check that a host answers", json!({ "type": "object" }));
///
/// // The wire form is the flat object the engine reads, with no `triggers` key when empty.
/// let wire = serde_json::to_string(&tool)?;
/// assert_eq!(
///     wire,
///     r#"{"name":"ping","description":"Check that a host answers","parameters":{"type":"object"}}"#
/// );
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Tool {
    name: String,
    description: String,
    parameters: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    triggers: Vec<String>,
}

impl Tool {
    /// Declares a tool from its name, description and JSON Schema.
    ///
    /// The description is what the model reasons over, so it is worth a full sentence: what the
    /// tool does, and when to reach for it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new(
    ///     "send_email",
    ///     "Send an email to one recipient",
    ///     json!({
    ///         "type": "object",
    ///         "properties": { "to": { "type": "string" }, "body": { "type": "string" } },
    ///         "required": ["to", "body"]
    ///     }),
    /// );
    ///
    /// assert_eq!(tool.name(), "send_email");
    /// assert!(tool.triggers().is_empty());
    /// ```
    #[must_use]
    pub fn new<N, D>(name: N, description: D, parameters: Value) -> Self
    where
        N: Into<String>,
        D: Into<String>,
    {
        Tool {
            name: name.into(),
            description: description.into(),
            parameters,
            triggers: Vec::new(),
        }
    }

    /// Adds a trigger: a case-insensitive regular expression that, when the request matches it,
    /// makes the engine call this tool.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new("set_light", "Turn a light on or off", json!({ "type": "object" }))
    ///     .with_trigger("light")
    ///     .with_trigger("lamp");
    ///
    /// assert_eq!(tool.triggers(), ["light", "lamp"]);
    /// ```
    #[must_use]
    pub fn with_trigger<T>(mut self, trigger: T) -> Self
    where
        T: Into<String>,
    {
        self.triggers.push(trigger.into());
        self
    }

    /// The tool's name, which is what a [`Call`](crate::needle::Call) carries back.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new("ping", "Check a host", json!({ "type": "object" }));
    /// assert_eq!(tool.name(), "ping");
    /// ```
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The description the model reasons over.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new("ping", "Check a host", json!({ "type": "object" }));
    /// assert_eq!(tool.description(), "Check a host");
    /// ```
    #[inline]
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The JSON Schema for the tool's arguments.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new("ping", "Check a host", json!({ "type": "object" }));
    /// assert_eq!(tool.parameters()["type"], "object");
    /// ```
    #[inline]
    #[must_use]
    pub fn parameters(&self) -> &Value {
        &self.parameters
    }

    /// The trigger words added with [`Tool::with_trigger`], in the order they were added.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Tool;
    /// use serde_json::json;
    ///
    /// let tool = Tool::new("ping", "Check a host", json!({ "type": "object" }));
    /// assert!(tool.triggers().is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn triggers(&self) -> &[String] {
        &self.triggers
    }

    /// Declares a tool whose schema is derived from a Rust type.
    ///
    /// The schema comes from [`schemars`], so the type the engine fills is the type the call is
    /// parsed back into with [`Call::parse`](crate::needle::Call::parse): one definition rather
    /// than a hand-written schema that has to be kept in step.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "schemars")]
    /// # fn main() {
    /// use cactus_rs::needle::Tool;
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct SetLight {
    ///     room: String,
    ///     on: bool,
    /// }
    ///
    /// let tool = Tool::of::<SetLight>("set_light", "Turn a room light on or off");
    /// assert_eq!(tool.parameters()["type"], "object");
    /// # }
    /// # #[cfg(not(feature = "schemars"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "schemars")]
    #[cfg_attr(docsrs, doc(cfg(feature = "schemars")))]
    #[must_use]
    pub fn of<T>(name: &str, description: &str) -> Self
    where
        T: schemars::JsonSchema,
    {
        // `Schema` serialises to a JSON object or a bare boolean, both of which are `Value`.
        let parameters = schemars::schema_for!(T).to_value();
        Tool::new(name, description, parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_form_is_flat_and_compact() {
        let tool = Tool::new("set_light", "Turn a light on or off", Value::Null);
        let wire = serde_json::to_string(&tool).expect("a Tool always serialises");
        assert_eq!(
            wire,
            r#"{"name":"set_light","description":"Turn a light on or off","parameters":null}"#
        );
    }

    #[test]
    fn triggers_appear_only_when_present() {
        let tool = Tool::new("set_light", "Turn a light on or off", Value::Null)
            .with_trigger("light")
            .with_trigger("lamp");
        let wire = serde_json::to_string(&tool).expect("a Tool always serialises");
        assert!(wire.ends_with(r#""triggers":["light","lamp"]}"#));
    }

    #[test]
    fn a_declaration_round_trips() {
        let tool = Tool::new(
            "ping",
            "Check a host",
            serde_json::json!({ "type": "object" }),
        )
        .with_trigger("ping");
        let wire = serde_json::to_string(&tool).expect("a Tool always serialises");
        let back: Tool = serde_json::from_str(&wire).expect("its own output parses");
        assert_eq!(back, tool);
    }

    #[test]
    fn a_missing_triggers_key_deserialises_as_none() {
        let back: Tool = serde_json::from_str(r#"{"name":"a","description":"b","parameters":{}}"#)
            .expect("the flat form parses");
        assert!(back.triggers().is_empty());
    }
}
