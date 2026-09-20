//! The JSON envelope the engine answers with, as Rust types.
//!
//! ## Overview
//!
//! Every [`needle_complete`](cactus_sys::needle_complete) call writes one JSON object into the
//! caller's buffer, and [`Completion`] is that object parsed. A turn that reaches for tools looks
//! like this:
//!
//! ```json
//! {"type":"call","success":true,"function_calls":[{"name":"set_light",
//!  "arguments":{"room":"kitchen","on":true}}],"suppressed_calls":[],
//!  "reasoning":"room 'kitchen' from query; on true from 'on'","confidence":1.0,
//!  "prefill_tps":1966.8,"decode_tps":809.7,"peak_ram_mb":90.9,
//!  "validation":{"ungrounded":[],"negation":false}}
//! ```
//!
//! A turn that does not looks the same with `"type":"respond"`, no calls, no reasoning and a
//! confidence of zero. A failure is a third shape, `{"type":"error","error":"..."}`, which
//! [`Completion`]'s [`FromStr`] turns into [`Error::Complete`] rather than a value.
//!
//! Every field is `#[serde(default)]` and every type is `#[non_exhaustive]`: an engine that grows
//! a field, or drops one, still parses.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::needle::Completion;
//!
//! let envelope = r#"{"type":"respond","success":true,"function_calls":[],
//!     "reasoning":null,"confidence":0.0,"decode_tps":812.0}"#;
//!
//! let completion: Completion = envelope.parse()?;
//! assert!(completion.kind().is_respond());
//! assert!(completion.calls().is_empty());
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::str::FromStr;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// What the engine decided to do with a turn.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Kind;
///
/// assert!(Kind::Call.is_call());
/// assert!(!Kind::Call.is_respond());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Kind {
    /// The engine answered with tool calls.
    Call,
    /// The engine answered in prose, with no tool calls.
    Respond,
    /// A tag this crate does not know, from an engine newer than the pinned one.
    #[default]
    #[serde(other)]
    Unknown,
}

impl Kind {
    /// Whether the engine answered with tool calls.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Kind;
    ///
    /// assert!(Kind::Call.is_call());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_call(&self) -> bool {
        matches!(self, Kind::Call)
    }

    /// Whether the engine answered in prose.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Kind;
    ///
    /// assert!(Kind::Respond.is_respond());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_respond(&self) -> bool {
        matches!(self, Kind::Respond)
    }

    /// Whether the tag was one this crate does not know.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Kind;
    ///
    /// assert!(Kind::Unknown.is_unknown());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Kind::Unknown)
    }
}

/// One tool call the engine chose to make.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Call;
///
/// let call: Call = serde_json::from_str(
///     r#"{"name":"set_light","arguments":{"room":"kitchen","on":true}}"#,
/// )?;
///
/// assert_eq!(call.name(), "set_light");
/// assert_eq!(call.arguments_json(), r#"{"room":"kitchen","on":true}"#);
/// # Ok::<(), serde_json::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Call {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl Call {
    /// The name of the declared [`Tool`](crate::needle::Tool) the engine chose.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Call;
    ///
    /// let call: Call = serde_json::from_str(r#"{"name":"ping","arguments":{}}"#)?;
    /// assert_eq!(call.name(), "ping");
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The arguments the engine filled in, as JSON.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Call;
    ///
    /// let call: Call = serde_json::from_str(r#"{"name":"ping","arguments":{"host":"a"}}"#)?;
    /// assert_eq!(call.arguments()["host"], "a");
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn arguments(&self) -> &Value {
        &self.arguments
    }

    /// The arguments as a compact JSON string, ready to print or log.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Call;
    ///
    /// let call: Call = serde_json::from_str(r#"{"name":"ping","arguments":{"host":"a"}}"#)?;
    /// assert_eq!(call.arguments_json(), r#"{"host":"a"}"#);
    /// # Ok::<(), serde_json::Error>(())
    /// ```
    #[must_use]
    pub fn arguments_json(&self) -> String {
        // Serializing a `Value` cannot fail: it has no non-string map keys and no non-finite
        // numbers to reject.
        serde_json::to_string(&self.arguments).unwrap_or_default()
    }

    /// Parses the arguments into a Rust type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Call;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize)]
    /// struct SetLight {
    ///     room: String,
    ///     on: bool,
    /// }
    ///
    /// let call: Call = serde_json::from_str(
    ///     r#"{"name":"set_light","arguments":{"room":"kitchen","on":true}}"#,
    /// )?;
    ///
    /// let args: SetLight = call.parse()?;
    /// assert_eq!(args.room, "kitchen");
    /// assert!(args.on);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Arguments`] when the arguments the engine produced do not fit `T`.
    pub fn parse<T>(&self) -> Result<T>
    where
        T: DeserializeOwned,
    {
        serde_json::from_value(self.arguments.clone()).map_err(|source| Error::Arguments { source })
    }
}

/// What the turn cost, as the engine measured it.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Completion;
///
/// let completion: Completion =
///     r#"{"type":"respond","prefill_tps":1966.8,"decode_tps":809.7,"peak_ram_mb":90.9}"#
///         .parse()?;
///
/// assert_eq!(completion.stats().decode_tps(), 809.7);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Stats {
    #[serde(default)]
    prefill_tps: f32,
    #[serde(default)]
    decode_tps: f32,
    #[serde(default)]
    peak_ram_mb: f32,
}

impl Stats {
    /// Tokens per second while reading the prompt.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond","prefill_tps":1966.8}"#.parse()?;
    /// assert!(completion.stats().prefill_tps() > 0.0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn prefill_tps(&self) -> f32 {
        self.prefill_tps
    }

    /// Tokens per second while writing the answer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond","decode_tps":809.7}"#.parse()?;
    /// assert!(completion.stats().decode_tps() > 0.0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn decode_tps(&self) -> f32 {
        self.decode_tps
    }

    /// Peak resident memory during the turn, in megabytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond","peak_ram_mb":90.9}"#.parse()?;
    /// assert!(completion.stats().peak_ram_mb() > 0.0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn peak_ram_mb(&self) -> f32 {
        self.peak_ram_mb
    }
}

/// The engine's own check on whether the calls it made follow from the input.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Completion;
///
/// let grounded: Completion =
///     r#"{"type":"call","validation":{"ungrounded":[],"negation":false}}"#.parse()?;
/// assert!(grounded.validation().is_grounded());
///
/// let invented: Completion =
///     r#"{"type":"call","validation":{"ungrounded":["room"],"negation":false}}"#.parse()?;
/// assert!(!invented.validation().is_grounded());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Validation {
    #[serde(default)]
    ungrounded: Vec<String>,
    #[serde(default)]
    negation: bool,
}

impl Validation {
    /// Argument values the engine could not trace back to the input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion =
    ///     r#"{"type":"call","validation":{"ungrounded":["kitchen"]}}"#.parse()?;
    /// assert_eq!(completion.validation().ungrounded(), ["kitchen"]);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn ungrounded(&self) -> &[String] {
        &self.ungrounded
    }

    /// Whether the input negated the action the calls would take.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"call","validation":{"negation":true}}"#.parse()?;
    /// assert!(completion.validation().negation());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn negation(&self) -> bool {
        self.negation
    }

    /// Whether nothing was flagged: no invented values and no negation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion =
    ///     r#"{"type":"call","validation":{"ungrounded":[],"negation":false}}"#.parse()?;
    /// assert!(completion.validation().is_grounded());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn is_grounded(&self) -> bool {
        self.ungrounded.is_empty() && !self.negation
    }
}

/// One turn's answer from the engine.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::needle::Completion;
///
/// let completion: Completion = r#"{"type":"call","success":true,
///     "function_calls":[{"name":"set_light","arguments":{"room":"kitchen","on":true}}],
///     "confidence":1.0,"validation":{"ungrounded":[],"negation":false}}"#
///     .parse()?;
///
/// assert!(completion.kind().is_call());
/// assert_eq!(completion.calls().len(), 1);
/// assert_eq!(completion.grounded_calls().len(), 1);
/// assert_eq!(completion.confidence(), 1.0);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[non_exhaustive]
pub struct Completion {
    #[serde(rename = "type", default)]
    kind: Kind,
    // Kept for `FromStr`, which turns a failed envelope into an error rather than a value.
    #[serde(default = "succeeded")]
    success: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_code: Option<String>,
    #[serde(rename = "function_calls", default)]
    calls: Vec<Call>,
    #[serde(default)]
    suppressed_calls: Vec<Call>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    confidence: f32,
    #[serde(flatten)]
    stats: Stats,
    #[serde(default)]
    validation: Validation,
}

/// The default for a missing `success` field: an envelope that omits it did not fail.
fn succeeded() -> bool {
    true
}

impl Completion {
    /// What the engine decided to do with the turn.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond"}"#.parse()?;
    /// assert!(completion.kind().is_respond());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// Every tool call the engine made, grounded or not.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion =
    ///     r#"{"type":"call","function_calls":[{"name":"ping","arguments":{}}]}"#.parse()?;
    /// assert_eq!(completion.calls()[0].name(), "ping");
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn calls(&self) -> &[Call] {
        &self.calls
    }

    /// The calls, or an empty slice when [`Validation`] flagged the turn.
    ///
    /// Upstream's harness discards every call in a turn whose arguments it could not ground, or
    /// whose input negated the action, rather than filtering them one by one. This mirrors that:
    /// act on these when a wrong call would be expensive, and on [`Completion::calls`] when it
    /// would not.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let flagged: Completion = r#"{"type":"call",
    ///     "function_calls":[{"name":"ping","arguments":{}}],
    ///     "validation":{"ungrounded":["host"],"negation":false}}"#
    ///     .parse()?;
    ///
    /// assert_eq!(flagged.calls().len(), 1);
    /// assert!(flagged.grounded_calls().is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn grounded_calls(&self) -> &[Call] {
        if self.validation.is_grounded() {
            &self.calls
        } else {
            &[]
        }
    }

    /// Calls the engine considered and then dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"call","suppressed_calls":[]}"#.parse()?;
    /// assert!(completion.suppressed_calls().is_empty());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn suppressed_calls(&self) -> &[Call] {
        &self.suppressed_calls
    }

    /// Why the engine made these calls, in its own words, when it said.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"call","reasoning":"room from query"}"#.parse()?;
    /// assert_eq!(completion.reasoning(), Some("room from query"));
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Why the engine took no action, when it said.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond","reason":null}"#.parse()?;
    /// assert_eq!(completion.reason(), None);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// How sure the engine was, from zero to one. A turn with no calls reports zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"call","confidence":1.0}"#.parse()?;
    /// assert_eq!(completion.confidence(), 1.0);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    /// What the turn cost.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"respond","decode_tps":809.7}"#.parse()?;
    /// assert_eq!(completion.stats().decode_tps(), 809.7);
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The engine's grounding check on this turn.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::needle::Completion;
    ///
    /// let completion: Completion = r#"{"type":"call","validation":{"negation":false}}"#.parse()?;
    /// assert!(completion.validation().is_grounded());
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn validation(&self) -> &Validation {
        &self.validation
    }
}

/// Parses an engine envelope, turning a reported failure into an error.
impl FromStr for Completion {
    type Err = Error;

    /// # Errors
    ///
    /// Returns [`Error::Envelope`] when the text is not the JSON this crate expects, and
    /// [`Error::Complete`] when it is a well-formed envelope reporting a failure.
    fn from_str(s: &str) -> Result<Self> {
        let completion: Completion =
            serde_json::from_str(s).map_err(|error| Error::envelope(s, error))?;

        if completion.success && completion.error.is_none() {
            return Ok(completion);
        }

        let detail = match (&completion.error, &completion.error_code) {
            (Some(error), Some(code)) => format!("{error} ({code})"),
            (Some(error), None) => error.clone(),
            (None, Some(code)) => format!("the engine reported failure {code}"),
            (None, None) => "the engine reported a failure without saying why".to_owned(),
        };
        Err(Error::Complete { detail })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The envelope a real macOS arm64 engine answered `turn the kitchen light on` with.
    const CALL: &str = concat!(
        r#"{"type":"call","success":true,"error":null,"error_code":null,"reason":null,"#,
        r#""function_calls":[{"name":"set_light","arguments":{"room":"kitchen","on":true}}],"#,
        r#""suppressed_calls":[],"#,
        r#""reasoning":"room 'kitchen' from query; on true from 'on'","confidence":1.0000,"#,
        r#""prefill_tps":1966.8,"decode_tps":809.7,"peak_ram_mb":90.9,"#,
        r#""validation":{"ungrounded":[],"negation":false}}"#,
    );

    /// The same engine, asked something no tool covers.
    const RESPOND: &str = r#"{"type":"respond","success":true,"error":null,"error_code":null,
        "reason":null,"function_calls":[],"suppressed_calls":[],"reasoning":null,
        "confidence":0.0,"prefill_tps":1902.1,"decode_tps":794.3,"peak_ram_mb":90.9,
        "validation":{"ungrounded":[],"negation":false}}"#;

    #[test]
    fn a_tool_call_envelope_parses() {
        let completion: Completion = CALL.parse().expect("the sample envelope parses");

        assert!(completion.kind().is_call());
        assert_eq!(completion.calls().len(), 1);
        assert_eq!(completion.calls()[0].name(), "set_light");
        assert_eq!(
            completion.calls()[0].arguments_json(),
            r#"{"room":"kitchen","on":true}"#
        );
        assert_eq!(completion.confidence(), 1.0);
        assert_eq!(completion.stats().decode_tps(), 809.7);
        assert_eq!(completion.stats().prefill_tps(), 1966.8);
        assert_eq!(completion.stats().peak_ram_mb(), 90.9);
        assert!(completion.validation().is_grounded());
        assert_eq!(completion.grounded_calls().len(), 1);
        assert!(completion.suppressed_calls().is_empty());
        assert!(completion.reasoning().is_some());
        assert_eq!(completion.reason(), None);
    }

    #[test]
    fn a_reply_envelope_parses() {
        let completion: Completion = RESPOND.parse().expect("the sample envelope parses");

        assert!(completion.kind().is_respond());
        assert!(completion.calls().is_empty());
        assert_eq!(completion.reasoning(), None);
        assert_eq!(completion.confidence(), 0.0);
    }

    #[test]
    fn an_error_envelope_becomes_a_complete_error() {
        let envelope = r#"{"type":"error","error":"needle_init not called"}"#;
        let error = envelope
            .parse::<Completion>()
            .expect_err("an error envelope is not a completion");

        let Error::Complete { detail } = error else {
            panic!("expected Error::Complete");
        };
        assert_eq!(detail, "needle_init not called");
    }

    #[test]
    fn success_false_becomes_a_complete_error() {
        let envelope = r#"{"type":"call","success":false,"error":"nope","error_code":"E7"}"#;
        let error = envelope
            .parse::<Completion>()
            .expect_err("success:false is not a completion");

        let Error::Complete { detail } = error else {
            panic!("expected Error::Complete");
        };
        assert_eq!(detail, "nope (E7)");
    }

    #[test]
    fn a_new_engine_tag_parses_as_unknown() {
        let completion: Completion = r#"{"type":"reflect","success":true}"#
            .parse()
            .expect("an unknown tag is not a parse failure");
        assert!(completion.kind().is_unknown());
    }

    #[test]
    fn a_missing_field_is_not_a_parse_failure() {
        let completion: Completion = r#"{"type":"respond"}"#.parse().expect("defaults fill in");
        assert_eq!(completion.confidence(), 0.0);
        assert_eq!(completion.stats().decode_tps(), 0.0);
        assert!(completion.validation().is_grounded());
    }

    #[test]
    fn junk_is_reported_with_its_text() {
        let error = "not json"
            .parse::<Completion>()
            .expect_err("junk is not a completion");
        let Error::Envelope { text, .. } = &error else {
            panic!("expected Error::Envelope");
        };
        assert_eq!(text, "not json");
    }

    #[test]
    fn a_negated_turn_has_no_grounded_calls() {
        let completion: Completion = r#"{"type":"call",
            "function_calls":[{"name":"ping","arguments":{}}],
            "validation":{"ungrounded":[],"negation":true}}"#
            .parse()
            .expect("the envelope parses");

        assert_eq!(completion.calls().len(), 1);
        assert!(completion.grounded_calls().is_empty());
    }
}
