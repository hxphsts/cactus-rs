//! Questions with a fixed set of answers, and what Mirage answered.
//!
//! ## Overview
//!
//! A [`Form`] is an ordered list of named fields, each one a question the model answers from
//! the picture:
//!
//! | Field | Asks | Answer |
//! | --- | --- | --- |
//! | [`Form::choice`] | which of these labels | the label, or none of them |
//! | [`Form::count`] | how many, within a range | a whole number |
//! | [`Form::text`] | a few words | short text |
//! | [`Form::bbox`] | where is the thing an earlier choice picked | a box |
//!
//! This is the fast path: a form skips the description entirely, and a one-choice form took
//! about a third of the time of a full description of the same image. Every answer comes with
//! the probability the model gave it, and a choice or a count with the probability of every
//! alternative, so the caller decides how sure is sure enough.
//!
//! Fields are answered in order and later ones can depend on earlier ones, which is why the
//! order is kept and why a box names the choice it belongs to.
//!
//! ## Name the field after the question
//!
//! The model reads a field's name: it is the question. On this repository's test image, a red
//! stop sign, the choices `stop`, `yield` and `speed limit` under the name `sign` were answered
//! `stop` with probability 0.75; the same choices under the name `what` were answered `yield`
//! with 0.71. Use the word a person would use for the thing being decided, and check the
//! answers on pictures of your own before trusting a threshold.
//!
//! ## What a form can decide, and how to tell
//!
//! A form is answered from the picture and from the words of the question, and for some
//! questions from the words alone. Measured on real photographs with the pinned model:
//!
//! - **Answered from the picture:** the scene and the things a [`Description`](super::Description)
//!   names. Indoors or outdoors, which room, which of several objects and where it is, a road
//!   sign, the colour of a plain surface.
//! - **Not answered from the picture:** gestures, faces and what a person is doing. Asked
//!   `gesture` with `thumbs up`, `thumbs down`, `open palm` and `fist`, the model said
//!   `thumbs up` with 0.68 to 0.89 for photographs of a thumbs-down, a fist, a peace sign, an
//!   open hand and a dog, and with 0.87 for a blank grey frame. Counts of people were wrong on every portrait
//!   tried.
//!
//! The blank frame is the test. Ask the same form of a flat grey image of the same size: an
//! answer that barely differs from it was not read from the picture, however high its
//! probability. In those measurements, questions the model could not answer stayed within 0.3
//! of the blank-frame distribution (half the summed absolute difference), and questions it
//! answered well moved 0.5 or more. A large distance shows that the model looked, not that it is right.
//!
//! As a rule of thumb, describe the picture first: a form about something the description
//! does not mention will not work.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::mirage::Form;
//!
//! let form = Form::new()
//!     .choice("sign", ["stop", "yield", "speed limit"])
//!     .bbox("where", "sign")
//!     .count("cars", 0, 9);
//! assert_eq!(form.len(), 3);
//! ```
//!
//! Reading the answers, here from the document the engine wrote for a picture of a stop sign:
//!
//! ```rust
//! use cactus_rs::mirage::{Answers, Form};
//!
//! let form = Form::new().choice("sign", ["stop", "yield"]).count("cars", 0, 3);
//! let answers = Answers::parse(
//!     &form,
//!     r#"{"form":{
//!         "sign":{"value":"stop","probabilities":{"stop":0.92,"yield":0.03,"-":0.05}},
//!         "cars":{"value":0,"probabilities":{"0":0.72,"1":0.21,"2":0.03,"3":0.04}}
//!     }}"#,
//! )?;
//!
//! assert_eq!(answers.choice("sign"), Some("stop"));
//! assert_eq!(answers.count("cars"), Some(0));
//! assert_eq!(answers.get("sign").and_then(|a| a.probability()), Some(0.92));
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use serde_json::{Map, Value, json};

use super::description::BBox;
use crate::error::{Error, Result};

/// The label the model gives the probability of "none of these".
///
/// It appears among a choice's [`Answer::probabilities`]. When it wins, the choice has no value.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::mirage::ABSTAIN, "-");
/// ```
pub const ABSTAIN: &str = "-";

/// What one field of a [`Form`] asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Choice(Vec<String>),
    Count { min: u32, max: u32 },
    Text { max_chars: Option<u32> },
    Box { of: String },
}

/// An ordered list of questions for [`Mirage::extract`](super::Mirage::extract).
///
/// The engine checks the form when it is used, so a form it cannot answer comes back from
/// `extract` as [`Error::Describe`], naming the field. It refuses: an empty form, a repeated
/// field name, a name or a choice that is empty or contains one of `; | = # @ $ *` or a line
/// break, a choice list with fewer than one distinct entry, a count whose maximum is below its
/// minimum, a text length of zero, and a box that does not name an earlier choice.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Form;
///
/// let form = Form::new()
///     .choice("animal", ["dog", "cat"])
///     .count("legs", 0, 4);
/// assert_eq!(form.names().collect::<Vec<_>>(), ["animal", "legs"]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Form {
    fields: Vec<(String, Kind)>,
}

impl Form {
    /// A form with no fields yet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// assert!(Form::new().is_empty());
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Form { fields: Vec::new() }
    }

    /// Asks which of `choices` fits. The answer is one of them, or none.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// let form = Form::new().choice("light", ["red", "amber", "green"]);
    /// assert_eq!(form.len(), 1);
    /// ```
    #[must_use]
    pub fn choice<N, I, S>(self, name: N, choices: I) -> Self
    where
        N: Into<String>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let choices = choices.into_iter().map(Into::into).collect();
        self.field(name, Kind::Choice(choices))
    }

    /// Asks for a whole number from `min` to `max`, both included.
    ///
    /// The model weighs every number in the range, so keep the range small: the answer's
    /// probabilities have one entry per number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// let form = Form::new().count("people", 0, 12);
    /// assert_eq!(form.len(), 1);
    /// ```
    #[must_use]
    pub fn count<N>(self, name: N, min: u32, max: u32) -> Self
    where
        N: Into<String>,
    {
        self.field(name, Kind::Count { min, max })
    }

    /// Asks for a few words of free text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// let form = Form::new().text("plate");
    /// assert_eq!(form.len(), 1);
    /// ```
    #[must_use]
    pub fn text<N>(self, name: N) -> Self
    where
        N: Into<String>,
    {
        self.field(name, Kind::Text { max_chars: None })
    }

    /// Asks for free text of at most `max_chars` characters, at least 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// let form = Form::new().text_up_to("plate", 8);
    /// assert_eq!(form.len(), 1);
    /// ```
    #[must_use]
    pub fn text_up_to<N>(self, name: N, max_chars: u32) -> Self
    where
        N: Into<String>,
    {
        self.field(
            name,
            Kind::Text {
                max_chars: Some(max_chars),
            },
        )
    }

    /// Asks where the thing is that the earlier choice field `of` picked.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Form;
    ///
    /// let form = Form::new()
    ///     .choice("sign", ["stop", "yield"])
    ///     .bbox("where", "sign");
    /// assert_eq!(form.len(), 2);
    /// ```
    #[must_use]
    pub fn bbox<N, O>(self, name: N, of: O) -> Self
    where
        N: Into<String>,
        O: Into<String>,
    {
        self.field(name, Kind::Box { of: of.into() })
    }

    /// How many fields the form has.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the form has no fields. The engine refuses an empty form.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field names, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|(name, _)| name.as_str())
    }

    fn field<N>(mut self, name: N, kind: Kind) -> Self
    where
        N: Into<String>,
    {
        self.fields.push((name.into(), kind));
        self
    }

    /// The `form` object the engine reads, fields in order.
    ///
    /// A name used twice is written twice: a JSON object built from a map would silently keep
    /// one and answer a form the caller did not write, so the document is assembled by hand and
    /// the engine reports the repeat.
    pub(crate) fn document(&self) -> String {
        let fields: Vec<String> = self
            .fields
            .iter()
            .map(|(name, kind)| {
                let field = match kind {
                    Kind::Choice(choices) => json!({ "enum": choices }),
                    Kind::Count { min, max } => {
                        json!({ "type": "integer", "minimum": min, "maximum": max })
                    }
                    Kind::Text { max_chars: None } => json!({ "type": "string" }),
                    Kind::Text {
                        max_chars: Some(max),
                    } => json!({ "type": "string", "maxLength": max }),
                    Kind::Box { of } => json!({ "x-source": "bbox", "x-of": of }),
                };
                format!("{}:{field}", Value::from(name.as_str()))
            })
            .collect();
        format!("{{\"form\":{{{}}}}}", fields.join(","))
    }
}

/// What the model answered to one field.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::{Answers, Form};
///
/// let form = Form::new().choice("sign", ["stop", "yield"]);
/// let answers = Answers::parse(
///     &form,
///     r#"{"form":{"sign":{"value":null,"probabilities":{"stop":0.1,"yield":0.2,"-":0.7}}}}"#,
/// )?;
///
/// let answer = answers.get("sign").expect("the field was asked");
/// assert_eq!(answer.as_choice(), None);
/// assert_eq!(answer.probability(), None);
/// assert_eq!(answer.probabilities().len(), 3);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Answer {
    value: AnswerValue,
    probability: Option<f64>,
    probabilities: Vec<(String, f64)>,
}

/// The value of an [`Answer`], by the kind of field that asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnswerValue {
    /// The model picked none of the choices, or found nothing to say or to point at.
    None,
    /// The label a choice field picked.
    Choice(String),
    /// The number a count field settled on.
    Count(i64),
    /// The words a text field got.
    Text(String),
    /// The box a box field found.
    Box(BBox),
}

impl Answer {
    /// The value, whatever its kind.
    #[inline]
    #[must_use]
    pub const fn value(&self) -> &AnswerValue {
        &self.value
    }

    /// The label, for a choice field that picked one.
    #[must_use]
    pub fn as_choice(&self) -> Option<&str> {
        match &self.value {
            AnswerValue::Choice(label) => Some(label),
            _ => None,
        }
    }

    /// The number, for a count field.
    #[must_use]
    pub const fn as_count(&self) -> Option<i64> {
        match self.value {
            AnswerValue::Count(count) => Some(count),
            _ => None,
        }
    }

    /// The words, for a text field that got some.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match &self.value {
            AnswerValue::Text(text) => Some(text),
            _ => None,
        }
    }

    /// The box, for a box field that found one.
    #[must_use]
    pub const fn as_bbox(&self) -> Option<BBox> {
        match self.value {
            AnswerValue::Box(bbox) => Some(bbox),
            _ => None,
        }
    }

    /// The probability the model gave the value, from 0 to 1; `None` when there is no value.
    ///
    /// These are the model's own numbers, not calibrated frequencies: compare them with a
    /// threshold you have measured on your own pictures.
    #[inline]
    #[must_use]
    pub const fn probability(&self) -> Option<f64> {
        self.probability
    }

    /// The probability of every alternative, in the order the engine wrote them.
    ///
    /// For a choice: each label, then [`ABSTAIN`]. For a count: each number in the range, as
    /// text. Empty for text and box fields, which have one value and its probability.
    #[inline]
    #[must_use]
    pub fn probabilities(&self) -> &[(String, f64)] {
        &self.probabilities
    }
}

/// The answers to a [`Form`], in the form's order.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::{Answers, BBox, Form};
///
/// let form = Form::new().choice("sign", ["stop", "yield"]).bbox("where", "sign");
/// let answers = Answers::parse(
///     &form,
///     r#"{"form":{
///         "sign":{"value":"stop","probabilities":{"stop":0.92,"yield":0.03,"-":0.05}},
///         "where":{"value":[361,174,642,517],"probability":0.82}
///     }}"#,
/// )?;
///
/// assert_eq!(answers.bbox("where"), Some(BBox::new(361, 174, 642, 517)));
/// assert_eq!(answers.iter().map(|(name, _)| name).collect::<Vec<_>>(), ["sign", "where"]);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct Answers {
    fields: Vec<(String, Answer)>,
    tokens: u32,
}

impl Answers {
    /// Reads the document the engine wrote for `form`.
    ///
    /// [`Mirage::extract`](super::Mirage::extract) does this itself; the function is public for
    /// a document that was stored or sent from elsewhere.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Envelope`] when the text is not a form document, or lacks a field of
    /// `form`, or holds a value of the wrong kind for one.
    pub fn parse(form: &Form, text: &str) -> Result<Self> {
        let invalid = |problem: String| {
            Error::envelope(
                text,
                <serde_json::Error as serde::de::Error>::custom(problem),
            )
        };

        let mut document: Map<String, Value> =
            serde_json::from_str(text).map_err(|source| Error::envelope(text, source))?;
        let Some(Value::Object(mut answered)) = document.remove("form") else {
            return Err(invalid("the document has no `form` object".to_owned()));
        };

        let mut fields = Vec::with_capacity(form.fields.len());
        for (name, kind) in &form.fields {
            let Some(Value::Object(answer)) = answered.remove(name) else {
                return Err(invalid(format!("the form field `{name}` has no answer")));
            };
            let answer = read(kind, &answer)
                .ok_or_else(|| invalid(format!("the answer to `{name}` is not of its kind")))?;
            fields.push((name.clone(), answer));
        }
        Ok(Answers { fields, tokens: 0 })
    }

    /// The answer to the field called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Answer> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, answer)| answer)
    }

    /// The label a choice field picked; `None` if it picked none or is not a choice.
    #[must_use]
    pub fn choice(&self, name: &str) -> Option<&str> {
        self.get(name)?.as_choice()
    }

    /// The number a count field settled on.
    #[must_use]
    pub fn count(&self, name: &str) -> Option<i64> {
        self.get(name)?.as_count()
    }

    /// The words a text field got.
    #[must_use]
    pub fn text(&self, name: &str) -> Option<&str> {
        self.get(name)?.as_text()
    }

    /// The box a box field found.
    #[must_use]
    pub fn bbox(&self, name: &str) -> Option<BBox> {
        self.get(name)?.as_bbox()
    }

    /// Every field with its answer, in the form's order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Answer)> {
        self.fields
            .iter()
            .map(|(name, answer)| (name.as_str(), answer))
    }

    /// How many fields were answered.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether there are no answers.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// How many tokens the model generated for these answers.
    ///
    /// Zero for answers parsed from text.
    #[inline]
    #[must_use]
    pub const fn tokens(&self) -> u32 {
        self.tokens
    }

    /// Records the engine's return value on parsed answers.
    pub(crate) fn with_tokens(mut self, tokens: u32) -> Self {
        self.tokens = tokens;
        self
    }
}

/// One field's answer, read as the kind of field that asked. `None` when it is of another kind.
fn read(kind: &Kind, answer: &Map<String, Value>) -> Option<Answer> {
    let value = answer.get("value")?;
    let probabilities: Vec<(String, f64)> = match answer.get("probabilities") {
        Some(Value::Object(all)) => all
            .iter()
            .map(|(label, p)| Some((label.clone(), p.as_f64()?)))
            .collect::<Option<_>>()?,
        Some(_) => return None,
        None => Vec::new(),
    };
    // A choice and a count carry the whole distribution; the value's share is looked up in it.
    let share = |label: &str| {
        probabilities
            .iter()
            .find(|(candidate, _)| candidate == label)
            .map(|(_, p)| *p)
    };
    let single = answer.get("probability").and_then(Value::as_f64);

    let (value, probability) = match (kind, value) {
        (_, Value::Null) => (AnswerValue::None, None),
        // A choice comes back as text even when its labels were written as numbers.
        (Kind::Choice(_), Value::String(label)) => {
            (AnswerValue::Choice(label.clone()), share(label))
        }
        (Kind::Count { .. }, Value::Number(count)) => {
            let count = count.as_i64()?;
            (AnswerValue::Count(count), share(&count.to_string()))
        }
        // The model writes the abstain label when it has nothing to say.
        (Kind::Text { .. }, Value::String(text)) if text == ABSTAIN => (AnswerValue::None, None),
        (Kind::Text { .. }, Value::String(text)) => (AnswerValue::Text(text.clone()), single),
        (Kind::Box { .. }, edges @ Value::Array(_)) => {
            let bbox: BBox = serde_json::from_value(edges.clone()).ok()?;
            (AnswerValue::Box(bbox), single)
        }
        _ => return None,
    };
    Some(Answer {
        value,
        probability,
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_document_keeps_the_order_and_the_engines_keys() {
        let form = Form::new()
            .choice("sign", ["stop", "yield"])
            .count("cars", 0, 9)
            .text("note")
            .text_up_to("plate", 8)
            .bbox("where", "sign");
        assert_eq!(
            form.document(),
            concat!(
                r#"{"form":{"sign":{"enum":["stop","yield"]},"#,
                r#""cars":{"type":"integer","minimum":0,"maximum":9},"#,
                r#""note":{"type":"string"},"#,
                r#""plate":{"type":"string","maxLength":8},"#,
                r#""where":{"x-source":"bbox","x-of":"sign"}}}"#
            )
        );
        // Still JSON, with the fields in the order they were added.
        let parsed: Value = serde_json::from_str(&form.document()).expect("valid JSON");
        let names: Vec<&String> = parsed["form"]
            .as_object()
            .expect("an object")
            .keys()
            .collect();
        assert_eq!(names, ["sign", "cars", "note", "plate", "where"]);
    }

    #[test]
    fn names_and_choices_are_escaped_not_spliced() {
        let form = Form::new().choice("a \"quoted\" name", ["back\\slash", "new\nline"]);
        let parsed: Value = serde_json::from_str(&form.document()).expect("valid JSON");
        assert_eq!(
            parsed["form"]["a \"quoted\" name"]["enum"],
            json!(["back\\slash", "new\nline"])
        );
    }

    #[test]
    fn a_repeated_name_reaches_the_engine_twice() {
        let form = Form::new().count("n", 0, 1).count("n", 0, 2);
        assert_eq!(form.document().matches("\"n\":").count(), 2);
    }

    /// What the pinned engine wrote for a three-field form on the test image.
    const MEASURED: &str = r#"{"form":{"sign":{"value":"stop","probabilities":{"stop":0.6929260502934657,"yield":0.07971662240306383,"speed limit":0.05349344111841511,"-":0.17386388618505536}},"count":{"value":1,"probabilities":{"0":0.013807542113156747,"1":0.8809761677056342,"2":0.08638469225223203}},"note":{"value":"-","probability":0.1339605669525078}}}"#;

    #[test]
    fn the_measured_document_parses_by_kind() {
        let form = Form::new()
            .choice("sign", ["stop", "yield", "speed limit"])
            .count("count", 0, 2)
            .text("note");
        let answers = Answers::parse(&form, MEASURED).expect("the engine's own output");

        assert_eq!(answers.len(), 3);
        assert_eq!(answers.choice("sign"), Some("stop"));
        assert_eq!(answers.count("count"), Some(1));
        // The abstain label in a text field is no text.
        assert_eq!(answers.text("note"), None);
        assert_eq!(
            answers.get("note").map(Answer::value),
            Some(&AnswerValue::None)
        );

        let sign = answers.get("sign").expect("asked");
        assert_eq!(sign.probability(), Some(0.692_926_050_293_465_7));
        assert_eq!(sign.probabilities().len(), 4);
        assert_eq!(sign.probabilities()[3].0, ABSTAIN);
        assert_eq!(
            answers.get("count").and_then(Answer::probability),
            Some(0.880_976_167_705_634_2)
        );

        // Asking a field for another kind's value gives nothing, not a panic.
        assert_eq!(answers.count("sign"), None);
        assert_eq!(answers.bbox("sign"), None);
        assert_eq!(answers.choice("missing"), None);
    }

    #[test]
    fn a_box_and_a_missing_box_both_parse() {
        let form = Form::new().choice("sign", ["stop"]).bbox("where", "sign");
        let found = Answers::parse(
            &form,
            r#"{"form":{"sign":{"value":"stop","probabilities":{"stop":0.9,"-":0.1}},"where":{"value":[361,174,642,517],"probability":0.82}}}"#,
        )
        .expect("a box");
        assert_eq!(found.bbox("where"), Some(BBox::new(361, 174, 642, 517)));
        assert_eq!(found.get("where").and_then(Answer::probability), Some(0.82));

        let missing = Answers::parse(
            &form,
            r#"{"form":{"sign":{"value":null,"probabilities":{"stop":0.1,"-":0.9}},"where":{"value":null,"probability":null}}}"#,
        )
        .expect("no box");
        assert_eq!(missing.choice("sign"), None);
        assert_eq!(missing.bbox("where"), None);
        assert_eq!(missing.get("where").and_then(Answer::probability), None);
    }

    #[test]
    fn numeric_labels_come_back_as_text() {
        let form = Form::new().choice("n", ["1", "2", "3"]);
        let answers = Answers::parse(
            &form,
            r#"{"form":{"n":{"value":"3","probabilities":{"1":0.1,"2":0.2,"3":0.6,"-":0.1}}}}"#,
        )
        .expect("a choice among numbers");
        assert_eq!(answers.choice("n"), Some("3"));
        assert_eq!(answers.get("n").and_then(Answer::probability), Some(0.6));
    }

    #[test]
    fn a_document_that_does_not_fit_the_form_is_an_envelope_error() {
        let form = Form::new().choice("sign", ["stop"]).count("cars", 0, 3);
        for text in [
            "not json",
            "{}",
            r#"{"form":[]}"#,
            // A field of the form is missing.
            r#"{"form":{"sign":{"value":"stop","probabilities":{}}}}"#,
            // A count answered with text.
            r#"{"form":{"sign":{"value":"stop","probabilities":{}},"cars":{"value":"two","probabilities":{}}}}"#,
            // A probability that is not a number.
            r#"{"form":{"sign":{"value":"stop","probabilities":{"stop":"high"}},"cars":{"value":1,"probabilities":{}}}}"#,
        ] {
            let error = Answers::parse(&form, text).expect_err(text);
            assert!(matches!(error, Error::Envelope { .. }), "{text}");
        }
    }

    #[test]
    fn answers_iterate_in_the_forms_order() {
        let form = Form::new().count("b", 0, 1).count("a", 0, 1);
        let answers = Answers::parse(
            &form,
            r#"{"form":{"a":{"value":0,"probabilities":{"0":1.0}},"b":{"value":1,"probabilities":{"1":1.0}}}}"#,
        )
        .expect("both answered");
        let names: Vec<&str> = answers.iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["b", "a"]);
        assert!(!answers.is_empty());
        assert_eq!(answers.tokens(), 0);
    }
}
