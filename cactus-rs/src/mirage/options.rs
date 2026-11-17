//! What one description should contain, and how hard the model should look.
//!
//! ## Overview
//!
//! [`DescribeOptions`] collects the settings `cactus_describe` reads from its JSON options. Each
//! one is optional: a setting that is never given is left out of the document, and the engine
//! uses its own default. The engine checks the values itself, so one out of range comes back as
//! [`Error::Describe`] naming the setting, in the engine's words.
//!
//! [`Section`] chooses which parts of the result are produced at all, and [`Field`] which
//! details each object carries. Asking for less is faster: the scene alone took a little over
//! half the time of the full description of the same image.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::mirage::{DescribeOptions, Field, MaskFormat, Section};
//!
//! let options = DescribeOptions::new()
//!     .with_sections([Section::Objects, Section::Text])
//!     .with_fields([Field::Bbox, Field::Score])
//!     .with_min_score(0.5)
//!     .with_max_objects(8);
//! assert_eq!(options.sections(), Some(&[Section::Objects, Section::Text][..]));
//! # let _ = MaskFormat::Polygons;
//! ```

use std::ffi::CString;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::error::{Error, Result};
use crate::whistle::Language;
use crate::whistle::options::join_keywords;

/// One part of a [`Description`](super::Description).
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Section;
///
/// assert_eq!(Section::Objects.as_str(), "objects");
/// assert_eq!(Section::ALL.len(), 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Section {
    /// What the picture shows as a whole: setting, surface, lighting, style.
    Scene,
    /// The things that were found, each with a name and, by default, a box, a score and a mask.
    Objects,
    /// Things the model names but could not place with a box.
    Regions,
    /// The text that was read, line by line.
    Text,
    /// How the objects relate: which is attached to, standing on or holding which.
    Relations,
    /// What is happening, when the picture shows an action.
    Events,
    /// The model's short account of how it read the picture.
    Reasoning,
}

impl Section {
    /// Every section, in the order the engine writes them.
    pub const ALL: [Section; 7] = [
        Section::Scene,
        Section::Objects,
        Section::Regions,
        Section::Text,
        Section::Relations,
        Section::Events,
        Section::Reasoning,
    ];

    /// The name the engine knows the section by.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Section;
    ///
    /// assert_eq!(Section::Reasoning.as_str(), "reasoning");
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Section::Scene => "scene",
            Section::Objects => "objects",
            Section::Regions => "regions",
            Section::Text => "text",
            Section::Relations => "relations",
            Section::Events => "events",
            Section::Reasoning => "reasoning",
        }
    }
}

/// One detail of an object or a line of text.
///
/// An object always has its id and name. Everything else is a field: by default the engine
/// writes the box, the score, the mask, the position and the description.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Field;
///
/// assert_eq!(Field::Bbox.as_str(), "bbox");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Field {
    /// The bounding box, also of each line of text.
    Bbox,
    /// The detection score, also of each line of text.
    Score,
    /// The outline or bitmap, in the chosen [`MaskFormat`].
    Mask,
    /// One point inside the object.
    Point,
    /// Where the object is, in words: "center", "below sign".
    Position,
    /// The object's attributes: colour, shape, material.
    Description,
    /// How many instances the entity has, when the model counts them.
    Count,
}

impl Field {
    /// The name the engine knows the field by.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Field;
    ///
    /// assert_eq!(Field::Description.as_str(), "description");
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Field::Bbox => "bbox",
            Field::Score => "score",
            Field::Mask => "mask",
            Field::Point => "point",
            Field::Position => "position",
            Field::Description => "description",
            Field::Count => "count",
        }
    }
}

/// How an object's mask is written.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::MaskFormat;
///
/// assert_eq!(MaskFormat::default(), MaskFormat::Largest);
/// assert_eq!(MaskFormat::Bitmap.as_str(), "bitmap");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MaskFormat {
    /// One outline: the largest connected part of the mask. The engine's default.
    #[default]
    Largest,
    /// One outline around all the parts of the mask, joined.
    Merged,
    /// One outline per connected part.
    Polygons,
    /// Every pixel, run-length encoded at the size of the image.
    Bitmap,
}

impl MaskFormat {
    /// The name the engine knows the format by.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::MaskFormat;
    ///
    /// assert_eq!(MaskFormat::Polygons.as_str(), "polygons");
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            MaskFormat::Largest => "largest",
            MaskFormat::Merged => "merged",
            MaskFormat::Polygons => "polygons",
            MaskFormat::Bitmap => "bitmap",
        }
    }
}

/// The settings of one description. Every one is optional.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::{DescribeOptions, Section};
///
/// let options = DescribeOptions::new().with_sections([Section::Text]);
/// assert_eq!(options.sections(), Some(&[Section::Text][..]));
/// assert_eq!(DescribeOptions::new(), DescribeOptions::default());
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct DescribeOptions {
    sections: Option<Vec<Section>>,
    fields: Option<Vec<Field>>,
    mask_format: Option<MaskFormat>,
    min_score: Option<f32>,
    min_text_score: Option<f32>,
    overlap: Option<f32>,
    suppress_overlaps: Option<bool>,
    suppress_across_names: Option<bool>,
    max_objects: Option<u32>,
    max_text_lines: Option<u32>,
    max_tokens: Option<u32>,
    mask_threshold: Option<f32>,
    mask_tolerance: Option<f32>,
    keyword_bias: Option<f32>,
    language: Option<Language>,
    keywords: Vec<String>,
}

impl DescribeOptions {
    /// The defaults: every section, the engine's choice for everything else.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new();
    /// assert_eq!(options.sections(), None);
    /// assert_eq!(options.language(), None);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        DescribeOptions {
            sections: None,
            fields: None,
            mask_format: None,
            min_score: None,
            min_text_score: None,
            overlap: None,
            suppress_overlaps: None,
            suppress_across_names: None,
            max_objects: None,
            max_text_lines: None,
            max_tokens: None,
            mask_threshold: None,
            mask_tolerance: None,
            keyword_bias: None,
            language: None,
            keywords: Vec::new(),
        }
    }

    /// Produces only these sections (`sections`).
    ///
    /// A section that is left out is absent from the result, and the work behind it is skipped
    /// where the engine can. An empty list produces an empty result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::{DescribeOptions, Section};
    ///
    /// let options = DescribeOptions::new().with_sections([Section::Scene, Section::Text]);
    /// assert_eq!(options.sections().map(<[Section]>::len), Some(2));
    /// ```
    #[must_use]
    pub fn with_sections<I>(mut self, sections: I) -> Self
    where
        I: IntoIterator<Item = Section>,
    {
        self.sections = Some(sections.into_iter().collect());
        self
    }

    /// Gives each object and line of text only these details (`fields`).
    ///
    /// An empty list leaves each object its id and name alone.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::{DescribeOptions, Field};
    ///
    /// let options = DescribeOptions::new().with_fields([Field::Bbox, Field::Score]);
    /// assert_eq!(options.fields(), Some(&[Field::Bbox, Field::Score][..]));
    /// ```
    #[must_use]
    pub fn with_fields<I>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = Field>,
    {
        self.fields = Some(fields.into_iter().collect());
        self
    }

    /// Chooses how masks are written (`mask_format`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::{DescribeOptions, MaskFormat};
    ///
    /// let options = DescribeOptions::new().with_mask_format(MaskFormat::Bitmap);
    /// assert_eq!(options.mask_format(), Some(MaskFormat::Bitmap));
    /// ```
    #[must_use]
    pub const fn with_mask_format(mut self, format: MaskFormat) -> Self {
        self.mask_format = Some(format);
        self
    }

    /// Drops objects scoring below this, from 0 to 1 (`conf`).
    ///
    /// An object that loses its box this way can still be named, under
    /// [`Description::regions`](super::Description::regions).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// assert_eq!(DescribeOptions::new().with_min_score(0.5).min_score(), Some(0.5));
    /// ```
    #[must_use]
    pub const fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }

    /// Drops lines of text scoring below this, from 0 to 1 (`text_conf`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_min_text_score(0.8);
    /// assert_eq!(options.min_text_score(), Some(0.8));
    /// ```
    #[must_use]
    pub const fn with_min_text_score(mut self, score: f32) -> Self {
        self.min_text_score = Some(score);
        self
    }

    /// Sets how much two boxes may overlap before one suppresses the other, from 0 to 1 (`iou`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// assert_eq!(DescribeOptions::new().with_overlap(0.4).overlap(), Some(0.4));
    /// ```
    #[must_use]
    pub const fn with_overlap(mut self, overlap: f32) -> Self {
        self.overlap = Some(overlap);
        self
    }

    /// Turns the suppression of overlapping boxes on or off (`nms`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_overlap_suppression(false);
    /// assert_eq!(options.overlap_suppression(), Some(false));
    /// ```
    #[must_use]
    pub const fn with_overlap_suppression(mut self, on: bool) -> Self {
        self.suppress_overlaps = Some(on);
        self
    }

    /// Lets a box suppress an overlapping box of a different name too (`agnostic_nms`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_suppression_across_names(true);
    /// assert_eq!(options.suppression_across_names(), Some(true));
    /// ```
    #[must_use]
    pub const fn with_suppression_across_names(mut self, on: bool) -> Self {
        self.suppress_across_names = Some(on);
        self
    }

    /// Reports at most this many objects, at least 1 (`max_objects`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// assert_eq!(DescribeOptions::new().with_max_objects(4).max_objects(), Some(4));
    /// ```
    #[must_use]
    pub const fn with_max_objects(mut self, objects: u32) -> Self {
        self.max_objects = Some(objects);
        self
    }

    /// Reports at most this many lines of text, at least 1 (`max_text_lines`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_max_text_lines(10);
    /// assert_eq!(options.max_text_lines(), Some(10));
    /// ```
    #[must_use]
    pub const fn with_max_text_lines(mut self, lines: u32) -> Self {
        self.max_text_lines = Some(lines);
        self
    }

    /// Lets the model write at most this many tokens, at least 1 (`max_tokens`).
    ///
    /// A budget that runs out is not an error: the result is still well-formed, with whatever
    /// the model had written. Text is read by a separate part of the model and is not affected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// assert_eq!(DescribeOptions::new().with_max_tokens(64).max_tokens(), Some(64));
    /// ```
    #[must_use]
    pub const fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Sets how sure a pixel must be to belong to a mask, from 0 to 1 (`mask_threshold`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_mask_threshold(0.7);
    /// assert_eq!(options.mask_threshold(), Some(0.7));
    /// ```
    #[must_use]
    pub const fn with_mask_threshold(mut self, threshold: f32) -> Self {
        self.mask_threshold = Some(threshold);
        self
    }

    /// Sets how far an outline may stray from the mask to save points, 0 or more
    /// (`mask_tolerance`).
    ///
    /// `0` keeps every point of the outline; larger values give simpler polygons.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_mask_tolerance(0.0);
    /// assert_eq!(options.mask_tolerance(), Some(0.0));
    /// ```
    #[must_use]
    pub const fn with_mask_tolerance(mut self, tolerance: f32) -> Self {
        self.mask_tolerance = Some(tolerance);
        self
    }

    /// Sets how strongly the keywords pull, from 0 to 100 (`keyword_bias`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_keyword_bias(10.0);
    /// assert_eq!(options.keyword_bias(), Some(10.0));
    /// ```
    #[must_use]
    pub const fn with_keyword_bias(mut self, bias: f32) -> Self {
        self.keyword_bias = Some(bias);
        self
    }

    /// Writes names and descriptions in this language, or with `None` in English.
    ///
    /// Text read from the picture is reported as written, whatever the language.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::{DescribeOptions, Language};
    ///
    /// let german = DescribeOptions::new().with_language(Language::De);
    /// assert_eq!(german.language(), Some(Language::De));
    /// assert_eq!(german.with_language(None).language(), None);
    /// ```
    #[must_use]
    pub fn with_language<L>(mut self, language: L) -> Self
    where
        L: Into<Option<Language>>,
    {
        self.language = language.into();
        self
    }

    /// Adds one word or phrase to bias the model towards.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new().with_keyword("forklift");
    /// assert_eq!(options.keywords(), ["forklift"]);
    /// ```
    #[must_use]
    pub fn with_keyword<S>(mut self, keyword: S) -> Self
    where
        S: Into<String>,
    {
        self.keywords.push(keyword.into());
        self
    }

    /// Replaces the keywords.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::DescribeOptions;
    ///
    /// let options = DescribeOptions::new()
    ///     .with_keyword("dropped")
    ///     .with_keywords(["pallet", "forklift"]);
    /// assert_eq!(options.keywords(), ["pallet", "forklift"]);
    /// ```
    #[must_use]
    pub fn with_keywords<I, S>(mut self, keywords: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    /// The sections asked for, or `None` for all of them.
    #[inline]
    #[must_use]
    pub fn sections(&self) -> Option<&[Section]> {
        self.sections.as_deref()
    }

    /// The fields asked for, or `None` for the engine's default set.
    #[inline]
    #[must_use]
    pub fn fields(&self) -> Option<&[Field]> {
        self.fields.as_deref()
    }

    /// The mask format, or `None` for the engine's default, [`MaskFormat::Largest`].
    #[inline]
    #[must_use]
    pub const fn mask_format(&self) -> Option<MaskFormat> {
        self.mask_format
    }

    /// The lowest object score kept, if set.
    #[inline]
    #[must_use]
    pub const fn min_score(&self) -> Option<f32> {
        self.min_score
    }

    /// The lowest text score kept, if set.
    #[inline]
    #[must_use]
    pub const fn min_text_score(&self) -> Option<f32> {
        self.min_text_score
    }

    /// The overlap at which one box suppresses another, if set.
    #[inline]
    #[must_use]
    pub const fn overlap(&self) -> Option<f32> {
        self.overlap
    }

    /// Whether overlapping boxes are suppressed, if set.
    #[inline]
    #[must_use]
    pub const fn overlap_suppression(&self) -> Option<bool> {
        self.suppress_overlaps
    }

    /// Whether suppression crosses names, if set.
    #[inline]
    #[must_use]
    pub const fn suppression_across_names(&self) -> Option<bool> {
        self.suppress_across_names
    }

    /// The most objects reported, if set.
    #[inline]
    #[must_use]
    pub const fn max_objects(&self) -> Option<u32> {
        self.max_objects
    }

    /// The most lines of text reported, if set.
    #[inline]
    #[must_use]
    pub const fn max_text_lines(&self) -> Option<u32> {
        self.max_text_lines
    }

    /// The token budget, if set.
    #[inline]
    #[must_use]
    pub const fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// The mask threshold, if set.
    #[inline]
    #[must_use]
    pub const fn mask_threshold(&self) -> Option<f32> {
        self.mask_threshold
    }

    /// The outline tolerance, if set.
    #[inline]
    #[must_use]
    pub const fn mask_tolerance(&self) -> Option<f32> {
        self.mask_tolerance
    }

    /// The keyword bias, if set.
    #[inline]
    #[must_use]
    pub const fn keyword_bias(&self) -> Option<f32> {
        self.keyword_bias
    }

    /// The language of names and descriptions, or `None` for English.
    #[inline]
    #[must_use]
    pub const fn language(&self) -> Option<Language> {
        self.language
    }

    /// The keywords as they were given, before trimming.
    #[inline]
    #[must_use]
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }

    /// The JSON object the engine reads, holding only the settings that were given.
    pub(crate) fn document(&self) -> Map<String, Value> {
        let mut document = Map::new();
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                document.insert(key.to_owned(), value);
            }
        };
        let names = |names: Vec<&'static str>| Value::from(names);

        set(
            "sections",
            self.sections
                .as_ref()
                .map(|sections| names(sections.iter().map(|s| s.as_str()).collect())),
        );
        set(
            "fields",
            self.fields
                .as_ref()
                .map(|fields| names(fields.iter().map(|f| f.as_str()).collect())),
        );
        set(
            "mask_format",
            self.mask_format.map(|format| json!(format.as_str())),
        );
        set("conf", self.min_score.map(|v| json!(v)));
        set("text_conf", self.min_text_score.map(|v| json!(v)));
        set("iou", self.overlap.map(|v| json!(v)));
        set("nms", self.suppress_overlaps.map(Value::from));
        set("agnostic_nms", self.suppress_across_names.map(Value::from));
        set("max_objects", self.max_objects.map(Value::from));
        set("max_text_lines", self.max_text_lines.map(Value::from));
        set("max_tokens", self.max_tokens.map(Value::from));
        set("mask_threshold", self.mask_threshold.map(|v| json!(v)));
        set("mask_tolerance", self.mask_tolerance.map(|v| json!(v)));
        set("keyword_bias", self.keyword_bias.map(|v| json!(v)));
        document
    }

    /// The options in the shape the C API takes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidKeyword`] for a keyword with a line break inside it, and
    /// [`Error::InteriorNul`] for one with a NUL byte.
    pub(crate) fn encode(&self) -> Result<Encoded> {
        Encoded::new(
            self.document(),
            self.language.map(|language| language.as_c_str().to_owned()),
            join_keywords(&self.keywords)?,
        )
    }
}

/// The three strings one `cactus_describe` call takes besides the image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Encoded {
    pub(crate) options: CString,
    pub(crate) language: Option<CString>,
    pub(crate) keywords: Option<CString>,
}

impl Encoded {
    /// Serialises `document` as the options string.
    ///
    /// The engine ignores an options string it cannot parse and describes the image with its
    /// defaults, so the document is always written by `serde_json`, never spliced together.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InteriorNul`] when a string inside the document holds a NUL byte.
    pub(crate) fn new(
        document: Map<String, Value>,
        language: Option<CString>,
        keywords: Option<CString>,
    ) -> Result<Self> {
        let options = CString::new(Value::Object(document).to_string())
            .map_err(|_| Error::InteriorNul { field: "options" })?;
        Ok(Encoded {
            options,
            language,
            keywords,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_options_send_an_empty_object() {
        let encoded = DescribeOptions::new().encode().expect("nothing to refuse");
        assert_eq!(encoded.options.to_str(), Ok("{}"));
        assert_eq!(encoded.language, None);
        assert_eq!(encoded.keywords, None);
    }

    #[test]
    fn every_setting_reaches_the_document_under_the_engines_name() {
        let options = DescribeOptions::new()
            .with_sections([Section::Objects, Section::Text])
            .with_fields([Field::Bbox, Field::Mask])
            .with_mask_format(MaskFormat::Polygons)
            .with_min_score(0.5)
            .with_min_text_score(0.25)
            .with_overlap(0.75)
            .with_overlap_suppression(false)
            .with_suppression_across_names(true)
            .with_max_objects(3)
            .with_max_text_lines(4)
            .with_max_tokens(5)
            .with_mask_threshold(0.5)
            .with_mask_tolerance(2.0)
            .with_keyword_bias(10.0);
        let document = Value::Object(options.document());
        assert_eq!(
            document,
            json!({
                "sections": ["objects", "text"],
                "fields": ["bbox", "mask"],
                "mask_format": "polygons",
                "conf": 0.5,
                "text_conf": 0.25,
                "iou": 0.75,
                "nms": false,
                "agnostic_nms": true,
                "max_objects": 3,
                "max_text_lines": 4,
                "max_tokens": 5,
                "mask_threshold": 0.5,
                "mask_tolerance": 2.0,
                "keyword_bias": 10.0,
            })
        );
    }

    #[test]
    fn an_empty_list_is_sent_as_one() {
        let document = DescribeOptions::new()
            .with_sections([])
            .with_fields([])
            .document();
        assert_eq!(
            Value::Object(document),
            json!({"sections": [], "fields": []})
        );
    }

    #[test]
    fn language_and_keywords_travel_beside_the_document() {
        let encoded = DescribeOptions::new()
            .with_language(Language::Pl)
            .with_keywords([" Łódź ", "", "café"])
            .encode()
            .expect("valid keywords");
        assert_eq!(encoded.options.to_str(), Ok("{}"));
        assert_eq!(encoded.language.as_deref(), Some(c"pl"));
        assert_eq!(
            encoded.keywords.as_deref().map(|k| k.to_str()),
            Some(Ok("Łódź\ncafé"))
        );
    }

    #[test]
    fn a_keyword_with_a_line_break_is_refused() {
        let error = DescribeOptions::new()
            .with_keyword("a\nb")
            .encode()
            .expect_err("a line break");
        assert!(matches!(error, Error::InvalidKeyword { .. }));
    }

    #[test]
    fn names_match_what_serde_writes() {
        for section in Section::ALL {
            assert_eq!(json!(section), json!(section.as_str()));
        }
        for field in [
            Field::Bbox,
            Field::Score,
            Field::Mask,
            Field::Point,
            Field::Position,
            Field::Description,
            Field::Count,
        ] {
            assert_eq!(json!(field), json!(field.as_str()));
        }
        for format in [
            MaskFormat::Largest,
            MaskFormat::Merged,
            MaskFormat::Polygons,
            MaskFormat::Bitmap,
        ] {
            assert_eq!(json!(format), json!(format.as_str()));
        }
    }
}
