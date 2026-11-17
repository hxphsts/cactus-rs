//! What Mirage saw: the typed form of the JSON document `cactus_describe` writes.
//!
//! ## Overview
//!
//! A [`Description`] holds up to seven sections, each present only if it was asked for:
//!
//! - the **scene**, a few free-form attributes of the picture as a whole
//! - **objects**, each an [`Object`] with a name and, by default, a [`BBox`], a score, a
//!   [`Mask`], a position in words and its attributes
//! - **regions**, things the model names but did not place with a box
//! - **text**, one [`TextLine`] per line that was read
//! - **relations**, [`Relation`]s between objects by id
//! - **events**, and the model's **reasoning**
//!
//! ## Coordinates
//!
//! Every position (boxes, points, polygon outlines) is on a scale from 0 to 1000 along each axis
//! of the image that was given, whatever its size or shape: `x = 500` is the horizontal middle.
//! [`BBox::to_pixels`] converts a box back. Bitmap masks are the exception: they are written at
//! the image's own size, pixel for pixel.
//!
//! ## What is not fixed
//!
//! The model writes the scene and each object's description as whatever attributes it finds
//! worth naming, so those are maps, not structs: one picture has `"time_of_day"`, the next does
//! not. Keys this crate does not know are kept, in [`Description::other`] and
//! [`Object::other`], so a field added by a later engine is not lost.
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::mirage::Description;
//!
//! let description: Description = r#"{
//!     "objects": [{"id": 1, "name": "stop sign", "bbox": [361, 174, 642, 517], "score": 0.732}],
//!     "text": [{"string": "STOP", "bbox": [370, 283, 632, 407], "score": 0.79}]
//! }"#
//! .parse()?;
//!
//! let sign = &description.objects()[0];
//! assert_eq!(sign.name(), "stop sign");
//! assert_eq!(sign.bbox().map(|b| b.to_pixels(512, 384)), Some([185, 67, 329, 199]));
//! assert_eq!(description.text()[0].string(), "STOP");
//! # Ok::<(), cactus_rs::Error>(())
//! ```

use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// The upper end of the coordinate scale: positions run from 0 to this along each axis.
///
/// # Examples
///
/// ```rust
/// assert_eq!(cactus_rs::mirage::SCALE, 1000);
/// ```
pub const SCALE: u16 = 1000;

/// One description of one image.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Description;
///
/// let description: Description = r#"{"scene": {"background": "blue sky"}}"#.parse()?;
/// assert_eq!(description.scene()["background"], "blue sky");
/// assert!(description.objects().is_empty());
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Description {
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    scene: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    objects: Vec<Object>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    regions: Vec<Region>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    text: Vec<TextLine>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    relations: Vec<Relation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    events: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(flatten)]
    other: Map<String, Value>,
    /// Not part of the document: the engine's return value.
    #[serde(skip)]
    tokens: u32,
}

impl Description {
    /// Attributes of the picture as a whole, such as `background`, `surface` or `time_of_day`.
    ///
    /// The model chooses which to write, so look keys up; do not expect a fixed set. Empty when
    /// the scene was not asked for.
    #[inline]
    #[must_use]
    pub const fn scene(&self) -> &Map<String, Value> {
        &self.scene
    }

    /// The things that were found, in the order the engine wrote them.
    ///
    /// Several entries may share one [`Object::id`]: the id names an entity, and an entity with
    /// two instances, such as two trees, has two boxes.
    #[inline]
    #[must_use]
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    /// Things the model names but did not place with a box.
    #[inline]
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The text that was read, one entry per line.
    #[inline]
    #[must_use]
    pub fn text(&self) -> &[TextLine] {
        &self.text
    }

    /// How the objects relate, by [`Object::id`].
    #[inline]
    #[must_use]
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    /// What is happening in the picture, as the engine wrote it.
    ///
    /// The shape of an event is not documented upstream and none was produced while this crate
    /// was written, so events are passed through as JSON.
    #[inline]
    #[must_use]
    pub fn events(&self) -> &[Value] {
        &self.events
    }

    /// The model's short account of how it read the picture, when it was asked for.
    ///
    /// Free text for a log or a second model, with parts separated by ` | `. It is not a
    /// reliable statement about the other sections: on a picture whose text was read correctly
    /// it has said "cannot read the sign's text".
    #[inline]
    #[must_use]
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Top-level keys of the document that this crate has no accessor for.
    #[inline]
    #[must_use]
    pub const fn other(&self) -> &Map<String, Value> {
        &self.other
    }

    /// How many tokens the model generated for this description.
    ///
    /// Zero for a description parsed from text.
    #[inline]
    #[must_use]
    pub const fn tokens(&self) -> u32 {
        self.tokens
    }

    /// The first object with this id.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::Description;
    ///
    /// let description: Description = r#"{
    ///     "objects": [{"id": 1, "name": "sign"}, {"id": 2, "name": "pole"}],
    ///     "relations": [{"subject": 1, "predicate": "attached_to", "object": 2}]
    /// }"#
    /// .parse()?;
    ///
    /// let relation = &description.relations()[0];
    /// let subject = description.object(relation.subject()).map(|o| o.name());
    /// let object = description.object(relation.object()).map(|o| o.name());
    /// assert_eq!((subject, relation.predicate(), object), (Some("sign"), "attached_to", Some("pole")));
    /// # Ok::<(), cactus_rs::Error>(())
    /// ```
    #[must_use]
    pub fn object(&self, id: u32) -> Option<&Object> {
        self.objects.iter().find(|object| object.id == id)
    }

    /// Records the engine's return value on a parsed description.
    pub(crate) fn with_tokens(mut self, tokens: u32) -> Self {
        self.tokens = tokens;
        self
    }
}

/// Parses the document `cactus_describe` writes.
///
/// # Errors
///
/// Returns [`Error::Envelope`] when the text is not that document.
impl FromStr for Description {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|source| Error::envelope(text, source))
    }
}

/// One thing that was found.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Description;
///
/// let description: Description = r#"{"objects": [{
///     "id": 1, "name": "stop sign", "score": 0.732, "position": "center",
///     "description": {"color": "red", "shape": "octagonal"}
/// }]}"#
/// .parse()?;
///
/// let sign = &description.objects()[0];
/// assert_eq!(sign.position(), Some("center"));
/// assert_eq!(sign.description()["color"], "red");
/// assert_eq!(sign.bbox(), None);
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Object {
    id: u32,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bbox: Option<BBox>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mask: Option<Mask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    point: Option<[u16; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    position: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    description: Map<String, Value>,
    #[serde(flatten)]
    other: Map<String, Value>,
}

impl Object {
    /// The entity this box belongs to; what [`Relation`]s refer to.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// What the object is, in the language asked for.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the object is, if boxes were asked for.
    #[inline]
    #[must_use]
    pub const fn bbox(&self) -> Option<BBox> {
        self.bbox
    }

    /// How sure the detector is, from 0 to 1, if scores were asked for.
    #[inline]
    #[must_use]
    pub const fn score(&self) -> Option<f32> {
        self.score
    }

    /// The object's outline or bitmap, if masks were asked for.
    #[inline]
    #[must_use]
    pub const fn mask(&self) -> Option<&Mask> {
        self.mask.as_ref()
    }

    /// One `[x, y]` point inside the object on the 0 to 1000 scale, if points were asked for.
    #[inline]
    #[must_use]
    pub const fn point(&self) -> Option<[u16; 2]> {
        self.point
    }

    /// Where the object is, in words, if positions were asked for.
    #[inline]
    #[must_use]
    pub fn position(&self) -> Option<&str> {
        self.position.as_deref()
    }

    /// The object's attributes, such as `color`, `shape` or `material`.
    ///
    /// The model chooses which to write. Empty when descriptions were not asked for.
    #[inline]
    #[must_use]
    pub const fn description(&self) -> &Map<String, Value> {
        &self.description
    }

    /// Keys of the object that this crate has no accessor for, such as `count`.
    #[inline]
    #[must_use]
    pub const fn other(&self) -> &Map<String, Value> {
        &self.other
    }
}

/// Something the model names but did not place with a box.
///
/// An object whose detection scored below the threshold appears here instead of under
/// [`Description::objects`].
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Description;
///
/// let description: Description =
///     r#"{"regions": [{"id": 2, "name": "pole", "position": "below sign"}]}"#.parse()?;
/// assert_eq!(description.regions()[0].name(), "pole");
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Region {
    id: u32,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    position: Option<String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    description: Map<String, Value>,
    #[serde(flatten)]
    other: Map<String, Value>,
}

impl Region {
    /// The entity's id, shared with objects and relations.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// What the region is.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where it is, in words.
    #[inline]
    #[must_use]
    pub fn position(&self) -> Option<&str> {
        self.position.as_deref()
    }

    /// Its attributes, as for [`Object::description`].
    #[inline]
    #[must_use]
    pub const fn description(&self) -> &Map<String, Value> {
        &self.description
    }

    /// Keys of the region that this crate has no accessor for.
    #[inline]
    #[must_use]
    pub const fn other(&self) -> &Map<String, Value> {
        &self.other
    }
}

/// One line of text read from the picture.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Description;
///
/// let description: Description =
///     r#"{"text": [{"string": "STOP", "bbox": [370, 283, 632, 407], "score": 0.79}]}"#.parse()?;
/// let line = &description.text()[0];
/// assert_eq!(line.string(), "STOP");
/// assert_eq!(line.bbox().map(|b| b.width()), Some(262));
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TextLine {
    string: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bbox: Option<BBox>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
}

impl TextLine {
    /// The text as it is written in the picture.
    #[inline]
    #[must_use]
    pub fn string(&self) -> &str {
        &self.string
    }

    /// Where the line is, if boxes were asked for.
    #[inline]
    #[must_use]
    pub const fn bbox(&self) -> Option<BBox> {
        self.bbox
    }

    /// How sure the reader is, from 0 to 1, if scores were asked for.
    #[inline]
    #[must_use]
    pub const fn score(&self) -> Option<f32> {
        self.score
    }
}

/// One statement about two entities: subject, predicate, object.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Description;
///
/// let description: Description =
///     r#"{"relations": [{"subject": 1, "predicate": "attached_to", "object": 2}]}"#.parse()?;
/// let relation = &description.relations()[0];
/// assert_eq!((relation.subject(), relation.predicate(), relation.object()), (1, "attached_to", 2));
/// # Ok::<(), cactus_rs::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Relation {
    subject: u32,
    predicate: String,
    object: u32,
}

impl Relation {
    /// The id of the entity the statement is about.
    #[inline]
    #[must_use]
    pub const fn subject(&self) -> u32 {
        self.subject
    }

    /// The relation, such as `attached_to` or `standing_on`.
    #[inline]
    #[must_use]
    pub fn predicate(&self) -> &str {
        &self.predicate
    }

    /// The id of the entity it relates to.
    #[inline]
    #[must_use]
    pub const fn object(&self) -> u32 {
        self.object
    }
}

/// A box on the 0 to 1000 scale: left, top, right, bottom.
///
/// In JSON it is the array `[left, top, right, bottom]`.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::BBox;
///
/// let bbox = BBox::new(250, 100, 750, 900);
/// assert_eq!((bbox.width(), bbox.height()), (500, 800));
/// assert_eq!(bbox.to_pixels(1920, 1080), [480, 108, 1440, 972]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "[u16; 4]", into = "[u16; 4]")]
pub struct BBox {
    left: u16,
    top: u16,
    right: u16,
    bottom: u16,
}

impl BBox {
    /// A box from its four edges on the 0 to 1000 scale.
    #[must_use]
    pub const fn new(left: u16, top: u16, right: u16, bottom: u16) -> Self {
        BBox {
            left,
            top,
            right,
            bottom,
        }
    }

    /// The left edge.
    #[inline]
    #[must_use]
    pub const fn left(self) -> u16 {
        self.left
    }

    /// The top edge.
    #[inline]
    #[must_use]
    pub const fn top(self) -> u16 {
        self.top
    }

    /// The right edge.
    #[inline]
    #[must_use]
    pub const fn right(self) -> u16 {
        self.right
    }

    /// The bottom edge.
    #[inline]
    #[must_use]
    pub const fn bottom(self) -> u16 {
        self.bottom
    }

    /// The width on the 0 to 1000 scale; zero if the edges are out of order.
    #[inline]
    #[must_use]
    pub const fn width(self) -> u16 {
        self.right.saturating_sub(self.left)
    }

    /// The height on the 0 to 1000 scale; zero if the edges are out of order.
    #[inline]
    #[must_use]
    pub const fn height(self) -> u16 {
        self.bottom.saturating_sub(self.top)
    }

    /// The box as `[left, top, right, bottom]` in pixels of an image of this size, rounded to
    /// the nearest pixel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::mirage::BBox;
    ///
    /// assert_eq!(BBox::new(0, 0, 1000, 1000).to_pixels(640, 480), [0, 0, 640, 480]);
    /// ```
    #[must_use]
    pub const fn to_pixels(self, width: u32, height: u32) -> [u32; 4] {
        [
            to_pixel(self.left, width),
            to_pixel(self.top, height),
            to_pixel(self.right, width),
            to_pixel(self.bottom, height),
        ]
    }
}

/// One coordinate on the 0 to 1000 scale as a pixel of an axis `extent` long, rounded.
const fn to_pixel(coordinate: u16, extent: u32) -> u32 {
    let scale = SCALE as u64;
    ((coordinate as u64 * extent as u64 + scale / 2) / scale) as u32
}

impl From<[u16; 4]> for BBox {
    fn from([left, top, right, bottom]: [u16; 4]) -> Self {
        BBox::new(left, top, right, bottom)
    }
}

impl From<BBox> for [u16; 4] {
    fn from(bbox: BBox) -> Self {
        [bbox.left, bbox.top, bbox.right, bbox.bottom]
    }
}

/// An object's shape, in the [`MaskFormat`](super::MaskFormat) that was asked for.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Mask;
///
/// let mask: Mask = serde_json::from_str("[426, 178, 361, 266, 360, 422]").unwrap();
/// assert_eq!(mask.outlines(), vec![vec![[426, 178], [361, 266], [360, 422]]]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum Mask {
    /// One outline as `x, y, x, y, ...` on the 0 to 1000 scale: what
    /// [`MaskFormat::Largest`](super::MaskFormat::Largest) and
    /// [`MaskFormat::Merged`](super::MaskFormat::Merged) give.
    Outline(Vec<u16>),
    /// One such outline per connected part: what
    /// [`MaskFormat::Polygons`](super::MaskFormat::Polygons) gives.
    Outlines(Vec<Vec<u16>>),
    /// Every pixel of the image, run-length encoded: what
    /// [`MaskFormat::Bitmap`](super::MaskFormat::Bitmap) gives.
    Bitmap(Bitmap),
}

impl Mask {
    /// The outlines as lists of `[x, y]` points on the 0 to 1000 scale.
    ///
    /// One list for [`Mask::Outline`], one per part for [`Mask::Outlines`], and none for a
    /// [`Mask::Bitmap`]. A trailing coordinate without a partner is dropped.
    #[must_use]
    pub fn outlines(&self) -> Vec<Vec<[u16; 2]>> {
        let points = |flat: &Vec<u16>| {
            flat.chunks_exact(2)
                .map(|pair| [pair[0], pair[1]])
                .collect::<Vec<_>>()
        };
        match self {
            Mask::Outline(flat) => vec![points(flat)],
            Mask::Outlines(parts) => parts.iter().map(points).collect(),
            Mask::Bitmap(_) => Vec::new(),
        }
    }

    /// The bitmap, for a [`Mask::Bitmap`].
    #[must_use]
    pub const fn bitmap(&self) -> Option<&Bitmap> {
        match self {
            Mask::Bitmap(bitmap) => Some(bitmap),
            _ => None,
        }
    }
}

/// A mask at the size of the image, run-length encoded column by column.
///
/// `counts` alternates runs of pixels outside and inside the mask, starting with a run outside
/// (which may be zero long), walking down the first column, then the second, and so on. This is
/// the uncompressed run-length encoding the COCO dataset uses. [`Bitmap::to_rows`] undoes it.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::mirage::Bitmap;
///
/// // A 3 x 2 image (two rows, three columns) with the middle column set.
/// let bitmap: Bitmap = serde_json::from_str(r#"{"size": [2, 3], "counts": [2, 2, 2]}"#).unwrap();
/// assert_eq!((bitmap.width(), bitmap.height()), (3, 2));
/// assert_eq!(
///     bitmap.to_rows(),
///     Some(vec![false, true, false, false, true, false])
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Bitmap {
    /// `[height, width]`, in that order.
    size: [u32; 2],
    counts: Vec<u32>,
}

impl Bitmap {
    /// The width of the bitmap in pixels: the width of the image.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.size[1]
    }

    /// The height of the bitmap in pixels: the height of the image.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.size[0]
    }

    /// The runs as the engine wrote them: outside, inside, outside, ..., column by column.
    #[inline]
    #[must_use]
    pub fn counts(&self) -> &[u32] {
        &self.counts
    }

    /// How many pixels are inside the mask.
    #[must_use]
    pub fn area(&self) -> u64 {
        self.counts
            .iter()
            .skip(1)
            .step_by(2)
            .map(|run| u64::from(*run))
            .sum()
    }

    /// The mask as one `bool` per pixel, row by row from the top: `width * height` of them.
    ///
    /// Returns `None` when the runs do not add up to exactly `width * height` pixels, or when
    /// that count does not fit in memory.
    #[must_use]
    pub fn to_rows(&self) -> Option<Vec<bool>> {
        let width = usize::try_from(self.width()).ok()?;
        let height = usize::try_from(self.height()).ok()?;
        let pixels = width.checked_mul(height)?;
        let total = self
            .counts
            .iter()
            .try_fold(0_u64, |sum, run| sum.checked_add(u64::from(*run)))?;
        if total != pixels as u64 {
            return None;
        }

        let mut rows = vec![false; pixels];
        let mut at = 0_usize;
        for (index, run) in self.counts.iter().enumerate() {
            let run = *run as usize;
            if index % 2 == 1 {
                // `at` walks the image column by column; `rows` is stored row by row.
                for position in at..at + run {
                    rows[(position % height) * width + position / height] = true;
                }
            }
            at += run;
        }
        Some(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The document the pinned engine wrote for the test image, abridged to two objects.
    const MEASURED: &str = r#"{"scene":{"background":"blue sky","surface":"green grass","style":"flat vector illustration","time_of_day":"day"},"objects":[{"id":1,"name":"stop sign","bbox":[361,174,642,517],"score":0.732,"mask":[426,178,361,266,360,422,395,470,434,520,566,521,573,516,638,432,641,422,640,266,587,193,574,177],"position":"center","description":{"color":"red","shape":"octagonal","text":"white"}},{"id":2,"name":"pole","bbox":[483,572,514,825],"score":0.384,"mask":[488,577,487,589,490,672,487,818,496,823,504,823,512,821,514,818,513,703,515,578,512,574,504,573,496,573],"position":"below sign","description":{"color":"gray","material":"metal"}}],"regions":[],"text":[{"string":"STOP","bbox":[370,283,632,407],"score":0.790}],"relations":[{"subject":1,"predicate":"attached_to","object":2}],"events":[],"reasoning":"lang: en | red octagonal stop sign with white text on gray pole"}"#;

    #[test]
    fn the_measured_document_parses_in_full() {
        let description: Description = MEASURED.parse().expect("the engine's own output");

        assert_eq!(description.scene()["time_of_day"], "day");
        assert_eq!(description.objects().len(), 2);
        assert!(description.regions().is_empty());
        assert!(description.events().is_empty());
        assert!(description.other().is_empty());
        assert_eq!(description.tokens(), 0);
        assert!(
            description
                .reasoning()
                .is_some_and(|r| r.starts_with("lang: en"))
        );

        let sign = &description.objects()[0];
        assert_eq!((sign.id(), sign.name()), (1, "stop sign"));
        assert_eq!(sign.bbox(), Some(BBox::new(361, 174, 642, 517)));
        assert_eq!(sign.score(), Some(0.732));
        assert_eq!(sign.position(), Some("center"));
        assert_eq!(sign.description()["shape"], "octagonal");
        assert!(sign.other().is_empty());
        assert_eq!(sign.mask().map(|m| m.outlines()[0].len()), Some(12));

        let line = &description.text()[0];
        assert_eq!(line.string(), "STOP");
        assert_eq!(line.score(), Some(0.79));

        let relation = &description.relations()[0];
        assert_eq!(description.object(relation.subject()), Some(sign));
        assert_eq!(description.object(9), None);
    }

    #[test]
    fn a_section_that_was_not_asked_for_is_empty() {
        let description: Description = r#"{"text":[{"string":"STOP"}]}"#.parse().expect("text");
        assert!(description.scene().is_empty());
        assert!(description.objects().is_empty());
        assert_eq!(description.reasoning(), None);
        assert_eq!(description.text()[0].bbox(), None);
        assert_eq!(description.text()[0].score(), None);

        let empty: Description = "{}".parse().expect("no sections at all");
        assert_eq!(empty, Description::default());
    }

    #[test]
    fn unknown_keys_are_kept_not_lost() {
        let description: Description =
            r#"{"objects":[{"id":1,"name":"tree","count":2}],"later":true}"#
                .parse()
                .expect("a document from a later engine");
        assert_eq!(description.other()["later"], true);
        assert_eq!(description.objects()[0].other()["count"], 2);
    }

    #[test]
    fn a_description_survives_a_round_trip() {
        let description: Description = MEASURED.parse().expect("the engine's own output");
        let written = serde_json::to_string(&description).expect("plain data");
        let read: Description = written.parse().expect("what was just written");
        assert_eq!(read, description);
    }

    #[test]
    fn text_that_is_not_a_description_is_an_envelope_error() {
        for text in [
            "",
            "not json",
            r#"{"objects":{}}"#,
            r#"{"objects":[{"id":1}]}"#,
        ] {
            let error = text.parse::<Description>().expect_err(text);
            assert!(matches!(error, Error::Envelope { .. }), "{text}");
        }
    }

    #[test]
    fn every_mask_format_has_its_variant() {
        let outline: Mask = serde_json::from_str("[1,2,3,4,5,6]").expect("flat pairs");
        assert_eq!(outline, Mask::Outline(vec![1, 2, 3, 4, 5, 6]));
        assert_eq!(outline.outlines(), vec![vec![[1, 2], [3, 4], [5, 6]]]);
        assert_eq!(outline.bitmap(), None);

        let outlines: Mask = serde_json::from_str("[[1,2,3,4],[5,6,7,8,9]]").expect("parts");
        assert_eq!(
            outlines.outlines(),
            vec![vec![[1, 2], [3, 4]], vec![[5, 6], [7, 8]]]
        );

        let bitmap: Mask =
            serde_json::from_str(r#"{"size":[2,2],"counts":[1,2,1]}"#).expect("a bitmap");
        assert!(bitmap.outlines().is_empty());
        assert_eq!(bitmap.bitmap().map(Bitmap::area), Some(2));
    }

    #[test]
    fn a_bitmap_is_decoded_column_by_column() {
        // Height 3, width 2. Runs: 1 out, 3 in, 2 out. Column 0 is rows 0..3, column 1 the rest.
        let bitmap: Bitmap =
            serde_json::from_str(r#"{"size":[3,2],"counts":[1,3,2]}"#).expect("a bitmap");
        assert_eq!(
            bitmap.to_rows(),
            Some(vec![false, true, true, false, true, false])
        );
        assert_eq!(bitmap.area(), 3);
    }

    #[test]
    fn runs_that_do_not_fill_the_image_are_refused() {
        for counts in ["[1,3,1]", "[1,3,3]", "[]", "[4294967295,4294967295,7]"] {
            let bitmap: Bitmap =
                serde_json::from_str(&format!(r#"{{"size":[3,2],"counts":{counts}}}"#))
                    .expect("a bitmap");
            assert_eq!(bitmap.to_rows(), None, "{counts}");
        }
        let empty: Bitmap = serde_json::from_str(r#"{"size":[0,0],"counts":[]}"#).expect("empty");
        assert_eq!(empty.to_rows(), Some(Vec::new()));
    }

    #[test]
    fn boxes_scale_to_any_image() {
        let bbox = BBox::new(361, 174, 642, 517);
        assert_eq!(bbox.to_pixels(512, 384), [185, 67, 329, 199]);
        assert_eq!(bbox.to_pixels(1000, 1000), [361, 174, 642, 517]);
        assert_eq!(bbox.to_pixels(0, 0), [0, 0, 0, 0]);
        assert_eq!(
            BBox::new(1000, 1000, 1000, 1000).to_pixels(u32::MAX, u32::MAX),
            [u32::MAX; 4]
        );
        assert_eq!(
            (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
            (361, 174, 642, 517)
        );
        // Edges out of order give no negative size.
        assert_eq!(BBox::new(9, 9, 1, 1).width(), 0);
        assert_eq!(BBox::new(9, 9, 1, 1).height(), 0);
    }
}
