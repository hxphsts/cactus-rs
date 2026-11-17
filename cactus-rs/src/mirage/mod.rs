//! Mirage: scene description, object detection, text reading and forms on the Cactus engine.
//!
//! ## Overview
//!
//! Mirage is Cactus Compute's 51M-parameter vision model. One pass over an image gives a
//! [`Description`]: what the scene is, which objects are in it and where, the text that can be
//! read, and how the objects relate. A [`Form`] asks instead for answers from fixed sets (which
//! of these labels, how many, where) and returns each with its probability.
//!
//! [`Mirage`] loads the model into the same engine as [`Needle`](crate::needle::Needle) and
//! [`Whistle`](crate::whistle::Whistle). The three coexist in one process and share one lock.
//!
//! ## Usage
//!
//! ```no_run
//! # #[cfg(feature = "download")]
//! # fn main() -> Result<(), cactus_rs::Error> {
//! use cactus_rs::mirage::{Form, Image, Mirage, Weights};
//!
//! let mut mirage = Mirage::builder(Weights::fetch()?).build()?; // 21.6 MB, cached
//!
//! // Packed RGB, three bytes per pixel, from your camera or image decoder.
//! let pixels: Vec<u8> = vec![0; 640 * 480 * 3];
//! let image = Image::from_rgb(&pixels, 640, 480)?;
//!
//! let description = mirage.describe(&image)?;
//! for object in description.objects() {
//!     println!("{} at {:?}", object.name(), object.bbox());
//! }
//! for line in description.text() {
//!     println!("reads {:?}", line.string());
//! }
//!
//! let form = Form::new().choice("sign", ["stop", "yield", "speed limit"]);
//! println!("{:?}", mirage.extract(&image, &form)?.choice("sign"));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "download"))]
//! # fn main() {}
//! ```
//!
//! ## What was measured
//!
//! Upstream publishes the C signature of `cactus_describe` and no description of what it
//! writes. Everything this module relies on was measured against the pinned engine, and
//! `cactus-sys/tests/probe_vision.rs` asserts it, so a later engine that behaves differently
//! turns that test red:
//!
//! - positions are on a 0 to 1000 scale along each axis of the image given ([`SCALE`])
//! - an option out of range, an unknown section or field and a form the model cannot answer
//!   are refused with a reason; an options string that is not JSON is silently ignored, which
//!   is why this crate always writes it with `serde_json`
//! - a result larger than the output buffer fails and states the size it needs; it is never
//!   truncated
//! - loading Mirage leaves a loaded Needle and Whistle as they were
//!
//! ## Limits
//!
//! - One [`Mirage`] per process, like the other models.
//! - CPU only. Upstream's 74 ms figure is for a phone NPU; the prebuilt archives this crate
//!   links contain no NPU or GPU backend. On an Apple-silicon laptop a full description of a
//!   512 x 384 image took about 130 ms, the scene alone 70 ms, and a one-field form 45 ms.
//! - Names and descriptions come in English, German, French, Spanish, Italian, Dutch or Polish
//!   ([`Language`]). Text in the picture is reported as written.
//! - The scores and probabilities are the model's own and are not calibrated.
//! - A form answers questions about the scene and the things a description names. It does not
//!   see gestures, faces or actions: it answers such questions from their wording, with high
//!   probability and no relation to the picture. See [`form`] for the measurements and for the
//!   blank-frame test that tells the two apart.

pub mod description;
pub mod engine;
pub mod form;
pub mod image;
pub mod options;
pub mod weights;

pub use self::description::{
    BBox, Bitmap, Description, Mask, Object, Region, Relation, SCALE, TextLine,
};
pub use self::engine::{Mirage, MirageBuilder};
pub use self::form::{ABSTAIN, Answer, AnswerValue, Answers, Form};
pub use self::image::Image;
pub use self::options::{DescribeOptions, Field, MaskFormat, Section};
pub use self::weights::Weights;
pub use crate::whistle::Language;
