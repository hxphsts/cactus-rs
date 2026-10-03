//! Which of the two models a weight archive, an engine slot or an error is about.
//!
//! ## Overview
//!
//! The linked engine holds one model per kind: a text model (Needle) and a speech model
//! (Whistle). Both ship as `.cact` archives with the same magic tag, so the crate names the model
//! wherever the two could be confused, most visibly in
//! [`Error::WrongModel`](crate::Error::WrongModel).
//!
//! ## Usage
//!
//! ```rust
//! use cactus_rs::Model;
//!
//! assert_eq!(Model::Whistle.archive(), "whistle.cact");
//! assert_eq!(Model::Needle.to_string(), "Needle");
//! ```

use std::fmt;

/// One of the models the engine can hold.
///
/// The enum is `#[non_exhaustive]`: matching it needs a `_` arm, and a model added by a later
/// engine is not a breaking change.
///
/// # Examples
///
/// ```rust
/// use cactus_rs::Model;
///
/// let model = Model::Needle;
/// assert_eq!(model.as_str(), "Needle");
/// assert_eq!(model.archive(), "needle3.cact");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Model {
    /// Needle 3, the text model: tool calling, structured extraction and text embedding.
    Needle,
    /// Whistle, the speech model: transcription, word timestamps and speech embedding.
    Whistle,
}

impl Model {
    /// The model's name, as it appears in messages.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::Model;
    ///
    /// assert_eq!(Model::Whistle.as_str(), "Whistle");
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Model::Needle => "Needle",
            Model::Whistle => "Whistle",
        }
    }

    /// The file name upstream publishes the model's weights under.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cactus_rs::Model;
    ///
    /// assert_eq!(Model::Needle.archive(), "needle3.cact");
    /// assert_eq!(Model::Whistle.archive(), "whistle.cact");
    /// ```
    #[inline]
    #[must_use]
    pub const fn archive(self) -> &'static str {
        match self {
            Model::Needle => "needle3.cact",
            Model::Whistle => "whistle.cact",
        }
    }
}

/// Writes the model's name, the same text as [`Model::as_str`].
impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_the_name() {
        for model in [Model::Needle, Model::Whistle] {
            assert_eq!(model.to_string(), model.as_str());
        }
    }

    #[test]
    fn archives_are_distinct() {
        assert_ne!(Model::Needle.archive(), Model::Whistle.archive());
    }
}
