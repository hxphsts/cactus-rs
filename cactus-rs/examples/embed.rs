//! Embeddings: three short strings, and how close the model thinks they are.
//!
//! The model that answers with tool calls also turns text into vectors, so a router can compare
//! a request against known phrases without loading a second model. Two of the strings below are
//! the same request in different words; the third asks for something else.
//!
//! Read the numbers with one caveat: Needle's vectors are not centred, so every pair of short
//! English sentences scores around 0.9 and it is the *ordering* that carries the signal, not the
//! absolute value. On one measured run the related pair scored 0.958 and
//! the two unrelated pairs 0.891 and 0.904: a real but narrow margin of 0.054. Subtract a
//! baseline before thresholding on it.
//!
//! Run it with `cargo run --example embed`. The weights are fetched once and cached under
//! `~/.cache/cactus-rs`; set `CACTUS_NEEDLE_WEIGHTS` to a local `needle3.cact` to skip the
//! download.

use cactus_rs::needle::{Needle, Weights};

/// The cosine of the angle between two vectors: 1.0 for the same direction, 0.0 for unrelated.
fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|a| a * a).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|b| b * b).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm * right_norm)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut needle = Needle::builder(Weights::fetch()?).build()?;
    println!("dimension {}", needle.embedding_dimension()?);

    let phrases = [
        "send an email to Ada",
        "email Ada a message",
        "close the bedroom blinds",
    ];

    let vectors = phrases
        .iter()
        .map(|phrase| needle.embed(phrase))
        .collect::<Result<Vec<_>, _>>()?;

    for (i, left) in phrases.iter().enumerate() {
        for (j, right) in phrases.iter().enumerate().skip(i + 1) {
            println!(
                "{:.3}  {left}  <->  {right}",
                cosine(&vectors[i], &vectors[j])
            );
        }
    }

    let related = cosine(&vectors[0], &vectors[1]);
    let unrelated = cosine(&vectors[0], &vectors[2]).max(cosine(&vectors[1], &vectors[2]));
    println!(
        "related beats the closest unrelated pair by {:.3}",
        related - unrelated
    );

    Ok(())
}
