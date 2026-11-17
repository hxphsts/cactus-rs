//! Describe an image with Mirage, then ask it a question with a fixed set of answers.
//!
//! Run it with `cargo run --example describe --features image`, which describes the vendored
//! `tests/data/images/stop.png`, or name your own PNG or JPEG, what to decide and the labels to
//! choose from:
//!
//! ```text
//! cargo run --example describe --features image -- photo.jpg animal cat dog bird
//! ```
//!
//! The name of the question matters: the model reads it. On the vendored image, `sign` with
//! `stop`, `yield` and `speed limit` answers `stop`; the same labels under the name `what`
//! answer `yield`.
//!
//! The weights are fetched once and cached under `~/.cache/cactus-rs`; set
//! `CACTUS_MIRAGE_WEIGHTS` to a local `mirage.cact` to skip the download.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use cactus_rs::mirage::{Form, Image, Mirage, Weights};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/images/stop.png"),
        PathBuf::from,
    );
    let question = args.next().unwrap_or_else(|| "sign".to_owned());
    let mut labels: Vec<String> = args.collect();
    if labels.is_empty() {
        labels = ["stop", "yield", "speed limit"].map(String::from).to_vec();
    }

    let image = Image::open(&path)?;
    let (width, height) = (image.width(), image.height());
    let mut mirage = Mirage::builder(Weights::fetch()?).build()?; // cached under ~/.cache/cactus-rs

    // One pass: the scene, the objects, the text and how the objects relate.
    let started = Instant::now();
    let description = mirage.describe(&image)?;
    println!(
        "{width} x {height}, {} tokens in {:.0} ms",
        description.tokens(),
        started.elapsed().as_secs_f64() * 1000.0
    );

    for (key, value) in description.scene() {
        println!("scene     {key}: {}", value.as_str().unwrap_or_default());
    }
    for object in description.objects() {
        // Positions are on a 0 to 1000 scale; `to_pixels` brings a box back to the image.
        let pixels = object.bbox().map(|bbox| bbox.to_pixels(width, height));
        println!(
            "object    #{} {} {:?} score {:.2}",
            object.id(),
            object.name(),
            pixels.unwrap_or_default(),
            object.score().unwrap_or_default()
        );
    }
    for line in description.text() {
        println!("text      {:?}", line.string());
    }
    for relation in description.relations() {
        let name = |id| description.object(id).map_or("?", |object| object.name());
        println!(
            "relation  {} {} {}",
            name(relation.subject()),
            relation.predicate(),
            name(relation.object())
        );
    }

    // A decision: which of the labels fits, how sure the model is, and where the thing is.
    let form = Form::new()
        .choice(&question, &labels)
        .bbox("where", &question);
    let started = Instant::now();
    let answers = mirage.extract(&image, &form)?;
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;

    let answer = answers
        .get(&question)
        .ok_or("the form asked the question")?;
    match answer.as_choice() {
        Some(label) => println!(
            "decision  {question}: {label} (p = {:.2}) in {elapsed:.0} ms",
            answer.probability().unwrap_or_default()
        ),
        None => println!("decision  {question}: none of {labels:?} in {elapsed:.0} ms"),
    }
    for (label, probability) in answer.probabilities() {
        println!("          {probability:.3}  {label}");
    }
    if let Some(bbox) = answers.bbox("where") {
        println!("where     {:?}", bbox.to_pixels(width, height));
    }

    Ok(())
}
