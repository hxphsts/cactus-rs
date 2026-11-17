//! Mirage against the live engine, alone and beside Needle and Whistle.
//!
//! These tests are about what the vision model sees in a real image and the rules the C API
//! imposes on it: one vision model per process, one engine shared with the other models, options
//! the engine checks itself, and results that are never truncated.
//!
//! The engine is one process-global runtime, and `cargo test` runs a file's tests on parallel
//! threads, so every test that touches it holds [`ENGINE`] for its whole life. Files run in
//! separate processes, so this lock only has to cover this one.
//!
//! The image is vendored as `tests/data/images/stop.png`; `tests/data/README.md` records how it
//! was drawn and where its shapes are. The assertions are about that geometry, with slack: the
//! model's exact numbers differ a little between platforms. Tests skip when
//! `CACTUS_MIRAGE_WEIGHTS` names no readable archive. Nothing here downloads anything.

use std::env;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use cactus_rs::mirage::{
    self, BBox, DescribeOptions, Field, Form, Image, Language, MaskFormat, Mirage, Section,
};
use cactus_rs::{Error, Model, needle, whistle};

/// Serialises every engine-touching test in this binary.
static ENGINE: Mutex<()> = Mutex::new(());

/// The size of `stop.png`.
const SIZE: (u32, u32) = (512, 384);

/// The octagon in `stop.png` as `[left, top, right, bottom]` pixels.
const SIGN: [u32; 4] = [184, 60, 328, 204];

/// Takes the engine lock, ignoring a poisoning left by an earlier failed test.
fn lock() -> MutexGuard<'static, ()> {
    ENGINE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The Mirage archive, or `None` when this machine has none to hand.
///
/// Only `CACTUS_MIRAGE_WEIGHTS` is consulted: `Weights::fetch()` would reach for the network,
/// and a test suite should not start a download.
fn weights() -> Option<mirage::Weights> {
    let path = env::var_os("CACTUS_MIRAGE_WEIGHTS")?;
    mirage::Weights::from_file(path).ok()
}

/// The line a skipped test prints, so a green run with no engine is not mistaken for a real one.
fn skipped(what: &str) {
    println!("skipping {what}: set CACTUS_MIRAGE_WEIGHTS to a mirage.cact to run it");
}

/// `tests/data/images/stop.png` as packed RGB.
fn stop_sign() -> Image<'static> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "data",
        "images",
        "stop.png",
    ]
    .iter()
    .collect();
    let decoded = image::open(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .into_rgb8();
    assert_eq!(decoded.dimensions(), SIZE, "the fixture changed size");
    Image::from_rgb_vec(decoded.into_raw(), SIZE.0, SIZE.1).expect("decoded RGB matches its size")
}

/// How much of the smaller of two pixel boxes lies inside the other, from 0 to 1.
fn overlap(a: [u32; 4], b: [u32; 4]) -> f64 {
    let width = a[2].min(b[2]).saturating_sub(a[0].max(b[0]));
    let height = a[3].min(b[3]).saturating_sub(a[1].max(b[1]));
    let area = |r: [u32; 4]| f64::from((r[2] - r[0]) * (r[3] - r[1]));
    f64::from(width * height) / area(a).min(area(b))
}

/// Whether a box from the model covers the octagon and little else.
fn is_the_sign(bbox: BBox) -> bool {
    let pixels = bbox.to_pixels(SIZE.0, SIZE.1);
    let area = f64::from((pixels[2] - pixels[0]) * (pixels[3] - pixels[1]));
    let sign = f64::from((SIGN[2] - SIGN[0]) * (SIGN[3] - SIGN[1]));
    overlap(pixels, SIGN) > 0.8 && area < sign * 1.5
}

#[test]
fn the_stop_sign_is_found_read_and_related() {
    let Some(weights) = weights() else {
        return skipped("the full description");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");

    let description = mirage.describe(&stop_sign()).expect("a description");
    println!(
        "{}",
        serde_json::to_string(&description).expect("plain data")
    );

    assert!(description.tokens() > 0);
    assert!(!description.scene().is_empty(), "no scene attributes");

    // The octagon is found as a sign, with a box on it, a score, a mask and a description.
    let sign = description
        .objects()
        .iter()
        .find(|object| object.name().contains("sign") && object.bbox().is_some_and(is_the_sign))
        .expect("an object named as a sign, boxed on the octagon");
    let score = sign.score().expect("scores are on by default");
    assert!((0.0..=1.0).contains(&score), "score {score}");
    assert_eq!(
        sign.description().get("color").and_then(|c| c.as_str()),
        Some("red")
    );
    let outline = &sign.mask().expect("masks are on by default").outlines()[0];
    assert!(outline.len() >= 4, "an outline of {} points", outline.len());
    assert!(outline.iter().flatten().all(|c| *c <= mirage::SCALE));

    // The word on it is read, where it is written.
    let line = description
        .text()
        .iter()
        .find(|line| line.string() == "STOP")
        .expect("the line STOP");
    let text = line
        .bbox()
        .expect("text has a box")
        .to_pixels(SIZE.0, SIZE.1);
    assert!(overlap(text, SIGN) > 0.9, "STOP at {text:?}");

    // Every relation is between entities the description names.
    assert!(!description.relations().is_empty(), "no relations");
    for relation in description.relations() {
        for id in [relation.subject(), relation.object()] {
            let named = description.object(id).is_some()
                || description.regions().iter().any(|region| region.id() == id);
            assert!(named, "relation {relation:?} names an unknown entity");
        }
    }
    assert!(description.reasoning().is_some());
    assert!(description.other().is_empty(), "{:?}", description.other());
}

#[test]
fn sections_and_fields_narrow_the_result() {
    let Some(weights) = weights() else {
        return skipped("the sections test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");
    let image = stop_sign();

    let text_only = DescribeOptions::new().with_sections([Section::Text]);
    let description = mirage
        .describe_with_options(&image, &text_only)
        .expect("text only");
    assert_eq!(description.text().len(), 1);
    assert!(description.objects().is_empty() && description.scene().is_empty());
    assert_eq!(description.reasoning(), None);

    // Boxes only: an object keeps its id and name and gains nothing else.
    let boxes = DescribeOptions::new()
        .with_sections([Section::Objects])
        .with_fields([Field::Bbox]);
    let description = mirage
        .describe_with_options(&image, &boxes)
        .expect("boxes only");
    assert!(!description.objects().is_empty());
    for object in description.objects() {
        assert!(object.bbox().is_some());
        assert_eq!(
            (object.score(), object.mask(), object.position()),
            (None, None, None)
        );
        assert!(object.description().is_empty());
    }

    let point = DescribeOptions::new()
        .with_sections([Section::Objects])
        .with_fields([Field::Point])
        .with_max_objects(1);
    let description = mirage
        .describe_with_options(&image, &point)
        .expect("one point");
    assert_eq!(description.objects().len(), 1);
    assert!(description.objects()[0].point().is_some());

    // The options the Mirage was built with were not changed by any of that.
    assert_eq!(mirage.options(), &DescribeOptions::new());
}

#[test]
fn every_mask_format_arrives_as_its_variant() {
    let Some(weights) = weights() else {
        return skipped("the mask format test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");
    let image = stop_sign();
    let masks = |format| {
        DescribeOptions::new()
            .with_sections([Section::Objects])
            .with_fields([Field::Bbox, Field::Mask])
            .with_max_objects(2)
            .with_mask_format(format)
    };

    for format in [MaskFormat::Largest, MaskFormat::Merged] {
        let description = mirage
            .describe_with_options(&image, &masks(format))
            .expect("one outline per object");
        for object in description.objects() {
            assert!(
                matches!(object.mask(), Some(mirage::Mask::Outline(_))),
                "{format:?}"
            );
        }
    }

    let description = mirage
        .describe_with_options(&image, &masks(MaskFormat::Polygons))
        .expect("outlines per part");
    for object in description.objects() {
        assert!(matches!(object.mask(), Some(mirage::Mask::Outlines(_))));
    }

    // A bitmap is the size of the image, decodes, and lies inside its object's box.
    let description = mirage
        .describe_with_options(&image, &masks(MaskFormat::Bitmap))
        .expect("bitmaps");
    assert!(!description.objects().is_empty());
    for object in description.objects() {
        let bitmap = object
            .mask()
            .and_then(mirage::Mask::bitmap)
            .expect("a bitmap mask");
        assert_eq!((bitmap.width(), bitmap.height()), SIZE);
        let rows = bitmap.to_rows().expect("runs that fill the image");
        assert_eq!(
            rows.iter().filter(|set| **set).count() as u64,
            bitmap.area()
        );
        assert!(bitmap.area() > 0);

        let [left, top, right, bottom] = object
            .bbox()
            .expect("boxes were asked for")
            .to_pixels(SIZE.0, SIZE.1);
        // Two pixels of slack: the box is rounded to the 0 to 1000 scale, the bitmap is not.
        let inside =
            |x: u32, y: u32| x + 2 >= left && x <= right + 2 && y + 2 >= top && y <= bottom + 2;
        for (index, _) in rows.iter().enumerate().filter(|(_, set)| **set) {
            let (x, y) = (index as u32 % SIZE.0, index as u32 / SIZE.0);
            assert!(
                inside(x, y),
                "{} has a mask pixel at {x},{y}",
                object.name()
            );
        }
    }
}

#[test]
fn a_form_decides_and_points() {
    let Some(weights) = weights() else {
        return skipped("the form test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");

    let form = Form::new()
        .choice("sign", ["stop", "yield", "speed limit"])
        .bbox("where", "sign")
        .count("signs", 0, 3)
        .text_up_to("note", 12);
    let answers = mirage.extract(&stop_sign(), &form).expect("answers");
    println!("{answers:?}");

    assert_eq!(answers.len(), 4);
    assert!(answers.tokens() > 0);
    assert_eq!(answers.choice("sign"), Some("stop"));

    let sign = answers.get("sign").expect("asked");
    let labels: Vec<&str> = sign
        .probabilities()
        .iter()
        .map(|(l, _)| l.as_str())
        .collect();
    assert_eq!(labels, ["stop", "yield", "speed limit", mirage::ABSTAIN]);
    let total: f64 = sign.probabilities().iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-6, "probabilities sum to {total}");
    assert!(sign.probability().is_some_and(|p| p > 0.5));

    assert!(
        answers.bbox("where").is_some_and(is_the_sign),
        "{:?}",
        answers.bbox("where")
    );
    assert!(answers.count("signs").is_some_and(|n| (0..=3).contains(&n)));
    assert_eq!(
        answers.get("signs").map(|a| a.probabilities().len()),
        Some(4)
    );

    // Labels that do not fit. The model may abstain or pick the least wrong one (which of the
    // two depends on the wording), so what is checked is that the answer is consistent: no value
    // means no probability, and a value's probability is its share of the distribution.
    let wrong = Form::new().choice("animal", ["cat", "dog"]);
    let answers = mirage.extract(&stop_sign(), &wrong).expect("answers");
    let animal = answers.get("animal").expect("asked");
    let share = |label: &str| {
        animal
            .probabilities()
            .iter()
            .find(|(candidate, _)| candidate == label)
            .map(|(_, p)| *p)
    };
    assert!(share(mirage::ABSTAIN).is_some(), "no abstain share");
    match animal.as_choice() {
        Some(label) => assert_eq!(animal.probability(), share(label)),
        None => assert_eq!(animal.probability(), None),
    }
}

#[test]
fn the_engine_names_what_it_refuses() {
    let Some(weights) = weights() else {
        return skipped("the refusals test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");
    let image = stop_sign();

    for (options, named) in [
        (DescribeOptions::new().with_min_score(2.0), "conf"),
        (
            DescribeOptions::new().with_min_text_score(-1.0),
            "text_conf",
        ),
        (DescribeOptions::new().with_overlap(7.0), "iou"),
        (DescribeOptions::new().with_max_objects(0), "max_objects"),
        (
            DescribeOptions::new().with_max_text_lines(0),
            "max_text_lines",
        ),
        (DescribeOptions::new().with_max_tokens(0), "max_tokens"),
        (
            DescribeOptions::new().with_mask_threshold(7.0),
            "mask_threshold",
        ),
        (
            DescribeOptions::new().with_mask_tolerance(-1.0),
            "mask_tolerance",
        ),
        (
            DescribeOptions::new().with_keyword_bias(101.0),
            "keyword_bias",
        ),
        (DescribeOptions::new().with_min_score(f32::NAN), "conf"),
    ] {
        let error = mirage
            .describe_with_options(&image, &options)
            .expect_err(named);
        let detail = error.detail().unwrap_or_default().to_owned();
        assert!(
            matches!(error, Error::Describe { .. }),
            "{named}: {error:?}"
        );
        assert!(detail.contains(named), "{named} is not named in {detail:?}");
    }

    for (form, named) in [
        (Form::new(), "no fields"),
        (Form::new().choice("a", Vec::<String>::new()), "field a"),
        (Form::new().count("n", 0, 1).count("n", 0, 1), "twice"),
        (Form::new().choice("a;b", ["x", "y"]), "a;b"),
        (Form::new().bbox("where", "nothing"), "where"),
        (Form::new().text_up_to("t", 0), "maxLength"),
    ] {
        let error = mirage.extract(&image, &form).expect_err(named);
        let detail = error.detail().unwrap_or_default().to_owned();
        assert!(
            matches!(error, Error::Describe { .. }),
            "{named}: {error:?}"
        );
        assert!(detail.contains(named), "{named} is not named in {detail:?}");
    }

    // A refusal leaves the model usable.
    assert!(mirage.describe(&image).is_ok());
}

#[test]
fn a_description_can_be_asked_for_in_german() {
    let Some(weights) = weights() else {
        return skipped("the language test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights)
        .options(
            DescribeOptions::new()
                .with_language(Language::De)
                .with_sections([Section::Objects, Section::Text])
                .with_fields([]),
        )
        .build()
        .expect("the vision model");

    let description = mirage.describe(&stop_sign()).expect("a description");
    let names: Vec<&str> = description.objects().iter().map(|o| o.name()).collect();
    assert!(
        names.iter().any(|name| name.contains("schild")),
        "no German sign among {names:?}"
    );
    // The text in the picture is reported as written, not translated.
    assert_eq!(description.text()[0].string(), "STOP");
}

#[test]
fn a_tiny_image_and_a_large_one_are_both_described() {
    let Some(weights) = weights() else {
        return skipped("the image size test");
    };
    let _engine = lock();
    let mut mirage = Mirage::builder(weights).build().expect("the vision model");

    let one = Image::from_rgb(&[143, 180, 217], 1, 1).expect("one pixel");
    assert!(mirage.describe(&one).is_ok());

    // 1920 x 1080 of one colour, with bitmap masks: the result is written at the image's size
    // and must come back whole, whatever the buffer started as.
    let flat = vec![200_u8; 1920 * 1080 * 3];
    let large = Image::from_rgb(&flat, 1920, 1080).expect("a full HD frame");
    let bitmaps = DescribeOptions::new()
        .with_sections([Section::Objects])
        .with_mask_format(MaskFormat::Bitmap);
    let description = mirage
        .describe_with_options(&large, &bitmaps)
        .expect("a description of a large image");
    for object in description.objects() {
        let bitmap = object
            .mask()
            .and_then(mirage::Mask::bitmap)
            .expect("a bitmap");
        assert_eq!((bitmap.width(), bitmap.height()), (1920, 1080));
        assert!(bitmap.to_rows().is_some());
    }
}

#[test]
fn a_second_mirage_finds_the_vision_model_busy() {
    let Some(first) = weights() else {
        return skipped("the busy test");
    };
    let _engine = lock();
    let mirage = Mirage::builder(first).build().expect("the vision model");

    let second = weights().expect("read a moment ago");
    let error = Mirage::builder(second)
        .build()
        .expect_err("the slot is taken");
    assert!(matches!(error, Error::EngineBusy));

    // Dropping the first frees the slot, and the same archive is not loaded again.
    drop(mirage);
    let again = weights().expect("read a moment ago");
    assert!(Mirage::builder(again).build().is_ok());
}

#[test]
fn another_models_archive_is_not_a_mirage() {
    let Some(mirage_weights) = weights() else {
        return skipped("the wrong archive test");
    };
    let _engine = lock();

    for (variable, model) in [
        ("CACTUS_NEEDLE_WEIGHTS", Model::Needle),
        ("CACTUS_WHISTLE_WEIGHTS", Model::Whistle),
    ] {
        let Some(path) = env::var_os(variable) else {
            println!("skipping the {model} half of the wrong archive test: set {variable}");
            continue;
        };
        let other = mirage::Weights::from_file(&path).expect("the tag is shared");
        let error = Mirage::builder(other)
            .build()
            .expect_err("not a Mirage archive");
        assert!(
            matches!(
                error,
                Error::WrongModel {
                    expected: Model::Mirage
                }
            ),
            "{model}: {error:?}"
        );
    }

    // And the other way round: Mirage's archive is refused by the other two builders.
    let path = env::var_os("CACTUS_MIRAGE_WEIGHTS").expect("checked by weights()");
    let as_needle = needle::Weights::from_file(&path).expect("the tag is shared");
    let error = needle::Needle::builder(as_needle)
        .build()
        .expect_err("not a Needle archive");
    assert!(matches!(
        error,
        Error::WrongModel {
            expected: Model::Needle
        }
    ));
    let as_whistle = whistle::Weights::from_file(&path).expect("the tag is shared");
    let error = whistle::Whistle::builder(as_whistle)
        .build()
        .expect_err("not a Whistle archive");
    assert!(matches!(
        error,
        Error::WrongModel {
            expected: Model::Whistle
        }
    ));

    // None of that loaded anything in the vision slot's way.
    assert!(Mirage::builder(mirage_weights).build().is_ok());
}

#[test]
fn all_three_models_live_in_one_process() {
    let Some(mirage_weights) = weights() else {
        return skipped("the three models test");
    };
    let (Some(needle_path), Some(whistle_path)) = (
        env::var_os("CACTUS_NEEDLE_WEIGHTS"),
        env::var_os("CACTUS_WHISTLE_WEIGHTS"),
    ) else {
        println!(
            "skipping the three models test: set CACTUS_NEEDLE_WEIGHTS and \
             CACTUS_WHISTLE_WEIGHTS as well"
        );
        return;
    };
    let _engine = lock();

    // Needle first, so that loading the other two has an init prefix to disturb.
    let mut needle = needle::Needle::builder(
        needle::Weights::from_file(needle_path).expect("the Needle archive"),
    )
    .tool(needle::Tool::new(
        "set_light",
        "Turn a light on or off",
        serde_json::json!({
            "type": "object",
            "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}},
            "required": ["room", "on"]
        }),
    ))
    .build()
    .expect("the text model");
    let before = needle
        .complete("turn on the kitchen light")
        .expect("a completion before the other models load");

    let mut whistle = whistle::Whistle::builder(
        whistle::Weights::from_file(whistle_path).expect("the Whistle archive"),
    )
    .build()
    .expect("the speech model");
    let mut mirage = Mirage::builder(mirage_weights)
        .build()
        .expect("the vision model");

    // Each model answers, in an order that interleaves them, and Needle answers as it did.
    let description = mirage.describe(&stop_sign()).expect("a description");
    assert!(
        description
            .text()
            .iter()
            .any(|line| line.string() == "STOP")
    );
    let transcript = whistle
        .transcribe(&vec![0.0; 16_000])
        .expect("a transcript");
    assert_eq!(transcript.text(), "");
    needle.reset();
    let after = needle
        .complete("turn on the kitchen light")
        .expect("a completion after the other models loaded");
    let names = |completion: &needle::Completion| -> Vec<String> {
        completion
            .calls()
            .iter()
            .map(|call| call.name().to_owned())
            .collect()
    };
    assert_eq!(names(&before), names(&after));
    assert_eq!(names(&after), ["set_light"]);
    assert!(mirage.describe(&stop_sign()).is_ok());
}
