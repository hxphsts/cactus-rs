//! What the engine costs, measured through the safe API.
//!
//! Three groups, in the order they must run:
//!
//! - `complete`: one tool-call turn, rewound first, which is the loop a request handler runs
//! - `embed`: one short phrase through the same loaded model
//! - `prefix`: [`NeedleBuilder::build`] and the matching drop, swept over the number of tools,
//!   which is the standing cost paid once per process rather than per turn
//!
//! The engine is one process-global model, so the whole run shares a single [`Needle`] for the
//! first two groups and drops it before the third, which builds and drops its own.
//!
//! Every iteration is tens of milliseconds, so the sample size is ten and the measurement window
//! short. Run it with `cargo bench -p cactus-rs -- --quick`, and set `CACTUS_NEEDLE_WEIGHTS` to a
//! local `needle3.cact`; with no archive to hand the benchmarks print a line and do nothing,
//! because a benchmark run is not a reason to download 35 MB.

use std::env;
use std::hint::black_box;
use std::time::Duration;

use cactus_rs::needle::{Needle, Tool, Weights};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::json;

/// The system prompt every configuration here shares.
const SYSTEM: &str = "You control the lights.";

/// The query the `complete` group runs, chosen because it produces exactly one grounded call.
const QUERY: &str = "turn the kitchen light on";

/// Tool counts the `prefix` sweep covers: one declaration, and a handful.
const TOOL_COUNTS: [usize; 2] = [1, 5];

/// The weight archive, or `None` when this machine has none to hand.
fn weights() -> Option<Weights> {
    let path = env::var_os("CACTUS_NEEDLE_WEIGHTS")?;
    Weights::from_file(path).ok()
}

/// One tool declaration, numbered so a sweep can ask for several distinct ones.
fn tool(index: usize) -> Tool {
    Tool::new(
        format!("set_light_{index}"),
        "Turn a room light on or off",
        json!({
            "type": "object",
            "properties": { "room": { "type": "string" }, "on": { "type": "boolean" } },
            "required": ["room", "on"]
        }),
    )
}

fn engine_benchmarks(c: &mut Criterion) {
    let Some(weights) = weights() else {
        println!("skipping: set CACTUS_NEEDLE_WEIGHTS to a needle3.cact to benchmark the engine");
        return;
    };

    {
        let mut needle = Needle::builder(weights.clone())
            .system(SYSTEM)
            .tool(tool(0))
            .build()
            .expect("the engine is free");

        let mut group = c.benchmark_group("complete");
        group.bench_function("tool_call", |bencher| {
            bencher.iter(|| {
                needle.reset();
                black_box(needle.complete(QUERY).expect("the engine answers"))
            });
        });
        group.finish();

        let mut group = c.benchmark_group("embed");
        group.bench_function("short_phrase", |bencher| {
            bencher.iter(|| black_box(needle.embed("kitchen").expect("the model embeds")));
        });
        group.finish();
    }
    // The shared Needle is gone, so the sweep below may take the engine for itself.

    let mut group = c.benchmark_group("prefix");
    for count in TOOL_COUNTS {
        let tools: Vec<Tool> = (0..count).map(tool).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &tools,
            |bencher, tools| {
                bencher.iter(|| {
                    // The clone is part of what a build costs here: `build` takes the archive by
                    // value and fingerprints it to decide whether the engine already has it.
                    let needle = Needle::builder(weights.clone())
                        .system(SYSTEM)
                        .tools(tools.clone())
                        .build()
                        .expect("the engine is free");
                    black_box(needle.prefix_tokens())
                    // and dropped here, which rewinds the conversation and frees the engine
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = engine_benchmarks
}
criterion_main!(benches);
