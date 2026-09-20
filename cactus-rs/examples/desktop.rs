//! A simulated desktop agent: twelve tools, a multi-step session, and a real agent loop.
//!
//! Nothing here touches your machine. Every tool acts on a [`Desktop`] value held in memory, so
//! the example can show what the model decided, what that did to the state, and how the result
//! was fed back for the next step.
//!
//! Three things to watch in the output:
//!
//! - **Retrieval**: with more than five tools declared, the engine shows the model only the five
//!   it scores most relevant to each turn. A tool that is not retrieved cannot be called.
//! - **Parallel calls**: one request can come back as several calls.
//! - **The loop**: after the calls run, their results go back in as the next input, and the
//!   model may act again or stop. The same conversation carries across requests until `reset`.
//!
//! Run it with `cargo run -p cactus-rs --example desktop`, or pass your own requests:
//! `cargo run -p cactus-rs --example desktop -- "mute the sound" "open notes"`.
//!
//! The built-in requests know which calls a correct agent makes, so each one ends in `PASS` or
//! `FAIL` with the expected and actual calls. Every request starts a fresh conversation, and the
//! tool-result loop runs inside it. Two flags change the conditions:
//!
//! - `--session` keeps one conversation across all eight requests.
//! - `--focused` declares only the five tools relevant to each request.
//!
//! Measured with Needle 3 on the pinned engine:
//!
//! | Mode | macOS arm64 | Linux x86_64 |
//! | --- | --- | --- |
//! | default, twelve tools | 8/8 | 8/8 |
//! | `--session` | 4/8 | 5/8 |
//! | `--focused` | 7/8 | 7/8 |
//!
//! ## What made it work
//!
//! The first version of these tools scored 6/8 at best. Two schemas were the cause, and the fix
//! was configuration, not code. The model copies surface words from the request, so:
//!
//! - **Spell closed sets the way people speak.** `press_key {key, modifiers: [..]}` lost the
//!   modifier, and a flat optional `modifier` got invented on plain keys. One enum,
//!   `shortcut: ["cmd s", "enter", ..]`, scores 1.00 confidence. Likewise `"left half"` beats
//!   `"left_half"`.
//! - **Give each intent one tool, and say so.** "open the browser on the right half" went to
//!   `open_app` until `put_window` said "opens the application if needed".
//! - **Pin stubborn phrasings with `triggers`.** Two regexes on "left/right half" and "full
//!   screen" settle "snap", "move" and "put" alike.
//! - **Keep conversations short.** The engine holds a 256-token sliding window over an 845-token
//!   pinned prefix. One conversation per request is reliable. One conversation for everything is
//!   not: it halves the score, and it is the only mode where the two machines disagree.
//!
//! Declaring fewer tools did not help. `--focused` hands the model five tools chosen by hand and
//! scores lower than the default, where the engine picks five of the twelve for each turn: its
//! own ranking leaves out distractors that a hand-written list lets in.

use std::collections::BTreeMap;
use std::env;

use cactus_rs::needle::{Call, Needle, Tool, Weights};
use serde_json::{Value, json};

const SYSTEM: &str = "You operate a desktop computer through the declared tools. Map each \
explicit request to the calls that carry it out, in order. Copy text the user gives verbatim. \
Do not guess missing values. Unsupported or negated requests return no call.";

/// Steps the loop may take for one request before it gives up.
const MAX_STEPS: usize = 4;

const APPS: [&str; 6] = ["browser", "notes", "terminal", "mail", "music", "settings"];

/// The tool declarations. Shapes follow upstream's advice: closed sets are enums, numbers have
/// bounds, and free text is copied rather than invented.
fn tools() -> Vec<Tool> {
    let app = json!({ "type": "string", "enum": APPS, "description": "The application." });

    vec![
        Tool::new(
            "open_app",
            "Launch an application and bring it to the front.",
            json!({ "type": "object", "properties": { "app": app }, "required": ["app"] }),
        ),
        Tool::new(
            "close_app",
            "Quit a running application.",
            json!({ "type": "object", "properties": { "app": app }, "required": ["app"] }),
        ),
        Tool::new(
            "type_text",
            "Type text into the focused application, exactly as the user gave it.",
            json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "minLength": 1, "description": "Text to type, verbatim." }
                },
                "required": ["text"]
            }),
        ),
        // One closed set, spelled the way people say it. An array of modifiers lost the
        // modifier, and an optional `modifier` field got invented on plain keys.
        Tool::new(
            "press_shortcut",
            "Press a keyboard shortcut or a single key.",
            json!({
                "type": "object",
                "properties": {
                    "shortcut": {
                        "type": "string",
                        "enum": ["cmd s", "cmd c", "cmd v", "cmd t", "cmd w", "enter", "escape", "tab", "space", "backspace"],
                        "description": "The shortcut, as spoken."
                    }
                },
                "required": ["shortcut"]
            }),
        ),
        Tool::new(
            "click",
            "Click the mouse at a screen position.",
            json!({
                "type": "object",
                "properties": {
                    "x": { "type": "integer", "minimum": 0, "maximum": 1919, "description": "Pixels from the left." },
                    "y": { "type": "integer", "minimum": 0, "maximum": 1079, "description": "Pixels from the top." },
                    "button": { "type": "string", "enum": ["left", "right"], "description": "Mouse button." }
                },
                "required": ["x", "y"]
            }),
        ),
        Tool::new(
            "scroll",
            "Scroll the focused window.",
            json!({
                "type": "object",
                "properties": {
                    "direction": { "type": "string", "enum": ["up", "down"] },
                    "lines": { "type": "integer", "minimum": 1, "maximum": 50 }
                },
                "required": ["direction", "lines"]
            }),
        ),
        Tool::new(
            "set_volume",
            "Set the system sound volume as a percentage. Zero is silent.",
            json!({
                "type": "object",
                "properties": { "percent": { "type": "integer", "minimum": 0, "maximum": 100 } },
                "required": ["percent"]
            }),
        ),
        Tool::new(
            "set_brightness",
            "Set the display brightness as a percentage.",
            json!({
                "type": "object",
                "properties": { "percent": { "type": "integer", "minimum": 5, "maximum": 100 } },
                "required": ["percent"]
            }),
        ),
        Tool::new(
            "toggle_setting",
            "Turn a system setting on or off.",
            json!({
                "type": "object",
                "properties": {
                    "setting": { "type": "string", "enum": ["wifi", "bluetooth", "dark_mode", "do_not_disturb"] },
                    "state": { "type": "string", "enum": ["on", "off"] }
                },
                "required": ["setting", "state"]
            }),
        ),
        Tool::new(
            "open_url",
            "Open a web address in the browser.",
            json!({
                "type": "object",
                "properties": { "url": { "type": "string", "pattern": "^https?://", "description": "Full address." } },
                "required": ["url"]
            }),
        ),
        // Enum values in the user's words, a description that owns "open it over there", and
        // triggers for the phrasings the model otherwise hands to `open_app`.
        Tool::new(
            "put_window",
            "Put an application's window on the left half, the right half, the center or full \
             screen. Opens the application if needed.",
            json!({
                "type": "object",
                "properties": {
                    "app": app,
                    "region": { "type": "string", "enum": ["left half", "right half", "full screen", "center"] }
                },
                "required": ["app", "region"]
            }),
        )
        .with_trigger(r"\b(left|right) half\b")
        .with_trigger(r"\bfull ?screen\b"),
        Tool::new(
            "take_screenshot",
            "Capture the screen to a file.",
            json!({
                "type": "object",
                "properties": { "area": { "type": "string", "enum": ["full", "window"] } },
                "required": ["area"]
            }),
        ),
    ]
}

/// The pretend machine the tools act on.
#[derive(Debug, Default)]
struct Desktop {
    running: Vec<String>,
    focused: Option<String>,
    typed: BTreeMap<String, String>,
    windows: BTreeMap<String, String>,
    settings: BTreeMap<String, String>,
    volume: i64,
    brightness: i64,
    screenshots: usize,
}

impl Desktop {
    fn new() -> Self {
        Self {
            volume: 60,
            brightness: 80,
            ..Self::default()
        }
    }

    /// Carries out one call and returns what a real tool would report back.
    fn execute(&mut self, call: &Call) -> Value {
        let args = call.arguments();
        let text = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_owned);
        let number = |key: &str| args.get(key).and_then(Value::as_i64);

        match call.name() {
            "open_app" => {
                let Some(app) = text("app") else {
                    return missing("app");
                };
                if !self.running.contains(&app) {
                    self.running.push(app.clone());
                }
                self.focused = Some(app.clone());
                json!({ "ok": true, "focused": app })
            }
            "close_app" => {
                let Some(app) = text("app") else {
                    return missing("app");
                };
                if !self.running.contains(&app) {
                    return json!({ "error": format!("{app} is not running") });
                }
                self.running.retain(|running| *running != app);
                if self.focused.as_deref() == Some(app.as_str()) {
                    self.focused = self.running.last().cloned();
                }
                json!({ "ok": true, "closed": app })
            }
            "type_text" => {
                let Some(typed) = text("text") else {
                    return missing("text");
                };
                let Some(app) = self.focused.clone() else {
                    return json!({ "error": "no application is focused" });
                };
                self.typed.entry(app.clone()).or_default().push_str(&typed);
                json!({ "ok": true, "typed_into": app, "characters": typed.chars().count() })
            }
            "press_shortcut" => {
                let Some(shortcut) = text("shortcut") else {
                    return missing("shortcut");
                };
                json!({ "ok": true, "pressed": shortcut })
            }
            "click" => match (number("x"), number("y")) {
                (Some(x), Some(y)) => json!({ "ok": true, "clicked": [x, y] }),
                _ => missing("x, y"),
            },
            "scroll" => json!({ "ok": true }),
            "set_volume" => {
                let Some(percent) = number("percent") else {
                    return missing("percent");
                };
                self.volume = percent;
                json!({ "ok": true, "volume": percent })
            }
            "set_brightness" => {
                let Some(percent) = number("percent") else {
                    return missing("percent");
                };
                self.brightness = percent;
                json!({ "ok": true, "brightness": percent })
            }
            "toggle_setting" => match (text("setting"), text("state")) {
                (Some(setting), Some(state)) => {
                    self.settings.insert(setting.clone(), state.clone());
                    json!({ "ok": true, "setting": setting, "state": state })
                }
                _ => missing("setting, state"),
            },
            "open_url" => {
                let Some(url) = text("url") else {
                    return missing("url");
                };
                if !self.running.iter().any(|app| app == "browser") {
                    self.running.push("browser".to_owned());
                }
                self.focused = Some("browser".to_owned());
                json!({ "ok": true, "loaded": url })
            }
            "put_window" => match (text("app"), text("region")) {
                (Some(app), Some(region)) => {
                    // The tool's contract: it opens the application when it is not running.
                    if !self.running.contains(&app) {
                        self.running.push(app.clone());
                    }
                    self.focused = Some(app.clone());
                    self.windows.insert(app.clone(), region.clone());
                    json!({ "ok": true, "app": app, "region": region })
                }
                _ => missing("app, region"),
            },
            "take_screenshot" => {
                self.screenshots += 1;
                json!({ "ok": true, "file": format!("screenshot-{}.png", self.screenshots) })
            }
            other => json!({ "error": format!("unknown tool: {other}") }),
        }
    }
}

fn missing(what: &str) -> Value {
    json!({ "error": format!("missing argument: {what}") })
}

/// One expected call, in the shape the comparison uses.
fn expect(name: &str, arguments: Value) -> Value {
    json!({ "name": name, "arguments": arguments })
}

/// One request of the built-in session.
struct Request {
    text: &'static str,
    /// The calls a correct agent makes, in order. Empty means "make no call".
    expected: Vec<Value>,
    /// The five tools or fewer that `--focused` declares for this request.
    focus: &'static [&'static str],
}

/// Requests for one session. Later requests lean on what earlier ones did.
fn session() -> Vec<Request> {
    let request = |text, expected, focus| Request {
        text,
        expected,
        focus,
    };

    vec![
        request(
            "open the notes app",
            vec![expect("open_app", json!({ "app": "notes" }))],
            &[
                "open_app",
                "close_app",
                "put_window",
                "type_text",
                "open_url",
            ],
        ),
        request(
            "type \"Buy oat milk and call Priya at 4pm\"",
            vec![expect(
                "type_text",
                json!({ "text": "Buy oat milk and call Priya at 4pm" }),
            )],
            &["type_text", "press_shortcut", "open_app", "click", "scroll"],
        ),
        request(
            "save it with cmd s, then take a screenshot of the window",
            vec![
                expect("press_shortcut", json!({ "shortcut": "cmd s" })),
                expect("take_screenshot", json!({ "area": "window" })),
            ],
            &[
                "press_shortcut",
                "take_screenshot",
                "type_text",
                "click",
                "close_app",
            ],
        ),
        request(
            "put notes on the left half and open the browser on the right half",
            vec![
                expect(
                    "put_window",
                    json!({ "app": "notes", "region": "left half" }),
                ),
                expect(
                    "put_window",
                    json!({ "app": "browser", "region": "right half" }),
                ),
            ],
            &[
                "put_window",
                "open_app",
                "close_app",
                "open_url",
                "take_screenshot",
            ],
        ),
        request(
            "go to https://docs.rs/cactus-rs and scroll down 20 lines",
            vec![
                expect("open_url", json!({ "url": "https://docs.rs/cactus-rs" })),
                expect("scroll", json!({ "direction": "down", "lines": 20 })),
            ],
            &["open_url", "scroll", "open_app", "click", "type_text"],
        ),
        request(
            "turn on do not disturb, set the volume to 15 percent and dim the screen to 40 percent",
            vec![
                expect(
                    "toggle_setting",
                    json!({ "setting": "do_not_disturb", "state": "on" }),
                ),
                expect("set_volume", json!({ "percent": 15 })),
                expect("set_brightness", json!({ "percent": 40 })),
            ],
            &[
                "toggle_setting",
                "set_volume",
                "set_brightness",
                "open_app",
                "take_screenshot",
            ],
        ),
        request(
            "don't close the browser",
            vec![],
            &["close_app", "open_app", "put_window", "open_url", "scroll"],
        ),
        request(
            "order me a pizza",
            vec![],
            &[
                "open_url",
                "open_app",
                "type_text",
                "click",
                "press_shortcut",
            ],
        ),
    ]
}

/// Renders calls one per line for the verdict, or `(no call)` for an empty list.
fn render(calls: &[Value]) -> String {
    if calls.is_empty() {
        return "(no call)".to_owned();
    }
    calls
        .iter()
        .map(|call| {
            format!(
                "{} {}",
                call["name"].as_str().unwrap_or("?"),
                call["arguments"]
            )
        })
        .collect::<Vec<_>>()
        .join("\n             ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // By default every request starts a fresh conversation, which is how upstream's own harness
    // drives the engine. `--session` keeps one conversation across all of them instead, and
    // `--focused` declares only that request's five tools.
    let flag = |name: &str| env::args().any(|arg| arg == name);
    let (session_mode, focused) = (flag("--session"), flag("--focused"));

    // Requests given on the command line have no expectations; the built-in session does.
    let given: Vec<String> = env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"))
        .collect();
    let requests: Vec<(String, Option<Request>)> = if given.is_empty() {
        session()
            .into_iter()
            .map(|request| (request.text.to_owned(), Some(request)))
            .collect()
    } else {
        given.into_iter().map(|text| (text, None)).collect()
    };

    let weights = Weights::fetch()?;
    let all_tools = tools();
    let mut needle = Some(
        Needle::builder(weights.clone())
            .system(SYSTEM)
            .tools(all_tools.clone())
            .build()?,
    );
    println!(
        "{} tools declared, prefix {} tokens\n",
        all_tools.len(),
        needle.as_ref().map_or(0, Needle::prefix_tokens)
    );

    let mut desktop = Desktop::new();
    let mut decode_tps = Vec::new();
    let mut correct = 0_usize;
    let mut scored = 0_usize;

    for (request, known) in &requests {
        println!("> {request}");
        let mut made: Vec<Value> = Vec::new();

        if let (true, Some(known)) = (focused, known) {
            // Only one `Needle` may exist, so the old one goes before the new one is built.
            drop(needle.take());
            let subset = all_tools
                .iter()
                .filter(|tool| known.focus.contains(&tool.name()));
            needle = Some(
                Needle::builder(weights.clone())
                    .system(SYSTEM)
                    .tools(subset.cloned())
                    .build()?,
            );
        }
        let Some(needle) = needle.as_mut() else { break };
        if !session_mode {
            needle.reset();
        }
        let mut input = request.clone();

        for step in 1..=MAX_STEPS {
            let completion = needle.complete(&input)?;
            decode_tps.push(completion.stats().decode_tps());

            if let Some(reasoning) = completion.reasoning() {
                println!("  step {step} thinks: {reasoning}");
            }
            for call in completion.suppressed_calls() {
                println!(
                    "  step {step} suppressed by the engine: {} {}",
                    call.name(),
                    call.arguments_json()
                );
            }
            if !completion.validation().is_grounded() {
                println!(
                    "  step {step} flagged: ungrounded {:?}, negation {}",
                    completion.validation().ungrounded(),
                    completion.validation().negation(),
                );
            }

            let calls = completion.grounded_calls();
            if calls.is_empty() {
                println!(
                    "  step {step}: no call (confidence {:.2})",
                    completion.confidence()
                );
                break;
            }

            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                let result = desktop.execute(call);
                made.push(expect(call.name(), call.arguments().clone()));
                println!(
                    "  step {step}: {} {}  ->  {result}   (confidence {:.2})",
                    call.name(),
                    call.arguments_json(),
                    completion.confidence(),
                );
                results.push(result);
            }

            // The loop: what the tools reported becomes the next input, as upstream's `run` does.
            input = serde_json::to_string(&results)?;
        }

        // Exact match, in order, across every step of the loop. `Value` equality ignores key
        // order, so only the calls, their sequence and their argument values are judged.
        if let Some(expected) = known.as_ref().map(|known| &known.expected) {
            scored += 1;
            if made == *expected {
                correct += 1;
                println!("  PASS");
            } else {
                println!("  FAIL");
                println!("    expected {}", render(expected));
                println!("    got      {}", render(&made));
            }
        }
        println!();
    }

    let mean = decode_tps.iter().sum::<f32>() / decode_tps.len().max(1) as f32;
    println!("final state: {desktop:#?}");
    if scored > 0 {
        println!("{correct}/{scored} requests answered with exactly the expected calls");
    }
    println!(
        "{} engine turns, mean decode {mean:.0} tok/s",
        decode_tps.len()
    );
    Ok(())
}
