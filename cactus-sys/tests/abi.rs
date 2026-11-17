//! The hand-written `extern "C"` block must declare exactly what the vendored header declares.
//!
//! Nothing compiles `include/cactus.h`: the bindings in `src/engine.rs` are written by hand, so a
//! header that moved without them is a silent ABI mismatch. This test reads both files as text,
//! translates each C prototype into the Rust types it must bind to, and compares the two lists
//! function by function. It links nothing and needs no engine archive.

use std::collections::BTreeMap;

const HEADER: &str = include_str!("../include/cactus.h");
const BINDINGS: &str = include_str!("../src/engine.rs");

/// The prefix of every exported function, and of the export macro in upper case.
const PREFIX: &str = "cactus_";

/// A function as `(parameter types, return type)`, in Rust spelling; `""` returns nothing.
type Signature = (Vec<String>, String);

/// Maps one C type, as the header spells it, onto the Rust type that binds it.
fn rust_type(c: &str) -> String {
    let normalised = c.split_whitespace().collect::<Vec<_>>().join(" ");
    let rust = match normalised.replace(" *", "*").as_str() {
        "void" => "",
        "int" => "c_int",
        "char*" => "*mut c_char",
        "float*" => "*mut c_float",
        "const char*" => "*const c_char",
        "const float*" => "*const c_float",
        "const unsigned char*" => "*const c_uchar",
        "unsigned long long" => "c_ulonglong",
        other => panic!("the header uses a C type this test cannot translate: `{other}`"),
    };
    rust.to_owned()
}

/// The header without its `/* ... */` comments.
fn without_comments(source: &str) -> String {
    let mut out = String::new();
    let mut rest = source;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        let end = rest[start..]
            .find("*/")
            .expect("the header has an unterminated comment");
        rest = &rest[start + end + 2..];
    }
    out.push_str(rest);
    out
}

/// Every `CACTUS_API <ret> cactus_<name>(<params>);` in the header.
fn header_signatures() -> BTreeMap<String, Signature> {
    let export = format!("{}API ", PREFIX.to_uppercase());
    let source = without_comments(HEADER);
    let mut found = BTreeMap::new();

    for declaration in source.split(';') {
        // Directives go first: the `#define` of the export macro also contains its name. What
        // is left of the first prototype still carries the `extern "C" {` that opens the block.
        let statement: String = declaration
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join(" ");
        let Some((_, prototype)) = statement.split_once(&export) else {
            continue;
        };
        let (head, params) = prototype
            .split_once('(')
            .expect("a prototype has a parameter list");
        let params = params.trim().trim_end_matches(')').trim();
        let name_at = head.find(PREFIX).expect("a prototype names its function");
        let (ret, name) = head.split_at(name_at);

        let params = if params == "void" {
            Vec::new()
        } else {
            params
                .split(',')
                .map(|param| {
                    // Drop the parameter name: everything after the last `*` or space.
                    let param = param.trim();
                    let cut = param
                        .rfind(['*', ' '])
                        .expect("a parameter has a type and a name");
                    rust_type(&param[..=cut])
                })
                .collect()
        };
        found.insert(name.trim().to_owned(), (params, rust_type(ret)));
    }
    found
}

/// Every `pub fn cactus_<name>(<params>) [-> <ret>];` in the bindings.
fn binding_signatures() -> BTreeMap<String, Signature> {
    let code: String = BINDINGS
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut found = BTreeMap::new();

    for declaration in code.split("pub fn ").skip(1) {
        let declaration = declaration
            .split_once(';')
            .expect("an extern declaration ends with `;`")
            .0;
        let (name, rest) = declaration
            .split_once('(')
            .expect("a declaration has a parameter list");
        let (params, ret) = rest.rsplit_once(')').expect("a parameter list is closed");

        let params = params
            .split(',')
            .map(str::trim)
            .filter(|param| !param.is_empty())
            .map(|param| {
                let (_, ty) = param.split_once(':').expect("a parameter has a type");
                ty.trim().to_owned()
            })
            .collect();
        let ret = ret.trim().trim_start_matches("->").trim().to_owned();
        found.insert(name.trim().to_owned(), (params, ret));
    }
    found
}

#[test]
fn test_bindings_declare_what_the_header_declares() {
    let header = header_signatures();
    let bindings = binding_signatures();

    assert!(
        !header.is_empty(),
        "no prototypes were read from the header"
    );
    assert_eq!(
        header.keys().collect::<Vec<_>>(),
        bindings.keys().collect::<Vec<_>>(),
        "the header and the bindings declare different functions"
    );
    for (name, signature) in &header {
        assert_eq!(
            signature, &bindings[name],
            "`{name}` is declared differently in the header and in the bindings"
        );
    }
}

#[test]
fn test_kind_bits_match_the_header() {
    for (name, value) in [
        ("TEXT", cactus_sys::CACTUS_TEXT),
        ("SPEECH", cactus_sys::CACTUS_SPEECH),
        ("VISION", cactus_sys::CACTUS_VISION),
    ] {
        let define = format!("#define {}{name} {value}", PREFIX.to_uppercase());
        assert!(
            HEADER.lines().any(|line| line.trim() == define),
            "the header does not say `{define}`"
        );
    }
}
