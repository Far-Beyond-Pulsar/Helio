//! Source-only architecture gate. No renderer/GPU dependency.
use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

fn tokens(value: &impl ToTokens) -> String {
    value.to_token_stream().to_string()
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("test")
            || (attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Path>()
                    .is_ok_and(|p| p.is_ident("test")))
    })
}

fn fields(fields: &syn::Fields) -> String {
    fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            format!(
                "{}: {}",
                field
                    .ident
                    .as_ref()
                    .map_or(i.to_string(), ToString::to_string),
                tokens(&field.ty)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Default)]
struct Audit {
    file: String,
    shapes: BTreeMap<String, String>,
    violations: BTreeSet<String>,
    forbidden: BTreeSet<String>,
}

impl Audit {
    fn reject(&mut self, detail: impl AsRef<str>) {
        self.violations
            .insert(format!("{}: {}", self.file, detail.as_ref()));
    }

    fn shape(&mut self, name: &syn::Ident, body: String) {
        let key = format!("{}::{name}", self.file);
        if self.shapes.insert(key.clone(), body).is_some() {
            self.reject(format!(
                "duplicate shape {key}; audit conditional definitions explicitly"
            ));
        }
    }

    fn routing(&self) -> bool {
        self.file.contains("/graph/") || self.file == "helio-core/src/registry.rs"
    }
}

impl<'ast> Visit<'ast> for Audit {
    fn visit_item(&mut self, item: &'ast syn::Item) {
        let attrs: &[syn::Attribute] = match item {
            syn::Item::Mod(v) => &v.attrs,
            syn::Item::Fn(v) => &v.attrs,
            syn::Item::Struct(v) => &v.attrs,
            syn::Item::Enum(v) => &v.attrs,
            syn::Item::Type(v) => &v.attrs,
            syn::Item::Const(v) => &v.attrs,
            syn::Item::Impl(v) => &v.attrs,
            syn::Item::Trait(v) => &v.attrs,
            _ => &[],
        };
        if !test_only(attrs) {
            visit::visit_item(self, item);
        }
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.shape(
            &item.ident,
            format!(
                "struct {} {{ {} }}",
                tokens(&item.generics),
                fields(&item.fields)
            ),
        );
        visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        let variants = item
            .variants
            .iter()
            .map(|v| format!("{}({})", v.ident, fields(&v.fields)))
            .collect::<Vec<_>>()
            .join(", ");
        self.shape(
            &item.ident,
            format!("enum {} {{ {variants} }}", tokens(&item.generics)),
        );
        visit::visit_item_enum(self, item);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.shape(
            &item.ident,
            format!("type {} = {}", tokens(&item.generics), tokens(&item.ty)),
        );
        visit::visit_item_type(self, item);
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        self.reject(format!(
            "union {} requires an architecture review",
            item.ident
        ));
    }

    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        let name = ident.to_string();
        if name.starts_with("helio_pass_") || self.forbidden.contains(&name) {
            self.reject(format!("concrete dependency/symbol {name}"));
        }
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func {
            let names: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if names.windows(2).any(|n| {
                matches!(n[0].as_str(), "ResourceKey" | "BufferKey")
                    && matches!(n[1].as_str(), "new" | "of")
            }) && call.args.iter().any(has_string)
            {
                self.reject("core declares a concrete resource/component key");
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if self.routing()
            && matches!(
                call.method.to_string().as_str(),
                "get"
                    | "get_mut"
                    | "insert"
                    | "entry"
                    | "contains"
                    | "contains_key"
                    | "write"
                    | "read"
                    | "route_named_texture"
                    | "write_texture_binding"
                    | "declare_external_input"
                    | "starts_with"
                    | "ends_with"
            )
            && call.args.first().is_some_and(has_string)
        {
            self.reject(format!("literal resource routing in {}()", call.method));
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        if self.routing()
            && (has_string(&expr.left) || has_string(&expr.right))
            && matches!(expr.op, syn::BinOp::Eq(_) | syn::BinOp::Ne(_))
        {
            self.reject("literal comparison in resource routing");
        }
        visit::visit_expr_binary(self, expr);
    }

    fn visit_arm(&mut self, arm: &'ast syn::Arm) {
        if self.routing() {
            let mut literals = StringFinder(false);
            literals.visit_pat(&arm.pat);
            if literals.0 {
                self.reject("literal match arm in resource routing");
            }
        }
        visit::visit_arm(self, arm);
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        // Item-generating macros are otherwise opaque to syn. Never silently
        // accept a macro-generated replacement for a closed field table.
        self.reject(format!(
            "item macro {} requires expansion and audit",
            tokens(&item.mac.path)
        ));
    }
}

struct StringFinder(bool);
impl<'ast> Visit<'ast> for StringFinder {
    fn visit_lit_str(&mut self, _: &'ast syn::LitStr) {
        self.0 = true;
    }
}
fn has_string(expr: &syn::Expr) -> bool {
    let mut finder = StringFinder(false);
    finder.visit_expr(expr);
    finder.0
}

fn files(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    for entry in fs::read_dir(root).unwrap_or_else(|e| panic!("{}: {e}", root.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            result.extend(files(&path));
        } else {
            result.push(path);
        }
    }
    result.sort();
    result
}

fn approved_shapes(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (key, value) = line
                .split_once(" | ")
                .expect("shape record: path::name | signature");
            (key.to_owned(), value.to_owned())
        })
        .collect()
}

fn check_shapes(audit: &mut Audit, approved: &BTreeMap<String, String>) {
    for (key, value) in &audit.shapes {
        if approved.get(key) != Some(value) {
            audit.violations.insert(format!(
                "{key}: unapproved data shape (new type, field, variant, or alias)"
            ));
        }
    }
    for key in approved.keys() {
        if !audit.shapes.contains_key(key) {
            audit
                .violations
                .insert(format!("{key}: stale generic-shape approval"));
        }
    }
}

fn main() {
    let tool = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = tool.parent().unwrap().parent().unwrap();
    let approved = approved_shapes(&fs::read_to_string(tool.join("generic-shapes.txt")).unwrap());
    let mut audit = Audit::default();
    for krate in ["helio-core"] {
        let manifest =
            fs::read_to_string(root.join("crates").join(krate).join("Cargo.toml")).unwrap();
        // Only the production `[dependencies]` table is load-bearing for the
        // "core never depends on a pass" rule -- `[dev-dependencies]` (e.g.
        // this crate's own `wgsl_validation` test aggregating every pass's
        // shader snippet for coverage) never ships and is not a core->pass
        // dependency in the sense this check guards against.
        let production_deps = manifest
            .split_once("[dependencies]")
            .map(|(_, rest)| rest)
            .unwrap_or("")
            .split("\n[")
            .next()
            .unwrap_or("");
        if production_deps.contains("helio-pass-") || production_deps.contains("helio_pass_") {
            audit
                .violations
                .insert(format!("{krate}/Cargo.toml: concrete pass dependency"));
        }
        for file in files(&root.join("crates").join(krate).join("src")) {
            audit.file = file
                .strip_prefix(root.join("crates"))
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if file.extension().is_some_and(|ext| ext == "rs") {
                let source = fs::read_to_string(&file).unwrap();
                let parsed =
                    syn::parse_file(&source).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
                audit.visit_file(&parsed);
            } else if audit.file == "helio-core/src/shader/prelude.wgsl" {
                // Explicit, reviewed exception: the camera struct and
                // depth/NDC conventions every pass shares, with no
                // pass-specific content -- the one case this rule's original
                // comment anticipated. Every other snippet (Hi-Z, PBR, wind,
                // ...) is a pass-declared `ShaderSnippet` living in its
                // owning pass crate, never embedded here.
            } else {
                // Shader implementations belong to their consumers.
                audit.reject("embedded non-Rust implementation requires relocation/audit");
            }
        }
    }
    if std::env::args().any(|arg| arg == "--inventory") {
        for (key, value) in &audit.shapes {
            println!("{key} | {value}");
        }
        return;
    }
    // Derive forbidden type names from actual unapproved declarations. A
    // renamed scene bag cannot pass just by avoiding historical names.
    audit.forbidden = audit
        .shapes
        .keys()
        .filter(|key| !approved.contains_key(*key))
        .map(|key| key.rsplit("::").next().unwrap().to_owned())
        .collect();
    for krate in ["helio-core"] {
        for file in files(&root.join("crates").join(krate).join("src")) {
            if file.extension().is_some_and(|ext| ext == "rs") {
                audit.file = file
                    .strip_prefix(root.join("crates"))
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let parsed = syn::parse_file(&fs::read_to_string(file).unwrap()).unwrap();
                // Scan references without recording declarations twice.
                let mut references = Audit {
                    file: audit.file.clone(),
                    forbidden: audit.forbidden.clone(),
                    ..Audit::default()
                };
                references.visit_file(&parsed);
                audit.violations.extend(references.violations);
            }
        }
    }
    check_shapes(&mut audit, &approved);
    for violation in &audit.violations {
        eprintln!("{violation}");
    }
    if !audit.violations.is_empty() {
        eprintln!(
            "Helio core boundary: FAILED ({} findings)",
            audit.violations.len()
        );
        std::process::exit(1);
    }
    println!("Helio core boundary: clean");
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scan(source: &str) -> Audit {
        let mut audit = Audit {
            file: "helio-core/src/registry.rs".into(),
            ..Audit::default()
        };
        audit.visit_file(&syn::parse_file(source).unwrap());
        audit
    }

    #[test]
    fn renamed_closed_bag_and_enum_are_rejected() {
        for source in [
            "struct Innocent { new_effect: Option<&'static Buffer> }",
            "enum Slot { NewEffect, OtherEffect }",
            "type Innocent = [Buffer; 12];",
            "struct Innocent(Buffer, Buffer);",
        ] {
            let mut audit = scan(source);
            check_shapes(&mut audit, &BTreeMap::new());
            assert!(!audit.violations.is_empty(), "{source}");
        }
    }

    #[test]
    fn approved_registry_cannot_acquire_a_concrete_field() {
        let approved = scan("struct Registry { slots: HashMap<Key, Value> }").shapes;
        let mut changed =
            scan("struct Registry { slots: HashMap<Key, Value>, new_effect: Buffer }");
        check_shapes(&mut changed, &approved);
        assert!(!changed.violations.is_empty());
        let mut unchanged = scan("struct Registry { slots: HashMap<Key, Value> }");
        check_shapes(&mut unchanged, &approved);
        assert!(unchanged.violations.is_empty());
    }

    #[test]
    fn unknown_resource_names_are_rejected_without_a_denylist() {
        for source in [
            "fn f() { let k = ResourceKey::<Buffer>::new(\"brand_new_effect\"); }",
            "fn f() { match name { \"brand_new_effect\" => (), _ => () } }",
            "fn f() { if name == r#\"brand_new_effect\"# {} }",
            "fn f() { slots.insert(\"brand_new_effect\", value); }",
            "generate_slots! { brand_new_effect: Buffer }",
            "use helio_pass_brand_new::Inputs;",
        ] {
            assert!(!scan(source).violations.is_empty(), "{source}");
        }
    }

    #[test]
    fn generic_routing_and_test_fixtures_are_allowed() {
        let audit = scan(
            r#"
            // struct Closed { fake: Buffer }
            fn route(name: &str) { slots.insert(name, value); }
            fn publish(name: &str) { slots.route_named_texture(name, view, "Graph"); }
            #[cfg(test)] mod tests {
                struct Fixture { concrete: Buffer }
                fn test() { slots.insert("fixture", value); }
            }
        "#,
        );
        assert!(audit.violations.is_empty());
        assert!(audit.shapes.is_empty());
    }
}
