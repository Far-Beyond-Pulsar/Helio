set windows-shell := ["pwsh.exe", "-NoLogo", "-NoProfile", "-Command"]

wasm-script := if os() == "windows" { ".\\build-wasm.ps1" } else { "./build-wasm.sh" }

alias help := default

default:
    @just --list --unsorted --justfile {{justfile()}}

ci: fmt-check check clippy nextest

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

fix:
    cargo fix --workspace --allow-dirty --allow-staged

check:
    cargo check --workspace

check-release:
    cargo check --workspace --release

check-package pkg:
    cargo check -p {{pkg}}

check-package-release pkg:
    cargo check -p {{pkg}} --release

build:
    cargo build --workspace

build-release:
    cargo build --workspace --release

build-package pkg:
    cargo build -p {{pkg}}

build-package-release pkg:
    cargo build -p {{pkg}} --release

build-example example='editor_demo':
    cargo build -p examples --bin {{example}}

build-example-release example='editor_demo':
    cargo build -p examples --bin {{example}} --release

run example='editor_demo' *args:
    cargo run -p examples --bin {{example}} -- {{args}}

run-release example='editor_demo' *args:
    cargo run -p examples --bin {{example}} --release -- {{args}}

run-package pkg *args:
    cargo run -p {{pkg}} -- {{args}}

test:
    cargo test --workspace

test-package pkg:
    cargo test -p {{pkg}}

nextest:
    cargo nextest run

nextest-package pkg:
    cargo nextest run -p {{pkg}}

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

clippy-package pkg:
    cargo clippy -p {{pkg}} --all-targets -- -D warnings

doc:
    cargo doc --workspace --no-deps

doc-open:
    cargo doc --workspace --no-deps --open

tree pkg='helio':
    cargo tree -p {{pkg}}

status:
    git status --short

diff:
    git diff --stat

audit:
    cargo audit

outdated:
    cargo outdated --workspace

clean:
    cargo clean

doctor:
    cargo --version
    rustc --version
    git --version
    just --version
    cargo nextest --version
    cargo set-version --help
    wasm-bindgen --version

install-tools:
    cargo install just
    cargo install cargo-nextest
    cargo install cargo-edit
    cargo install wasm-bindgen-cli
    cargo install cargo-audit
    cargo install cargo-outdated
    cargo install basic-http-server

wasm:
    {{wasm-script}} --release

wasm-dev:
    {{wasm-script}} --dev

wasm-demo demo:
    {{wasm-script}} --release {{demo}}

wasm-demo-dev demo:
    {{wasm-script}} --dev {{demo}}

serve-wasm port='8000':
    npx --yes http-server target/wasm-prebuilt --host 127.0.0.1 --port {{port}}

version package version:
    cargo set-version -p {{package}} {{version}}

version-bump package part:
    cargo set-version -p {{package}} --bump {{part}}

version-workspace version:
    cargo set-version -p helio {{version}}

version-workspace-bump part:
    cargo set-version -p helio --bump {{part}}

version-all version:
    cargo set-version --workspace {{version}}

version-all-bump part:
    cargo set-version --workspace --bump {{part}}

stage-version package:
    git add crates/{{package}}/Cargo.toml Cargo.lock

stage-workspace-versions:
    git add Cargo.lock crates/*/Cargo.toml

commit msg:
    git commit -m "{{msg}}"

tag name:
    git tag {{name}}

push:
    git push

push-tags:
    git push --follow-tags

release package version:
    cargo set-version -p {{package}} {{version}}
    cargo check -p {{package}} --quiet
    git add crates/{{package}}/Cargo.toml Cargo.lock
    git commit -m "chore(release): {{package}} {{version}}"

release-bump package part:
    cargo set-version -p {{package}} --bump {{part}}
    cargo check -p {{package}} --quiet
    git add crates/{{package}}/Cargo.toml Cargo.lock
    git commit -m "chore(release): bump {{package}} ({{part}})"

release-workspace version:
    cargo set-version -p helio {{version}}
    cargo check -p helio --quiet
    git add crates/helio/Cargo.toml Cargo.lock
    git commit -m "chore(release): helio {{version}}"

release-workspace-bump part:
    cargo set-version -p helio --bump {{part}}
    cargo check -p helio --quiet
    git add crates/helio/Cargo.toml Cargo.lock
    git commit -m "chore(release): bump helio ({{part}})"

release-all version:
    cargo set-version --workspace {{version}}
    cargo check --workspace --quiet
    git add Cargo.lock crates/*/Cargo.toml
    git commit -m "chore(release): workspace {{version}}"

release-all-bump part:
    cargo set-version --workspace --bump {{part}}
    cargo check --workspace --quiet
    git add Cargo.lock crates/*/Cargo.toml
    git commit -m "chore(release): bump workspace ({{part}})"