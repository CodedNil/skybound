# Repository Guidelines

This repository is a Rust binary crate (package: `skybound`) built with Bevy. Use this guide when contributing.

## Project Structure & Module Organization
- `Cargo.toml` / `Cargo.lock` — crate metadata and deps.
- `src/` — main application code (entry: `src/main.rs`).
- `assets/` — runtime assets (shaders, textures, models).
- `justfile`, `shell.nix`, `.envrc` — developer shortcuts and environment config.
- `target/` — build artifacts (ignored in VCS).

## Build, Test, and Development Commands
- `just` — default dev shortcut (`just` runs `cargo run`).
- `cargo run` — build and run in debug.
- `cargo run --release` — run optimized (use for performance testing).
- `cargo build` / `cargo build --release` — compile artifacts.
- `cargo fmt` — format code with `rustfmt`.
- `cargo clippy --all-targets -- -D warnings` — lint and fail on warnings.
- `direnv allow` or `nix develop` — enable the repo's environment if present.

## Coding Style & Naming Conventions
- Indentation: 4 spaces (follow `rustfmt`).
- Naming: `snake_case` for functions/files, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- Safety: `unsafe` is forbidden (`unsafe_code = "forbid"` in `Cargo.toml`).
- Clippy: the project uses pedantic rules; address lints before opening PRs.

## Testing Guidelines
- This is an application with no tests currently; add tests only for deterministic, pure logic modules.
- If you add tests: unit tests inline (`#[cfg(test)]`), integration tests in `tests/`, run with `cargo test`.

## Commit & Pull Request Guidelines
- Commit messages: imperative, concise. Example: `Improve cloud raymarching performance`.
- Branches: `feature/<short-desc>`, `fix/<short-desc>`.
- PR checklist: description, linked issue (if any), testing notes, and visual proof (screenshots/video) for rendering changes.

## Security & Configuration Tips
- Never commit secrets; use `.envrc` and environment variables.
- Run `cargo audit` when updating dependencies.
