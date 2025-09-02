
# Repository Guidelines

This repository is a Rust binary crate (package: `skybound`) built with Bevy. Use this guide when contributing.

## Project Layout
- `Cargo.toml` / `Cargo.lock` — crate metadata and deps.
- `src/` — main application code (entry: `src/main.rs`).
- `src/render` — rendering pipeline, shaders, and GPU integration.
- `justfile`, `shell.nix`, `.envrc` — developer shortcuts and environment config.
- `target/` — build artifacts (ignored in VCS).

## Build & Development Commands
- `just` — default dev shortcut (`just` runs `cargo run`).
- `cargo run` / `cargo run --release` — run in debug or optimized mode.
- `cargo build` / `cargo build --release` — compile artifacts.
- `cargo fmt` — format code (run before commits).
- `cargo clippy --all-targets -- -D warnings` — lint and fail on warnings.
- `direnv allow` or `nix develop` — enable the repo environment if present.

## Code Style & Readability

We value clear, concise, and maintainable code. In addition to standard Rust conventions (4-space indentation, `snake_case` for functions/files, `CamelCase` for types, constants in `SCREAMING_SNAKE_CASE`, and `unsafe` forbidden), follow these practical principles inspired by the CodeAesthetic channel (https://www.youtube.com/@CodeAesthetic/videos):

- Reduce nesting: prefer early returns and small helper functions to keep top-level control flow shallow.
- Name for clarity: choose meaningful variable and function names so the code documents itself rather than relying on comments.
- Small functions: favor single-responsibility functions that are easy to read and test.
- Minimize comments: prefer self-descriptive names and small function extracts over long explanatory comments.
- Flatten conditionals: prefer match/guards and simple boolean logic to deeply nested branches.
- Immutable by default: prefer immutable bindings, and make mutability explicit where needed.

Additional repository rules:
- All public and internal functions (including methods) should have a one-line `///` doc comment summarizing purpose.
- Keep functions concise; if a function grows beyond ~50 lines, consider extracting helpers.

## Testing Guidance
- This is primarily an application; add tests only for deterministic, pure logic modules.
- Unit tests should be inline (`#[cfg(test)]`) and integration tests in `tests/`.
- Run tests with `cargo test`.

## Commits & Pull Requests
- Commit messages: imperative, concise (e.g. `Improve cloud raymarching performance`).
- Branches: `feature/<short-desc>`, `fix/<short-desc>`.
- PR checklist: description, linked issue (if any), testing notes, and visual proof (screenshots/video) for rendering changes.

## Security & Dependencies
- Do not commit secrets; use environment variables and `.envrc`.
- Run `cargo audit` when updating dependencies.

If you have questions about style or a non-obvious refactor, open a short PR and request feedback rather than making broad changes unilaterally.
