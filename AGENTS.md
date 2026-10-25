# Skybound

A flight game over an alien sky world. A Bevy host (`src/graphics`, `src/game`) drives a rust-gpu shader crate (`src/scene`) compiled to SPIR-V by `build.rs`, which raymarches the whole scene: the aur ocean, clouds, and the creatures and plants of its ecosystem.

## The world

There is no ground. Life drifts or flies above the aur, a glowing, lethal ocean that lights the sky from below and drives the updraft everything depends on. Water is scarce, food drifts in the air, and the taxon descriptions in `src/scene/life` hold each lineage's lore.

Design creatures from this world, not Earth's. No ground means no legs; light and danger come from below, so tetralates have inverted countershading (pale back, dark belly) and downward-looking eyes; scarce water means louvred breathing vents that recover moisture. Every feature should have a reason that follows from the lore, and related species should still differ in ways their niches explain.

## Building

- Use the dev profile only (`cargo build`, `cargo test`); it is already opt-level 3. Never use `--release`: switching profiles relinks everything and costs many minutes.
- Any edit under `src/scene` re-runs the shader build, which takes about a minute.
- Run from `target/debug/skybound`. If the shell predates the flake hook, set `DLSS_SDK=$PWD/vendor/dlss`.
- Shader code can't loop over arrays by value, slice arrays, or call `saturating_sub` or `hypot`; files using `sin`, `sqrt` and so on need `#[cfg(target_arch = "spirv")] use spirv_std::num_traits::Float;`.

## Looking at the result

- The benchmark mode renders offscreen and saves a screenshot: `SKYBOUND_SCREENSHOT=/path/out.png target/debug/skybound --benchmark --scene <scene> --warmup 6 --duration 2`. Scenes are listed in `src/benchmark.rs`; `--altitude`, `--pitch` and `--yaw` give a free camera, and `--live` lets time run. It prints frame times.
- Creature previews render on the CPU: `cargo build --example species_preview`, then `target/debug/examples/species_preview --size 384 --outputs render --out <dir>`. With no `--out` it regenerates the assets committed beside each species file.

## Layout

- `src/scene/life`: each species is a short file with a `surface` built from shared primitives in `life/mod.rs` and a `LOOK` (colours and wings). Shading is in `life/surface.rs`.
- `src/game/ecosystem.rs`: the simulated population, its needs and behaviour, and which creatures are drawn each frame.
- `src/scene/render`: clouds (`clouds.rs`, `march.rs`), the aur ocean, and the passes that draw everything.

## Style

- Imports go in one merged block at the top, using `crate::` paths (rustfmt sorts them).
- Comments are full prose sentences that explain the physics or intent, matching the existing code.
