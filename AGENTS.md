# Skybound

Bevy host (`src/graphics`, `src/game`) plus a rust-gpu shader crate (`src/scene`) compiled to SPIR-V by `build.rs`.

## Building

- Use the dev profile only (`cargo build`, `cargo test`); it is already opt-level 3. Never use `--release`: switching profiles relinks everything and costs many minutes.
- Any edit under `src/scene` re-runs the rust-gpu shader build. Incrementally that takes about 30 seconds; only a cold build with dependencies takes minutes.
- Run from `target/debug/skybound`. If the shell predates the flake hook, set `DLSS_SDK=$PWD/vendor/dlss`.

## Looking at the result

The benchmark mode renders offscreen and saves a screenshot:

```
SKYBOUND_SCREENSHOT=/path/out.png target/debug/skybound --benchmark --scene <scene> --warmup 6 --duration 2
```

- Scenes are listed in `scene_altitude` in `src/benchmark.rs`. For clouds use `cloud-edge`, `cloud-top`, `cloud-interior`, `cloud-bank`, `cloud-tower`, `cloud-storm`, `cloud-high`, `cloud-rain`. For the aur and veil use `garden`, `aur-skim`, `aur-horizon`.
- Free camera: `--altitude <m> --pitch <deg> --yaw <deg>`. Other options: `--size WxH`, `--time <s>`, and `--weather a,b,c,d` to force the weather.
- It prints `BENCHMARK ... mean_ms=` for a frame-time comparison.

## Creatures

- Each species is a short file under `src/scene/life/` with a `surface` built from shared primitives in `life/mod.rs` (`torso`, `mouth`, `eyes`, `annulated`, `sucker`, `eyespots`, `rounded`, `capsule`), `curved_tube`, `oral_tentacles` and `rig_tentacles` from `animalia/tetralata`, and a `LOOK` constant: its material family, pigment, accent and wing pairs. Shading, coats and self-shadowing live in `life/surface.rs`.
- Design from this world, not Earth's. No ground, so no legs. Light and danger come from the glowing aur below, so tetralates have inverted countershading (pale back, dark belly) and downward eyes. Water is scarce, hence the louvred breathing vents. Every feature should have a reason that follows from the lore in the taxon descriptions.
- Use `rounded` rather than `ellipsoid` for parts that are smooth-blended. `ellipsoid` switches to a box bound away from the surface, and blends and soft shadows pick that up as facets.
- Wings: a species' `LOOK` lists up to two pairs as (span, chord, root z, opacity). Roots sit on the body's midline at that z, which must lie within the body. `with_wings` blends each wing's bone frame (`wing_bones`: arm, leading finger and trailing fingers) into the body, and `visit_wings` draws the cambered membranes over the same hinge. `specimen` grows the bounds to fit the wings.
- Weapons and other adult-only structures should grow with `c.life.y`. Live-born young would otherwise be born with blades.
- Hanging structures (cloth, tentacles, `rig_ribbons` strands) are re-simulated every frame from a rest pose. Each is settled from two starts on a fixed grid and crossfaded by the grid phase, which keeps it smooth over time without per-entity state. Don't settle over a window that slides with the clock: that is what made the cloth jerky.
- Mouth tentacles are simulated chains. `rig_tentacles` in a species' `animate` must use the same mouth and scale as its `oral_tentacles` call. The knots live in the instance's cloth slots, which tetralates don't otherwise use.
- `gather_herd` in `game/ecosystem.rs` logs a per-species census of the living every minute (`census` in the log), which shows whether every lineage is surviving.
- **The world** draws the player pair (`frame.players`) and up to `HERD` ecosystem creatures, chosen each frame by screen coverage in `game/ecosystem.rs` (`gather_herd`). Other species render through `life::render_opaque` and `visit_thin`: one tracer over `any_surface`, which switches on species, with materials chosen by `LOOK`.
- **GPU compile time:** rust-gpu inlines every call, so `trace_material` evaluates the surface from exactly one place (one loop that steps through marching, normal, occlusion and shadow phases). Adding more surface call sites to it, or more tracer instances over `any_surface`, multiplies the shader build by the number of species. GPU code also can't loop over arrays by value, take slices of arrays, index zero-length arrays, or call `saturating_sub` or `hypot`; use index loops and fixed-size arrays.
- **Previews** are rendered on the CPU by `examples/species_preview`. `cargo build --example species_preview`, then `target/debug/examples/species_preview --size 384 --outputs render --out <dir>` takes about a minute for every species. With no `--out` it regenerates all the committed assets beside each species file: render, anatomy, lifecycle and variation grids, and a half-size idle GIF.
- Shader-target files that call `sin`, `sqrt` and so on need `#[cfg(target_arch = "spirv")] use spirv_std::num_traits::Float;`.

## Rendering layout

- `src/scene/render/clouds.rs`: cloud density (per-layer genus shapes, cumulus `heap`, storm towers), lighting (`cloud_light`), cloud map, bounds and light-volume passes.
- `src/scene/render/march.rs`: `march_clouds` (the cloud-only pass, which writes a slab of radiance and transmittance plus four quarter-opacity depths) and `integrate` (aur veil, arcs and pole beams, joining the cloud slab at those depths).
- The density functions have three detail levels. `Probe` must never understate cloud, and the `free` distances must stay conservative, or the march skips cloud.

## Style

- Imports go in one merged block at the top, using `crate::` paths (rustfmt sorts them).
- Comments are full prose sentences that explain the physics or intent, matching the existing code.
