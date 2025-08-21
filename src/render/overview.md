**Rendering Overview (detailed)**

**Quick Summary**
- The renderer produces clouds, atmosphere scattering, and simple solids using a low-resolution compute raymarch pass plus a fullscreen composite pass that upsamples and temporally filters results to full resolution.
- Compute entrypoint: `shaders/rendering.wgsl` (dispatched from the `RaymarchNode` / compute pipeline in `src/render/raymarch.rs`).
- Composite entrypoint: `shaders/composite.wgsl` (runs as a fullscreen fragment pass in `src/render/composite.rs`).

**High-level Pipeline**
1. CPU side (`src/render/raymarch.rs`): prepare view uniforms, create low-res textures (color, motion, depth) and full-res history buffers, bind noise textures and samplers, dispatch compute pipeline.
2. Compute (`rendering.wgsl`): reconstruct rays, run solids SDF raymarch, call volumetric integrator, composite volumetrics with background sky/solids, write low-res `color`, `motion`, and `depth` targets.
3. Composite (`composite.wgsl`): bilateral upsample low-res color to full-res, reproject previous-frame history using motion vectors and Catmull–Rom filtering, apply variance clipping (YCoCg), write final color + history textures.

**Shader Registry (what to edit for effect changes)**
- `rendering.wgsl` — Orchestrates per-pixel ray construction, Henyey–Greenstein phase functions, and final outputs. Calls into `raymarch_solids`, `raymarch_volumetrics`, and sky sampling.
- `volumetrics.wgsl` — `raymarch_volumetrics`: adaptive raymarch integrator for clouds, fog, and poles. Handles entry/exit intervals, density sampling, light marching, transmittance accumulation, aur/aural lighting, and outputs color+alpha and weighted depth.
- `raymarch.wgsl` — Simple SDF solids renderer (repeating spheres example), used to combine opaque geometry with volumetrics.
- `clouds.wgsl` — Cloud shape & density: layered cloud model, multi-layer parameters, base/detail/weather noises, wind vectors, layer scaling, returns density (and simple mode).
- `aur_fog.wgsl` — Fog/aurora and lightning flashes: turbulence, Poisson-disk flashes, fog sampling that returns density, color, emission.
- `poles.wgsl` — Polar aurora/poles sampling and pole-shaped volume intersection helper.
- `sky.wgsl` — Physical atmosphere renderer (Rayleigh/Mie/ozone): integrates scattering along view ray and computes sun light color that volumetrics use as illumination.
- `composite.wgsl` — Fullscreen upsample + TAA: bilateral upsample, motion reprojection, temporal blending with confidence metric, YCoCg variance clipping.
- `utils.wgsl` — Shared helpers: hashing, blue-noise, matrix helpers, sphere/plane intersection helpers, ray-shell intersection, quaternion rotation, `View` / `AtmosphereData` structs.

**Rust Modules (what runs the shaders and manages resources)**
- `src/render/mod.rs` — Plugin registration: loads and registers shaders, sets up extract systems and render graph nodes (`RaymarchNode` and `CompositeNode`) and initializes resources.
- `src/render/raymarch.rs` — Core glue for the compute pass:
  - `CloudsViewUniform` and `CloudsViewUniforms`: GPU-side view uniform layout, jitter (Halton) for TAA.
  - `manage_textures`: create/resize low-res color/motion/depth and full-res history textures.
  - `RaymarchNode` (ViewNode): builds bind groups and dispatches the compute shader over the low-res targets.
  - `RaymarchPipeline` (FromWorld): creates the compute pipeline and bind group layout matching `rendering.wgsl` bindings.
- `src/render/composite.rs` — Composite pass node + `CompositePipeline`:
  - Binds low-res color/motion/depth and full-res history textures.
  - Runs a fullscreen pass that outputs both main color and the history buffer (ping-ponged).
- `src/render/noise/*` — Procedural texture generation used by the volumetric shaders:
  - `mod.rs` — orchestration to generate/load `base`, `detail`, and `weather` textures.
  - `simplex.rs` — 3D Simplex noise + FBM, seamless tiling helpers.
  - `worley.rs` — 3D Worley (cellular) noise + FBM.
  - `utils.rs` — helpers for generating/saving textures, interleaving channels, image sampler descriptor used by Bevy.

**Noise Textures**
- Noise textures are generated on startup (or loaded from `assets/textures/*.bin`) by `setup_noise_textures` and inserted as `NoiseTextures` resource.
- Textures:
  - `base_texture` (3D R8): base cloud shapes (perlin+worley mix).
  - `details_texture` (3D RGBA8): multi-channel detail/fog layers.
  - `weather_texture` (2D RGBA8): coverage/height/variation for weather per-layer.
- These textures are sampled in `clouds.wgsl`, `volumetrics.wgsl` and `aur_fog.wgsl` with repeat addressing.

**Bindings & Textures (summary of important shader bindings)**
- Compute (`rendering.wgsl`) bindings (group 0):
  - (0) `view`: `View` uniform.
  - (1) `linear_sampler`: sampler.
  - (2) `base_texture`: texture_3d<f32> (base noise).
  - (3) `details_texture`: texture_3d<f32> (detail noise).
  - (4) `weather_texture`: texture_2d<f32>`.
  - (5) `output_color`: storage texture `rgba16float` (write)
  - (6) `output_motion`: storage texture `rg16float` (write)
  - (7) `output_depth`: storage texture `r32float` (write)
- Composite (`composite.wgsl`) bindings (group 0): color, history, motion, depth textures + samplers.

**Temporal Anti-Aliasing & Motion**
- Jittering: a Halton sequence of 8 offsets is applied to the projection matrix in `prepare_clouds_view_uniforms`, producing sub-pixel samples for TAA.
- Motion vectors: when volumetrics produce a depth < t_max, the shader reconstructs the world position and projects it into the previous frame's clip space to compute `uv - uv_prev`, stored in `output_motion`.
- Composite pass reprojects previous history using the motion vector and performs Catmull-Rom filtering + YCoCg variance clipping and a history confidence heuristic to avoid ghosting.

**Design Notes / Where to change what**
- To change cloud appearance: edit `src/render/shaders/clouds.wgsl` (layer parameters, noise scales, detail amounts).
- To change lighting and sky color: edit `src/render/shaders/sky.wgsl` (Rayleigh/Mie coefficients, steps, exposure).
- To tune performance vs quality: adjust workgroup size (in compute shader), `MAX_STEPS`/`STEP_SIZE_*` in `volumetrics.wgsl`, and the low-res target scale in `raymarch.rs` (`primary_window.physical_* / 2`).
- To debug: render low-res `color`, `motion`, or `depth` directly to a debug view; composite pass reads/writes history textures you can also inspect.

**Quick File Reference (paths)**n- Shaders: `src/render/shaders/*.wgsl` (see registry above).
- Render glue: `src/render/raymarch.rs` and `src/render/composite.rs`.
- Plugin entry: `src/render/mod.rs` (registers nodes, loads shaders).
- Noise generation: `src/render/noise/{mod.rs,simplex.rs,worley.rs,utils.rs}`

