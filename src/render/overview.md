**Rendering Overview**

**Quick summary**
- The renderer draws volumetric clouds as a fullscreen fragment pass. A fullscreen triangle pass executes `shaders/rendering.wgsl` as a fragment shader and writes into the main view and a motion target. Noise textures and a GPU cloud data storage buffer are prepared on the CPU side and bound to that pass.

**Quick file reference**
- Shaders: `src/render/shaders/*.wgsl`
- Render glue and pipeline: `src/render/raymarch.rs`, `src/render/mod.rs`
- Cloud actor & GPU buffer: `src/render/clouds.rs`
- Noise generation: `src/render/noise/{mod.rs,simplex.rs,worley.rs,utils.rs}`

**High-level pipeline**
1. CPU / main App:
   - `setup_clouds` (Startup): create randomized clouds and insert `CloudsState` and initial `CloudsBufferData` resources.
   - `setup_noise_textures` (Startup): generate/load procedural 3D noise textures and insert `NoiseTextures` (base/detail) as resources.
   - `update_clouds` (Update): advance cloud positions, perform frustum culling, fill `CloudsBufferData` with visible/cloud count.
2. Render extraction & prepare (RenderApp):
   - `extract_clouds_view_uniform` (ExtractSchedule): extract per-view matrices and planet data for cloud rendering.
   - `update_clouds_buffer` (Render Prepare): upload `CloudsBufferData` into a GPU `CloudsBuffer` (storage buffer).
   - `prepare_clouds_view_uniforms` (Render PrepareResources): write per-view dynamic `CloudsViewUniform` blocks (matrices, Halton jitter, TAA data).
3. Render graph (RenderApp):
   - `RaymarchNode` (a ViewNode) is registered on the core 3D graph and runs as a fullscreen fragment pass using `RaymarchPipeline`.
   - The fragment shader (`shaders/rendering.wgsl`) is bound with: dynamic view uniform, a linear sampler, the cloud storage buffer, and noise 3D textures, and it renders into the main color target and a motion render target. The prepass depth and motion texture views are used for depth/motion inputs.

**Important shaders**
- `rendering.wgsl` — Fragment shader used by the fullscreen pass; orchestrates volumetric ray marching, sky sampling, and outputs color and motion.
- `volumetrics.wgsl` — Volumetric integration and density lighting.
- `clouds.wgsl` — Cloud shape/coverage/detail sampling (noise layering, coverage masks).
- `raymarch.wgsl` — SDF helpers and small solid examples.
- `sky.wgsl` — Atmosphere and scattering utilities.
- `utils.wgsl` — Shared helpers used across shaders.

All shader files live in `src/render/shaders/`.

**Rust-side resources & types**
- `CloudsState` (main world): CPU representation of all clouds.
- `CloudsBufferData` (ExtractResource): packed array of visible clouds + count; produced on the main App and extracted to the render world.
- `CloudsBuffer` (Render resource): GPU `Buffer` containing the uploaded `CloudsBufferData` used as a storage buffer in the shader.
- `NoiseTextures` (ExtractResource): handles to generated 3D noise `Image`s (`base` and `detail`).
- `CloudsViewUniform` / `CloudsViewUniforms`: per-view `ShaderType` uniform written to a dynamic uniform buffer; contains matrices, jitter/TAA, planet data, and time/frame.
- `PreviousViewData`: render-only resource storing previous-frame matrices used for motion vector computation.

**Bindings (summary from `RaymarchPipeline` / `rendering.wgsl`)**
- Bind group layout (fragment stage):
  - (0) dynamic `CloudsViewUniform` (uniform buffer binding)
  - (1) linear `Sampler` (filtering, repeat addressing)
  - (2) `CloudsBuffer` (storage buffer with `CloudsBufferData`)
  - (3) base noise `texture_3d` (R8Unorm-like data)
  - (4) detail noise `texture_3d` (RGBA8Unorm-like data)

The fragment pipeline writes to two color targets: `TextureFormat::Rgba16Float` (main color) and `TextureFormat::Rg16Float` (motion). The depth-stencil attachment uses the prepass depth view (`Depth32Float`).

**Noise & procedural textures**
- Noise is generated in `src/render/noise/mod.rs` using `simplex.rs` and `worley.rs` helpers and saved/loaded via `utils.rs` helpers.
- Currently `base` is generated as a single-channel (R) 3D texture (used for large shapes) and `detail` is a multi-channel 3D texture (RGBA) containing smaller-scale features/fog channels.
