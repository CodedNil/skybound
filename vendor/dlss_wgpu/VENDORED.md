# Vendored dlss_wgpu

Bevy's wrapper around NVIDIA DLSS for wgpu, version 6.0.0 from crates.io
(<https://github.com/bevyengine/dlss_wgpu>), dual licensed MIT or Apache 2.0.
The root `Cargo.toml` patches Bevy's dependency on it to this copy.

## Local patches

- **Render at the mode's resolution.** `DlssSuperResolution::render_resolution`
  returned the bottom of DLSS's dynamic resolution range rather than the
  optimal resolution the feature was created for. Current DLSS libraries set
  that bottom at half the output size for every mode, so Quality, Balanced
  and Performance all rendered at Performance's resolution. It now returns
  the optimal one, as ray reconstruction already did.
- **No D3D bindings.** Upstream commit `4046a40` (after 6.0.0): bindgen
  skips the D3D items, whose symbols the Linux SDK lacks.
