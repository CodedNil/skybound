use spirv_std::glam::{FloatExt, Vec3};

use crate::scene::render::utils::Textures;

pub fn sample_storm(
    point: Vec3,
    view: &crate::scene::ViewUniform,
    strength: f32,
    footprint: f32,
    textures: &Textures,
) -> f32 {
    if strength <= 0.001 {
        return 0.0;
    }
    let Some(storm) = textures.storm(point, view) else {
        return 0.0;
    };
    let local = storm.coordinates(point);
    let height = local.dot(storm.up);
    let radial = local - storm.up * height;
    let tower = (radial / 0.55 + storm.up * height).length();
    let anvil = (radial / 0.72 + storm.up * ((height - 0.4) / 0.55)).length();
    let interior = 1.0 - tower.min(anvil);
    if interior <= 0.0 {
        return 0.0;
    }
    let flow = point * 0.000_08 - storm.up * view.time() * 0.003;
    let noise = textures.base_filtered(flow, footprint * 0.000_08);
    let body = (interior - (1.0 - noise.x) * 0.35).smoothstep(0.0, 0.15);
    body * strength * (0.16 + noise.y * 0.12)
}
