use crate::scene::ViewUniform;
pub use crate::scene::climate::{
    CLOUD_BOTTOM as CLOUD_BOTTOM_HEIGHT, CLOUD_TOP as CLOUD_TOP_HEIGHT,
};
use crate::scene::render::utils::Textures;
use spirv_std::glam::{FloatExt, Vec3};
pub fn extinction(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
    records: &[crate::scene::cloud::Cloud],
    grid: &[u32],
) -> f32 {
    use crate::scene::cloud::{GRID_CELLS, cell, cell_index};
    if view.cloud_time.z < 0.5 {
        return 0.0;
    }
    let Some(index) = cell_index(cell(point)) else {
        return 0.0;
    };
    let start = grid[index];
    let end = grid[index + 1];
    let mut extinction = 0.0;
    for offset in start..end {
        let id = grid[GRID_CELLS + 1 + offset as usize] as usize;
        let cloud = records[id];
        extinction += sample_cloud(cloud, point, view, footprint, textures);
    }
    extinction
}

/// Native density: the ellipsoid is only a bound; macro billows form the silhouette.
pub fn sample_cloud(
    cloud: crate::scene::cloud::Cloud,
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
) -> f32 {
    let local = cloud.coordinates(point, view.cloud_time.x);
    let interior = 1.0 - local.length();
    if interior <= 0.0 {
        return 0.0;
    }
    let seed = Vec3::splat(cloud.motion.w * 17.0);
    let radii = Vec3::new(cloud.center.w, cloud.east.w, cloud.up.w);
    let macro_shape =
        textures.base_filtered(local * 0.55 + seed, footprint / radii.min_element() * 0.55);
    let scale = cloud.appearance.z;
    let fine = textures.details_filtered(local * radii * scale + seed, footprint * scale);
    let warped = local + (macro_shape.truncate() - Vec3::splat(0.5)) * 0.3;
    let mut lobes = 0.63 - (warped - Vec3::new(0.0, 0.0, -0.25)).length();
    let mut random = Vec3::new(
        cloud.motion.w * 7.31,
        cloud.motion.w * 17.17,
        cloud.motion.w * 31.37,
    )
    .fract_gl();
    for _ in 0..5 {
        random = (random * Vec3::new(3.71, 5.17, 7.13) + Vec3::new(0.19, 0.37, 0.61)).fract_gl();
        let center = (random - Vec3::splat(0.5)) * Vec3::new(1.0, 0.9, 0.8) + Vec3::Z * 0.2;
        lobes = lobes.max(0.35 + random.y * 0.3 - (warped - center).length());
    }
    let convection = if cloud.appearance.z > 0.0002 {
        0.8 + cloud.appearance.w * 0.2
    } else {
        0.15
    };
    let shape = interior * (1.0 - convection) + lobes * 1.4 * convection;
    let eroded =
        shape.min(interior) - (1.0 - macro_shape.x) * 0.35 - fine.x * cloud.appearance.y * 0.2;
    let density = eroded.smoothstep(0.0, 0.16)
        * local.z.smoothstep(-0.95, -0.6)
        * (0.65 + macro_shape.y * 0.6);
    let strength = if view.weather.x >= 0.0 {
        let high = (1200.0 - cloud.up.w).smoothstep(0.0, 800.0);
        view.weather.x * (1.0 - high) + view.weather.z * high
    } else {
        1.0
    };
    cloud.appearance.x * density * strength
}
