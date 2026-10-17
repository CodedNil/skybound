use super::utils::Textures;
use super::volumetrics::{cloud_lighting, clouds};
use crate::scene::{FrameUniform, PLANET_RADIUS, ViewUniform, froxel as grid};
use spirv_std::glam::{UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::{Image, Sampler, spirv};

fn medium(
    point: Vec3,
    direction: Vec3,
    footprint: f32,
    view: &ViewUniform,
    textures: &Textures,
    records: &[crate::scene::cloud::Cloud],
    grid: &[u32],
) -> Vec4 {
    let altitude = point.length() - PLANET_RADIUS;
    let profile = view.camera_position.w as u32;
    let mut clearance = 65_000.0;
    let mut scattering = view.ambient_color.xyz() * 0.45 + Vec3::new(0.025, 0.035, 0.055);
    if profile & 2 == 0
        && altitude > clouds::CLOUD_BOTTOM_HEIGHT - footprint * 2.0
        && altitude < clouds::CLOUD_TOP_HEIGHT + footprint * 2.0
    {
        // A distant depth cell can straddle the entire cloud layer. Evaluate
        // illumination on the layer rather than missing it at both probes.
        let point = point.normalize()
            * (PLANET_RADIUS
                + altitude.clamp(
                    clouds::CLOUD_BOTTOM_HEIGHT + 1.0,
                    clouds::CLOUD_TOP_HEIGHT - 1.0,
                ));
        let density = clouds::extinction(point, view, footprint, textures, records, grid);
        scattering = cloud_lighting(point, direction, view, density, textures, records, grid);
    }
    if profile & 4 == 0
        && profile & 128 == 0
        && altitude < super::volumetrics::aur_ocean::OCEAN_TOP_HEIGHT + footprint * 3.0
    {
        let coordinates = crate::scene::planet::surface_coordinates(point);
        let curve = textures.ejection_field(coordinates, f32::MAX);
        // The lookup covers nine cells. An unvisited cell can contain an arc
        // even when that local lookup is empty, so bound its possible reach.
        let within = (coordinates.truncate() / crate::scene::aur::EJECTION_CELL).fract_gl();
        let unvisited = (1.5 - (within - Vec2::splat(0.5)).abs().max_element())
            * crate::scene::aur::EJECTION_CELL
            - crate::scene::aur::EJECTION_REACH;
        // Longitude stretches near the poles. Convert the field's distance to
        // a conservative world-space bound and cover the complete froxel.
        let metric = point.truncate().length() / point.length().max(1.0);
        clearance = ((curve.x.min(unvisited) - crate::scene::aur::EJECTION_WIDTH * 6.0) * metric
            - footprint * 3.0)
            .clamp(0.0, 65_000.0);
    } else if profile & 4 == 0 && profile & 128 == 0 {
        clearance = (altitude - super::volumetrics::aur_ocean::OCEAN_TOP_HEIGHT - footprint * 3.0)
            .clamp(0.0, 65_000.0);
    }
    scattering.extend(clearance)
}

#[spirv(compute(threads(4, 4, 4)))]
pub fn froxel_medium(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] transmittance: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] ejections: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] shape: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 8)] columns: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 9)] _unused: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] output: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)]
    cloud_records: &[crate::scene::cloud::Cloud],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] cloud_grid: &[u32],
) {
    if id.x >= grid::WIDTH || id.y >= grid::HEIGHT || id.z >= grid::DEPTH {
        return;
    }
    let view = &frame.view;
    let uv = (id.truncate().as_vec2() + 0.5) / Vec2::new(grid::WIDTH as f32, grid::HEIGHT as f32);
    let origin = view.to_planet(view.world_position.xyz());
    let direction = view.planet_rotation.conjugate() * view.ray_direction(uv, false);
    let textures = Textures {
        base,
        details,
        weather,
        transmittance,
        ejections,
        sampler,
        cache_shape: shape,
        cache_columns: columns,
        cached: true,
    };
    let start = grid::distance(id.z as f32);
    let end = grid::distance(id.z as f32 + 1.0);
    let distance = (start + end) * 0.5;
    let footprint = (distance * 2.0 / grid::HEIGHT as f32)
        .max((end - start) * 0.5)
        .max(4.0);
    // Average over the depth cell rather than sparsely probing a thin veil.
    let a = medium(
        origin + direction * (start * 0.75 + end * 0.25),
        direction,
        footprint,
        view,
        &textures,
        cloud_records,
        cloud_grid,
    );
    let b = medium(
        origin + direction * (start * 0.25 + end * 0.75),
        direction,
        footprint,
        view,
        &textures,
        cloud_records,
        cloud_grid,
    );
    let value = ((a.xyz() + b.xyz()) * 0.5).extend(a.w.min(b.w));
    output[((id.z * grid::HEIGHT + id.y) * grid::WIDTH + id.x) as usize] = UVec2::new(
        grid::half(value.x) | (grid::half(value.y) << 16),
        grid::half(value.z) | (grid::half(value.w) << 16),
    );
}
