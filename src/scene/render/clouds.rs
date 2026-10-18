use spirv_std::glam::{FloatExt, UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

use crate::scene::aur::{EJECTION_CELL, EJECTION_REACH, EJECTION_WIDTH};
use crate::scene::climate::{
    CLOUD_BOTTOM, CLOUD_GRID_CELLS, CLOUD_TOP, Cloud, cloud_cell, cloud_cell_index,
};
use crate::scene::math::simpson;
use crate::scene::planet::{light_position, surface_coordinates};
use crate::scene::render::march::RaymarchResult;
use crate::scene::render::ocean;
use crate::scene::render::ocean::OCEAN_TOP_HEIGHT;
use crate::scene::render::textures::{AtmosphereData, Textures};
use crate::scene::{FrameUniform, PLANET_RADIUS, ViewUniform, froxel as grid};

pub fn extinction(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
    records: &[Cloud],
    grid: &[u32],
) -> f32 {
    if view.cloud_time.z < 0.5 {
        return 0.0;
    }
    let Some(index) = cloud_cell_index(cloud_cell(point)) else {
        return 0.0;
    };
    let start = grid[index];
    let end = grid[index + 1];
    let mut extinction = 0.0;
    for offset in start..end {
        let id = grid[CLOUD_GRID_CELLS + 1 + offset as usize] as usize;
        let cloud = records[id];
        extinction += sample_cloud(cloud, point, view, footprint, textures);
    }
    extinction
}

/// Native density: the ellipsoid is only a bound; macro billows form the silhouette.
pub fn sample_cloud(
    cloud: Cloud,
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

pub(super) fn cloud_lighting(
    position: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    density: f32,
    textures: &Textures,
    records: &[Cloud],
    grid: &[u32],
) -> Vec3 {
    let source = light_position(position);
    let light = (source - position).normalize();
    let mut optical = density * 100.0;
    let mut distance = 150.0;
    let mut step = 300.0;
    for _ in 0..4 {
        optical += extinction(
            position + light * distance,
            view,
            step * 0.3,
            textures,
            records,
            grid,
        ) * step;
        distance += step;
        step *= 2.5;
    }
    let cosine = direction.dot(light);
    let denominator = (1.0 + 0.6 * 0.6 - 1.2 * cosine).max(0.01);
    let phase = (1.0 - 0.6 * 0.6) / (denominator * denominator.sqrt());
    let sunlight = textures.pole_light(position, source) * view.sky_horizon.y;
    let direct = (-optical).exp() * (0.12 * phase + 0.04);
    let multiple = (-optical * 0.12).exp() * 0.18;
    let altitude = position.length() - PLANET_RADIUS;
    let mut aur_depth = density * 160.0;
    if altitude < 25_000.0 {
        aur_depth += extinction(
            position - position.normalize() * 400.0,
            view,
            100.0,
            textures,
            records,
            grid,
        ) * 640.0;
    }
    let (_, palette) = ocean::sample(position, view, 4000.0, textures);
    let source = if palette.w > 0.0 {
        palette.xyz()
    } else {
        vec3(0.40, 0.34, 0.52) * view.sky_horizon.y
    };
    let upward = source
        * (-altitude.max(0.0) / 16_000.0).exp()
        * ((-aur_depth).exp() * 0.7 + (-aur_depth * 0.2).exp() * 0.3);
    sunlight * (direct + multiple)
        + upward
        + view.ambient_color.xyz() * 0.45
        + vec3(0.025, 0.035, 0.055)
}

pub fn short_volume(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    length: f32,
    textures: &Textures,
    records: &[Cloud],
    grid: &[u32],
) -> RaymarchResult {
    let origin = view.to_planet(ro);
    let direction = view.planet_rotation.conjugate() * rd;
    let middle = origin + direction * (length * 0.5);
    let footprint = (length / 8.0).max(4.0);
    let a = extinction(origin, view, footprint, textures, records, grid);
    let m = extinction(middle, view, footprint, textures, records, grid);
    let b = extinction(
        origin + direction * length,
        view,
        footprint,
        textures,
        records,
        grid,
    );
    let optical = simpson(a, m, b, length);
    let air = textures.air_transmittance(origin, direction, length);
    let transmission = (-optical).exp() * air;
    let lighting = if optical > 0.0001 {
        cloud_lighting(
            middle,
            direction,
            view,
            a.max(m).max(b),
            textures,
            records,
            grid,
        )
    } else {
        atmosphere.haze
    };
    RaymarchResult {
        color: (lighting * (1.0 - transmission)).extend(transmission),
        depth: length,
    }
}

fn medium(
    point: Vec3,
    direction: Vec3,
    footprint: f32,
    view: &ViewUniform,
    textures: &Textures,
    records: &[Cloud],
    grid: &[u32],
) -> Vec4 {
    let altitude = point.length() - PLANET_RADIUS;
    let profile = view.camera_position.w as u32;
    let mut clearance = 65_000.0;
    let mut scattering = view.ambient_color.xyz() * 0.45 + Vec3::new(0.025, 0.035, 0.055);
    if profile & 2 == 0
        && altitude > CLOUD_BOTTOM - footprint * 2.0
        && altitude < CLOUD_TOP + footprint * 2.0
    {
        // A distant depth cell can straddle the entire cloud layer. Evaluate
        // illumination on the layer rather than missing it at both probes.
        let point = point.normalize()
            * (PLANET_RADIUS + altitude.clamp(CLOUD_BOTTOM + 1.0, CLOUD_TOP - 1.0));
        let density = extinction(point, view, footprint, textures, records, grid);
        scattering = cloud_lighting(point, direction, view, density, textures, records, grid);
    }
    if profile & 4 == 0 && profile & 128 == 0 && altitude < OCEAN_TOP_HEIGHT + footprint * 3.0 {
        let coordinates = surface_coordinates(point);
        let curve = textures.ejection_field(coordinates);
        // The lookup covers nine cells. An unvisited cell can contain an arc
        // even when that local lookup is empty, so bound its possible reach.
        let within = (coordinates.truncate() / EJECTION_CELL).fract_gl();
        let unvisited = (1.5 - (within - Vec2::splat(0.5)).abs().max_element()) * EJECTION_CELL
            - EJECTION_REACH;
        // Longitude stretches near the poles. Convert the field's distance to
        // a conservative world-space bound and cover the complete froxel.
        let metric = point.truncate().length() / point.length().max(1.0);
        clearance = ((curve.x.min(unvisited) - EJECTION_WIDTH * 6.0) * metric - footprint * 3.0)
            .clamp(0.0, 65_000.0);
    } else if profile & 4 == 0 && profile & 128 == 0 {
        clearance = (altitude - OCEAN_TOP_HEIGHT - footprint * 3.0).clamp(0.0, 65_000.0);
    }
    scattering.extend(clearance)
}

#[spirv(compute(threads(4, 4, 4), entry_point_name = "froxel_medium"))]
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] output: &mut [UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 10)] cloud_records: &[Cloud],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 11)] cloud_grid: &[u32],
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
