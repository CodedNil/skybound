use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::aur::{CURRENT, EJECTION_HEIGHT, EJECTION_WIDTH, energy_color};
use crate::scene::math::{cube, square};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub const OCEAN_TOP_HEIGHT: f32 = EJECTION_HEIGHT + EJECTION_WIDTH * 3.0;

pub(crate) fn wave_height(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    wave_base(position, time, footprint, textures)
        + wave_detail(position, time, footprint, textures)
}

pub(crate) fn wave_base(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let noise = textures.noise(footprint);
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    let drift = noise
        .billows(flowing * 0.0000025 + Vec3::Z * time * 0.001, 0.0000025)
        .xyz()
        * 2.0
        - 1.0;
    let mut warped = flowing * 0.000014 + drift * 0.8;
    warped += vec3(
        (warped.y + time * 0.07).sin(),
        (warped.z - time * 0.06).sin(),
        (warped.x + time * 0.05).sin(),
    ) * 0.4;
    let broad = noise.billows(warped + Vec3::Z * time * 0.004, 0.000022).y;
    let folds = noise
        .billows(
            warped * 0.61 + warped.yzx() * 0.46 - Vec3::Z * time * 0.006,
            0.000017,
        )
        .z;
    (broad - 0.5) * 4200.0 + (folds - 0.5) * 3300.0 - 120.0
}

/// Fine relief changes the volume boundary, independently of broad lighting.
pub(crate) fn wave_detail(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    (textures
        .noise(footprint)
        .billows(flowing * 0.00016 + Vec3::Z * time * 0.012, 0.00016)
        .y
        - 0.5)
        * 240.0
}

/// Blend differently oriented samples so the periodic detail texture cannot
/// impose a visible grid. Altitude stays in the coordinates: this is a volume.
fn ocean_detail(point: Vec3, time: f32, footprint: f32, textures: &Textures) -> Vec4 {
    let altitude = point.length() - PLANET_RADIUS;
    let coordinates = ((point - CURRENT * time) * 0.00006
        + Vec3::Z * (altitude * 0.0003 - time * 0.025))
        * vec3(3.0, 3.0, 1.0);
    (textures.details_filtered(coordinates, footprint * 0.00036)
        + textures.details_filtered(
            crate::scene::noise::rotated(coordinates),
            footprint * 0.00027,
        ))
        * 0.5
}

/// Cache the original depth-lit material, rather than a flat colour swatch.
pub(crate) fn column_material(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    normal: Vec3,
    textures: &Textures,
) -> Vec4 {
    let time = view.time();
    let flowing = point - CURRENT * time;
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
    );
    let detail = ocean_detail(point, time, footprint, textures);
    let noise = detail.w * 2.0 - 1.0;
    let wisp = cube((1.0 - noise.abs() * 2.0).max(0.0));
    let granules = flow.w.smoothstep(0.25, 0.75);
    let facing = normal.dot(vec3(-0.48, 0.36, 0.8)).max(0.0);
    let hue = vec3(0.12, 0.09, 0.23).lerp(vec3(0.38, 0.24, 0.32), flow.y.smoothstep(0.35, 0.65));
    let core = hue.lerp(vec3(0.40, 0.61, 0.57), flow.z.smoothstep(0.4, 0.65));
    let energy = (flow.w * 0.7 + detail.w * 0.3).smoothstep(0.30, 0.70);
    // The column is evaluated 100 m inside the veil, matching the depth used
    // to normalise its lighting in the native pass.
    let depth_glow = 1.0 - (-135.0_f32 / 180.0).exp();
    let pulse = 0.85 + 0.15 * (time * 1.4 + flow.y * 9.0).sin();
    let mut flash = Vec3::ZERO;
    for index in 0..32 {
        let event = view.aur_flashes[index];
        if event[1].w <= 0.0 {
            break;
        }
        let delta = (point - event[0].xyz()) * event[0].w;
        flash += event[1].xyz()
            * (-delta.length_squared()).exp()
            * (0.35 + granules * 0.45 + wisp * 0.2);
    }
    let light = (hue * (0.65 + facing * 0.85)
        + core * depth_glow * (0.65 + cube(energy) * 3.5) * (0.35 + facing * 1.5)
        + vec3(0.26, 0.40, 0.42) * wisp * energy * 2.0)
        * pulse
        + flash;
    let density = (1.0 - (-100.0 + noise * 65.0).smoothstep(-170.0, 90.0))
        * (0.075 + granules * 0.09 + wisp * 0.06 + 0.03)
        + crate::scene::math::fourth(granules) * wisp * 0.035;
    (light * view.sky_horizon.y + hue * view.ambient_color.xyz() * 0.2).extend(density.max(0.01))
}

pub(crate) fn ejections(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec3,
    textures: &Textures,
) -> VolumeSample {
    if curve.z <= 0.0 || curve.x > EJECTION_WIDTH * 3.0 {
        return VolumeSample::default();
    }
    let time = view.time();
    let flowing = planet_point - CURRENT * time;
    let turbulence = textures.details_filtered(
        flowing * 0.0003 + Vec3::Z * (curve.y * 2.0 - time * 0.12),
        footprint * 0.0003,
    );
    let width = EJECTION_WIDTH * (0.6 + turbulence.w * 0.25);
    let halo = (-square(curve.x / width)).exp();
    let core = (-square(curve.x / (width * 0.28))).exp();
    let strands = cube((1.0 - turbulence.y).max(0.0));
    let surge = 0.5 + 0.5 * (curve.y * 45.0 - time * 6.0 + turbulence.w * 8.0).sin();
    let density = (halo * (0.018 + strands * 0.08) + core * 0.085) * curve.z;
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
    );
    let tint = energy_color(flow.y + (turbulence.w - 0.5) * 0.3, flow.z);
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: (tint * (2.0 + strands * 12.0 + surge * 3.0)
            + vec3(0.8, 1.1, 0.95) * core * surge * 4.0)
            * view.sky_horizon.y,
    }
}

pub fn sample_body(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec3,
    wave: f32,
    height_span: f32,
    palette: Vec4,
    textures: &Textures,
) -> VolumeSample {
    let altitude = planet_point.length() - PLANET_RADIUS;
    let height = altitude - wave;
    if altitude > 3700.0 || height - height_span > 900.0 {
        return VolumeSample::default();
    }
    let time = view.time();
    let detail = ocean_detail(planet_point, time, footprint, textures);
    let noise = detail.w * 2.0 - 1.0;
    let low = (height + noise * 65.0 - height_span + 170.0) / 260.0;
    let high = (height + noise * 65.0 + height_span + 170.0) / 260.0;
    let body = 1.0 - crate::scene::volume::smoothstep_mean(low, high);
    let wisp = cube((1.0 - noise.abs() * 2.0).max(0.0));
    let depth = (-height).max(0.0);
    let plume = (-height.max(0.0) / 180.0).exp();
    let material = if palette.w > 0.0 {
        palette
    } else {
        column_material(
            planet_point.normalize() * (PLANET_RADIUS + wave - 100.0),
            view,
            footprint,
            planet_point.normalize(),
            textures,
        )
    };
    let base = (material.w - 0.03).max(0.075);
    let density = (body * (base + depth.min(1200.0) * 0.0003) + plume * wisp * 0.008)
        * (1.0 - altitude.smoothstep(3300.0, 3700.0));
    let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
    let arc_glow = (-square(curve.x / (EJECTION_WIDTH * 0.85))).exp() * curve.z;
    let source = material.xyz() * (0.65 + 0.35 * depth_glow / 0.5276) * (0.8 + detail.w * 0.4)
        + vec3(0.3, 0.5, 0.48) * (wisp * plume * 0.15 + arc_glow * 0.35) * view.sky_horizon.y;
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: source,
    }
}
