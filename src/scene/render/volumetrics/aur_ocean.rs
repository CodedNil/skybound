use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::aur::{CURRENT, EJECTION_HEIGHT, EJECTION_WIDTH};
use crate::scene::math::{cube, fourth, square};
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

/// Keep fine relief separate from the broad gradients used for illumination.
pub(crate) fn wave_detail(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    (textures
        .noise(footprint)
        .billows(flowing * 0.00016 + Vec3::Z * time * 0.012, 0.00016)
        .y
        - 0.5)
        * 240.0
}

pub(crate) fn ejections(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec3,
    textures: &Textures,
) -> VolumeSample {
    if curve.z <= 0.0 || curve.x > EJECTION_WIDTH * 6.0 {
        return VolumeSample::default();
    }
    let flowing = planet_point - CURRENT * view.time();
    let turbulence = textures.details_filtered(flowing * 0.0003, footprint.max(120.0) * 0.0003);
    // A constant-width tube joins the veil naturally; no altitude-dependent
    // skirt, root flare or separate hot-coloured emitter at either endpoint.
    let width = EJECTION_WIDTH * (0.8 + turbulence.w * 0.2);
    let distance = curve.x * curve.x;
    let halo = (-distance / (width * width)).exp();
    let core = (-distance / (width * width * 0.24)).exp();
    let filament = turbulence.z.smoothstep(0.2, 0.75);
    let density = (halo * (0.085 + filament * 0.025) + core * 0.045) * curve.z;
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * view.time() * 0.003,
        footprint * 0.0001,
    );
    let hue = vec3(0.12, 0.09, 0.23).lerp(vec3(0.38, 0.24, 0.32), flow.y.smoothstep(0.35, 0.65));
    let tint = hue.lerp(vec3(0.40, 0.61, 0.57), flow.z.smoothstep(0.4, 0.65)) * view.sky_horizon.y;
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: tint * (1.4 + filament * 0.5 + core * 0.5),
    }
}

pub fn sample_ocean(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec3,
    wave: f32,
    normal: Vec3,
    palette: spirv_std::glam::Vec4,
    textures: &Textures,
) -> VolumeSample {
    let pos = if curve.z > 0.0 {
        crate::scene::planet::surface_coordinates(planet_point)
    } else {
        Vec3::new(0.0, 0.0, planet_point.length() - PLANET_RADIUS)
    };
    if pos.z > OCEAN_TOP_HEIGHT {
        return VolumeSample::default();
    }
    let time = view.time();
    let jets = if view.camera_position.w as u32 & 128 == 0 {
        ejections(planet_point, view, footprint, curve, textures)
    } else {
        VolumeSample::default()
    };
    if pos.z > 3700.0 {
        return jets;
    }
    let flowing = planet_point - CURRENT * time;
    let altitude = pos.z - wave;
    if altitude > 900.0 {
        return jets;
    }
    let coordinates = flowing * 0.00006 + Vec3::Z * (pos.z * 0.0003 - time * 0.025);
    if palette.w > 0.0 {
        let detail =
            textures.details_filtered(coordinates * vec3(3.0, 3.0, 1.0), footprint * 0.00036);
        let noise = detail.w * 2.0 - 1.0;
        let wisp = cube((1.0 - noise.abs() * 2.0).max(0.0));
        let body = 1.0 - (altitude + noise * 65.0).smoothstep(-170.0, 90.0);
        let depth = (-altitude).max(0.0);
        let plume = (-altitude.max(0.0) / 180.0).exp();
        let base = (palette.w - 0.03).max(0.075);
        let body_density = (body * (base + depth.min(1200.0) * 0.0003) + plume * wisp * 0.008)
            * (1.0 - pos.z.smoothstep(3300.0, 3700.0));
        let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
        let arc_glow = (-square(curve.x / (EJECTION_WIDTH * 0.85))).exp() * curve.z;
        let source = palette.xyz() * (0.65 + 0.35 * depth_glow / 0.5276) * (0.8 + detail.w * 0.4)
            + vec3(0.3, 0.5, 0.48) * (wisp * plume * 0.15 + arc_glow * 0.35) * view.sky_horizon.y;
        let density = body_density + jets.density;
        return VolumeSample {
            density,
            color: Vec3::ZERO,
            emission: (source * body_density + jets.emission * jets.density)
                / density.max(0.0000001),
        };
    }
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
    );
    let detail = textures.details_filtered(coordinates * vec3(3.0, 3.0, 1.0), footprint * 0.00036);
    let noise = detail.w * 2.0 - 1.0;
    let wisp = cube((1.0 - noise.abs() * 2.0).max(0.0));
    let granules = flow.w.smoothstep(0.25, 0.75);
    let body = 1.0 - (altitude + noise * 65.0).smoothstep(-170.0, 90.0);
    let depth = (-altitude).max(0.0);
    let plume = (-altitude.max(0.0) / 180.0).exp();
    let body_density = (body
        * (0.075 + granules * 0.09 + wisp * 0.06 + depth.min(1200.0) * 0.0003)
        + plume * fourth(granules) * wisp * 0.035)
        * (1.0 - pos.z.smoothstep(3300.0, 3700.0));
    let mut sample = VolumeSample {
        density: body_density + jets.density,
        ..VolumeSample::default()
    };
    if sample.density > 0.0001 {
        let facing = normal.dot(vec3(-0.48, 0.36, 0.8)).max(0.0);
        let hue =
            vec3(0.12, 0.09, 0.23).lerp(vec3(0.38, 0.24, 0.32), flow.y.smoothstep(0.35, 0.65));
        let core = hue.lerp(vec3(0.40, 0.61, 0.57), flow.z.smoothstep(0.4, 0.65));
        let energy = (flow.w * 0.7 + detail.w * 0.3).smoothstep(0.30, 0.70);
        let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
        let pulse = 0.85 + 0.15 * (time * 1.4 + flow.y * 9.0).sin();
        let mut flash = Vec3::ZERO;
        for index in 0..32 {
            let event = view.aur_flashes[index];
            if event[1].w <= 0.0 {
                break;
            }
            let distance = planet_point.distance(event[0].xyz()) * event[0].w;
            let structure = 0.35 + granules * 0.45 + wisp * 0.2;
            flash += event[1].xyz() * (-distance * distance).exp() * structure;
        }
        let arc_glow = (-square(curve.x / (EJECTION_WIDTH * 0.85))).exp() * curve.z;
        let emission = (hue * (0.65 + facing * 0.85)
            + core * depth_glow * (0.65 + cube(energy) * 3.5) * (0.35 + facing * 1.5)
            + vec3(0.26, 0.40, 0.42) * wisp * plume * energy * 2.0)
            * pulse
            + flash
            + core * arc_glow * (0.2 + granules * 0.3);
        sample.color = hue;
        sample.emission = (emission * body_density * view.sky_horizon.y
            + jets.emission * jets.density)
            / sample.density;
    }
    sample
}
