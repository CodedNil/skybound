use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::aur::{CURRENT, EJECTION_HEIGHT, EJECTION_WIDTH, ejection_field};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub const OCEAN_TOP_HEIGHT: f32 = EJECTION_HEIGHT + EJECTION_WIDTH * 3.0;

fn wave_height(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    let drift = textures
        .billows(
            flowing * 0.0000025 + Vec3::Z * time * 0.001,
            footprint * 0.0000025,
        )
        .xyz()
        * 2.0
        - 1.0;
    let mut warped = flowing * 0.000014 + drift * 0.8;
    warped += vec3(
        (warped.y + time * 0.07).sin(),
        (warped.z - time * 0.06).sin(),
        (warped.x + time * 0.05).sin(),
    ) * 0.4;
    let broad = textures
        .billows(warped + Vec3::Z * time * 0.004, footprint * 0.000022)
        .y;
    let folds = textures
        .billows(
            warped * 0.61 + warped.yzx() * 0.46 - Vec3::Z * time * 0.006,
            footprint * 0.000017,
        )
        .z;
    let ripples = textures
        .billows(
            flowing * 0.00016 + Vec3::Z * time * 0.012,
            footprint * 0.00016,
        )
        .y;
    (broad - 0.5) * 4200.0 + (folds - 0.5) * 3300.0 + (ripples - 0.5) * 240.0 - 120.0
}

fn ejections(pos: Vec3, view: &ViewUniform, footprint: f32, textures: &Textures) -> VolumeSample {
    let curve = ejection_field(pos, view.time());
    if curve.x > EJECTION_WIDTH * 3.0 {
        return VolumeSample::default();
    }
    let turbulence = textures.details_filtered(
        vec3(
            curve.y * 12.0 - view.time() * 2.5,
            pos.z * 0.0003,
            curve.x * 0.002,
        ),
        footprint * 0.0004,
    );
    let width = EJECTION_WIDTH * (0.65 + turbulence.w * 0.35);
    let distance = curve.x * curve.x;
    let halo = (-distance / (width * width)).exp();
    let core = (-distance / (width * width * 0.18)).exp();
    let filament = turbulence.z.smoothstep(0.20, 0.75);
    let surge = (curve.y * 16.0 - view.time() * 8.0 + turbulence.x * 3.0).sin() * 0.5 + 0.5;
    let density = (halo * (0.024 + filament.powi(3) * 0.105) + core * 0.036)
        * curve.z
        * (0.45 + surge * 0.55);
    let roots = ((curve.y - 0.5).abs() * 2.0).powi(2);
    let fringe = vec3(0.16, 0.40, 0.34).lerp(vec3(0.48, 0.18, 0.30), turbulence.y);
    let hot = vec3(2.3, 0.7, 0.15).lerp(vec3(0.3, 1.7, 1.05), surge);
    VolumeSample {
        density,
        color: fringe,
        emission: (fringe * (1.8 + filament * 3.0) + hot * (core * 5.4 + roots * 6.0))
            * view.sky_horizon.y,
        ..VolumeSample::default()
    }
}

pub fn sample_ocean(
    pos: Vec3,
    planet_point: Vec3,
    view: &ViewUniform,
    only_density: bool,
    footprint: f32,
    textures: &Textures,
) -> VolumeSample {
    if pos.z > OCEAN_TOP_HEIGHT {
        return VolumeSample::default();
    }
    let time = view.time();
    let jets = ejections(pos, view, footprint, textures);
    if pos.z > 3700.0 {
        return jets;
    }
    let flowing = planet_point - CURRENT * time;
    let wave = wave_height(planet_point, time, footprint, textures);
    let altitude = pos.z - wave;
    let coordinates = flowing * 0.00006 + Vec3::Z * (pos.z * 0.0003 - time * 0.025);
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
    );
    let detail = textures.details_filtered(coordinates * vec3(3.0, 3.0, 1.0), footprint * 0.0016);
    let noise = detail.w * 2.0 - 1.0;
    let wisp = (1.0 - noise.abs() * 2.0).max(0.0).powi(3);
    let granules = flow.w.smoothstep(0.25, 0.75);
    let body = 1.0 - (altitude + noise * 65.0).smoothstep(-170.0, 90.0);
    let depth = (-altitude).max(0.0);
    let plume = (-altitude.max(0.0) / 180.0).exp();
    let body_density = (body
        * (0.075 + granules * 0.09 + wisp * 0.06 + depth.min(1200.0) * 0.0003)
        + plume * granules.powi(4) * wisp * 0.035)
        * (1.0 - pos.z.smoothstep(3300.0, 3700.0));
    let mut sample = VolumeSample {
        density: body_density + jets.density,
        ..VolumeSample::default()
    };
    if !only_density && sample.density > 0.0001 {
        let normal = if altitude.abs() < 350.0 {
            let epsilon = footprint.max(200.0);
            let gradient = vec3(
                wave_height(planet_point + Vec3::X * epsilon, time, footprint, textures) - wave,
                wave_height(planet_point + Vec3::Y * epsilon, time, footprint, textures) - wave,
                wave_height(planet_point + Vec3::Z * epsilon, time, footprint, textures) - wave,
            ) / epsilon;
            (planet_point.normalize() - gradient * 10.0).normalize()
        } else {
            planet_point.normalize()
        };
        let facing = normal.dot(vec3(-0.48, 0.36, 0.8)).max(0.0);
        let hue =
            vec3(0.12, 0.09, 0.23).lerp(vec3(0.38, 0.24, 0.32), flow.y.smoothstep(0.35, 0.65));
        let core = hue.lerp(vec3(0.40, 0.61, 0.57), flow.z.smoothstep(0.4, 0.65));
        let energy = (flow.w * 0.7 + detail.w * 0.3).smoothstep(0.30, 0.70);
        let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
        let pulse = 0.85 + 0.15 * (time * 1.4 + flow.y * 9.0).sin();
        let mut flash = 0.0;
        for index in 0..32 {
            let event = view.aur_flashes[index];
            if event.w <= 0.0 {
                break;
            }
            let center = crate::scene::planet::surface_point(event.xy().extend(-120.0));
            let distance = planet_point.distance(center) / event.z;
            let ripple = (distance * 11.0 - time * 24.0).sin() * 0.5 + 0.5;
            flash += (-distance * distance).exp() * event.w * (0.3 + ripple.powi(3));
        }
        let emission = (hue * (0.65 + facing * 0.85)
            + core * depth_glow * (0.65 + energy.powi(3) * 3.5) * (0.35 + facing * 1.5)
            + vec3(0.26, 0.40, 0.42) * wisp * plume * energy * 2.0)
            * pulse
            + vec3(0.45, 0.95, 1.25) * flash * 3.0;
        sample.color = hue;
        sample.emission = (emission * body_density * view.sky_horizon.y
            + jets.emission * jets.density)
            / sample.density;
    }
    sample
}
