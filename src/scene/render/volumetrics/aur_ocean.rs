use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4Swizzles, vec2, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::ViewUniform;
use crate::scene::aur::{CURRENT, curl, ejection_field};
use crate::scene::render::utils::Textures;

pub const OCEAN_TOP_HEIGHT: f32 = 3800.0;

fn wave_height(position: Vec2, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position - CURRENT.xy() * time;
    let drift = textures
        .base_filtered(
            (flowing * 0.000025).extend(time * 0.003),
            footprint * 0.000025,
        )
        .yz()
        * 2.0
        - 1.0;
    let warped = curl(flowing * 0.00014 + drift * 0.8, time);
    let broad = textures
        .base_filtered(warped.extend(time * 0.012), footprint * 0.00022)
        .y;
    let folds = textures
        .base_filtered(
            vec2(
                warped.x * 1.73 + warped.y * 0.61,
                warped.y * 1.73 - warped.x * 0.61,
            )
            .extend(-time * 0.018),
            footprint * 0.0003,
        )
        .z;
    (broad - 0.5) * 1800.0 + (folds - 0.5) * 500.0 - 120.0
}

fn ejections(pos: Vec3, view: &ViewUniform, footprint: f32, textures: &Textures) -> Vec2 {
    let curve = ejection_field(pos, view.time());
    if curve.x > 1875.0 {
        return Vec2::ZERO;
    }
    let turbulence = textures.details_filtered(
        (pos - CURRENT * view.time() * 0.3) * 0.002,
        footprint * 0.002,
    );
    let width = 325.0 + turbulence.w * 300.0;
    let distance = curve.x * curve.x;
    let halo = (-distance / (width * width)).exp();
    let core = (-distance / (width * width * 0.18)).exp();
    let filament = turbulence.z.smoothstep(0.15, 0.75);
    let density = (halo * (0.012 + filament.powi(3) * 0.08) + core * 0.02) * curve.z;
    let surge = (curve.y * 16.0 - view.time() * 4.0).sin() * 0.5 + 0.5;
    let roots = ((curve.y - 0.5).abs() * 2.0).powi(2);
    vec2(
        density,
        density * (2.0 + core * (3.0 + surge * 5.0) + roots * 6.0 + filament * 2.0),
    )
}

pub fn sample_ocean(
    pos: Vec3,
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
    if pos.z > 1900.0 {
        return VolumeSample {
            density: jets.x,
            color: vec3(0.4, 0.5, 0.6),
            emission: vec3(0.75, 0.86, 1.05) * (jets.y / jets.x.max(0.00001)) * view.sky_horizon.y,
            ..VolumeSample::default()
        };
    }
    let flowing = pos - CURRENT * time;
    let wave = wave_height(pos.xy(), time, footprint, textures);
    let altitude = pos.z - wave;
    let coordinates = curl(flowing.xy() * 0.00018, time).extend(pos.z * 0.0008 - time * 0.025);
    let flow = textures.base_filtered(
        flowing * vec3(0.0001, 0.0001, 0.0003) - Vec3::Z * time * 0.0072,
        footprint * 0.0003,
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
        * (1.0 - pos.z.smoothstep(2000.0, OCEAN_TOP_HEIGHT));
    let mut sample = VolumeSample {
        density: body_density + jets.x,
        ..VolumeSample::default()
    };
    if !only_density && sample.density > 0.0001 {
        let normal = if altitude.abs() < 350.0 {
            let epsilon = footprint.max(24.0);
            let gradient = vec2(
                wave_height(pos.xy() + Vec2::X * epsilon, time, footprint, textures) - wave,
                wave_height(pos.xy() + Vec2::Y * epsilon, time, footprint, textures) - wave,
            ) / epsilon;
            (-gradient).extend(1.0).normalize()
        } else {
            Vec3::Z
        };
        let facing = normal.dot(vec3(-0.48, 0.36, 0.8)).max(0.0);
        let hue =
            vec3(0.16, 0.12, 0.27).lerp(vec3(0.34, 0.21, 0.30), flow.y.smoothstep(0.25, 0.75));
        let core = hue.lerp(vec3(0.40, 0.61, 0.57), flow.z.smoothstep(0.46, 0.75));
        let energy = flow.w.smoothstep(0.40, 0.75);
        let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
        let pulse = 0.85 + 0.15 * (time * 1.4 + flow.y * 9.0).sin();
        let mut flash = 0.0;
        for index in 0..32 {
            let event = view.aur_flashes[index];
            if event.w <= 0.0 {
                break;
            }
            let distance = (pos.xy() - event.xy()).length_squared() + (pos.z + 120.0).powi(2);
            flash += (-distance / (event.z * event.z)).exp() * event.w;
        }
        let emission = (hue * (0.08 + facing * 0.35)
            + core * depth_glow * (0.15 + energy.powi(3) * 3.5) * (0.2 + facing * 1.4)
            + vec3(0.26, 0.40, 0.42) * wisp * plume * energy * 0.5)
            * pulse
            + vec3(1.0, 1.8, 3.0) * flash * 4.0;
        sample.color = hue;
        sample.emission = (emission * body_density + vec3(0.75, 0.86, 1.05) * jets.y)
            / sample.density
            * view.sky_horizon.y;
    }
    sample
}
