use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::polar::{REACTION_SPREAD, reaction_coordinates};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub const POLE_WIDTH: f32 = 400_000.0;

pub fn sample_poles(
    relative: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
) -> VolumeSample {
    let axis = Vec3::Z;
    let along = relative.dot(axis);
    let radial = relative - axis * along;
    let height = (along.abs() - PLANET_RADIUS).max(0.0);
    let width = 1200.0 + height.sqrt() * 5.0;
    let distance = radial.length_squared();
    let noise = textures.base_filtered(
        (radial + axis * (along - view.time() * 1200.0)) * 0.000025,
        footprint * 0.000025,
    );
    let filament = 0.3 + noise.y.smoothstep(0.25, 0.75) * 0.7;
    let core = (-distance / (width * width)).exp() * filament * 0.002;
    let sheath = (-distance / (width * width * 36.0)).exp() * (0.5 + noise.z) * 0.00012;
    let halo = (-distance / (width * width * 625.0)).exp() * 0.000004;
    let reaction = reaction_coordinates(relative, axis);
    let mut impact = 0.0;
    let mut corona = 0.0;
    if reaction.x < REACTION_SPREAD * 1.6 && reaction.y.abs() < 45_000.0 {
        let surface = relative.normalize();
        let turbulence = textures.base_filtered(
            surface * 39.0 - axis * view.time() * 0.025,
            footprint / (PLANET_RADIUS + height) * 39.0,
        );
        let spread = (-((reaction.x / REACTION_SPREAD).powi(2)) * 2.5).exp();
        let shell = reaction.y - (turbulence.w - 0.5) * 14_000.0;
        let thickness = 4500.0 + turbulence.y * 5500.0;
        let layer = (-(shell / thickness).powi(2)).exp();
        let fronts =
            (reaction.x * 0.00018 - view.time() * 1.8 + turbulence.z * 5.0).sin() * 0.5 + 0.5;
        let reference = if axis.z.abs() < 0.95 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let east = axis.cross(reference).normalize();
        let azimuth = radial.dot(axis.cross(east)).atan2(radial.dot(east));
        let rays =
            (azimuth * 27.0 + turbulence.x * 4.0 + reaction.x * 0.000015 - view.time() * 0.3).sin()
                * 0.5
                + 0.5;
        corona = layer * spread * (0.00012 + fronts.powi(3) * rays.powi(3) * 0.0012);
        impact = (-(reaction.x / 12_000.0).powi(2) - (reaction.y / 12_000.0).powi(2)).exp()
            * (0.008 + noise.w * 0.004);
    }
    let density = core + sheath + halo + impact + corona;
    if density < 0.0000001 {
        return VolumeSample::default();
    }
    VolumeSample {
        density,
        color: vec3(0.1, 0.15, 0.25),
        emission: (vec3(1.0, 0.85, 0.6) * (core * 90.0)
            + vec3(0.2, 0.5, 1.0) * (sheath * 25.0)
            + vec3(0.15, 0.08, 0.4) * (halo * 4.0)
            + vec3(1.0, 0.96, 0.82) * (impact * 110.0)
            + vec3(0.32, 0.52, 1.0).lerp(vec3(1.0, 0.66, 0.35), noise.z) * (corona * 8.0))
            / density,
        ..VolumeSample::default()
    }
}

pub fn poles_raymarch_entry(ro: Vec3, rd: Vec3, view: &ViewUniform, t_max: f32) -> Vec2 {
    let axis = view.planet_rotation.mul_vec3(Vec3::Z).normalize();
    let oc = ro - view.planet_center();
    let ad = axis.dot(rd);
    let ao = axis.dot(oc);
    let a = 1.0 - ad * ad;
    let b = 2.0 * (oc.dot(rd) - ao * ad);
    let c = oc.dot(oc) - ao * ao - POLE_WIDTH * POLE_WIDTH;
    if a < 0.000001 {
        return if c <= 0.0 {
            vec2(0.0, t_max)
        } else {
            vec2(t_max, 0.0)
        };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2(t_max, 0.0);
    }
    let s = disc.sqrt();
    let entry = (-b - s) / (2.0 * a);
    let exit = (-b + s) / (2.0 * a);
    if exit <= 0.0 {
        return vec2(t_max, 0.0);
    }
    vec2(entry.max(0.0), exit.min(t_max))
}
