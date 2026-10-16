use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
use spirv_std::num_traits::Float;

use super::VolumeSample;
use crate::scene::math::{cube, fourth, square};
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
    let core_envelope = (-distance / (width * width)).exp() * 0.002;
    let sheath_envelope = (-distance / (width * width * 36.0)).exp() * 0.00012;
    let halo = (-distance / (width * width * 625.0)).exp() * 0.000004;
    let near_reaction = distance < square(REACTION_SPREAD * 1.6 * 1.04)
        && (relative.length() - crate::scene::polar::REACTION_RADIUS).abs() < 45_000.0;
    if !near_reaction && core_envelope + sheath_envelope * 1.5 + halo < 0.0000001 {
        return VolumeSample::default();
    }
    let noise = textures.base_filtered(
        (radial + axis * (along - view.time() * 1200.0)) * 0.000025,
        footprint * 0.000025,
    );
    let filament = 0.3 + noise.y.smoothstep(0.25, 0.75) * 0.7;
    let core = core_envelope * filament;
    let sheath = sheath_envelope * (0.5 + noise.z);
    let reaction = if near_reaction {
        reaction_coordinates(relative, axis)
    } else {
        Vec2::splat(f32::MAX)
    };
    let mut impact = 0.0;
    let mut corona = 0.0;
    if reaction.x < REACTION_SPREAD * 1.6 {
        let outward = radial / radial.length().max(1.0);
        let flow = relative - outward * view.time() * 1800.0;
        let turbulence = textures.billows(flow * 0.000018, footprint * 0.000018);
        let detail = textures.billows(flow * 0.000075, footprint * 0.000075);
        let spread = (-square(reaction.x / REACTION_SPREAD) * 2.5).exp();
        let shell = reaction.y - (turbulence.w - 0.5) * 18_000.0;
        let thickness = 6500.0 + turbulence.y * 6500.0;
        let layer = (-square(shell / thickness)).exp();
        let front = (reaction.x * 0.00010 - view.time() * 1.3 + turbulence.z * 6.0).sin();
        let folds = fourth(1.0 - (detail.y * 2.0 - 1.0).abs());
        let ribbons = (-square((shell / thickness + front * 0.35) * 2.0)).exp();
        corona = spread
            * (layer * (0.00018 + folds * 0.0003) + ribbons * square(cube(front.abs())) * 0.0005);
        impact = (-square(reaction.x / 12_000.0) - square(reaction.y / 12_000.0)).exp()
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
            + vec3(0.30, 0.52, 1.0).lerp(vec3(1.0, 0.73, 0.48), noise.z) * (corona * 12.0))
            / density,
        ..VolumeSample::default()
    }
}

pub fn poles_raymarch_entry(ro: Vec3, rd: Vec3, t_max: f32) -> Vec2 {
    let axis = Vec3::Z;
    let oc = ro;
    let ad = axis.dot(rd);
    let ao = axis.dot(oc);
    let radial_origin = oc - axis * ao;
    let radial_direction = rd - axis * ad;
    let a = radial_direction.length_squared();
    let b = 2.0 * radial_origin.dot(radial_direction);
    let c = radial_origin.length_squared() - POLE_WIDTH * POLE_WIDTH;
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
