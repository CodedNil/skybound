use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    math::{cube, fourth, square},
    planet::{REACTION_RADIUS, REACTION_SPREAD, reaction_coordinates},
    render::{march::VolumeSample, textures::Textures},
};
use spirv_std::{
    glam::{FloatExt, Vec2, Vec3, Vec4Swizzles, vec2, vec3},
    num_traits::Float,
};

/// The marched part of the poles: the shell where each beam strikes the
/// reaction and spreads into a corona.
pub const POLE_WIDTH: f32 = REACTION_SPREAD * 1.6 * 1.04;

/// The beams' light along a ray, integrated in closed form: each is a set
/// of Gaussian falloffs about the polar axis, so its integral along any
/// line is a Gaussian at the ray's closest approach times an erf window.
/// Returns the radiance per unit of optical depth, the optical depth, and
/// where along the ray the beam lies.
pub fn beam(ro: Vec3, rd: Vec3, limit: f32, view: &ViewUniform) -> (Vec3, f32, f32) {
    let radial_origin = ro.truncate();
    let radial_direction = rd.truncate();
    let a = radial_direction.length_squared().max(1.0e-6);
    let closest = (-radial_origin.dot(radial_direction) / a).clamp(0.0, limit);
    let point = ro + rd * closest;
    let distance = point.truncate().length_squared();
    let height = (point.z.abs() - PLANET_RADIUS).max(0.0);
    // The beams stand on the poles, not through the planet.
    if point.z.abs() < PLANET_RADIUS * 0.9 {
        return (Vec3::ZERO, 0.0, limit);
    }
    // Barely spreading as it climbs, and as bright per unit length all the
    // way up, so it reads as reaching on into space.
    let width = 1200.0 + height.sqrt() * 1.5;
    let thinning = 1200.0 / width;
    let root = a.sqrt();
    let window = |w: f32| {
        let scale = root / w;
        let full = (-distance / (w * w)).exp() * w * 0.886_227 / root;
        full * (erf((limit - closest) * scale) + erf(closest * scale))
    };
    // Pulses travel up it; a ray crosses the beam too briefly to see more.
    let flow = point.z.abs() * 0.000_04 - view.time() * 0.9;
    let pulse = 0.75 + 0.15 * flow.sin() + 0.1 * (flow * 2.7 + 1.3).sin();
    let core = window(width) * 0.002 * thinning * pulse;
    let sheath = window(width * 6.0) * 0.000_12 * thinning * (0.8 + 0.2 * (flow * 0.6).cos());
    let halo = window(width * 25.0) * 0.000_004 * thinning;
    let column = core + sheath + halo;
    if column < 1.0e-9 {
        return (Vec3::ZERO, 0.0, limit);
    }
    let emission = (vec3(1.0, 0.85, 0.6) * (core * 90.0)
        + vec3(0.2, 0.5, 1.0) * (sheath * 25.0)
        + vec3(0.15, 0.08, 0.4) * (halo * 4.0))
        / column;
    let radiance =
        emission * view.sky_horizon.y + vec3(0.1, 0.15, 0.25) * view.ambient_color.xyz() * 0.2;
    (radiance, column * 0.018, closest)
}

/// Abramowitz and Stegun's approximation to the error function.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_7 + t * (1.421_413_7 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

pub fn sample_poles(
    relative: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
) -> VolumeSample {
    let axis = Vec3::Z;
    let along = relative.dot(axis);
    let radial = relative - axis * along;
    let distance = radial.length_squared();
    let near_reaction = distance < square(POLE_WIDTH)
        && (relative.length() - REACTION_RADIUS).abs() < 45_000.0 + footprint * 2.0;
    if !near_reaction {
        return VolumeSample::default();
    }
    let noise = textures.base_filtered(
        (radial + axis * (along - view.time() * 1200.0)) * 0.000025,
        footprint * 0.000025,
    );
    let reaction = reaction_coordinates(relative, axis);
    let mut impact = 0.0;
    let mut corona = 0.0;
    if reaction.x < REACTION_SPREAD * 1.6 {
        let outward = radial / radial.length().max(1.0);
        let flow = relative - outward * view.time() * 1800.0;
        let turbulence = textures.billows(flow * 0.000018, footprint * 0.000018);
        let detail = textures.billows(flow * 0.000075, footprint * 0.000075);
        let spread = (-square(reaction.x / REACTION_SPREAD) * 2.5).exp();
        let shell = reaction.y - (turbulence.w - 0.5) * 18_000.0;
        let base_thickness = 6500.0 + turbulence.y * 6500.0;
        let thickness = (base_thickness * base_thickness + footprint * footprint).sqrt();
        let layer = (-square(shell / thickness)).exp();
        let front = (reaction.x * 0.00010 - view.time() * 1.3 + turbulence.z * 6.0).sin();
        let folds = fourth(1.0 - (detail.y * 2.0 - 1.0).abs());
        let ribbons = (-square((shell / thickness + front * 0.35) * 2.0)).exp();
        corona = spread
            * (layer * (0.00018 + folds * 0.0003) + ribbons * square(cube(front.abs())) * 0.0005)
            * (base_thickness / thickness);
        impact = (-square(reaction.x / 12_000.0) - square(reaction.y / 12_000.0)).exp()
            * (0.008 + noise.w * 0.004);
    }
    let density = impact + corona;
    if density < 0.0000001 {
        return VolumeSample::default();
    }
    VolumeSample {
        density,
        color: vec3(0.1, 0.15, 0.25),
        emission: (vec3(1.0, 0.96, 0.82) * (impact * 110.0)
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
