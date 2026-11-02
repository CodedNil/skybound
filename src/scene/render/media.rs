//! What fills the air besides the cloud and the aur: the dusty fog of the Mineral Fields, the rain
//! about the camera, and the auroras over the equator. Each is integrated along a ray on its own,
//! as one layer the march takes in where it passes it.

use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    features::{VENT_FIELDS, apart, vent_field},
    math::{Power, intersect_sphere},
};
use spirv_std::{
    glam::{FloatExt, Vec3, Vec4Swizzles, vec3},
    num_traits::Float,
};

/// A medium integrated along a ray: the light it gives, how thick it is, and how far along the ray
/// it lies.
#[derive(Clone, Copy)]
pub struct Layer {
    pub light: Vec3,
    pub depth: f32,
    pub at: f32,
}

impl Layer {
    pub const NONE: Self = Self {
        light: Vec3::ZERO,
        depth: 0.0,
        at: f32::MAX,
    };

    /// A layer giving `light` in all, over a thin sliver of extinction, at `at`.
    fn glowing(light: Vec3, at: f32) -> Self {
        const THIN: f32 = 0.01;
        Self {
            light: light / (1.0 - (-THIN).exp()),
            depth: THIN,
            at,
        }
    }
}

/// The dusty fog over the Mineral Fields: thick and low over the vents, thinning upward and outward
/// from them, the colour of the minerals in it and lit from beneath by the aur, whose light it
/// mostly blocks.
pub fn dust(ro: Vec3, rd: Vec3, limit: f32, view: &ViewUniform) -> Layer {
    let mut depth = 0.0;
    let mut at = 0.0;
    for index in 0..VENT_FIELDS {
        let field = vent_field(index);
        let reach = field.radius * 1.6;
        let span = intersect_sphere(ro - field.up * PLANET_RADIUS, rd, reach);
        let (start, end) = (span.x.max(0.0), span.y.min(limit));
        if span.x > span.y || start >= end {
            continue;
        }
        let step = (end - start) / 8.0;
        for sample in 0..8 {
            let t = start + step * (sample as f32 + 0.5);
            let point = ro + rd * t;
            let height = point.length();
            let up = point / height;
            let altitude = height - PLANET_RADIUS;
            let within = 1.0 - (apart(up, field.up) / reach).smoothstep(0.35, 1.0);
            let density = within * (-altitude.max(0.0) / 450.0).exp() * 0.0004;
            depth += density * step;
            at += t * density * step;
        }
    }
    if depth <= 1.0e-6 {
        return Layer::NONE;
    }
    let glow = view.sky_horizon.y;
    let ambient = view.ambient_color.xyz();
    let light = vec3(0.22, 0.17, 0.13) * (ambient * 0.7 + view.pole_color.xyz() * 0.05)
        + vec3(0.3, 0.2, 0.34) * glow * 0.35;
    Layer {
        light,
        depth,
        at: at / depth,
    }
}

/// The height the auroras' lower edges hang at, and the height they fade out by.
const AURORA_LOW: f32 = 22_000.0;
const AURORA_HIGH: f32 = 42_000.0;

/// The auroras over the equator: curtains of light hung high in the thin air, each a sheet folded
/// along the band, sharp at its lower edge, green there, fading up through violet to a dim red, and
/// striated with rays. They fold and drift slowly, and brighten and fade in long surges.
pub fn aurora(ro: Vec3, rd: Vec3, limit: f32, view: &ViewUniform) -> Layer {
    let height = ro.length();
    if height > PLANET_RADIUS + AURORA_LOW {
        return Layer::NONE;
    }
    let low = intersect_sphere(ro, rd, PLANET_RADIUS + AURORA_LOW);
    let high = intersect_sphere(ro, rd, PLANET_RADIUS + AURORA_HIGH);
    // Below the sheets, a ray climbs out through the lower sphere and on through the upper.
    let start = low.y.max(0.0);
    let end = high.y.min(limit).min(start + 260_000.0);
    if low.x > low.y || start >= end || limit < start {
        return Layer::NONE;
    }
    // Only near the equatorial band is there anything to see.
    let mid = (ro + rd * (start + end) * 0.5) / height;
    if mid.z.abs() > 0.7 {
        return Layer::NONE;
    }
    let time = view.time();
    let step = (end - start) / 32.0;
    let mut light = Vec3::ZERO;
    let mut at = 0.0;
    let mut weight = 0.0;
    for sample in 0..32 {
        let t = start + step * (sample as f32 + 0.5);
        let point = ro + rd * t;
        let radius = point.length();
        let up = point / radius;
        let latitude = up.z.clamp(-1.0, 1.0).asin();
        let band = 1.0 - (latitude.abs() / 0.45).smoothstep(0.4, 1.0);
        if band <= 0.0 {
            continue;
        }
        let longitude = up.y.atan2(up.x);
        let rise =
            ((radius - PLANET_RADIUS - AURORA_LOW) / (AURORA_HIGH - AURORA_LOW)).clamp(0.0, 1.0);
        // Sheets running along the band, folded and drifting.
        let across = sheets(latitude, longitude, time);
        // A step may cross several sheets at once: what it sees of them is their average then, so
        // the thin sheets neither alias nor vanish between samples.
        let before = point - rd * step * 0.5;
        let after = point + rd * step * 0.5;
        let spread = (sheets_at(before, time) - sheets_at(after, time))
            .abs()
            .min(1.0);
        let sharp = (-(across.fract() - 0.5).power(2) * 160.0).exp();
        let sheet = sharp.lerp(0.14, spread);
        // Rays down each curtain, faded to their average where a pixel spans many of them.
        let fine = (t * PIXEL * 260.0 / PLANET_RADIUS).min(1.0);
        let rays = (0.4
            + 0.6
                * (longitude * 260.0 + across * 3.0 + time * 0.2)
                    .sin()
                    .power(2))
        .lerp(0.7, fine);
        let edge = rise.smoothstep(0.0, 0.04) * (-rise * 2.4).exp();
        let surge = 0.35
            + 0.65
                * (longitude * 2.0 - time * 0.012 + 1.3)
                    .sin()
                    .max(0.0)
                    .powf(1.5);
        let glow = sheet * rays * edge * surge * band;
        let colour = vec3(0.25, 1.0, 0.55)
            .lerp(vec3(0.55, 0.25, 0.85), rise.smoothstep(0.1, 0.5))
            .lerp(vec3(0.5, 0.08, 0.12), rise.smoothstep(0.5, 1.0));
        let emitted = glow * step * 0.0012;
        light += colour * emitted;
        at += t * emitted;
        weight += emitted;
    }
    if weight <= 1.0e-7 {
        return Layer::NONE;
    }
    Layer::glowing(light, at / weight)
}

/// Roughly how wide a pixel is, as an angle.
const PIXEL: f32 = 0.0015;

/// Where across the auroras' sheets a place at a latitude and longitude lies: whole numbers fall
/// between sheets, which run along the band, folded and drifting.
fn sheets(latitude: f32, longitude: f32, time: f32) -> f32 {
    latitude * 9.0
        + (longitude * 5.0 + time * 0.03).sin() * 0.55
        + (longitude * 17.0 - time * 0.07 + latitude * 12.0).sin() * 0.18
        + (longitude * 43.0 + time * 0.11).sin() * 0.05
}

/// [`sheets`] at a point relative to the planet's centre.
fn sheets_at(point: Vec3, time: f32) -> f32 {
    let up = point / point.length();
    sheets(up.z.clamp(-1.0, 1.0).asin(), up.y.atan2(up.x), time)
}

/// Rain about the camera at the `layer`th of four distances out from it: streaks falling past
/// through a band of air, driven slantwise by the wind, as dense as the rain is heavy.
pub fn rain(direction: Vec3, up: Vec3, layer: u32, limit: f32, view: &ViewUniform) -> Layer {
    let heavy = view.rain.x;
    let distance = 3.0 + (layer * layer) as f32 * 7.0;
    if heavy <= 0.01 || distance >= limit {
        return Layer::NONE;
    }
    let drift = view.rain.yzw();
    let point = direction * distance;
    let height = point.dot(up);
    let level = point - up * height;
    let east = up.cross(Vec3::Z + Vec3::X * 0.3);
    let east = east / east.length().max(1e-6);
    let north = up.cross(east);
    let around = level.dot(north).atan2(level.dot(east)) * distance;
    // Drops fall at nine metres a second, slanted by the wind.
    let fall = 9.0;
    let slant = drift.dot(east) / fall;
    let time = view.time();
    let lane = (around - height * slant) / 0.22;
    let column = lane.floor();
    let seed = ((column * 12.989 + layer as f32 * 78.233).sin() * 43_758.547)
        .fract()
        .abs();
    if seed > heavy * 0.8 {
        return Layer::NONE;
    }
    let drop = ((height + time * fall + seed * 37.0) / 2.6).fract();
    let streak = (1.0 - ((lane - column) - 0.5).abs() * 9.0).max(0.0) * drop.smoothstep(0.55, 0.95);
    if streak <= 0.0 {
        return Layer::NONE;
    }
    let light = view.ambient_color.xyz() * 1.4 + view.pole_color.xyz() * 0.2;
    Layer {
        light,
        depth: streak * heavy * 0.25 / (1.0 + layer as f32),
        at: distance,
    }
}
