//! The planet's fixed features as solids: the Mineral Fields' vent chimneys, the swamps' water
//! geysers, the floating lakes and the ice crystals over the equator. Each is placed by
//! [`crate::scene::features`], so what is drawn is where the simulation has it.
//!
//! Positions are worked as offsets from each feature's own centre, never as absolute points this
//! far from the planet's centre, which round to an eighth of a metre.

use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    aur::{SPIRE_METRIC, energy_color, spire_axes, spire_coordinates},
    features::{
        Geyser, ICE_CELL, ICE_HIGH, ICE_LOW, LAKES, Lake, SWAMPS, VENT_FIELDS, Vent, crystal,
        geyser, geysers, lake, swamp, vent, vent_field, vents,
    },
    life::surface::Hit,
    math::{Power, intersect_sphere},
    planet::light_towards,
};
use core::f32::consts::TAU;
use spirv_std::{
    glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4Swizzles, vec2, vec3},
    num_traits::Float,
};

/// A frame standing at a direction from the planet's centre: east, up and north.
#[derive(Clone, Copy)]
struct Frame {
    origin: Vec3,
    east: Vec3,
    up: Vec3,
    north: Vec3,
}

impl Frame {
    fn at(up: Vec3, altitude: f32) -> Self {
        let east = vec3(-up.y, up.x, 0.0);
        let east = east / east.length().max(1e-6);
        Self {
            origin: up * (PLANET_RADIUS + altitude),
            east,
            up,
            north: up.cross(east),
        }
    }

    /// A point given relative to the planet's centre, in the frame.
    fn local(self, offset: Vec3) -> Vec3 {
        vec3(
            offset.dot(self.east),
            offset.dot(self.up),
            offset.dot(self.north),
        )
    }

    /// A direction in the frame, back in the planet's axes.
    fn outward(self, local: Vec3) -> Vec3 {
        self.east * local.x + self.up * local.y + self.north * local.z
    }
}

/// What a ray met: how far along it, the solid's normal in the planet's axes, and which feature.
#[derive(Clone, Copy)]
struct Found {
    t: f32,
    normal: Vec3,
    kind: u32,
    detail: Vec3,
}

const VENT: u32 = 1;
const GEYSER: u32 = 2;
const LAKE: u32 = 3;
const ICE: u32 = 4;

/// A cheap, smooth wobble in three dimensions, for crusts and churning water.
fn ripple(p: Vec3) -> f32 {
    (p.x * 1.7 + (p.y * 1.3).sin()).sin() * (p.z * 1.9 + (p.x * 0.7).sin()).sin()
        + (p.y * 2.3 + p.z * 0.6).sin() * 0.5
}

/// A vent chimney in its own frame: a squat, tapering cone of mineral crust rising out of the aur
/// to a cratered mouth, crusted and knobbed.
fn chimney(q: Vec3, vent: Vent) -> f32 {
    let (h, r) = (vent.height, vent.radius);
    let rise = (q.y / h).clamp(0.0, 1.0);
    let girth = r * (1.0 - rise * 0.45) * (1.0 + (1.0 - rise).power(4) * 0.6);
    let across = vec2(q.x, q.z).length();
    let outer = (across - girth).max(q.y - h).max(-q.y - 60.0);
    let throat = (across - girth * 0.45).max(h - r * 0.9 - q.y);
    let crust = ripple(q * (3.0 / r)) * r * 0.06;
    outer.max(-throat) * 0.8 + crust
}

/// A geyser's jet in its own frame: water thrown up out of the aur in a churning column that
/// widens as it rises and bursts into a crown of spray at its height.
fn jet(q: Vec3, geyser: Geyser, time: f32) -> f32 {
    let (h, r) = (geyser.height, geyser.radius);
    let rise = (q.y / h).clamp(0.0, 1.0);
    let girth = r * (1.0 + rise * 2.2);
    let angle = q.z.atan2(q.x);
    // The surface churns upward with the water, fast.
    let churn = ((q.y * 0.08 - time * 25.0) + (angle * 5.0).sin() * 1.5).sin()
        * ((q.y * 0.031 - time * 13.0 + angle * 3.0).sin())
        * girth
        * 0.18;
    let column = (vec2(q.x, q.z).length() - girth - churn)
        .max(q.y - h)
        .max(-q.y - 30.0);
    // Where it leaves the aur, the water boils up in a dome about its root.
    let dome =
        (q.length() - r * 3.5 * (1.0 + ripple(q * (0.6 / r) - time * 2.0) * 0.2)).max(-q.y - 30.0);
    let column = smooth_union(column, dome, r);
    let crown = q - Vec3::Y * h;
    let burst = crown.length() - r * 5.0 * (1.0 + ripple(crown * (0.5 / r) + time * 0.6) * 0.25);
    column.min(burst * 0.7)
}

/// Two solids melted into one over `k`.
fn smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b.lerp(a, h) - k * h * (1.0 - h)
}

/// A floating lake in its own frame, its surface at zero: almost flat across the top, curving down
/// beneath to its deepest at the middle.
fn lens(q: Vec3, lake: Lake, time: f32) -> f32 {
    let across = vec2(q.x, q.z).length();
    let bowl = vec2(across / lake.radius, q.y / lake.depth);
    let belly = (bowl.length() - 1.0) * lake.depth.min(lake.radius) * 0.9;
    let surface = q.y - ((q.x * 0.11 + time * 0.9).sin() * (q.z * 0.09 - time * 0.7).sin()) * 0.15;
    belly.max(surface)
}

/// How hard it is raining over lake `index` now: showers come and go over each on its own slow
/// rhythm, rarely, for the lakes hang where rain is scarce.
fn shower(index: u32, time: f32) -> f32 {
    let phase = time * 0.0021 + index as f32 * 2.3;
    ((phase.sin() + (phase * 2.7).sin() * 0.5) * 0.7).smoothstep(0.55, 0.85)
}

/// The streams pouring into a lake in rain, in its own frame: rainwater gathered from the air about
/// it and drawn down into the low pressure at its heart, falling in arcs from beyond its rim onto
/// its water. Nothing when it is dry.
fn streams(q: Vec3, lake: Lake, rain: f32, time: f32) -> f32 {
    if rain <= 0.0 {
        return f32::MAX;
    }
    let mut d = f32::MAX;
    for stream in 0..5 {
        let angle = stream as f32 * 1.256_637 + 0.4 + (time * 0.05 + stream as f32).sin() * 0.1;
        let out = vec3(angle.cos(), 0.0, angle.sin());
        let from = out * lake.radius * 2.2 + Vec3::Y * lake.radius * 0.9;
        let to = out * lake.radius * 0.7;
        let bend = out * lake.radius * 1.6 + Vec3::Y * lake.radius * 0.3;
        let thick = (2.0 + lake.radius * 0.008) * rain;
        let mut previous = from;
        for piece in 1..5 {
            let t = piece as f32 / 4.0;
            let point = from * (1.0 - t).power(2) + bend * (2.0 * t * (1.0 - t)) + to * (t * t);
            let along = q - previous;
            let axis = point - previous;
            let h = (along.dot(axis) / axis.length_squared().max(1e-6)).clamp(0.0, 1.0);
            let wobble = (q.y * 0.3 - time * 6.0 + stream as f32).sin() * thick * 0.25;
            d = d.min((along - axis * h).length() - thick * (1.0 - t * 0.4) - wobble);
            previous = point;
        }
    }
    d
}

/// A hexagonal prism, `half` its across-the-flats radius and half length, lying along y.
fn hexagonal(p: Vec3, half: Vec2) -> f32 {
    let k = vec3(-0.866_025_4, 0.5, 0.577_350_3);
    let mut q = vec2(p.x.abs(), p.z.abs());
    q -= 2.0 * vec2(k.x, k.y).dot(q).min(0.0) * vec2(k.x, k.y);
    let d = vec2(
        (q - vec2(q.x.clamp(-k.z * half.x, k.z * half.x), half.x)).length()
            * (q.y - half.x).signum_gl(),
        p.y.abs() - half.y,
    );
    d.x.max(d.y).min(0.0) + vec2(d.x.max(0.0), d.y.max(0.0)).length()
}

trait Sign {
    fn signum_gl(self) -> f32;
}

impl Sign for f32 {
    /// The sign of a number as a shader has it, zero for zero, without the infinite check that
    /// shaders reject.
    fn signum_gl(self) -> f32 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

/// A crystal in its own frame, `size` across: a hexagonal plate with six branching arms, a stout
/// column capped with points, or a few columns grown from one root. Plates and columns alike hang
/// broad side down to the updraft holding them, rocking a little.
fn ice_shape(q: Vec3, size: f32, form: u32, seed: f32, time: f32) -> f32 {
    let rock = (time * 0.2 + seed * 9.0).sin() * 0.12;
    let (s, c) = rock.sin_cos();
    let q = vec3(q.x * c - q.y * s, q.x * s + q.y * c, q.z);
    if form == 0 {
        // A plate, its arms branching.
        let mut d = hexagonal(q, vec2(size * 0.32, size * 0.04));
        for arm in 0..6 {
            let angle = arm as f32 * TAU / 6.0 + seed;
            let (sa, ca) = angle.sin_cos();
            let p = vec3(q.x * ca + q.z * sa, q.y, -q.x * sa + q.z * ca);
            let spoke = vec3(
                (p.x - size * 0.3).abs() - size * 0.28,
                p.y.abs() - size * 0.018,
                p.z.abs() - size * 0.025,
            );
            d = d.min(spoke.max_element());
            for branch in 0..2 {
                let along = size * (0.25 + branch as f32 * 0.17);
                let side = vec3(p.x - along, p.y, p.z.abs());
                let twig = vec3(
                    (side.x * 0.866 - side.z * 0.5).abs() - size * 0.012,
                    side.y.abs() - size * 0.014,
                    (side.x * 0.5 + side.z * 0.866 - size * 0.08).abs() - size * 0.08,
                );
                d = d.min(twig.max_element());
            }
        }
        d
    } else {
        // Columns lying across the updraft, capped with points.
        let column = |p: Vec3, length: f32| {
            let r = size * 0.12;
            let radial = vec2(p.y, p.z).length();
            let body = hexagonal(vec3(p.y, p.x, p.z), vec2(r, length + r * 1.2));
            body.max((p.x.abs() + radial * 1.2 - (length + r * 1.2)) * 0.64)
        };
        let mut d = column(q, size * 0.4);
        if form == 2 {
            for branch in 0..2 {
                let angle = 1.0 + branch as f32 * 2.1 + seed;
                let (sa, ca) = angle.sin_cos();
                let p = vec3(q.x * ca + q.z * sa, q.y, -q.x * sa + q.z * ca);
                d = d.min(column(p - vec3(0.0, 0.0, size * 0.1), size * 0.28));
            }
        }
        d
    }
}

/// Sphere-trace one solid in its frame, from `t` to `end`, returning where it is met.
fn march<F: Fn(Vec3) -> f32>(
    start: Vec3,
    rd: Vec3,
    t: f32,
    end: f32,
    field: F,
    epsilon: f32,
) -> f32 {
    let mut t = t;
    for _ in 0..80 {
        if t >= end {
            return f32::MAX;
        }
        let d = field(start + rd * t);
        if d < epsilon * t.max(1.0) {
            return t;
        }
        t += d.max(0.05);
    }
    f32::MAX
}

/// The normal of a solid in its frame at `p`.
fn normal<F: Fn(Vec3) -> f32>(p: Vec3, e: f32, field: F) -> Vec3 {
    let a = vec3(1.0, -1.0, -1.0);
    let b = vec3(-1.0, -1.0, 1.0);
    let c = vec3(-1.0, 1.0, -1.0);
    let d = Vec3::ONE;
    let n =
        a * field(p + a * e) + b * field(p + b * e) + c * field(p + c * e) + d * field(p + d * e);
    n / n.length().max(1e-6)
}

/// Where a ray from `ro` along `rd` (in the planet's axes) first meets a feature, before `limit`.
fn find(ro: Vec3, rd: Vec3, limit: f32, time: f32, angle: f32) -> Option<Found> {
    let mut best: Option<Found> = None;
    let nearer = |t: f32, best: &Option<Found>| t < limit && best.is_none_or(|found| t < found.t);
    let height = ro.length();
    let up = ro / height;
    for index in 0..VENT_FIELDS {
        let field = vent_field(index);
        let centre = field.up * PLANET_RADIUS;
        let bounds = intersect_sphere(ro - centre, rd, field.radius + 500.0);
        if bounds.x > bounds.y || bounds.y < 0.0 || !nearer(bounds.x.max(0.0), &best) {
            continue;
        }
        for which in 0..vents(index) {
            let vent = vent(index, which);
            let frame = Frame::at(vent.up, 0.0);
            let start = frame.local(ro - frame.origin);
            let way = frame.local(rd);
            let reach = vent.radius * 2.0 + vent.height;
            let sphere = intersect_sphere(start - Vec3::Y * vent.height * 0.4, way, reach);
            if sphere.x > sphere.y || sphere.y < 0.0 || !nearer(sphere.x.max(0.0), &best) {
                continue;
            }
            let t = march(
                start,
                way,
                sphere.x.max(0.0),
                sphere.y,
                |q| chimney(q, vent),
                angle,
            );
            if nearer(t, &best) {
                let at = start + way * t;
                let n = normal(at, (t * angle).max(0.05), |q| chimney(q, vent));
                best = Some(Found {
                    t,
                    normal: frame.outward(n),
                    kind: VENT,
                    detail: at / vent.height.max(1.0),
                });
            }
        }
    }
    for index in 0..SWAMPS {
        let field = swamp(index);
        let centre = field.up * PLANET_RADIUS;
        let bounds = intersect_sphere(ro - centre, rd, field.radius + 4500.0);
        if bounds.x > bounds.y || bounds.y < 0.0 || !nearer(bounds.x.max(0.0), &best) {
            continue;
        }
        for which in 0..geysers(index) {
            let geyser = geyser(index, which);
            let frame = Frame::at(geyser.up, 0.0);
            let start = frame.local(ro - frame.origin);
            let way = frame.local(rd);
            let reach = geyser.height * 0.55 + geyser.radius * 8.0;
            let sphere = intersect_sphere(start - Vec3::Y * geyser.height * 0.5, way, reach);
            if sphere.x > sphere.y || sphere.y < 0.0 || !nearer(sphere.x.max(0.0), &best) {
                continue;
            }
            let t = march(
                start,
                way,
                sphere.x.max(0.0),
                sphere.y,
                |q| jet(q, geyser, time),
                angle,
            );
            if nearer(t, &best) {
                let at = start + way * t;
                let n = normal(at, (t * angle).max(0.1), |q| jet(q, geyser, time));
                best = Some(Found {
                    t,
                    normal: frame.outward(n),
                    kind: GEYSER,
                    detail: vec3(at.y / geyser.height, 0.0, 0.0),
                });
            }
        }
    }
    for index in 0..LAKES {
        let lake = lake(index);
        let frame = Frame::at(lake.up, lake.altitude);
        let start = frame.local(ro - frame.origin);
        let way = frame.local(rd);
        let rain = shower(index, time);
        let reach = if rain > 0.0 { 2.6 } else { 1.05 };
        let sphere = intersect_sphere(start, way, lake.radius * reach + lake.depth);
        if sphere.x > sphere.y || sphere.y < 0.0 || !nearer(sphere.x.max(0.0), &best) {
            continue;
        }
        let water = |q: Vec3| lens(q, lake, time).min(streams(q, lake, rain, time));
        let t = march(start, way, sphere.x.max(0.0), sphere.y, water, angle);
        if nearer(t, &best) {
            let at = start + way * t;
            let n = normal(at, (t * angle).max(0.05), water);
            // A stream falling into it is shaded as falling water, the lake as a lake.
            let falling = streams(at, lake, rain, time) < lens(at, lake, time);
            best = Some(Found {
                t,
                normal: frame.outward(n),
                kind: if falling { GEYSER } else { LAKE },
                detail: if falling {
                    vec3(at.y / lake.radius, 0.0, 0.0)
                } else {
                    at / lake.radius
                },
            });
        }
    }
    if let Some(found) = crystals(
        ro,
        rd,
        limit.min(best.map_or(f32::MAX, |found| found.t)),
        time,
        angle,
        up,
    ) {
        best = Some(found);
    }
    best
}

/// Where a ray first meets an ice crystal, crossing the ice band one cell at a time.
fn crystals(ro: Vec3, rd: Vec3, limit: f32, time: f32, angle: f32, up: Vec3) -> Option<Found> {
    // Only where the band is near: the equatorial ice is far from most of the sky.
    if up.z.abs() > 0.45 {
        return None;
    }
    let low = intersect_sphere(ro, rd, PLANET_RADIUS + ICE_LOW);
    let high = intersect_sphere(ro, rd, PLANET_RADIUS + ICE_HIGH);
    if high.x > high.y || high.y < 0.0 {
        return None;
    }
    let mut t = high.x.max(0.0);
    let height = ro.length() - PLANET_RADIUS;
    let end = if height > ICE_LOW && low.x <= low.y && low.x > 0.0 {
        low.x
    } else if height < ICE_LOW && low.x <= low.y {
        t = t.max(low.y);
        high.y
    } else {
        high.y
    }
    .min(limit)
    .min(30_000.0);
    for _ in 0..64 {
        if t >= end {
            return None;
        }
        let point = ro + rd * t;
        let place = spire_coordinates(point);
        let cell = (place.xy() / ICE_CELL).floor();
        let local = place.xy() - (cell + 0.5) * ICE_CELL;
        let (first, second) = spire_axes(point);
        let velocity =
            vec2(first.dot(rd), second.dot(rd)) * (PLANET_RADIUS / point.length() / SPIRE_METRIC);
        let exit = |position: f32, speed: f32| {
            if speed > 1.0e-6 {
                (ICE_CELL * 0.5 - position) / speed
            } else if speed < -1.0e-6 {
                (-ICE_CELL * 0.5 - position) / speed
            } else {
                f32::MAX
            }
        };
        let leave = exit(local.x, velocity.x).min(exit(local.y, velocity.y)) + 1.0;
        match crystal(cell, time) {
            None => t += leave.max(1.0),
            Some((middle, size, form, seed)) => {
                let q = vec3(local.x - middle.x, place.z - middle.z, local.y - middle.y);
                // Far from the crystal, its bounding sphere is enough to step by.
                let bound = q.length() - size * 0.7;
                let d = if bound > size * 0.2 {
                    bound
                } else {
                    ice_shape(q, size, form, seed, time)
                } * SPIRE_METRIC;
                if d < (t * angle * 0.5).max(0.02) {
                    let e = (t * angle).max(0.05);
                    let field = |o: Vec3| {
                        let p = spire_coordinates(point + o);
                        let l = p.xy() - (cell + 0.5) * ICE_CELL;
                        ice_shape(
                            vec3(l.x - middle.x, p.z - middle.z, l.y - middle.y),
                            size,
                            form,
                            seed,
                            time,
                        )
                    };
                    let n = normal(Vec3::ZERO, e, field);
                    return Some(Found {
                        t,
                        normal: n,
                        kind: ICE,
                        detail: vec3(seed, size, 0.0),
                    });
                }
                t += d.max(0.5).min(leave.max(0.5));
            }
        }
    }
    None
}

/// The features along a ray, as an opaque hit, if one is met before `limit`.
pub fn trace(origin: Vec3, direction: Vec3, view: &ViewUniform, limit: f32, angle: f32) -> Hit {
    let ro = view.to_planet(origin);
    let rd = view.planet_rotation.conjugate() * direction;
    let time = view.time();
    let Some(found) = find(ro, rd, limit, time, angle) else {
        return Hit::MISS;
    };
    let point = ro + rd * found.t;
    let color = shade(found, point, rd, view);
    let world = origin + direction * found.t;
    let color = color.lerp(
        vec3(0.42, 0.38, 0.58) * view.sky_horizon.y + view.ambient_color.xyz() * 0.3,
        1.0 - (-found.t / 32_500.0).exp(),
    );
    Hit {
        color_depth: color.extend(found.t),
        normal: view.planet_rotation * found.normal,
        previous_position: world,
        transmittance: Vec3::ZERO,
        part: 0,
    }
}

/// The colour of a feature where a ray met it.
fn shade(found: Found, point: Vec3, rd: Vec3, view: &ViewUniform) -> Vec3 {
    let n = found.normal;
    let up = point / point.length();
    let altitude = point.length() - PLANET_RADIUS;
    let light = light_towards(point);
    let pole = view.pole_color.xyz().min(Vec3::splat(1.5));
    let ambient = view.ambient_color.xyz().min(Vec3::splat(0.8));
    let glow = view.sky_horizon.y;
    // The aur lights everything from below, strongest close over it.
    let from_below = vec3(0.42, 0.36, 0.62) * glow * (0.35 + 0.65 * (0.5 - 0.5 * n.dot(up)));
    let reflected = rd - n * (2.0 * rd.dot(n));
    let facing = (-rd).dot(n).max(0.0);
    let fresnel = 0.04 + 0.96 * (1.0 - facing).power(5);
    let sky = ambient * vec3(0.8, 0.9, 1.15) * (0.4 + 0.6 * reflected.dot(up).max(0.0));
    let time = view.time();
    match found.kind {
        VENT => {
            // The aur's own crust, as the spires are: obsidian-dark glass with a drifting film of
            // violet and teal, veined with aur that pulses up towards the mouth, where it
            // smoulders in the throat; pale mineral crust is laid down in streaks about the rim.
            let height = found.detail.y;
            let film = (1.0 - facing).power(2);
            let streak = (height * 40.0 + ripple(found.detail * 30.0) * 2.0).sin() * 0.5 + 0.5;
            let albedo = vec3(0.035, 0.03, 0.05)
                .lerp(vec3(0.02, 0.06, 0.07), film)
                .lerp(
                    vec3(0.5, 0.44, 0.26),
                    streak.power(8) * height.smoothstep(0.5, 1.0) * 0.7,
                );
            let vein = (ripple(found.detail * 12.0) * 3.0)
                .sin()
                .smoothstep(0.86, 0.99)
                * (0.3 + (height * 30.0 - time * 1.5).sin().power(2));
            let mouth = height.smoothstep(0.82, 1.0) * (1.0 - n.dot(up).max(0.0));
            let charge = energy_color(0.7, 0.55 + 0.25 * (time * 0.7).sin());
            let heat = charge * glow * (vein * 1.2 + mouth * 2.5);
            let sheen = pole * reflected.dot(light).max(0.0).powf(48.0) * 0.6;
            let lit = albedo * (pole * n.dot(light).max(0.0) + ambient + from_below * 6.0);
            lit + heat + sheen + sky * fresnel
        }
        GEYSER => {
            // Churning white water, bluer deep in the column, lit from beneath by the aur, its
            // foam streaming upward.
            let foam = (found.detail.x * 30.0 - time * 8.0).sin() * 0.5 + 0.5;
            let water = vec3(0.55, 0.68, 0.78).lerp(vec3(0.92, 0.95, 0.98), foam * 0.6 + 0.3);
            water * (pole * (0.4 + 0.6 * n.dot(light).max(0.0)) + ambient + from_below * 1.4)
                + sky * fresnel
        }
        LAKE => {
            // Its still top mirrors the sky and catches the light in its ripples; beneath, the
            // water glows blue-green with the aur's light coming up through it, brightest where it
            // is thinnest at the rim, and the rim churns white where the walls of rising air hold
            // it in.
            let across = vec2(found.detail.x, found.detail.z);
            let reach = across.length();
            let top = n.dot(up) > 0.5;
            let ripple_normal = if top {
                let wave = vec2(
                    (across.x * 90.0 + time * 1.3).sin() + (across.y * 70.0 - time).sin() * 0.6,
                    (across.y * 85.0 + time * 1.1).sin()
                        + (across.x * 60.0 + time * 0.8).sin() * 0.6,
                ) * 0.04;
                let east = vec3(-up.y, up.x, 0.0);
                let east = east / east.length().max(1e-6);
                let tilted = n + east * wave.x + up.cross(east) * wave.y;
                tilted / tilted.length().max(1e-6)
            } else {
                n
            };
            let mirrored = rd - ripple_normal * (2.0 * rd.dot(ripple_normal));
            let facing = (-rd).dot(ripple_normal).max(0.0);
            let fresnel = 0.02 + 0.98 * (1.0 - facing).power(5);
            let sky = ambient * vec3(0.75, 0.85, 1.2) * (0.5 + 0.8 * mirrored.dot(up).max(0.0))
                + pole * mirrored.dot(light).max(0.0).powf(6.0) * 0.4;
            let glint = pole * mirrored.dot(light).max(0.0).powf(220.0) * 6.0;
            let thin = reach.smoothstep(0.55, 1.0);
            let deep = vec3(0.02, 0.12, 0.16).lerp(vec3(0.1, 0.45, 0.48), thin);
            let body = deep * (ambient * 0.5 + from_below * if top { 1.2 } else { 3.5 })
                + vec3(0.05, 0.3, 0.32) * glow * thin * 0.6;
            let foam = reach.smoothstep(0.9, 1.0)
                * (0.6
                    + 0.4 * (reach * 200.0 - time * 3.0 + across.x.atan2(across.y) * 12.0).sin());
            let surface =
                body * (1.0 - fresnel) + (sky + glint) * fresnel.max(if top { 0.2 } else { 0.05 });
            surface.lerp(
                vec3(0.85, 0.9, 0.95) * (ambient + pole * 0.5 + from_below),
                foam * 0.8,
            )
        }
        _ => {
            // Clear ice. It is the aur's light that fills it: what the faces show, through the
            // ice or mirrored in it, is the glowing aur below or the dark sky above, as each face
            // turns; the light bent through it is tinted blue the deeper it goes, threaded with
            // frost, and glints of the pole light and the aur's own sparkle off its edges.
            let seed = found.detail.x;
            let aur = vec3(0.55, 0.42, 0.78) * glow * 1.6;
            let seen = |way: Vec3| {
                let rise = way.dot(up);
                aur.lerp(ambient * vec3(0.7, 0.8, 1.1), rise.smoothstep(-0.25, 0.35))
                    + pole * way.dot(light).max(0.0).powf(12.0) * 0.8
            };
            let bent = rd + n * (rd.dot(n).abs() * 0.24);
            let bent = bent / bent.length().max(1e-6);
            let depth = (1.0 - facing).power(2);
            let tint = vec3(0.75, 0.92, 1.0).lerp(vec3(0.3, 0.6, 0.95), depth);
            let frost = (ripple(point * (3.0 / found.detail.y.max(1.0)) + seed * 7.0) * 2.5)
                .sin()
                .smoothstep(0.7, 1.0)
                * 0.35;
            let within = seen(bent) * tint + vec3(0.8, 0.9, 1.0) * frost * (ambient + aur * 0.5);
            let mirrored = seen(reflected);
            let glints = (pole * reflected.dot(light).max(0.0).powf(300.0) * 10.0
                + aur * reflected.dot(-up).max(0.0).powf(80.0) * 3.0)
                * (0.6 + 0.4 * (time * 3.0 + seed * 40.0 + altitude * 0.1).sin());
            within * (1.0 - fresnel) + mirrored * fresnel.max(0.06) + glints
        }
    }
}
