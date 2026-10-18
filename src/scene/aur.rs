//! The aur ocean: its current and arcs, crystal spires, and lightning.
use core::f32::consts::{PI, TAU};

use spirv_std::glam::{
    FloatExt, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::PLANET_RADIUS;
use crate::scene::math::square;
use crate::scene::planet::surface_point;

pub const CURRENT: Vec3 = vec3(35.0, -20.0, 0.0);

pub const EJECTION_CELL: f32 = 40_000.0;

pub const EJECTION_WIDTH: f32 = 1050.0;

pub const EJECTION_REACH: f32 = 36_000.0;

pub const EJECTION_HEIGHT: f32 = 12_150.0;

pub const EJECTION_GRID_HALF: i32 = (PI * PLANET_RADIUS / EJECTION_CELL) as i32 + 2;

pub const EJECTION_GRID_WIDTH: u32 = EJECTION_GRID_HALF as u32 * 2;

pub const EJECTION_GRID_HEIGHT: u32 = EJECTION_GRID_HALF as u32 + 1;

// Crystal spires that pierce the aur ocean, in sparse archipelagos.
//
// Positions are surface coordinates: longitude and colatitude scaled by the
// planet radius, and altitude. Every spire stays `SPIRE_MARGIN` inside its cell,
// so a cell's own spires and the distance to its border bound the field.
pub const SPIRE_CELL: f32 = 26_000.0;

/// Spires are buried well below the deepest wave trough.
pub const SPIRE_BOTTOM: f32 = -5000.0;

pub const SPIRE_TOP: f32 = 9600.0;

const SPIRE_REACH: f32 = 10_500.0;

const SPIRE_MARGIN: f32 = SPIRE_CELL * 0.5 - SPIRE_REACH;

#[derive(Clone, Copy, Default)]
pub struct Spire {
    pub base: Vec2,
    pub lean: Vec2,
    pub tip: f32,
    pub radius: f32,
    pub flutes: f32,
    pub twist: f32,
}

pub fn energy_color(warm: f32, charged: f32) -> Vec3 {
    let violet = vec3(0.12, 0.09, 0.23);
    let rose = vec3(0.38, 0.24, 0.32);
    violet
        .lerp(rose, warm.smoothstep(0.35, 0.65))
        .lerp(vec3(0.40, 0.61, 0.57), charged.smoothstep(0.4, 0.65))
}

pub fn curl(mut pos: Vec2, time: f32) -> Vec2 {
    let mut direction = vec2(0.6, 0.8);
    let mut frequency = 1.6;
    for octave in 0..4 {
        let perpendicular = vec2(-direction.y, direction.x);
        let phase = frequency * pos.dot(perpendicular)
            + time * (0.18 + octave as f32 * 0.035)
            + octave as f32;
        pos += direction * (0.6 * phase.sin() / frequency);
        direction = vec2(
            direction.x * -0.737 - direction.y * 0.676,
            direction.x * 0.676 + direction.y * -0.737,
        );
        frequency *= 1.4;
    }
    pos
}

/// The planet-scale character of the aur at a surface direction: a calm,
/// warm equatorial belt, storm tracks through the mid latitudes broken into
/// slowly drifting regions, and charged polar seas. Analytic so the host's
/// lightning catalogue agrees with the shaders.
/// x: storminess, y: warmth, z: charge, w: brightness.
pub fn aur_climate(up: Vec3, time: f32) -> Vec4 {
    let latitude = up.z.clamp(-1.0, 1.0).asin().abs();
    let tropics = 1.0 - latitude.smoothstep(0.12, 0.42);
    let tracks = (-square((latitude - 0.78) / 0.28)).exp();
    let polar = latitude.smoothstep(0.95, 1.35);
    let drift = time * 0.0004;
    let p = up * 3.1;
    let regions = ((p.x * 1.7 + drift).sin() * (p.y * 2.3 - drift * 0.7).sin()
        + (p.z * 2.9 + p.x * 1.3 + drift * 1.3).sin() * 0.6)
        * 0.3
        + 0.5;
    let cells = (p.x * 5.3 + p.z * 3.1 - drift * 2.0).sin() * (p.y * 4.7 + drift).sin() * 0.5 + 0.5;
    let storm =
        (tracks * 0.75 + regions * 0.7 + cells * 0.25 - tropics * 0.35 - 0.38).clamp(0.0, 1.0);
    let warmth = (tropics * 0.75 + (1.0 - regions) * 0.35).clamp(0.0, 1.0);
    let charge = (polar * 0.7 + storm * 0.4 + cells * 0.15).clamp(0.0, 1.0);
    Vec4::new(
        storm,
        warmth,
        charge,
        0.82 + storm * 0.35 + charge * 0.2 - tropics * 0.12,
    )
}

pub fn ambient(pos: Vec3, time: f32) -> Vec3 {
    let fade = (1.0 - pos.z.max(0.0) / 6000.0).clamp(0.0, 1.0);
    let flow = curl((pos - CURRENT * time).xy() * 0.001, time);
    let pulse = 0.8 + 0.2 * (time * 1.2 + flow.x * 7.0).sin();
    vec3(0.035, 0.008, 0.065) * fade * pulse
}

fn eruption_hash(cell: Vec2) -> Vec3 {
    let mut hash = (cell.xyx() * vec3(0.1031, 0.1030, 0.0973)).fract_gl();
    hash += hash.dot(hash.yzx() + 33.33);
    ((hash.xxy() + hash.yzz()) * hash.zyx()).fract_gl()
}

/// One eruption per cell and cycle, far more often in stormy seas. The
/// loop rises out of the veil, holds while plasma drains down its legs, then
/// parts at the apex and fades. Size, height and orientation vary per cycle.
pub fn ejection(cell: Vec2, time: f32) -> [Vec4; 2] {
    let seed = eruption_hash(cell + vec2(19.7, 63.1));
    let period = 45.0 + seed.x * 60.0;
    let cycle = time / period + seed.y;
    let phase = (cycle.fract_gl() * period / (6.0 + seed.z * 6.0)).min(1.0);
    if phase >= 1.0 {
        return [Vec4::ZERO; 2];
    }
    let random = eruption_hash(cell + vec2(37.1, 91.7) * (cycle.floor() + 1.0));
    let centre = (cell + random.xy()) * EJECTION_CELL;
    let up = surface_point(centre.extend(0.0)).normalize();
    if random.z > 0.25 + aur_climate(up, time).x * 0.9 {
        return [Vec4::ZERO; 2];
    }
    let angle = random.z * TAU * 7.0;
    let radius = 7000.0 + seed.x * 9000.0;
    [
        centre.extend(phase).extend(1.0),
        Vec4::new(
            angle.cos(),
            angle.sin(),
            radius,
            (4000.0 + seed.z * 8000.0).min(radius - 2200.0),
        ),
    ]
}

/// How much plasma a loop holds at a point along it.
pub fn ejection_fill(progress: f32, phase: f32) -> f32 {
    let alive = phase.smoothstep(0.0, 0.08) * (1.0 - phase.smoothstep(0.78, 1.0));
    let apex = (-square((progress - 0.5) / 0.16)).exp();
    alive * (1.0 - phase.smoothstep(0.55, 0.85) * apex)
}

/// The loop's circle sinks below the veil while it is still rising.
fn ejection_centre(event: [Vec4; 2]) -> f32 {
    let rise = 0.3 + 0.7 * event[0].z.smoothstep(0.0, 0.3);
    event[1].w * rise - event[1].z
}

fn ejection_coordinates(position: Vec3, event: [Vec4; 2]) -> Vec3 {
    let delta = position.xy() - event[0].truncate().truncate();
    let axis = event[1].truncate().truncate();
    let along = delta.dot(axis);
    let across = delta.dot(vec2(-axis.y, axis.x));
    vec3(along, position.z - ejection_centre(event), across)
}

pub fn ejection_field(position: Vec3, time: f32) -> Vec4 {
    ejection_field_with(position, 511, |cell| ejection(cell, time))
}

/// Distance to the nearest loop's axis, progress along it from one foot to
/// the other, its fill, and the angle around the axis for braided strands.
pub(crate) fn ejection_field_with(
    position: Vec3,
    mut active: u32,
    event: impl Fn(Vec2) -> [Vec4; 2],
) -> Vec4 {
    let mut nearest = 1.0e20;
    let mut local = Vec3::ZERO;
    let mut radial = 0.0;
    let mut phase = 1.0;
    let cell = (position.xy() / EJECTION_CELL).floor();
    while active != 0 {
        let index = active.trailing_zeros();
        active &= active - 1;
        let neighbour = cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
        let outside =
            ((position.xy() / EJECTION_CELL - neighbour - 0.5).abs() - 0.5).max(Vec2::ZERO);
        if outside.max_element() * EJECTION_CELL > EJECTION_REACH {
            continue;
        }
        let event = event(neighbour);
        if event[0].w < 0.01 {
            continue;
        }
        let candidate = ejection_coordinates(position, event);
        let offset = candidate.xy().length() - event[1].z;
        let distance_squared = offset * offset + candidate.z * candidate.z;
        if distance_squared < nearest {
            nearest = distance_squared;
            local = candidate;
            radial = offset;
            phase = event[0].z;
        }
    }
    if nearest > 1.0e19 {
        return Vec4::new(1.0e10, 0.0, 0.0, 0.0);
    }
    let progress = (local.y.atan2(local.x) / PI).clamp(0.0, 1.0);
    Vec4::new(
        nearest.sqrt(),
        progress,
        ejection_fill(progress, phase),
        local.z.atan2(radial),
    )
}

fn spire_hash(p: Vec2) -> Vec4 {
    let mut q =
        (Vec4::new(p.x, p.y, p.x, p.y) * Vec4::new(0.1031, 0.1030, 0.0973, 0.1099)).fract_gl();
    q += q.dot(q.wzxy() + 33.33);
    ((q.xxyz() + q.yzzw()) * q.zywx()).fract_gl()
}

/// Up to four spires around one jittered centre; regions of cells share an
/// occupancy so spires gather into archipelagos with open ocean between.
pub fn spire_count(cell: Vec2) -> usize {
    let region = spire_hash((cell / 4.0).floor() + 71.3);
    let seed = spire_hash(cell);
    if seed.x > region.x * region.x * region.x * region.x * 0.9 {
        0
    } else {
        1 + (seed.y * 3.99) as usize
    }
}

pub fn spire(cell: Vec2, index: usize) -> Spire {
    // Clusters wander far inside their cells so no lattice shows through.
    let centre = (spire_hash(cell).zw() - 0.5) * 8000.0;
    let main = spire_hash(cell + 3.1);
    let tallest = 5000.0 + main.x * 4500.0;
    if index == 0 {
        return Spire {
            base: centre,
            lean: (main.zw() - 0.5) * 0.08,
            tip: tallest,
            radius: 1500.0 + main.y * 1100.0,
            flutes: 5.0 + (main.w * 3.0).floor(),
            twist: (main.z - 0.5) * 9.0,
        };
    }
    // Lesser horns crowd the main spire and curl away from it.
    let random = spire_hash(cell + index as f32 * 17.17 + 9.4);
    let angle = random.x * TAU;
    let outward = vec2(angle.cos(), angle.sin());
    Spire {
        base: centre + outward * (1300.0 + random.y * 1500.0),
        lean: outward * (0.05 + random.z * 0.11),
        tip: 1500.0 + random.z * (tallest - 1500.0) * 0.75,
        radius: 600.0 + random.w * 600.0,
        flutes: 4.0 + (random.y * 4.0).floor(),
        twist: (random.w - 0.5) * 12.0,
    }
}

/// The axis bows like a horn and sways once along its height.
pub fn spire_axis(spire: &Spire, z: f32) -> (Vec2, f32) {
    let height = spire.tip - SPIRE_BOTTOM;
    let u = ((z - SPIRE_BOTTOM) / height).clamp(0.0, 1.0);
    let sway = vec2(-spire.lean.y, spire.lean.x) * ((u * core::f32::consts::PI).sin() * 0.25);
    (spire.base + (spire.lean * (u * u) + sway) * height, u)
}

/// Unfluted radius; the fog reads this smooth silhouette.
fn spire_radius(spire: &Spire, u: f32) -> f32 {
    spire.radius * (1.0 - u).max(0.0).powf(1.15)
}

/// Twisted flutes and stacked growth strata on the smooth silhouette.
pub fn spire_relief(spire: &Spire, angle: f32, u: f32) -> f32 {
    let flute = (spire.flutes * angle + spire.twist * u).cos();
    // Irregular growth bands rather than evenly stacked tiers.
    let layer = u * 7.0 + (u * 23.0 + spire.twist).sin() * 0.45;
    let strata = (layer - layer.floor()).smoothstep(0.8, 1.0);
    1.0 - 0.08 * flute - 0.035 * strata
}

pub fn spire_distance(spire: &Spire, local: Vec2, z: f32) -> f32 {
    let (centre, u) = spire_axis(spire, z);
    let q = local - centre;
    let length = q.length();
    let smooth = spire_radius(spire, u);
    // Taper, lean and relief all steepen the field; scale it back to a bound.
    // Relief never grows the silhouette by more than 8%, so far away the
    // smooth bound is enough and the angular terms can be skipped.
    let side = if length > smooth * 1.3 {
        (length - smooth * 1.08) * 0.55
    } else {
        (length - smooth * spire_relief(spire, q.y.atan2(q.x), u)) * 0.55
    };
    side.max(z - spire.tip).max(SPIRE_BOTTOM - z)
}

fn spire_cell(p: Vec3) -> (Vec2, Vec2) {
    let cell = (p.truncate() / SPIRE_CELL).floor();
    (cell, p.truncate() - (cell + 0.5) * SPIRE_CELL)
}

/// A lower bound on the distance to any spire, in surface units.
pub fn spire_field(p: Vec3) -> f32 {
    let (cell, local) = spire_cell(p);
    let mut distance = SPIRE_CELL * 0.5 - local.abs().max_element() + SPIRE_MARGIN;
    for index in 0..spire_count(cell) {
        distance = distance.min(spire_distance(&spire(cell, index), local, p.z));
    }
    distance
}

/// The nearest spire silhouette at this altitude: the gap to its surface and
/// the horizontal direction away from its axis.
pub fn nearest_spire(p: Vec3) -> Vec3 {
    let (cell, local) = spire_cell(p);
    let mut nearest = vec3(1.0e10, 0.0, 0.0);
    for index in 0..spire_count(cell) {
        let spire = spire(cell, index);
        let (centre, u) = spire_axis(&spire, p.z);
        let q = local - centre;
        let length = q.length().max(1.0);
        let gap = length - spire_radius(&spire, u);
        if gap < nearest.x {
            nearest = vec3(gap, q.x / length, q.y / length);
        }
    }
    nearest
}

/// Entry and exit along a ray through the band holding the spires. Altitude
/// is measured from the planet centre without squaring large radii.
pub fn spire_band(altitude: f32, upward: f32, max_distance: f32) -> Vec2 {
    let radius = PLANET_RADIUS + altitude;
    let b = radius * upward;
    let crossing = |height: f32| {
        let c = (altitude - height) * (radius + PLANET_RADIUS + height);
        let discriminant = b * b - c;
        if discriminant < 0.0 {
            return vec2(1.0, -1.0);
        }
        let root = discriminant.sqrt();
        let q = -b - root.copysign(b);
        let (near, far) = (q, c / q);
        vec2(near.min(far), near.max(far))
    };
    let top = crossing(SPIRE_TOP);
    if top.y < 0.0 || top.x > top.y {
        return vec2(1.0, -1.0);
    }
    let bottom = crossing(SPIRE_BOTTOM);
    let exit = if bottom.x <= bottom.y && bottom.x > 0.0 {
        bottom.x
    } else {
        top.y
    };
    vec2(top.x.max(0.0), exit.min(max_distance))
}

#[cfg(test)]
mod tests {
    use spirv_std::glam::ivec2;

    use super::*;

    #[test]
    fn every_spire_stays_inside_the_margin_of_its_cell() {
        for x in -40..40 {
            for y in -40..40 {
                let cell = vec2(x as f32, y as f32);
                for index in 0..spire_count(cell) {
                    let spire = spire(cell, index);
                    for z in [SPIRE_BOTTOM, 0.0, spire.tip] {
                        let (centre, u) = spire_axis(&spire, z);
                        let extent = centre.abs().max_element() + spire_radius(&spire, u) * 1.08;
                        assert!(extent < SPIRE_REACH, "spire leaves its cell: {extent}");
                    }
                }
            }
        }
    }

    #[test]
    fn spires_pierce_the_ocean_and_the_field_is_signed() {
        let mut found = 0;
        for x in -20..20 {
            for y in -20..20 {
                let cell = vec2(x as f32, y as f32);
                for index in 0..spire_count(cell) {
                    let (centre, _) = spire_axis(&spire(cell, index), 0.0);
                    let point = ((cell + 0.5) * SPIRE_CELL + centre).extend(0.0);
                    assert!(spire_field(point) < 0.0, "axis must be inside");
                    assert!(spire_field(point.with_z(SPIRE_TOP + 10.0)) > 0.0);
                    assert!(nearest_spire(point).x < 0.0);
                    found += 1;
                }
            }
        }
        assert!(found > 100, "too few spires: {found}");
    }

    #[test]
    fn band_matches_the_altitudes_it_crosses() {
        let down = spire_band(2400.0, -1.0, 1.0e7);
        assert!(down.x.abs() < 0.01 && (down.y - 7400.0).abs() < 1.0);
        let from_above = spire_band(80_000.0, -1.0, 1.0e7);
        assert!((from_above.x - (80_000.0 - SPIRE_TOP)).abs() < 1.0);
        assert!(spire_band(80_000.0, 1.0, 1.0e7).x > spire_band(80_000.0, 1.0, 1.0e7).y);
        let grazing = spire_band(2400.0, 0.0, 1.0e7);
        assert!(grazing.x.abs() < 0.01 && grazing.y > 100_000.0);
    }

    #[test]
    fn eruptions_rise_hold_then_part_at_the_apex() {
        assert!(ejection_fill(0.5, 0.0) < 0.01);
        assert!(ejection_fill(0.5, 0.3) > 0.99 && ejection_fill(0.1, 0.3) > 0.99);
        assert!(ejection_fill(0.5, 0.85) < 0.05 && ejection_fill(0.1, 0.85) > 0.4);
        assert!(ejection_fill(0.1, 1.0) < 0.01);
    }

    #[test]
    fn cached_arc_candidates_cover_the_planet_and_preserve_the_full_field() {
        let radius = super::super::PLANET_RADIUS;
        for longitude in [-PI, -PI * 0.5, 0.0, PI * 0.5, PI] {
            for colatitude in [0.0, 0.01, PI * 0.5, PI - 0.01, PI] {
                for time in [0.0, 10.0, 120.0, 50_000.0] {
                    let point = vec3(longitude * radius, colatitude * radius, 650.0);
                    let cell = (point.xy() / EJECTION_CELL).floor();
                    let mut active = 0;
                    for index in 0..9 {
                        let neighbour =
                            cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
                        let pixel = neighbour.as_ivec2() + ivec2(EJECTION_GRID_HALF, 1);
                        assert!(pixel.x >= 0 && pixel.x < EJECTION_GRID_WIDTH.cast_signed());
                        assert!(pixel.y >= 0 && pixel.y < EJECTION_GRID_HEIGHT.cast_signed());
                        active |= u32::from(ejection(neighbour, time)[0].w > 0.0) << index;
                    }
                    assert_eq!(
                        ejection_field(point, time),
                        ejection_field_with(point, active, |cell| ejection(cell, time))
                    );
                }
            }
        }
    }
}
