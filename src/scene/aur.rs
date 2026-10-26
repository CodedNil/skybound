use crate::scene::{
    PLANET_RADIUS,
    climate::{cube_direction, cube_face},
    math::square,
};
#[cfg(not(target_arch = "spirv"))]
use core::array;
use core::f32::consts::{FRAC_PI_2, PI, TAU};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::{DVec3, dvec2};
use spirv_std::glam::{
    FloatExt, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const CURRENT: Vec3 = vec3(4.375, -2.5, 0.0);

/// Arc cells are three spire cells across, so the face grid that places spires near the camera
/// places arcs too.
pub const EJECTION_CELL: f32 = SPIRE_CELL * 3.0;

pub const EJECTION_WIDTH: f32 = 131.25;

pub const EJECTION_REACH: f32 = 9000.0;

/// How far a loop's tilt and skew can carry its apex across the surface.
const EJECTION_LEAN: f32 = 3000.0;

/// How far any part of a loop can lie from its centre across the surface: the largest loop radius,
/// its lean and the reach of its glow.
pub const EJECTION_EXTENT: f32 = 5000.0 + EJECTION_LEAN + EJECTION_WIDTH * 3.0;

/// Stretching, tilting and skewing a loop shrinks distances along it by at most this factor, so the
/// arc field scaled by it stays a lower bound.
const EJECTION_METRIC: f32 = 0.65;

/// Rings of cells searched around each cell for its arc clearance.
const EJECTION_SEARCH: i32 = 4;

/// The highest a loop's apex rises: its tallest circle, stretched.
pub const EJECTION_HEIGHT: f32 = 6000.0;

/// Arc cells held in the lookup: a window this many cells across, centred on the camera.
///
/// It wraps, so a cell's texel is its coordinate modulo the window. Further cells alias to ones
/// whose loops lie a window away, which read as empty.
pub const EJECTION_WINDOW: u32 = 384;

// Crystal spires that pierce the aur ocean, in sparse archipelagos.
//
// Positions are spire coordinates: metres across the cube faces of the planet, side by side, and
// altitude. Cells are the same size everywhere, so spires never crowd towards the poles. Every
// spire stays `SPIRE_MARGIN` inside its cell, so a cell's own spires and the distance to its border
// bound the field.
pub const SPIRE_CELLS: f32 = 360.0;

pub const SPIRE_CELL: f32 = PLANET_RADIUS * FRAC_PI_2 / SPIRE_CELLS;

/// Distances in spire coordinates overstate world distances by at most this factor, where the cube
/// faces stretch towards their edges.
pub const SPIRE_METRIC: f32 = 0.65;

/// Faces sit this far apart in spire coordinates so no cell spans two.
const SPIRE_FACE: f32 = SPIRE_CELL * (SPIRE_CELLS + 4.0);

/// Spires are buried well below the deepest wave trough.
pub const SPIRE_BOTTOM: f32 = -625.0;

pub const SPIRE_TOP: f32 = 1200.0;

const SPIRE_REACH: f32 = 1312.5;

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

/// The planet-scale character of the aur at a surface direction.
///
/// A calm, warm equatorial belt, storm tracks through the mid latitudes broken into slowly drifting
/// regions, and charged polar seas. Analytic so the host's lightning catalogue agrees with the
/// shaders. x: storminess, y: warmth, z: charge, w: brightness.
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
    let fade = (1.0 - pos.z.max(0.0) / 750.0).clamp(0.0, 1.0);
    let flow = curl((pos - CURRENT * time).xy() * 0.008, time);
    let pulse = 0.8 + 0.2 * (time * 1.2 + flow.x * 7.0).sin();
    vec3(0.035, 0.008, 0.065) * fade * pulse
}

fn eruption_hash(cell: Vec2) -> Vec3 {
    let mut hash = (cell.xyx() * vec3(0.1031, 0.1030, 0.0973)).fract_gl();
    hash += hash.dot(hash.yzx() + 33.33);
    ((hash.xxy() + hash.yzz()) * hash.zyx()).fract_gl()
}

/// One eruption per cell and cycle, far more often in stormy seas.
///
/// The loop rises out of the veil, holds while plasma drains down its legs, then parts at the apex
/// and fades. Every eruption draws its own size, height, orientation, stretch, tilt and skew, and a
/// seed for its strands.
///
/// The loop's centre, phase and whether it is alive; its axis, radius and how deep its circle
/// sinks; and its stretch, tilt and seed.
pub fn ejection(cell: Vec2, time: f32) -> [Vec4; 3] {
    let seed = eruption_hash(cell + vec2(19.7, 63.1));
    let period = 45.0 + seed.x * 60.0;
    let cycle = time / period + seed.y;
    let phase = (cycle.fract_gl() * period / (6.0 + seed.z * 6.0)).min(1.0);
    if phase >= 1.0 {
        return [Vec4::ZERO; 3];
    }
    let eruption = cycle.floor() + 1.0;
    let random = eruption_hash(cell + vec2(37.1, 91.7) * eruption);
    // Loops keep clear of the cube faces' edges, where the face grid parts.
    let centre = (cell + random.xy()) * EJECTION_CELL;
    let face = (centre.x / SPIRE_FACE).floor();
    let across = vec2(centre.x - face * SPIRE_FACE, centre.y);
    if across.min_element() < EJECTION_EXTENT
        || across.max_element() > SPIRE_CELL * SPIRE_CELLS - EJECTION_EXTENT
    {
        return [Vec4::ZERO; 3];
    }
    let up = spire_point(centre.extend(0.0)).normalize();
    if random.z > (0.25 + aur_climate(up, time).x * 0.9) / 3.0 {
        return [Vec4::ZERO; 3];
    }
    let shape = eruption_hash(cell * 1.37 + vec2(11.3, 5.9) * eruption);
    let form = eruption_hash(cell * 0.71 + vec2(3.3, 8.1) * eruption);
    let angle = shape.x * TAU;
    // Most loops are modest; a few tower over the archipelagos.
    let radius = 875.0 + shape.y * shape.y * 4125.0;
    [
        random.xy().extend(phase).extend(1.0),
        Vec4::new(
            angle.cos(),
            angle.sin(),
            radius,
            (radius * (0.3 + shape.z * 0.5)).min(radius - 275.0),
        ),
        Vec4::new(0.0, 0.7 + form.x * 0.8, (form.y - 0.5) * 0.9, form.z),
    ]
}

/// A lower bound on the distance from anywhere in a cell to the glow of any loop, so rays cross
/// empty stretches of the arc layer at once.
pub fn ejection_clearance(cell: Vec2, time: f32) -> f32 {
    let side = EJECTION_SEARCH * 2 + 1;
    let mut clearance = EJECTION_SEARCH as f32 * EJECTION_CELL - EJECTION_EXTENT;
    for index in 0..side * side {
        let offset = vec2(
            (index % side - EJECTION_SEARCH) as f32,
            (index / side - EJECTION_SEARCH) as f32,
        );
        let event = ejection(cell + offset, time);
        if event[0].w <= 0.0 {
            continue;
        }
        // Measured from this cell's corner.
        let centre = (offset + event[0].xy()) * EJECTION_CELL;
        let nearest = centre.clamp(Vec2::ZERO, Vec2::splat(EJECTION_CELL));
        clearance = clearance
            .min(centre.distance(nearest) - event[1].z - EJECTION_LEAN - EJECTION_WIDTH * 3.0);
    }
    clearance.max(0.0) * EJECTION_METRIC
}

/// How much plasma a loop holds at a point along it.
pub fn ejection_fill(progress: f32, phase: f32) -> f32 {
    let alive = phase.smoothstep(0.0, 0.08) * (1.0 - phase.smoothstep(0.78, 1.0));
    let apex = (-square((progress - 0.5) / 0.16)).exp();
    alive * (1.0 - phase.smoothstep(0.55, 0.85) * apex)
}

/// A point in a loop's own frame, from its offset to the corner of the loop's cell and its
/// altitude: along its baseline, up through its circle and across it. The loop leans over by its
/// tilt about the baseline, and stretches taller or flatter and skews to one side above the veil.
fn ejection_coordinates(position: Vec3, event: [Vec4; 3]) -> Vec3 {
    let delta = position.xy() - event[0].xy() * EJECTION_CELL;
    let axis = event[1].truncate().truncate();
    let along = delta.dot(axis);
    let across = delta.dot(vec2(-axis.y, axis.x));
    let (sine, cosine) = event[2].z.sin_cos();
    let height = (position.z * cosine + across * sine) / event[2].y;
    let across = across * cosine - position.z * sine;
    let skew = (event[2].w * 7.31).fract_gl() - 0.5;
    // The loop's circle sinks below the veil while it is still rising.
    let rise = 0.3 + 0.7 * event[0].z.smoothstep(0.0, 0.3);
    vec3(
        along - height.max(0.0) * skew * 0.6,
        height - (event[1].w * rise - event[1].z),
        across,
    )
}

pub fn ejection_field(cell: Vec2, position: Vec3, time: f32) -> (Vec4, f32) {
    ejection_field_with(cell, position, 511, |cell| ejection(cell, time))
}

/// Distance to the nearest loop's axis, progress along it from one foot to the other, its fill, and
/// the angle around the axis for braided strands; and that loop's seed. The distance never
/// overstates the true one.
///
/// The point is given by its arc cell and its offset to that cell's corner, with its altitude, so
/// nothing far from the camera is ever subtracted.
pub(crate) fn ejection_field_with(
    cell: Vec2,
    position: Vec3,
    mut active: u32,
    event: impl Fn(Vec2) -> [Vec4; 3],
) -> (Vec4, f32) {
    let mut nearest = 1.0e20;
    let mut local = Vec3::ZERO;
    let mut radial = 0.0;
    let mut phase = 1.0;
    let mut seed = 0.0;
    while active != 0 {
        let index = active.trailing_zeros();
        active &= active - 1;
        let step = vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
        // The point from the neighbouring cell's corner.
        let relative = position.xy() - step * EJECTION_CELL;
        let outside = ((relative / EJECTION_CELL - 0.5).abs() - 0.5).max(Vec2::ZERO);
        if outside.max_element() * EJECTION_CELL > EJECTION_REACH {
            continue;
        }
        let event = event(cell + step);
        if event[0].w < 0.01 {
            continue;
        }
        let candidate = ejection_coordinates(relative.extend(position.z), event);
        let offset = candidate.xy().length() - event[1].z;
        let distance_squared = offset * offset + candidate.z * candidate.z;
        if distance_squared < nearest {
            nearest = distance_squared;
            local = candidate;
            radial = offset;
            phase = event[0].z;
            seed = event[2].w;
        }
    }
    if nearest > 1.0e19 {
        return (Vec4::new(1.0e10, 0.0, 0.0, 0.0), 0.0);
    }
    let progress = (local.y.atan2(local.x) / PI).clamp(0.0, 1.0);
    (
        Vec4::new(
            nearest.sqrt() * EJECTION_METRIC,
            progress,
            ejection_fill(progress, phase),
            local.z.atan2(radial),
        ),
        seed,
    )
}

fn spire_hash(p: Vec2) -> Vec4 {
    let mut q =
        (Vec4::new(p.x, p.y, p.x, p.y) * Vec4::new(0.1031, 0.1030, 0.0973, 0.1099)).fract_gl();
    q += q.dot(q.wzxy() + 33.33);
    ((q.xxyz() + q.yzzw()) * q.zywx()).fract_gl()
}

/// Up to four spires around one jittered centre; regions of cells share an occupancy so spires
/// gather into archipelagos with open ocean between.
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
    let centre = (spire_hash(cell).zw() - 0.5) * 1000.0;
    let main = spire_hash(cell + 3.1);
    let tallest = 625.0 + main.x * 562.5;
    if index == 0 {
        return Spire {
            base: centre,
            lean: (main.zw() - 0.5) * 0.08,
            tip: tallest,
            radius: 187.5 + main.y * 137.5,
            flutes: 5.0 + (main.w * 3.0).floor(),
            twist: (main.z - 0.5) * 9.0,
        };
    }
    // Lesser horns crowd the main spire and curl away from it.
    let random = spire_hash(cell + index as f32 * 17.17 + 9.4);
    let angle = random.x * TAU;
    let outward = vec2(angle.cos(), angle.sin());
    Spire {
        base: centre + outward * (162.5 + random.y * 187.5),
        lean: outward * (0.05 + random.z * 0.11),
        tip: 187.5 + random.z * (tallest - 187.5) * 0.75,
        radius: 75.0 + random.w * 75.0,
        flutes: 4.0 + (random.y * 4.0).floor(),
        twist: (random.w - 0.5) * 12.0,
    }
}

/// The axis bows like a horn and sways once along its height.
pub fn spire_axis(spire: &Spire, z: f32) -> (Vec2, f32) {
    let height = spire.tip - SPIRE_BOTTOM;
    let u = ((z - SPIRE_BOTTOM) / height).clamp(0.0, 1.0);
    let sway = vec2(-spire.lean.y, spire.lean.x) * ((u * PI).sin() * 0.25);
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
    // Taper, lean and relief all steepen the field; scale it back to a bound. Relief never grows
    // the silhouette by more than 8%, so far away the smooth bound is enough and the angular terms
    // can be skipped.
    let side = if length > smooth * 1.3 {
        (length - smooth * 1.08) * 0.55
    } else {
        (length - smooth * spire_relief(spire, q.y.atan2(q.x), u)) * 0.55
    };
    side.max(z - spire.tip).max(SPIRE_BOTTOM - z)
}

/// Spire coordinates of a point in the planet.
pub fn spire_coordinates(point: Vec3) -> Vec3 {
    let radius = point.length();
    let (face, position) = cube_face(point / radius);
    let across = (position * 0.5 + 0.5) * (SPIRE_CELL * SPIRE_CELLS);
    vec3(
        across.x + face as f32 * SPIRE_FACE,
        across.y,
        radius - PLANET_RADIUS,
    )
}

/// The point in the planet at spire coordinates.
pub fn spire_point(coordinates: Vec3) -> Vec3 {
    let face = (coordinates.x / SPIRE_FACE).floor().clamp(0.0, 5.0);
    let across =
        vec2(coordinates.x - face * SPIRE_FACE, coordinates.y) / (SPIRE_CELL * SPIRE_CELLS);
    cube_direction(face as u32, across * 2.0 - 1.0) * (PLANET_RADIUS + coordinates.z)
}

/// World directions in which the two horizontal spire coordinates grow.
pub fn spire_axes(point: Vec3) -> (Vec3, Vec3) {
    let up = point.normalize();
    let (face, _) = cube_face(up);
    let (first, second) = match face / 2 {
        0 => (Vec3::Y, Vec3::Z),
        1 => (Vec3::X, Vec3::Z),
        _ => (Vec3::X, Vec3::Y),
    };
    let tangent = |axis: Vec3| (axis - up * axis.dot(up)).normalize();
    (tangent(first), tangent(second))
}

/// A point on the face grid: the spire cell holding it, its offset from that cell's centre, and
/// its altitude.
///
/// Kept apart, the offset stays exact where whole coordinates this far across the planet would
/// round to metres.
#[derive(Clone, Copy)]
pub struct Place {
    pub cell: Vec2,
    pub local: Vec2,
    pub altitude: f32,
}

impl Place {
    /// The place of a point given in whole spire coordinates.
    pub fn at(p: Vec3) -> Self {
        let cell = (p.truncate() / SPIRE_CELL).floor();
        Self {
            cell,
            local: p.truncate() - (cell + 0.5) * SPIRE_CELL,
            altitude: p.z,
        }
    }

    /// The arc cell holding this place, and the place's offset from that cell's corner, with
    /// its altitude.
    pub fn arc(self) -> (Vec2, Vec3) {
        let cell = (self.cell / 3.0).floor();
        (
            cell,
            ((self.cell - cell * 3.0 + 0.5) * SPIRE_CELL + self.local).extend(self.altitude),
        )
    }
}

/// A cube face's two axes across it and its outward normal, as `cube_face` projects onto them.
fn face_frame(face: u32) -> (Vec3, Vec3, Vec3) {
    let sign = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    match face / 2 {
        0 => (Vec3::Y, Vec3::Z, Vec3::X * sign),
        1 => (Vec3::X, Vec3::Z, Vec3::Y * sign),
        _ => (Vec3::X, Vec3::Y, Vec3::Z * sign),
    }
}

/// Where a point sits in spire coordinates on each face it can be seen from: its whole cell, then
/// its offset into that cell.
///
/// Worked out in double precision, since single precision rounds positions this far from the
/// planet's centre to whole metres, and further to several metres across the faces laid side
/// by side.
#[cfg(not(target_arch = "spirv"))]
pub fn spire_origins(point: DVec3) -> [Vec4; 6] {
    use core::f64::consts::FRAC_PI_4;
    let cell = f64::from(SPIRE_CELL);
    let width = f64::from(SPIRE_CELL * SPIRE_CELLS);
    array::from_fn(|face| {
        let (first, second, normal) = face_frame(face as u32);
        let depth = point.dot(normal.as_dvec3());
        if depth <= 0.0 {
            return Vec4::ZERO;
        }
        let across = dvec2(
            (point.dot(first.as_dvec3()) / depth).atan(),
            (point.dot(second.as_dvec3()) / depth).atan(),
        );
        let across = (across / FRAC_PI_4 * 0.5 + 0.5) * width;
        let whole = (across / cell).floor();
        let offset = across - whole * cell;
        (whole.as_vec2() + vec2(face as f32 * (SPIRE_CELLS + 4.0), 0.0))
            .extend(offset.x as f32)
            .extend(offset.y as f32)
    })
}

/// The place of a point near the camera, from its offset to the camera.
///
/// Only differences of nearby positions are rounded, so spires and arcs stay exact up close, where
/// `spire_coordinates` would show metre steps.
///
/// `origins` are the camera's from `spire_origins`, `camera` its position and `altitude` its
/// precise height above the surface.
pub fn place_near(origins: &[Vec4; 6], camera: Vec3, altitude: f32, offset: Vec3) -> Place {
    let point = camera + offset;
    let (face, _) = cube_face(point.normalize());
    let (first, second, normal) = face_frame(face);
    // atan(a / b) - atan(a0 / b0), with the large terms cancelled exactly.
    let a0 = vec2(camera.dot(first), camera.dot(second));
    let b0 = camera.dot(normal);
    let a = vec2(offset.dot(first), offset.dot(second));
    let b = offset.dot(normal);
    let camera_squared = b0 * b0 + a0 * a0;
    let turn = (a * b0 - a0 * b) / (camera_squared + a0 * a + b0 * b);
    let across = vec2(turn.x.atan(), turn.y.atan()) * (SPIRE_CELL * SPIRE_CELLS / FRAC_PI_2);
    let origin = origins[face as usize];
    let total = origin.zw() + across;
    let whole = (total / SPIRE_CELL).floor();
    let rise = (2.0 * camera.dot(offset) + offset.dot(offset)) / (point.length() + camera.length());
    Place {
        cell: origin.xy() + whole,
        local: total - (whole + 0.5) * SPIRE_CELL,
        altitude: altitude + rise,
    }
}

/// A lower bound on the distance from a point in the planet to any spire: for collisions.
pub fn spire_clearance(point: Vec3) -> f32 {
    spire_field(spire_coordinates(point)) * SPIRE_METRIC
}

/// A lower bound on the distance to any spire, in spire coordinates.
pub fn spire_field(p: Vec3) -> f32 {
    spire_field_at(Place::at(p))
}

/// `spire_field` at a place.
pub fn spire_field_at(place: Place) -> f32 {
    let Place {
        cell,
        local,
        altitude: z,
    } = place;
    let mut distance = SPIRE_CELL * 0.5 - local.abs().max_element() + SPIRE_MARGIN;
    for index in 0..spire_count(cell) {
        distance = distance.min(spire_distance(&spire(cell, index), local, z));
    }
    distance
}

/// The nearest spire silhouette at this altitude: the gap to its surface and the horizontal
/// direction away from its axis.
pub fn nearest_spire(p: Vec3) -> Vec3 {
    nearest_spire_at(Place::at(p))
}

/// `nearest_spire` at a place.
pub fn nearest_spire_at(place: Place) -> Vec3 {
    let Place {
        cell,
        local,
        altitude: z,
    } = place;
    let mut nearest = vec3(1.0e10, 0.0, 0.0);
    for index in 0..spire_count(cell) {
        let spire = spire(cell, index);
        let (centre, u) = spire_axis(&spire, z);
        let q = local - centre;
        let length = q.length().max(0.125);
        let gap = length - spire_radius(&spire, u);
        if gap < nearest.x {
            nearest = vec3(gap, q.x / length, q.y / length);
        }
    }
    nearest
}

/// Entry and exit along a ray through the band holding the spires. Altitude is measured from the
/// planet centre without squaring large radii.
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
    use super::*;
    use spirv_std::glam::DVec3;

    #[test]
    fn near_spire_coordinates_match_absolute_ones_and_resolve_centimetres() {
        let camera = DVec3::new(0.3, -0.5, 0.81).normalize() * f64::from(PLANET_RADIUS + 250.0);
        let origins = spire_origins(camera);
        let near = |offset: Vec3| {
            let place = place_near(&origins, camera.as_vec3(), 250.0, offset);
            ((place.cell + 0.5) * SPIRE_CELL + place.local).extend(place.altitude)
        };
        for offset in [
            Vec3::ZERO,
            vec3(375.0, -250.0, 62.5),
            vec3(-5000.0, 3125.0, 125.0),
        ] {
            let absolute = spire_coordinates(camera.as_vec3() + offset);
            let gap = near(offset) - absolute;
            assert!(
                gap.truncate().length() < 2.5 && gap.z.abs() < 0.5,
                "{offset} {gap}"
            );
        }
        let at = |offset| place_near(&origins, camera.as_vec3(), 250.0, offset);
        let (still, moved) = (at(Vec3::ZERO), at(vec3(0.25, 0.0, 0.0)));
        let step = (moved.local - still.local).length();
        assert!(
            still.cell == moved.cell && (0.1..0.5).contains(&step),
            "{step}"
        );
    }

    #[test]
    fn spire_coordinates_round_trip_and_never_overstate_distance_by_more_than_the_metric() {
        for index in 0..2000 {
            let z = (index as f32 + 0.5) / 2000.0 * 2.0 - 1.0;
            let longitude = index as f32 * 2.399_963;
            let up = vec3(
                (1.0 - z * z).sqrt() * longitude.cos(),
                (1.0 - z * z).sqrt() * longitude.sin(),
                z,
            );
            let point = up * (PLANET_RADIUS + 150.0);
            let coordinates = spire_coordinates(point);
            assert!(spire_point(coordinates).distance(point) < 2.0);
            // Nudge within the face in every direction: the world moves at least SPIRE_METRIC as
            // far as the coordinates.
            for angle in 0..8 {
                let offset = vec2((angle as f32).cos(), (angle as f32).sin()) * 50.0;
                let moved = coordinates + offset.extend(0.0);
                let span = SPIRE_CELL * SPIRE_CELLS;
                let inside = moved.y > 0.0
                    && moved.y < span
                    && moved.x - (moved.x / SPIRE_FACE).floor() * SPIRE_FACE < span;
                if !inside
                    || (moved.x / SPIRE_FACE).floor() as i32
                        != (coordinates.x / SPIRE_FACE).floor() as i32
                {
                    continue;
                }
                let world = spire_point(moved).distance(point);
                assert!(world >= 50.0 * SPIRE_METRIC, "{world}");
            }
        }
    }

    #[test]
    fn arc_clearance_never_overstates_the_distance_to_a_loop() {
        for time in [0.0, 40.0, 130.0] {
            for x in 30..42 {
                for y in 30..42 {
                    let cell = vec2(x as f32, y as f32);
                    let clearance = ejection_clearance(cell, time);
                    for index in 0..25 {
                        let within = vec2((index % 5) as f32, (index / 5) as f32) / 4.0;
                        for z in [0.0, 750.0, 1750.0, 4000.0] {
                            let position = (within * EJECTION_CELL).extend(z);
                            let (curve, _) = ejection_field(cell, position, time);
                            assert!((curve.x - EJECTION_WIDTH * 3.0).max(0.0) >= clearance - 0.125);
                        }
                    }
                }
            }
        }
    }

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
                    assert!(spire_field(point.with_z(SPIRE_TOP + 1.25)) > 0.0);
                    assert!(nearest_spire(point).x < 0.0);
                    found += 1;
                }
            }
        }
        assert!(found > 100, "too few spires: {found}");
    }

    #[test]
    fn band_matches_the_altitudes_it_crosses() {
        let down = spire_band(300.0, -1.0, 1.0e7);
        assert!(down.x.abs() < 0.01 && (down.y - 925.0).abs() < 0.125);
        let from_above = spire_band(10_000.0, -1.0, 1.0e7);
        assert!((from_above.x - (10_000.0 - SPIRE_TOP)).abs() < 0.125);
        assert!(spire_band(10_000.0, 1.0, 1.0e7).x > spire_band(10_000.0, 1.0, 1.0e7).y);
        let grazing = spire_band(300.0, 0.0, 1.0e7);
        assert!(grazing.x.abs() < 0.01 && grazing.y > 12_500.0);
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
        let width = SPIRE_CELL * SPIRE_CELLS;
        for face in 0..6 {
            for across in [vec2(0.003, 0.5), vec2(0.5, 0.5), vec2(0.997, 0.2)] {
                for time in [0.0, 10.0, 120.0, 50_000.0] {
                    let whole = across * width + vec2(face as f32 * SPIRE_FACE, 0.0);
                    let cell = (whole / EJECTION_CELL).floor();
                    let point = (whole - cell * EJECTION_CELL).extend(81.25);
                    let mut active = 0;
                    for index in 0..9 {
                        let neighbour =
                            cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
                        active |= u32::from(ejection(neighbour, time)[0].w > 0.0) << index;
                    }
                    assert_eq!(
                        ejection_field(cell, point, time),
                        ejection_field_with(cell, point, active, |cell| ejection(cell, time))
                    );
                }
            }
        }
    }
}
