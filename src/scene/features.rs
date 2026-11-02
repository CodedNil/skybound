//! The planet's fixed features: the places the aur and the air make what they make, which never
//! move.
//!
//! Water reaches the sky through geysers: rare fields of them, each a cluster of jets thrown up out
//! of the aur so fast that touching one is death, and about each field a swamp, its air saturated
//! and its rain constant, the rain falling back into the aur's heat to boil up again. Minerals
//! reach it through thermal vents, chimneys of the aur's crust clustered in fields across the dim
//! low latitudes, which fling dust into a thick low haze and breathe the air in about them: the
//! Mineral Fields. Over a very few places where the updraft is strong and the air wet, an even
//! stronger ring of rising air walls water in, and a lake floats. And high over the cold, dry
//! equator the updraft carries ice: crystals grown large in still air, held up by the same rising
//! air that grew them.
//!
//! Every feature is a few numbers drawn from its index, so the host and the shaders agree on where
//! each is without being told, and few enough that anything can look at all of them.

use crate::scene::{PLANET_RADIUS, aur::spire_point, math::Power};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// Swamps: fields of water geysers.
pub const SWAMPS: u32 = 28;
/// The most geysers one swamp holds.
pub const GEYSERS: u32 = 10;
/// Fields of thermal vents: the Mineral Fields.
pub const VENT_FIELDS: u32 = 22;
/// The most vents one field holds.
pub const VENTS: u32 = 16;
/// Floating lakes, the rarest sight in the sky.
pub const LAKES: u32 = 7;

/// A field of geysers or vents: where its centre lies and how far it reaches, in metres across the
/// planet's surface.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub up: Vec3,
    pub radius: f32,
}

/// One geyser: where it rises from the aur, how high its jet is thrown, and how wide it is.
#[derive(Clone, Copy, Debug)]
pub struct Geyser {
    pub up: Vec3,
    pub height: f32,
    pub radius: f32,
}

/// One thermal vent: where its chimney stands, how high above the aur it is built, and how wide.
#[derive(Clone, Copy, Debug)]
pub struct Vent {
    pub up: Vec3,
    pub height: f32,
    pub radius: f32,
}

/// A floating lake: where it hangs, at what height, how wide it is and how deep at its middle.
#[derive(Clone, Copy, Debug)]
pub struct Lake {
    pub up: Vec3,
    pub altitude: f32,
    pub radius: f32,
    pub depth: f32,
}

/// A well-mixed number from another.
const fn mix(mut state: u32) -> u32 {
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^ (state >> 16)
}

/// Four numbers in [0, 1) drawn from an index and a salt.
fn draw(index: u32, salt: u32) -> Vec4 {
    let a = mix(index.wrapping_mul(0x9e37_79b9) ^ salt.wrapping_mul(0x85eb_ca6b));
    let b = mix(a);
    let c = mix(b);
    let d = mix(c);
    vec4(
        (a >> 8) as f32,
        (b >> 8) as f32,
        (c >> 8) as f32,
        (d >> 8) as f32,
    ) / 16_777_216.0
}

/// A direction from the planet's centre at a latitude whose sine is `rise` and a longitude a share
/// `around` of the way round.
fn direction(rise: f32, around: f32) -> Vec3 {
    let across = (1.0 - rise * rise).max(0.0).sqrt();
    let angle = around * TAU;
    vec3(across * angle.cos(), across * angle.sin(), rise)
}

/// Another direction a distance `reach` metres from `up` across the surface, turned `angle` about
/// it.
fn beside(up: Vec3, reach: f32, angle: f32) -> Vec3 {
    let east = vec3(-up.y, up.x, 0.0);
    let east = east / east.length().max(1e-6);
    let north = up.cross(east);
    let arc = reach / PLANET_RADIUS;
    let way = east * angle.cos() + north * angle.sin();
    let point = up * arc.cos() + way * arc.sin();
    point / point.length().max(1e-6)
}

/// How far apart two directions lie across the planet's surface, in metres.
///
/// The chord between them, lengthened by the first term of its arc's series, which is exact to
/// well under a metre across any field and costs no inverse sine.
pub fn apart(a: Vec3, b: Vec3) -> f32 {
    let chord = (a - b).length();
    chord * (1.0 + chord * chord / 24.0) * PLANET_RADIUS
}

/// Swamp `index`: anywhere but the dry poles.
pub fn swamp(index: u32) -> Field {
    let r = draw(index, 11);
    Field {
        up: direction((r.x * 2.0 - 1.0) * 0.85, r.y),
        radius: 4000.0 + r.z * 5000.0,
    }
}

/// How many geysers swamp `index` holds.
pub fn geysers(index: u32) -> u32 {
    5 + (draw(index, 12).x * (GEYSERS - 4) as f32) as u32
}

/// Geyser `geyser` of swamp `index`, clustered towards the swamp's heart.
pub fn geyser(index: u32, geyser: u32) -> Geyser {
    let field = swamp(index);
    let r = draw(index * GEYSERS + geyser, 13);
    Geyser {
        up: beside(field.up, r.x.sqrt() * field.radius * 0.55, r.y * TAU),
        height: 1500.0 + r.z * 2500.0,
        radius: 9.0 + r.w * 14.0,
    }
}

/// Vent field `index`: in the dim low latitudes, where the aurlight rules.
pub fn vent_field(index: u32) -> Field {
    let r = draw(index, 21);
    Field {
        up: direction((r.x * 2.0 - 1.0) * 0.4, r.y),
        radius: 3000.0 + r.z * 4000.0,
    }
}

/// How many vents field `index` holds.
pub fn vents(index: u32) -> u32 {
    8 + (draw(index, 22).x * (VENTS - 7) as f32) as u32
}

/// Vent `vent` of field `index`: chimneys in clusters, a few tall ones among many squat ones.
pub fn vent(index: u32, vent: u32) -> Vent {
    let field = vent_field(index);
    let r = draw(index * VENTS + vent, 23);
    // A few clusters within the field, each vent near one of them.
    let cluster = draw(index * 4 + (r.w * 3.99) as u32, 24);
    let heart = beside(field.up, cluster.x * field.radius * 0.6, cluster.y * TAU);
    let tall = r.z.power(3);
    Vent {
        up: beside(heart, r.x * field.radius * 0.18, r.y * TAU),
        height: 60.0 + tall * 340.0,
        radius: 18.0 + r.w * 26.0 + tall * 20.0,
    }
}

/// Lake `index`: near the equator, where the rain is too scarce to fill any basin, high in a
/// column of rising air.
pub fn lake(index: u32) -> Lake {
    let r = draw(index, 31);
    let side = if r.w < 0.5 { -1.0 } else { 1.0 };
    Lake {
        up: direction(side * (0.1 + r.x * 0.35), r.y),
        altitude: 1400.0 + r.z * 1800.0,
        radius: 120.0 + r.w.fract() * 260.0,
        depth: 25.0 + r.z * 40.0,
    }
}

/// The nearest geyser to a direction from the planet's centre, and how far its jet's axis lies
/// across the surface, if any swamp is near enough to hold one in reach.
pub fn nearest_geyser(up: Vec3) -> Option<(Geyser, f32)> {
    let mut best: Option<(Geyser, f32)> = None;
    for index in 0..SWAMPS {
        let field = swamp_at(index);
        if apart(up, field.up) > field.radius {
            continue;
        }
        for which in 0..geysers_at(index) {
            let geyser = geyser_at(index, which);
            let distance = apart(up, geyser.up);
            if best.is_none_or(|(_, nearest)| distance < nearest) {
                best = Some((geyser, distance));
            }
        }
    }
    best
}

/// The nearest floating lake to a direction from the planet's centre, and how far its middle lies
/// across the surface.
pub fn nearest_lake(up: Vec3) -> (Lake, f32) {
    let mut best = (lake_at(0), apart(up, lake_at(0).up));
    for index in 1..LAKES {
        let distance = apart(up, lake_at(index).up);
        if distance < best.1 {
            best = (lake_at(index), distance);
        }
    }
    best
}

/// The planet's sites, worked out once on the host, which looks them up far more often than they
/// could change; the shaders, with nowhere to keep them, work each out where they need it.
#[cfg(not(target_arch = "spirv"))]
mod sites {
    use super::{
        Field, GEYSERS, Geyser, LAKES, Lake, SWAMPS, VENT_FIELDS, geyser, geysers, lake, swamp,
        vent_field,
    };
    use std::sync::OnceLock;

    struct Sites {
        swamps: Vec<Field>,
        geysers: Vec<Geyser>,
        counts: Vec<u32>,
        vent_fields: Vec<Field>,
        lakes: Vec<Lake>,
    }

    fn sites() -> &'static Sites {
        static SITES: OnceLock<Sites> = OnceLock::new();
        SITES.get_or_init(|| Sites {
            swamps: (0..SWAMPS).map(swamp).collect(),
            geysers: (0..SWAMPS * GEYSERS)
                .map(|index| geyser(index / GEYSERS, index % GEYSERS))
                .collect(),
            counts: (0..SWAMPS).map(geysers).collect(),
            vent_fields: (0..VENT_FIELDS).map(vent_field).collect(),
            lakes: (0..LAKES).map(lake).collect(),
        })
    }

    pub fn swamp_at(index: u32) -> Field {
        sites().swamps[index as usize]
    }

    pub fn geyser_at(index: u32, which: u32) -> Geyser {
        sites().geysers[(index * GEYSERS + which) as usize]
    }

    pub fn geysers_at(index: u32) -> u32 {
        sites().counts[index as usize]
    }

    pub fn vent_field_at(index: u32) -> Field {
        sites().vent_fields[index as usize]
    }

    pub fn lake_at(index: u32) -> Lake {
        sites().lakes[index as usize]
    }
}

#[cfg(target_arch = "spirv")]
mod sites {
    use super::{Field, Geyser, Lake, geyser, geysers, lake, swamp, vent_field};

    pub fn swamp_at(index: u32) -> Field {
        swamp(index)
    }

    pub fn geyser_at(index: u32, which: u32) -> Geyser {
        geyser(index, which)
    }

    pub fn geysers_at(index: u32) -> u32 {
        geysers(index)
    }

    pub fn vent_field_at(index: u32) -> Field {
        vent_field(index)
    }

    pub fn lake_at(index: u32) -> Lake {
        lake(index)
    }
}

use sites::{geyser_at, geysers_at, lake_at, swamp_at, vent_field_at};

/// The nearest Mineral Field to a direction from the planet's centre, and how far its heart lies
/// across the surface.
pub fn nearest_vent_field(up: Vec3) -> (Field, f32) {
    let mut best = (vent_field_at(0), apart(up, vent_field_at(0).up));
    for index in 1..VENT_FIELDS {
        let field = vent_field_at(index);
        let distance = apart(up, field.up);
        if distance < best.1 {
            best = (field, distance);
        }
    }
    best
}

/// How strongly a place lies within a field that reaches `radius` from it: whole at its heart,
/// fading to nothing at its edge.
fn within(distance: f32, radius: f32) -> f32 {
    let q = (distance / radius.max(1.0)).min(1.0);
    1.0 - q * q * (3.0 - 2.0 * q)
}

/// What the planet's features make of the air at a place.
#[derive(Clone, Copy, Debug, Default)]
pub struct Influence {
    /// How deep within a swamp: its rain and saturated air.
    pub swamp: f32,
    /// How deep within a vent field: its dust and its breathing.
    pub vents: f32,
    /// Along the surface towards the nearest vent field's heart, for the air it draws in.
    pub to_vents: Vec3,
    /// How deep within a floating lake's column of rising air, and how near its walls.
    pub lake: f32,
    pub wall: f32,
    /// How thick the ice crystals hang here, high over the cold equator.
    pub ice: f32,
}

/// What the planet's features make of the air at a direction from its centre and a height above
/// the aur.
pub fn influence(up: Vec3, altitude: f32) -> Influence {
    let mut found = Influence::default();
    for index in 0..SWAMPS {
        let field = swamp_at(index);
        found.swamp = found
            .swamp
            .max(within(apart(up, field.up), field.radius * 1.4));
    }
    let mut nearest = 1.0e9;
    for index in 0..VENT_FIELDS {
        let field = vent_field_at(index);
        let distance = apart(up, field.up);
        let here = within(distance, field.radius * 1.6);
        found.vents = found.vents.max(here);
        if distance < nearest {
            nearest = distance;
            let toward = field.up - up * field.up.dot(up);
            found.to_vents = toward / toward.length().max(1e-6);
        }
    }
    for index in 0..LAKES {
        let lake = lake_at(index);
        let q = apart(up, lake.up) / lake.radius;
        let column = (1.0 - ((altitude - lake.altitude) / 2500.0).abs()).max(0.0);
        found.lake = found.lake.max(within(q, 2.5) * column);
        found.wall = found
            .wall
            .max((-((q - 1.15) / 0.2).power(2)).exp() * column);
    }
    found.ice = ice(up, altitude);
    found
}

/// How thick ice crystals hang at a direction from the planet's centre and a height: in a band of
/// cold, high air over the equator, broken into drifts.
pub fn ice(up: Vec3, altitude: f32) -> f32 {
    let latitude = up.z.clamp(-1.0, 1.0).asin().abs();
    let band = 1.0 - (latitude / 0.32).min(1.0);
    let height = within((altitude - 9500.0).abs(), 2600.0);
    let drifts = (up.x * 41.0).sin() * (up.y * 37.0 + up.z * 13.0).sin() * 0.5 + 0.5;
    band * height * drifts.smoothstep(0.35, 0.75)
}

/// The ice crystals' cells, in spire coordinates: each holds at most one.
pub const ICE_CELL: f32 = 900.0;
/// The heights the ice crystals hang between.
pub const ICE_LOW: f32 = 7000.0;
pub const ICE_HIGH: f32 = 12_500.0;

/// Four numbers in [0, 1) drawn from a cell, by integers alone so the host and the shaders agree
/// exactly: a sine of a coordinate this large is computed differently on each.
fn cell_hash(cell: Vec2) -> [f32; 4] {
    let key = (cell.x as i32 as u32).wrapping_mul(0x8da6_b343)
        ^ (cell.y as i32 as u32).wrapping_mul(0xd816_3841);
    let r = draw(key, 41);
    [r.x, r.y, r.z, r.w]
}

/// The ice crystal in a cell, if the ice is thick enough there for one: where its middle hangs (as
/// across the cell and in height), its size, its form and its own seed.
pub fn crystal(cell: Vec2, time: f32) -> Option<(Vec3, f32, u32, f32)> {
    let [a, b, c, d] = cell_hash(cell);
    let middle = (cell + 0.5) * ICE_CELL;
    // How thick the ice hangs over this cell, from the planet-wide band and its drifts.
    let around = spire_point(middle.extend(0.0)) / PLANET_RADIUS;
    let thick = ice(around, 9500.0);
    if a > thick * 0.5 {
        return None;
    }
    let size = 40.0 + b.power(2) * 120.0;
    // Bobbing slowly on the updraft that holds it up.
    let bob = (time * (0.12 + c * 0.08) + d * TAU).sin() * size * 0.08;
    let across = (vec2(b, c) - 0.5) * (ICE_CELL - size * 2.5);
    let height = ICE_LOW + 400.0 + d * (ICE_HIGH - ICE_LOW - 800.0) + bob;
    Some((across.extend(height), size, (c * 3.0) as u32, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_lie_where_their_lore_puts_them() {
        for index in 0..VENT_FIELDS {
            assert!(
                vent_field(index).up.z.abs() < 0.4,
                "a vent field strayed poleward"
            );
        }
        for index in 0..LAKES {
            let lake = lake(index);
            assert!(lake.up.z.abs() > 0.09 && lake.up.z.abs() < 0.46);
        }
        for index in 0..SWAMPS {
            let field = swamp(index);
            for geyser in 0..geysers(index) {
                let geyser = super::geyser(index, geyser);
                assert!(
                    apart(geyser.up, field.up) < field.radius,
                    "a geyser left its swamp"
                );
            }
        }
    }

    #[test]
    fn swamps_and_vents_are_felt_only_near_them() {
        let field = swamp(0);
        assert!(influence(field.up, 500.0).swamp > 0.9);
        let far = beside(field.up, 40_000.0, 0.0);
        let others =
            (0..SWAMPS).all(|index| apart(far, swamp(index).up) > swamp(index).radius * 1.4);
        if others {
            assert!(influence(far, 500.0).swamp < 0.01);
        }
        let vents = vent_field(3);
        let near = beside(vents.up, vents.radius * 0.8, 1.0);
        let toward = influence(near, 200.0).to_vents;
        assert!(
            toward.dot(vents.up - near) > 0.0,
            "the vents' pull points away"
        );
    }
}
