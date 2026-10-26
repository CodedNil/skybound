//! The planet's population at large.
//!
//! Life covers the whole sky, but only what lives near the player is lived out animal by animal.
//! Everywhere else the planet is divided into regions, each holding how many of every kind live
//! there: as many as the region can support, from how well its light, its aur and its food suit
//! them. Those counts grow towards what the region supports, and the travellers among the animals
//! drift towards better regions. As the player nears a region its herds, swarms and stands of
//! plants come into being where they range, in their groups, and as the player leaves them they
//! fold back into its count, so whatever was born or died among them is kept.

use crate::game::{
    ecosystem::{Physiology, Species, population},
    flight::{Mount, Player},
    world::{PlanetPosition, WorldData},
};
use bevy::{math::DVec3, prelude::*};
use rayon::prelude::*;
use skybound::scene::{
    PLANET_RADIUS,
    climate::Climate,
    life::{KINDS, Kind, Kingdom, animalia::nature::band, development::DAY},
};
use spirv_std::glam::Vec3 as SceneVec3;
use std::{
    collections::{HashMap, HashSet},
    f32::consts::TAU,
    f64::consts::TAU as TAU64,
    mem,
};

/// Regions along each edge of each of the six faces of the cube the sphere is divided as, each
/// about 25 km across.
const FACE: usize = 48;

const REGIONS: usize = 6 * FACE * FACE;

/// How near the player a group of a kind must range to be lived out in full: out to where one of
/// them would still cover a pixel, so giants are met from far off and the smallest only close by.
/// One that strays half as far again is folded back into its region.
fn near(kind: Kind) -> f64 {
    (f64::from(kind.render_radius()) / 0.0015).clamp(1500.0, NEAREST_GIANT)
}

fn far(kind: Kind) -> f64 {
    near(kind) * 1.5
}

/// The furthest any group is lived out.
const NEAREST_GIANT: f64 = 25_000.0;

/// The most organisms lived out in full at once.
const LIVING_LIMIT: usize = 1500;

/// Seconds between steps of the planet-wide counts, and between looking for groups to live out or
/// fold back.
const TICK: f32 = 10.0;

const LOOK: f32 = 1.0;

/// Seconds a group folded back waits before it can be lived out again, so one that has strayed
/// does not reappear at once where it ranges.
const REST: f32 = 300.0;

/// A group lived out in full: its region, its kind and which of the kind's groups there it is.
#[derive(Component, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Band {
    region: u32,
    kind: u16,
    index: u16,
}

#[derive(Resource)]
pub struct Range {
    /// How many of each kind live in each region, beyond those lived out near the player, by
    /// region and then kind.
    counts: Vec<f32>,
    /// How many each region could support of each kind, from its light, aur and wind alone.
    capacity: Vec<f32>,
    /// What each kind eats, by index.
    diets: Vec<Vec<usize>>,
    active: HashSet<Band>,
    resting: HashMap<Band, f32>,
    since: f32,
    looked: f32,
}

pub struct RangePlugin;

impl Plugin for RangePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Range::seeded())
            .add_systems(Update, (step, gather).chain());
    }
}

impl Range {
    /// The planet as it starts: every region holding most of what it can support.
    fn seeded() -> Self {
        let kinds = KINDS.len();
        let capacity: Vec<f32> = (0..REGIONS)
            .into_par_iter()
            .flat_map_iter(|region| {
                let up = centre(region);
                KINDS
                    .iter()
                    .map(move |&kind| suitability(kind, up) * density(kind))
            })
            .collect();
        // A plant's power is measured against the best the planet offers it, so each kind fills
        // the regions that suit it best.
        let mut capacity = capacity;
        for kind in 0..kinds {
            if KINDS[kind].kingdom() != Kingdom::Plantae {
                continue;
            }
            let best = (0..REGIONS)
                .map(|region| capacity[region * kinds + kind])
                .fold(0.0f32, f32::max)
                .max(1e-6);
            let full = density(KINDS[kind]);
            for region in 0..REGIONS {
                let share = capacity[region * kinds + kind] / best;
                capacity[region * kinds + kind] = share * share * full;
            }
        }
        let diets = KINDS
            .iter()
            .map(|&eater| {
                KINDS
                    .iter()
                    .enumerate()
                    .filter(|&(_, &food)| {
                        food != eater
                            && eater
                                .animal()
                                .is_some_and(|animal| (animal.eats)(eater, food))
                    })
                    .map(|(index, _)| index)
                    .collect()
            })
            .collect();
        let mut range = Self {
            counts: capacity.clone(),
            capacity,
            diets,
            active: HashSet::new(),
            resting: HashMap::new(),
            since: 0.0,
            looked: 0.0,
        };
        let food = range.food();
        for (index, count) in range.counts.iter_mut().enumerate() {
            let jitter = 0.6 + hash(index as u64, 7) * 0.4;
            *count *= jitter * food[index];
        }
        range
    }

    /// How well fed each kind could be in each region, from how much of what it eats lives there
    /// against what could; plants make their own.
    fn food(&self) -> Vec<f32> {
        let kinds = KINDS.len();
        (0..REGIONS * kinds)
            .into_par_iter()
            .map(|index| {
                let (region, kind) = (index / kinds, index % kinds);
                let diet = &self.diets[kind];
                if diet.is_empty() {
                    return 1.0;
                }
                let found: f32 = diet
                    .iter()
                    .map(|&food| {
                        let at = region * kinds + food;
                        self.counts[at] / density(KINDS[food]).max(1e-6)
                    })
                    .sum();
                found.clamp(0.05, 1.0)
            })
            .collect()
    }

    /// A step of the planet-wide counts: each grows or shrinks towards what its region supports,
    /// and the travellers drift towards the neighbouring region that suits them better.
    fn advance(&mut self, dt: f32) {
        let kinds = KINDS.len();
        let food = self.food();
        let capacity = &self.capacity;
        self.counts
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, count)| {
                let kind = KINDS[index % kinds];
                let supported = capacity[index] * food[index];
                let rate = 1.0 / (kind.biology().maturity_days * DAY);
                *count += rate * *count * (1.0 - *count / supported.max(1e-3)) * dt;
                *count = count.max(0.0);
            });
        for (kind_index, kind) in KINDS.iter().enumerate() {
            let speed = kind.speed();
            if speed <= 0.0 {
                continue;
            }
            // The share of a region's travellers that cross into a better one in a step.
            let share = (speed * dt / 25_000.0 * 0.1).min(0.25);
            let mut moved = vec![0.0; REGIONS];
            for region in 0..REGIONS {
                let at = region * kinds + kind_index;
                let here = self.counts[at];
                if here < 1.0 {
                    continue;
                }
                let room = |region: usize| {
                    let at = region * kinds + kind_index;
                    capacity[at] * food[at] - self.counts[at]
                };
                if let Some(best) = neighbours(region)
                    .into_iter()
                    .flatten()
                    .max_by(|&a, &b| room(a).total_cmp(&room(b)))
                    && room(best) > room(region)
                {
                    let flow = here * share;
                    moved[region] -= flow;
                    moved[best] += flow;
                }
            }
            for (region, flow) in moved.into_iter().enumerate() {
                let at = region * kinds + kind_index;
                self.counts[at] = (self.counts[at] + flow).max(0.0);
            }
        }
    }
}

/// How many of a kind a region that suits it perfectly supports: fewer of the larger, and few of
/// those that live fastened to others, for they are found on their hosts.
fn density(kind: Kind) -> f32 {
    let mass = kind.biology().adult_mass.max(0.3);
    let base = match kind.kingdom() {
        Kingdom::Animalia => 60.0,
        Kingdom::Plantae => 130.0,
    };
    base / mass.powf(0.45) * if kind.attaches() { 0.2 } else { 1.0 }
}

/// How well a region suits a kind: an animal by how near the light there is to the light it lives
/// in, a plant by how much of what powers it the region gives at the heights it grows.
fn suitability(kind: Kind, up: DVec3) -> f32 {
    let biology = kind.biology();
    let altitude = f32::midpoint(biology.altitude.0, biology.altitude.1);
    let point = (up * f64::from(PLANET_RADIUS + altitude)).as_vec3();
    let point = SceneVec3::new(point.x, point.y, point.z);
    let climate = Climate::on_planet(point, Climate::weather_on_planet(point, 0.0));
    match (kind.animal(), kind.plant()) {
        (Some(_), _) => {
            let (dimmest, brightest) = band(kind.light());
            let astray = (dimmest - climate.light)
                .max(climate.light - brightest)
                .max(0.0);
            (1.0 - astray * 5.0).max(0.0)
        }
        (_, Some(plant)) => (plant.powered)(&climate).max(0.0),
        _ => 0.0,
    }
}

/// How many of a kind keep together: herds and swarms in the lineages that took to them, stands
/// of plants, and the rest alone or in pairs.
fn group(kind: Kind, seed: u64) -> usize {
    let spread = hash(seed, 3);
    match (kind.animal(), kind.plant()) {
        (Some(animal), _) if animal.flocks => 6 + (spread * 10.0) as usize,
        (Some(_), _) => 1 + (spread * 2.5) as usize,
        (_, Some(_)) if kind.biology().adult_mass > 50.0 => 1 + (spread * 3.0) as usize,
        _ => 4 + (spread * 8.0) as usize,
    }
}

/// Step the planet-wide counts now and then.
fn step(mut range: ResMut<Range>, time: Res<Time>) {
    range.since += time.delta_secs();
    if range.since >= TICK {
        let dt = mem::take(&mut range.since);
        range.advance(dt);
    }
}

/// Where the range is centred: the player's mount, or the eye.
type Centre = Or<(With<Mount>, With<Camera3d>)>;

type Lived<'a> = (
    Entity,
    &'a Species,
    &'a Physiology,
    &'a PlanetPosition,
    Option<&'a Band>,
);

/// Live out the groups ranging near the player, and fold back into their regions those it has
/// left behind.
fn gather(
    mut commands: Commands,
    mut range: ResMut<Range>,
    world: Res<WorldData>,
    time: Res<Time>,
    eye: Query<(&PlanetPosition, Has<Mount>), Centre>,
    lived: Query<Lived, (Without<Player>, Without<Mount>)>,
) {
    let now = time.elapsed_secs();
    if now - range.looked < LOOK {
        return;
    }
    range.looked = now;
    // Centred on the player's mount, or on the eye when there is none.
    let Some(centre) = eye
        .iter()
        .max_by_key(|(_, mount)| *mount)
        .map(|(eye, _)| eye.position)
    else {
        return;
    };
    let kinds = KINDS.len();
    let mut members: HashMap<Band, usize> = HashMap::new();
    let mut living = 0;
    for (entity, species, body, position, band) in &lived {
        if position.position.distance(centre) > far(species.0) {
            if body.0.health > 0.0 {
                let at = region_of(position.position) * kinds + species.0 as usize;
                range.counts[at] += 1.0;
            }
            commands.entity(entity).despawn();
            continue;
        }
        living += 1;
        if let Some(band) = band {
            *members.entry(*band).or_default() += 1;
        }
    }
    let gone: Vec<Band> = range
        .active
        .iter()
        .filter(|band| !members.contains_key(band))
        .copied()
        .collect();
    for band in gone {
        range.active.remove(&band);
        range.resting.insert(band, now + REST);
    }
    range.resting.retain(|_, until| *until > now);
    // Every group ranging within reach, nearest first for its kind's reach, so when there is room
    // for only some the nearest come into being.
    let mut ranging_near = Vec::new();
    for region in near_regions(centre) {
        for (kind_index, &kind) in KINDS.iter().enumerate() {
            let at = region * kinds + kind_index;
            let mut remaining = range.counts[at];
            for index in 0..48u16 {
                let band = Band {
                    region: region as u32,
                    kind: kind_index as u16,
                    index,
                };
                let size = group(kind, band_seed(band));
                if remaining < size as f32 {
                    break;
                }
                remaining -= size as f32;
                if range.active.contains(&band) || range.resting.contains_key(&band) {
                    continue;
                }
                let anchor = ranging(band, kind, now);
                let nearness = anchor.distance(centre) / near(kind);
                if nearness <= 1.0 {
                    ranging_near.push((nearness, band, kind, anchor, size, at));
                }
            }
        }
    }
    ranging_near.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, band, kind, anchor, size, at) in ranging_near {
        if living + size > LIVING_LIMIT || range.counts[at] < size as f32 {
            continue;
        }
        range.counts[at] -= size as f32;
        range.active.insert(band);
        living += size;
        live_out(&mut commands, &world, band, kind, anchor, size);
    }
}

/// A group brought into being about where it ranges, its members close together, a few of them
/// young.
fn live_out(
    commands: &mut Commands,
    world: &WorldData,
    band: Band,
    kind: Kind,
    anchor: DVec3,
    size: usize,
) {
    let seed = band_seed(band);
    let up = anchor.normalize();
    let east = DVec3::Z.cross(up).try_normalize().unwrap_or(DVec3::X);
    let north = up.cross(east);
    let spread = if kind.kingdom() == Kingdom::Plantae {
        150.0
    } else {
        15.0 + size as f64 * 5.0
    };
    for member in 0..size {
        let key = seed ^ (member as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let angle = f64::from(hash(key, 1)) * TAU64;
        let distance = f64::from(hash(key, 2).sqrt()) * spread;
        let lift = (f64::from(hash(key, 4)) - 0.5) * spread * 0.4;
        let position = anchor + (east * angle.cos() + north * angle.sin()) * distance + up * lift;
        let young = hash(key, 5) < 0.15;
        let mass = if young {
            0.3
        } else {
            0.85 + hash(key, 6) * 0.3
        };
        commands.spawn((population(kind, world.local(position), mass), band));
    }
}

/// Where a group ranges now: a place of its own in its region, which travellers wander slowly
/// about and plants hold to, at a height in the band of air its kind lives in.
fn ranging(band: Band, kind: Kind, now: f32) -> DVec3 {
    let seed = band_seed(band);
    let wander = if kind.speed() > 0.0 { 0.15 } else { 0.0 };
    let phase = hash(seed, 9) * TAU;
    let period = 1800.0 + hash(seed, 10) * 3600.0;
    let drift = |offset: f32| wander * (now / period + phase + offset).sin();
    let u = (hash(seed, 11) + drift(0.0)).clamp(0.02, 0.98);
    let v = (hash(seed, 12) + drift(1.7)).clamp(0.02, 0.98);
    let up = direction(band.region as usize, u, v);
    let (low, high) = kind.biology().altitude;
    let low = low + 40.0;
    let high = (high - 40.0).max(low + 1.0);
    // Most life keeps low, in the thick air and the strongest updraft.
    let altitude = low + (high - low) * hash(seed, 13).powf(1.8);
    up * f64::from(PLANET_RADIUS + altitude)
}

/// The regions any group within reach of the player could range in, sampled closer together than
/// a region is wide so none is missed.
fn near_regions(centre: DVec3) -> Vec<usize> {
    let up = centre.normalize();
    let east = DVec3::Z.cross(up).try_normalize().unwrap_or(DVec3::X);
    let north = up.cross(east);
    let step = (NEAREST_GIANT + 1000.0) / 3.0;
    let mut regions: Vec<usize> = (-3..=3)
        .flat_map(|x| (-3..=3).map(move |y| (x, y)))
        .map(|(x, y)| region_of(centre + (east * f64::from(x) + north * f64::from(y)) * step))
        .collect();
    regions.sort_unstable();
    regions.dedup();
    regions
}

/// The region a point on or above the planet lies over.
fn region_of(point: DVec3) -> usize {
    let a = point.abs();
    let (face, u, v) = if a.x >= a.y && a.x >= a.z {
        (usize::from(point.x < 0.0), point.y / a.x, point.z / a.x)
    } else if a.y >= a.z {
        (2 + usize::from(point.y < 0.0), point.x / a.y, point.z / a.y)
    } else {
        (4 + usize::from(point.z < 0.0), point.x / a.z, point.y / a.z)
    };
    let cell = |t: f64| ((f64::midpoint(t, 1.0) * FACE as f64) as usize).min(FACE - 1);
    face * FACE * FACE + cell(u) * FACE + cell(v)
}

/// The way up at a place in a region, `u` and `v` across it from zero to one.
fn direction(region: usize, u: f32, v: f32) -> DVec3 {
    let face = region / (FACE * FACE);
    let within = region % (FACE * FACE);
    let u = ((within / FACE) as f64 + f64::from(u)) / FACE as f64 * 2.0 - 1.0;
    let v = ((within % FACE) as f64 + f64::from(v)) / FACE as f64 * 2.0 - 1.0;
    let sign = if face.is_multiple_of(2) { 1.0 } else { -1.0 };
    match face / 2 {
        0 => DVec3::new(sign, u, v),
        1 => DVec3::new(u, sign, v),
        _ => DVec3::new(u, v, sign),
    }
    .normalize()
}

fn centre(region: usize) -> DVec3 {
    direction(region, 0.5, 0.5)
}

/// The regions beside one on the same face.
fn neighbours(region: usize) -> [Option<usize>; 4] {
    let face = region / (FACE * FACE);
    let within = region % (FACE * FACE);
    let (i, j) = (within / FACE, within % FACE);
    let at = |i: usize, j: usize| face * FACE * FACE + i * FACE + j;
    [
        (i > 0).then(|| at(i - 1, j)),
        (i + 1 < FACE).then(|| at(i + 1, j)),
        (j > 0).then(|| at(i, j - 1)),
        (j + 1 < FACE).then(|| at(i, j + 1)),
    ]
}

fn band_seed(band: Band) -> u64 {
    (u64::from(band.region) << 32) ^ (u64::from(band.kind) << 16) ^ u64::from(band.index)
}

/// A number from zero to one, fixed by a seed and a salt.
fn hash(seed: u64, salt: u64) -> f32 {
    let mut x = seed ^ salt.wrapping_mul(0xd6e8_feb8_6659_fd93);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use skybound::scene::life::animalia::physiology::Light;

    #[test]
    fn every_region_round_trips_through_its_centre() {
        for region in (0..REGIONS).step_by(97) {
            assert_eq!(region_of(centre(region) * f64::from(PLANET_RADIUS)), region);
        }
    }

    #[test]
    fn each_kind_lives_where_its_light_is() {
        let range = Range::seeded();
        let kinds = KINDS.len();
        for (index, kind) in KINDS.iter().enumerate() {
            let total: f32 = (0..REGIONS)
                .map(|region| range.counts[region * kinds + index])
                .sum();
            assert!(total > 50.0, "{kind:?} has almost no range: {total}");
        }
        // Aur-lit animals keep to the equator's glow, and sunlit ones to the poles' light.
        let pole = region_of(DVec3::Z);
        let equator = region_of(DVec3::X);
        let at = |region: usize, kind: Kind| range.capacity[region * kinds + kind as usize];
        for kind in KINDS.iter().copied().filter(|kind| kind.animal().is_some()) {
            match kind.light() {
                Light::Aurlit => {
                    assert!(at(pole, kind) < at(equator, kind), "{kind:?} at the pole");
                }
                Light::Sunlit => {
                    assert!(
                        at(pole, kind) > at(equator, kind),
                        "{kind:?} at the equator"
                    );
                }
                _ => {}
            }
        }
    }
}
