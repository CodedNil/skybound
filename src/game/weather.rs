//! The planet's weather systems, indexed for the renderer and for anything
//! that needs to know how cloudy its sky is; and the lightning of its storms.
use bevy::{
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
};
use core::f32::consts::{PI, TAU};
use rayon::prelude::*;
use skybound::scene::{
    FEATURE_SCALE, PLANET_RADIUS, ViewUniform,
    aur::{aur_climate, energy_color},
    climate::{
        CLOUD_CELLS, CLOUD_GRID, CLOUD_SLOTS, CloudCover, CloudParcel, cloud_cell, cloud_parcel,
        cube_direction,
    },
    math::fourth,
    planet::surface_frame,
};
use spirv_std::glam::{FloatExt, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4};
use std::sync::{Arc, OnceLock};

/// Seconds between rebuilds of the parcel index. Systems are listed with
/// enough margin to cover how far they move and grow in that time.
const INDEX_PERIOD: f32 = 60.0;

const INDEX_MARGIN: f32 = 14_000.0;

/// Altitude margin on the layers listed for a cell.
const LAYER_MARGIN: f32 = 600.0;

/// Clear distances beyond this gain the march nothing.
const CLEARANCE_LIMIT: f32 = 400_000.0;

/// How deep inside the veil a discharge is rooted, in the world.
const ROOT_DEPTH: f32 = 900.0 / FEATURE_SCALE;

/// Every weather system in flight, and an index of the systems over each
/// cube-face cell of the planet.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct CloudField {
    pub parcels: Arc<[CloudParcel]>,
    /// Each cell's range of entries, then the entries: parcel ids.
    pub index: Arc<[u32]>,
    /// Each cell's clear distance, then the bottom and top of its cloud
    /// below the cirrus level, and the bottom of its cirrus.
    pub cells: Arc<[Vec4]>,
    /// The time the index was built at.
    pub built: f32,
}

pub struct WeatherPlugin;

struct LightningSeed {
    up: Vec3,
    period: f32,
    offset: f32,
    duration: f32,
    random: f32,
}

#[derive(Clone, Copy)]
pub struct Lightning {
    pub id: u32,
    pub position: Vec3,
    pub up: Vec3,
    pub strength: f32,
    pub reach: f32,
    pub radius: f32,
    pub random: f32,
    /// Which flicker of the flash this is; each one re-forks the channel.
    pub pulse: u32,
}

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CloudField::at(0.0))
            .add_systems(Update, advance_clouds);
    }
}

impl CloudField {
    pub fn at(time: f32) -> Self {
        let parcels: Arc<[CloudParcel]> = parcels(time).into();
        let (index, cells) = build_index(&parcels);
        Self {
            parcels,
            index: index.into(),
            cells: cells.into(),
            built: time,
        }
    }

    /// Cloud cover over a direction from the planet centre: see
    /// [`CloudCover::texels`]. The renderer adds the finer structure.
    pub fn cover(&self, up: Vec3) -> [Vec4; 2] {
        let cell = cloud_cell(up);
        let mut cover = CloudCover::default();
        for &id in &self.index[self.index[cell] as usize..self.index[cell + 1] as usize] {
            cover.add(self.parcels[id as usize], up);
        }
        cover.texels()
    }
}

impl Lightning {
    /// A rising discharge rooted inside the veil. The main channel keeps
    /// its course through a flash, but every flicker re-forks its joints a
    /// little, so a strike crackles instead of hanging still.
    pub fn segments(self) -> [[Vec4; 2]; 8] {
        let basis = surface_frame(self.up);
        let root = self.up * (PLANET_RADIUS - ROOT_DEPTH);
        let seed = self.id ^ self.random.to_bits();
        let pulse = seed ^ self.pulse.wrapping_mul(0x2c1b_3c6d);
        let mut points = [root; 7];
        let mut segments = [[Vec4::ZERO; 2]; 8];
        for index in 1..=6 {
            let t = index as f32 / 6.0;
            let east = strike_random(seed ^ (index as u32 * 17)) * 2.0 - 1.0
                + (strike_random(pulse ^ (index as u32 * 17)) - 0.5) * 0.7;
            let north = strike_random(seed ^ (index as u32 * 31)) * 2.0 - 1.0
                + (strike_random(pulse ^ (index as u32 * 31)) - 0.5) * 0.7;
            points[index] = root
                + self.up * (self.reach * t)
                + (basis.x_axis * east + basis.y_axis * north) * (self.reach * 0.045);
            segments[index - 1] = [
                points[index - 1]
                    .extend((24.0 + self.random * 18.0) * (1.0 - t * 0.45) / FEATURE_SCALE),
                points[index].extend(self.strength * 12.0 * (1.0 - t * 0.25)),
            ];
        }
        for branch in 0..2 {
            let index = 2 + branch * 2;
            let angle = strike_random(pulse ^ (branch as u32 + 113)) * TAU;
            let sideways = basis.x_axis * angle.cos() + basis.y_axis * angle.sin();
            let end = points[index] + (self.up * 0.13 + sideways * 0.10) * self.reach;
            segments[6 + branch] = [
                points[index].extend((10.0 + self.random * 6.0) / FEATURE_SCALE),
                end.extend(self.strength * 4.0),
            ];
        }
        segments
    }
}

fn parcels(time: f32) -> Vec<CloudParcel> {
    (0..CLOUD_SLOTS)
        .into_par_iter()
        .map(|slot| cloud_parcel(slot, time))
        .collect()
}

/// Each cell's centre direction and angular radius.
fn cell_geometry() -> &'static [(Vec3, f32)] {
    static CELLS: OnceLock<Vec<(Vec3, f32)>> = OnceLock::new();
    CELLS.get_or_init(|| {
        (0..CLOUD_CELLS as u32)
            .map(|cell| {
                let face = cell / (CLOUD_GRID * CLOUD_GRID);
                let (x, y) = (cell % CLOUD_GRID, cell / CLOUD_GRID % CLOUD_GRID);
                let at = |dx: f32, dy: f32| {
                    let position = vec2(x as f32 + dx, y as f32 + dy) / CLOUD_GRID as f32;
                    cube_direction(face, position * 2.0 - 1.0)
                };
                let centre = at(0.5, 0.5);
                let radius = [at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0), at(1.0, 1.0)]
                    .iter()
                    .map(|corner| corner.dot(centre).clamp(-1.0, 1.0).acos())
                    .fold(0.0, f32::max);
                (centre, radius)
            })
            .collect()
    })
}

fn angle(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

/// List the systems reaching each cell, and bound how far each cell's sky
/// is clear and which altitudes its cloud can fill.
fn build_index(parcels: &[CloudParcel]) -> (Vec<u32>, Vec<Vec4>) {
    let geometry = cell_geometry();
    let active: Vec<CloudParcel> = parcels
        .iter()
        .copied()
        .filter(|p| p.shape.y > 0.0)
        .collect();
    let ids: Vec<u32> = (0..parcels.len() as u32)
        .filter(|&id| parcels[id as usize].shape.y > 0.0)
        .collect();
    let lists: Vec<Vec<u32>> = geometry
        .par_iter()
        .map(|&(centre, radius)| {
            ids.iter()
                .copied()
                .filter(|&id| {
                    let parcel = parcels[id as usize];
                    let reach = (parcel.centre.w + INDEX_MARGIN) / PLANET_RADIUS + radius;
                    reach >= PI || centre.dot(parcel.centre.xyz()) > reach.cos()
                })
                .collect()
        })
        .collect();
    let cells = geometry
        .par_iter()
        .zip(&lists)
        .map(|(&(centre, radius), list)| {
            let touching: Vec<CloudParcel> = list
                .iter()
                .map(|&id| parcels[id as usize])
                .filter(|parcel| {
                    let reach = (parcel.centre.w + INDEX_MARGIN) / PLANET_RADIUS + radius;
                    centre.dot(parcel.centre.xyz()) > reach.min(PI).cos()
                })
                .collect();
            if touching.is_empty() {
                // Distance from anywhere in the cell to the nearest system,
                // less what it can move before the next rebuild.
                let horizon = (CLEARANCE_LIMIT / PLANET_RADIUS + radius).min(PI).cos();
                let nearest = active
                    .iter()
                    .filter(|parcel| centre.dot(parcel.centre.xyz()) > horizon)
                    .map(|parcel| {
                        angle(centre, parcel.centre.xyz()) * PLANET_RADIUS - parcel.centre.w
                    })
                    .fold(CLEARANCE_LIMIT + radius * PLANET_RADIUS, f32::min);
                let clearance = nearest - radius * PLANET_RADIUS - INDEX_MARGIN;
                return vec4(clearance.max(1.0), 1.0e9, -1.0e9, 1.0e9);
            }
            let extent = touching.iter().map(|parcel| parcel.extent()).fold(
                vec4(1.0e9, -1.0e9, 1.0e9, -1.0e9),
                |bounds, extent| {
                    vec4(
                        bounds.x.min(extent.x),
                        bounds.y.max(extent.y),
                        bounds.z.min(extent.z),
                        bounds.w.max(extent.w),
                    )
                },
            );
            vec4(
                0.0,
                extent.x - LAYER_MARGIN,
                extent.y + LAYER_MARGIN,
                extent.z - LAYER_MARGIN,
            )
        })
        .collect();
    let total: usize = lists.iter().map(Vec::len).sum();
    let mut index = Vec::with_capacity(CLOUD_CELLS + 1 + total);
    let mut offset = (CLOUD_CELLS + 1) as u32;
    for list in &lists {
        index.push(offset);
        offset += list.len() as u32;
    }
    index.push(offset);
    index.extend(lists.into_iter().flatten());
    (index, cells)
}

fn advance_clouds(time: Res<Time>, mut field: ResMut<CloudField>) {
    let now = time.elapsed_secs_wrapped();
    if (now - field.built).abs() > INDEX_PERIOD {
        *field = CloudField::at(now);
    } else {
        field.parcels = parcels(now).into();
    }
}

fn strike_random(id: u32) -> f32 {
    let mut value = id.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    value = ((value >> ((value >> 28) + 4)) ^ value).wrapping_mul(277_803_737);
    ((value >> 22) ^ value) as f32 / u32::MAX as f32
}

fn lightning_catalog() -> &'static [LightningSeed] {
    static SEEDS: OnceLock<Vec<LightningSeed>> = OnceLock::new();
    SEEDS.get_or_init(|| {
        (0..8192)
            .map(|id| {
                let z = 1.0 - 2.0 * (id as f32 + 0.5) / 8192.0;
                let longitude = id as f32 * 2.399_963_1 + strike_random(id ^ 0x117) * 0.1;
                let radius = (1.0 - z * z).sqrt();
                let period = 24.0 + strike_random(id ^ 0x319) * 30.0;
                LightningSeed {
                    up: vec3(radius * longitude.cos(), radius * longitude.sin(), z),
                    period,
                    offset: strike_random(id ^ 0x713) * period,
                    duration: 0.7 + strike_random(id ^ 0x151) * 1.6,
                    random: strike_random(id ^ 0xa17),
                }
            })
            .collect()
    })
}

pub fn lightning_events(time: f32) -> impl Iterator<Item = Lightning> {
    lightning_catalog()
        .iter()
        .enumerate()
        .filter_map(move |(id, seed)| {
            // Storm regions discharge far more often than calm seas.
            let storm = aur_climate(seed.up, time).x;
            if strike_random(id as u32 ^ 0x5a5a) > 0.12 + storm * 1.3 {
                return None;
            }
            let age = (time + seed.offset).rem_euclid(seed.period);
            if age >= seed.duration {
                return None;
            }
            let phase = age / seed.duration;
            let envelope = phase.smoothstep(0.0, 0.12) * (1.0 - phase.smoothstep(0.7, 1.0));
            let flicker_phase = time * 19.0 + seed.random * TAU;
            let flicker = fourth(flicker_phase.sin().max(0.0));
            // Each recurring storm produces a new channel on its next discharge.
            let cycle = ((time + seed.offset) / seed.period).floor() as u32;
            let strike = strike_random((id as u32) ^ cycle.wrapping_mul(0x9e37_79b9));
            let reach = (8000.0 + strike * 20_000.0) / FEATURE_SCALE;
            Some(Lightning {
                id: id as u32,
                position: seed.up * (PLANET_RADIUS + reach * 0.5),
                up: seed.up,
                reach,
                radius: (6000.0 + seed.random * 10_000.0) / FEATURE_SCALE,
                strength: envelope * (0.25 + flicker * 0.75),
                random: strike,
                pulse: (flicker_phase / TAU).floor() as u32,
            })
        })
}

/// Choose the visible lightning for this frame: flashes light the veil and
/// bolts are drawn as glowing segments.
pub fn prepare_flashes(view: &mut ViewUniform) {
    view.aur_flashes = [[Vec4::ZERO; 2]; 32];
    view.aur_bolts = [[Vec4::ZERO; 2]; 32];
    let camera = view.to_planet(view.world_position.xyz());
    let mut visible: Vec<_> = lightning_events(view.time())
        .filter_map(|event| {
            let offset = event.position - camera;
            let distance = offset.length();
            // Cull the solid planet, including the upper part of a tall strike.
            if event.up.dot(camera) < PLANET_RADIUS - event.reach {
                return None;
            }
            let local = view.from_planet(event.position);
            let clip = view.clip_from_world * local.extend(1.0);
            let margin = (event.reach + event.radius) * 3.0 / distance.max(1.0);
            if distance > event.reach + event.radius
                && (clip.w <= 0.0
                    || (clip.truncate().truncate() / clip.w).abs().max_element() > 1.0 + margin)
            {
                return None;
            }
            let importance =
                event.strength * (0.5 + 0.5 * (event.radius / distance.max(1.0)).sqrt().min(1.0));
            Some((importance, event))
        })
        .collect();
    visible.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
    let mut bolt_count = 0;
    for (index, (_, event)) in visible.iter().take(32).enumerate() {
        // The veil it lights is in feature space.
        let root = event.up * (PLANET_RADIUS - ROOT_DEPTH) * FEATURE_SCALE;
        view.aur_flashes[index] = [
            root.extend((event.radius * FEATURE_SCALE).recip()),
            (energy_color(event.random, 0.45 + event.random * 0.2) * event.strength * 9.0)
                .extend(event.strength),
        ];
        if bolt_count + 8 > view.aur_bolts.len() {
            continue;
        }
        for segment in event.segments() {
            view.aur_bolts[bolt_count] = [
                view.from_planet(segment[0].truncate()).extend(segment[0].w),
                view.from_planet(segment[1].truncate()).extend(segment[1].w),
            ];
            bolt_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_PI_2;
    use skybound::scene::ViewUniform;
    use spirv_std::glam::{Mat4, camera::rh::proj::directx::perspective_infinite_reverse};

    fn sphere(index: u32, count: u32) -> Vec3 {
        let z = (index as f32 + 0.5) / count as f32 * 2.0 - 1.0;
        let longitude = index as f32 * 2.399_963;
        let radius = (1.0 - z * z).sqrt();
        vec3(radius * longitude.cos(), radius * longitude.sin(), z)
    }

    #[test]
    fn systems_travel_polewards_and_dissolve_before_the_poles() {
        for slot in (0..CLOUD_SLOTS).step_by(7) {
            let early = cloud_parcel(slot, 1000.0);
            let later = cloud_parcel(slot, 1600.0);
            if early.shape.y <= 0.0 || later.shape.y <= 0.0 {
                continue;
            }
            // A slot may begin a new journey at the equator in between.
            let (from, to) = (early.centre.z.abs(), later.centre.z.abs());
            assert!(
                to >= from || to < 0.25,
                "slot {slot} drifted towards the equator"
            );
        }
        for slot in 0..CLOUD_SLOTS {
            let parcel = cloud_parcel(slot, 5000.0);
            if parcel.centre.z.abs() > 0.998 {
                assert!(parcel.shape.y < 0.05, "a system survived at the pole");
            }
        }
    }

    #[test]
    fn skies_vary_from_clear_to_overcast_and_cloud_up_towards_the_poles() {
        let field = CloudField::at(500.0);
        let mut bands = [(0.0f32, 0u32); 4];
        let (mut clear, mut overcast) = (0, 0);
        for index in 0..4000 {
            let up = sphere(index, 4000);
            let [lower, upper] = field.cover(up);
            let total = lower.max_element().max(upper.x);
            clear += u32::from(total < 0.02 && upper.y.max(upper.z) < 0.02);
            overcast += u32::from(total > 0.8);
            let band = ((up.z.abs().asin() / FRAC_PI_2) * 4.0).min(3.0) as usize;
            bands[band].0 += total;
            bands[band].1 += 1;
        }
        assert!(clear > 200, "too few clear skies: {clear}");
        // Overcast is the exception: most skies hold a few broken layers.
        assert!(overcast > 40, "too few overcast skies: {overcast}");
        assert!(overcast < 800, "too many overcast skies: {overcast}");
        let mean = |band: (f32, u32)| band.0 / band.1 as f32;
        assert!(
            mean(bands[2]) > mean(bands[0]),
            "polar skies must be cloudier"
        );
    }

    #[test]
    fn index_clearances_never_hide_cloud_until_the_next_rebuild() {
        let field = CloudField::at(800.0);
        let later = CloudField {
            parcels: parcels(800.0 + INDEX_PERIOD).into(),
            ..field.clone()
        };
        for index in 0..20_000 {
            let up = sphere(index, 20_000);
            let clearance = field.cells[cloud_cell(up)].x;
            if clearance <= 0.0 {
                continue;
            }
            for parcel in later.parcels.iter().filter(|parcel| parcel.shape.y > 0.003) {
                let distance = angle(up, parcel.centre.xyz()) * PLANET_RADIUS;
                assert!(
                    distance - parcel.centre.w >= clearance,
                    "cloud within a clear cell"
                );
            }
        }
    }

    #[test]
    fn lightning_remains_planet_fixed_when_local_coordinates_change() {
        let mut first = ViewUniform {
            planet_center_value: (-Vec3::Z * (PLANET_RADIUS + 476_000.0)).extend(0.0),
            clip_from_world: perspective_infinite_reverse(1.2, 1.7, 0.1),
            times: Vec4::new(17.5, 0.0, 0.0, 0.0),
            ..ViewUniform::default()
        };
        prepare_flashes(&mut first);
        assert!(first.aur_bolts[0][0].w > 0.0);
        let offset = vec3(10_000.0, 20_000.0, -5000.0);
        let mut shifted = first;
        shifted.world_position = offset.extend(0.0);
        shifted.planet_center_value += offset.extend(0.0);
        shifted.clip_from_world *= Mat4::from_translation(-offset);
        prepare_flashes(&mut shifted);
        for (a, b) in first.aur_bolts.iter().zip(shifted.aur_bolts.iter()) {
            if a[0].w <= 0.0 {
                break;
            }
            for endpoint in 0..2 {
                assert!((a[endpoint].truncate() + offset).distance(b[endpoint].truncate()) < 1.0);
                assert!(
                    (first.to_planet(a[endpoint].truncate()).length() - PLANET_RADIUS).abs()
                        < 30_000.0
                );
            }
        }
    }

    #[test]
    fn discharges_rise_from_the_veil_and_branches_join_the_main_channel() {
        for event in lightning_events(17.5) {
            let segments = event.segments();
            assert_eq!(segments, event.segments());
            let root = segments[0][0].truncate();
            assert!((root.length() - PLANET_RADIUS + ROOT_DEPTH).abs() < 0.5);
            for (index, segment) in segments[..6].iter().enumerate() {
                let direction = segment[1].truncate() - segment[0].truncate();
                assert!(direction.normalize().dot(event.up) > 0.75);
                if index > 0 {
                    assert_eq!(segment[0].truncate(), segments[index - 1][1].truncate());
                }
            }
            assert_eq!(segments[6][0].truncate(), segments[1][1].truncate());
            assert_eq!(segments[7][0].truncate(), segments[3][1].truncate());
            let tip = segments[5][1].truncate() - root;
            assert!((tip.dot(event.up) - event.reach).abs() < 0.5);
        }
    }

    #[test]
    fn lightning_has_reproducible_activity_across_both_hemispheres() {
        let first: Vec<_> = lightning_events(17.5).collect();
        let second: Vec<_> = lightning_events(17.5).collect();
        assert!(first.len() > 150 && first.len() < 700);
        assert_eq!(
            first.iter().map(|event| event.id).collect::<Vec<_>>(),
            second.iter().map(|event| event.id).collect::<Vec<_>>()
        );
        assert!(first.iter().any(|event| event.up.z > 0.8));
        assert!(first.iter().any(|event| event.up.z < -0.8));
        for (a, b) in first.iter().zip(second.iter()) {
            assert_eq!(a.position, b.position);
            assert!((a.up.length() - 1.0).abs() < 0.0001);
            assert!((0.0..=1.0).contains(&a.strength));
        }
    }
}
