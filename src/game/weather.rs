//! Slow, deterministic planetary moisture transport and individually advected clouds.
use core::f32::consts::{PI, TAU};
use std::env;
use std::sync::{Arc, OnceLock};

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::tasks::futures_lite::future::poll_once;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on};
use rayon::prelude::*;
use skybound::scene::aur::{aur_climate, energy_color};
use skybound::scene::climate::{CLOUD_GRID_CELLS, Cloud, cloud_cell, cloud_cell_index};
use skybound::scene::math::{fbm, fourth};
use skybound::scene::planet::surface_frame;
use skybound::scene::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{FloatExt, IVec3, Vec3, Vec4, Vec4Swizzles, vec3};

pub const TICK: f32 = 2.0;

const LATITUDES: usize = 32;

const LONGITUDES: usize = 64;

pub const CLOUD_COUNT: usize = 16_384;

#[derive(Resource)]
pub struct WeatherSimulation(pub Arc<PlanetWeather>, Option<Task<PlanetWeather>>);

impl Default for WeatherSimulation {
    fn default() -> Self {
        let mut arguments = env::args();
        let seed = arguments
            .find(|arg| arg == "--seed")
            .and_then(|_| arguments.next())
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        Self(Arc::new(PlanetWeather::new(seed, 0.0)), None)
    }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct CloudSnapshot {
    pub clouds: Arc<[Cloud]>,
    pub grid: Arc<[u32]>,
    pub time: f32,
    pub revision: u64,
}

impl FromWorld for CloudSnapshot {
    fn from_world(world: &mut World) -> Self {
        let model = &world.resource::<WeatherSimulation>().0;
        Self {
            clouds: model.clouds.clone().into(),
            grid: model.grid.clone().into(),
            time: model.time,
            revision: model.revision,
        }
    }
}

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WeatherSimulation>()
            .init_resource::<CloudSnapshot>()
            .add_systems(Update, simulate);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WeatherCell {
    pub moisture: f32,
    pub temperature: f32,
    pub pressure: f32,
    pub precipitation: f32,
    pub wind: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Cyclone {
    pub center: Vec3,
    pub radius: f32,
    pub intensity: f32,
    phase: f32,
}

#[derive(Clone)]
pub struct PlanetWeather {
    pub cells: Vec<WeatherCell>,
    pub clouds: Vec<Cloud>,
    pub cyclones: Vec<Cyclone>,
    pub grid: Vec<u32>,
    pub time: f32,
    pub revision: u64,
    seed: u32,
    ages: Vec<f32>,
    generations: Vec<u32>,
}

impl PlanetWeather {
    pub fn new(seed: u32, time: f32) -> Self {
        let cyclones: Vec<_> = (0..12)
            .map(|id| {
                let latitude = (0.3 + random(seed ^ (id * 19 + 7)) * 0.55)
                    * if id & 1 == 0 { -1.0 } else { 1.0 };
                let longitude = random(seed ^ (id * 71 + 11)) * TAU;
                Cyclone {
                    center: vec3(
                        latitude.cos() * longitude.cos(),
                        latitude.cos() * longitude.sin(),
                        latitude.sin(),
                    ),
                    radius: 110_000.0 + random(id + 91) * 160_000.0,
                    intensity: 0.35 + random(id + 111) * 0.6,
                    phase: random(id + 151) * TAU,
                }
            })
            .collect();
        let cells = (0..LATITUDES * LONGITUDES)
            .map(|index| {
                let up = direction(index / LONGITUDES, index % LONGITUDES);
                WeatherCell {
                    moisture: regional_moisture(up, seed, &cyclones),
                    temperature: 284.0 + up.z.abs() * 25.0,
                    pressure: 1.0,
                    wind: circulation(up, &cyclones).0,
                    precipitation: 0.0,
                }
            })
            .collect();
        let mut result = Self {
            cells,
            clouds: vec![Cloud::default(); CLOUD_COUNT],
            cyclones,
            grid: Vec::new(),
            time,
            revision: 0,
            seed,
            ages: vec![0.0; CLOUD_COUNT],
            generations: vec![0; CLOUD_COUNT],
        };
        for id in 0..CLOUD_COUNT {
            let up = result.birth_direction(id, 0, false);
            result.clouds[id] = result.make_cloud(id, up, 0);
            result.ages[id] = random(id as u32 ^ seed ^ 0xa71) * 21_600.0;
        }
        result.rebuild_index();
        result
    }
    fn birth_direction(&self, id: usize, generation: u32, equatorial: bool) -> Vec3 {
        let key = (id as u32).wrapping_mul(137) ^ self.seed ^ generation.wrapping_mul(7919);
        let mut up = Vec3::X;
        for attempt in 0..12 {
            let key = key ^ (attempt * 3917 + 17);
            let z = (random(key) * 2.0 - 1.0) * if equatorial { 0.14 } else { 0.96 };
            let longitude = random(key ^ 0x157) * TAU;
            up = vec3(
                (1.0 - z * z).sqrt() * longitude.cos(),
                (1.0 - z * z).sqrt() * longitude.sin(),
                z,
            );
            let (.., convection, arms) = circulation(up, &self.cyclones);
            let cover = formation(sample(&self.cells, up).moisture, convection, arms);
            if random(key ^ 0x371) < cover {
                break;
            }
        }
        up
    }
    fn make_cloud(&self, id: usize, up: Vec3, generation: u32) -> Cloud {
        let seed =
            random((id as u32).wrapping_mul(137) ^ self.seed ^ generation.wrapping_mul(7919));
        let weather = sample(&self.cells, up);
        let (wind, convection, arms) = circulation(up, &self.cyclones);
        let layer = random(id as u32 ^ self.seed ^ 0xc16);
        let high = layer > 0.94;
        let middle = layer > 0.73 && !high;
        let vertical = random(id as u32 ^ self.seed ^ generation ^ 0x513);
        let horizontal = random(id as u32 ^ self.seed ^ generation ^ 0xa17);
        let aspect = random(id as u32 ^ self.seed ^ generation ^ 0x239);
        let cover = formation(weather.moisture, convection, arms);
        let development =
            ((weather.moisture - 0.35) * 0.9 + convection * 0.65 + (vertical - 0.5) * 0.4)
                .clamp(0.0, 1.0);
        let height = if high {
            300.0 + vertical * 1100.0
        } else if middle {
            1500.0 + vertical * 3500.0
        } else {
            2000.0 + development * 10_000.0
        };
        let base = if high {
            36_000.0 + seed * 20_000.0
        } else if middle {
            19_000.0 + seed * 3000.0
        } else {
            5500.0 + seed * 1800.0
        };
        let center = up * (PLANET_RADIUS + base + height);
        let frame = surface_frame(up);
        let axis = if wind.length_squared() > 0.1 {
            wind.normalize()
        } else {
            frame.x_axis
        };
        let radius = if high {
            22_000.0 + seed * 25_000.0
        } else {
            (3500.0 + horizontal * 17_000.0 + convection * 10_000.0) * (1.0 + cover * 0.65)
        };
        let density = if high {
            0.000_08
        } else {
            0.002 + development * 0.002
        };
        Cloud {
            center: center.extend(radius),
            up: up.extend(height),
            east: axis.extend(
                radius
                    * if high {
                        0.12 + aspect * 0.3
                    } else {
                        0.45 + aspect * 0.65
                    },
            ),
            appearance: Vec4::new(
                density * cover * (0.7 + arms * 0.5),
                0.2 + aspect * 0.5,
                if high { 0.000_15 } else { 0.000_35 },
                development,
            ),
            motion: wind.extend(seed),
        }
    }
    pub fn tick(&mut self) {
        self.time += TICK;
        for storm in &mut self.cyclones {
            let frame = surface_frame(storm.center);
            storm.center = (storm.center
                + (frame.y_axis * storm.center.z.signum() * 8.0 + frame.x_axis * 5.0)
                    * (TICK / PLANET_RADIUS))
                .normalize();
            let weather = sample(&self.cells, storm.center);
            let target = ((weather.moisture - 0.35) * 1.8).clamp(0.0, 1.0)
                * (1.0 - storm.center.z.abs().smoothstep(0.85, 0.98));
            storm.intensity += (target - storm.intensity) * (TICK / 7200.0);
            storm.phase += TICK * storm.intensity / 3600.0;
        }
        let old = &self.cells;
        let storms = &self.cyclones;
        self.cells = (0..old.len())
            .into_par_iter()
            .map(|index| {
                let up = direction(index / LONGITUDES, index % LONGITUDES);
                let (wind, convergence, _) = circulation(up, storms);
                let previous = sample(old, (up - wind * (TICK / PLANET_RADIUS)).normalize());
                let equatorial_feed = (-up.z.powi(2) / 0.05).exp() * 0.000_015;
                let polar_drying = up.z.abs().smoothstep(0.8, 0.99) * 0.000_1;
                let condensation =
                    (previous.moisture - 0.6).max(0.0) * (0.000_01 + convergence * 0.000_08);
                let precipitation = previous.precipitation
                    + (condensation * 10_000.0 - previous.precipitation) * TICK / 600.0;
                WeatherCell {
                    moisture: (previous.moisture
                        + TICK * (equatorial_feed - polar_drying - condensation))
                        .clamp(0.0, 1.0),
                    temperature: 284.0 + up.z.abs() * 25.0 - condensation * 10_000.0,
                    pressure: 1.0 - convergence * 0.12,
                    precipitation,
                    wind,
                }
            })
            .collect();
        let cells = &self.cells;
        let storms = &self.cyclones;
        self.clouds
            .par_iter_mut()
            .zip(self.ages.par_iter_mut())
            .for_each(|(cloud, age)| {
                *age += TICK;
                let altitude = cloud.center.truncate().length() - PLANET_RADIUS;
                let up = (cloud.center.truncate() + cloud.motion.truncate() * TICK).normalize();
                let (wind, convection, arms) = circulation(up, storms);
                let weather = sample(cells, up);
                cloud.center = (up * (PLANET_RADIUS + altitude)).extend(cloud.center.w);
                cloud.up = up.extend(cloud.up.w);
                cloud.east = wind.normalize_or_zero().extend(cloud.east.w);
                cloud.motion = wind.extend(cloud.motion.w);
                let drying = up.z.abs().smoothstep(0.82, 0.98);
                let target = formation(weather.moisture, convection, arms)
                    * (0.7 + arms * 0.5)
                    * (1.0 - drying);
                let density_scale = if cloud.appearance.z < 0.0002 {
                    0.000_08
                } else {
                    0.002 + cloud.appearance.w * 0.002
                };
                cloud.appearance.x += (target * density_scale - cloud.appearance.x) * TICK / 600.0;
                cloud.appearance.w += (convection - cloud.appearance.w) * TICK / 1800.0;
            });
        for id in 0..self.clouds.len() {
            if self.clouds[id].up.z.abs() > 0.97
                || self.ages[id] > 43_200.0 + self.clouds[id].motion.w * 43_200.0
            {
                self.generations[id] += 1;
                let up = self.birth_direction(id, self.generations[id], true);
                self.clouds[id] = self.make_cloud(id, up, self.generations[id]);
                self.ages[id] = 0.0;
            }
        }
        self.rebuild_index();
        self.revision += 1;
    }
    fn rebuild_index(&mut self) {
        let bounds: Vec<_> = self
            .clouds
            .iter()
            .map(|cloud| {
                let extent = cloud.extent() + Vec3::splat(512.0);
                (
                    cloud_cell(cloud.center.truncate() - extent),
                    cloud_cell(cloud.center.truncate() + extent),
                )
            })
            .collect();
        let mut counts = vec![0u32; CLOUD_GRID_CELLS];
        let visit = |lower: IVec3, upper: IVec3, f: &mut dyn FnMut(usize)| {
            for z in lower.z..=upper.z {
                for y in lower.y..=upper.y {
                    for x in lower.x..=upper.x {
                        if let Some(index) = cloud_cell_index(IVec3::new(x, y, z)) {
                            f(index);
                        }
                    }
                }
            }
        };
        for &(lower, upper) in &bounds {
            visit(lower, upper, &mut |index| counts[index] += 1);
        }
        self.grid.clear();
        self.grid.resize(CLOUD_GRID_CELLS + 1, 0);
        for (index, count) in counts.iter().enumerate() {
            self.grid[index + 1] = self.grid[index] + count;
        }
        let total = self.grid[CLOUD_GRID_CELLS] as usize;
        let mut cursors = self.grid[..CLOUD_GRID_CELLS].to_vec();
        self.grid.resize(CLOUD_GRID_CELLS + 1 + total, 0);
        for (id, &(lower, upper)) in bounds.iter().enumerate() {
            visit(lower, upper, &mut |index| {
                self.grid[CLOUD_GRID_CELLS + 1 + cursors[index] as usize] = id as u32;
                cursors[index] += 1;
            });
        }
    }
}

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

impl Lightning {
    /// A rising discharge rooted inside the veil. The main channel keeps
    /// its course through a flash, but every flicker re-forks its joints a
    /// little, so a strike crackles instead of hanging still.
    pub fn segments(self) -> [[Vec4; 2]; 8] {
        let basis = surface_frame(self.up);
        let root = self.up * (PLANET_RADIUS - 900.0);
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
                points[index - 1].extend((24.0 + self.random * 18.0) * (1.0 - t * 0.45)),
                points[index].extend(self.strength * 12.0 * (1.0 - t * 0.25)),
            ];
        }
        for branch in 0..2 {
            let index = 2 + branch * 2;
            let angle = strike_random(pulse ^ (branch as u32 + 113)) * TAU;
            let sideways = basis.x_axis * angle.cos() + basis.y_axis * angle.sin();
            let end = points[index] + (self.up * 0.13 + sideways * 0.10) * self.reach;
            segments[6 + branch] = [
                points[index].extend(10.0 + self.random * 6.0),
                end.extend(self.strength * 4.0),
            ];
        }
        segments
    }
}

fn simulate(
    time: Res<Time<Virtual>>,
    mut simulation: ResMut<WeatherSimulation>,
    mut snapshot: ResMut<CloudSnapshot>,
) {
    let now = time.elapsed_secs();
    if let Some(mut pending) = simulation.1.take() {
        if let Some(model) = block_on(poll_once(&mut pending)) {
            simulation.0 = Arc::new(model);
        } else {
            simulation.1 = Some(pending);
        }
    }
    if simulation.1.is_none() && (now - simulation.0.time >= TICK || now < simulation.0.time) {
        let current = Arc::clone(&simulation.0);
        simulation.1 = Some(AsyncComputeTaskPool::get().spawn(async move {
            let mut model = (*current).clone();
            if now < model.time || now - model.time > 120.0 {
                model.time = now;
                model.revision += 1;
            } else {
                model.tick();
            }
            model
        }));
    }
    if snapshot.revision != simulation.0.revision {
        snapshot.clouds = simulation.0.clouds.clone().into();
        snapshot.grid = simulation.0.grid.clone().into();
        snapshot.time = simulation.0.time;
        snapshot.revision = simulation.0.revision;
    }
}

fn random(value: u32) -> f32 {
    let mut value = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    value = ((value >> ((value >> 28) + 4)) ^ value).wrapping_mul(277_803_737);
    ((value >> 22) ^ value) as f32 / u32::MAX as f32
}

fn direction(row: usize, column: usize) -> Vec3 {
    let latitude = ((row as f32 + 0.5) / LATITUDES as f32 - 0.5) * PI;
    let longitude = (column as f32 + 0.5) / LONGITUDES as f32 * TAU;
    vec3(
        latitude.cos() * longitude.cos(),
        latitude.cos() * longitude.sin(),
        latitude.sin(),
    )
}

fn sample(cells: &[WeatherCell], up: Vec3) -> WeatherCell {
    let y = ((up.z.clamp(-1.0, 1.0).asin() / PI + 0.5) * LATITUDES as f32 - 0.5)
        .clamp(0.0, (LATITUDES - 1) as f32);
    let x = up.y.atan2(up.x).rem_euclid(TAU) / TAU * LONGITUDES as f32 - 0.5;
    let mut result = WeatherCell::default();
    for row in 0..2 {
        for column in 0..2 {
            let iy = (y.floor() as usize + row).min(LATITUDES - 1);
            let ix = (x.floor() as i32 + column).rem_euclid(64) as usize;
            let weight = if row == 0 { 1.0 - y.fract() } else { y.fract() }
                * if column == 0 {
                    1.0 - (x - x.floor())
                } else {
                    x - x.floor()
                };
            let cell = cells[iy * LONGITUDES + ix];
            result.moisture += cell.moisture * weight;
            result.temperature += cell.temperature * weight;
            result.pressure += cell.pressure * weight;
            result.precipitation += cell.precipitation * weight;
            result.wind += cell.wind * weight;
        }
    }
    result
}

fn circulation(up: Vec3, cyclones: &[Cyclone]) -> (Vec3, f32, f32) {
    let frame = surface_frame(up);
    let poleward = frame.y_axis
        * up.z.signum()
        * (12.0 + up.z.abs() * 35.0)
        * (1.0 - up.z * up.z).max(0.0).sqrt();
    let jet = frame.x_axis * (up.z * 6.0).sin() * 18.0;
    let mut wind = poleward + jet;
    let mut convergence = 0.0f32;
    let mut arms = 0.0f32;
    for storm in cyclones {
        let cosine = up.dot(storm.center);
        if cosine <= 0.0 {
            continue;
        }
        let delta = storm.center - up * cosine;
        let distance = delta.length() * PLANET_RADIUS;
        if distance > storm.radius * 3.0 || distance < 1.0 {
            continue;
        }
        let inward = delta.normalize();
        let envelope = (-(distance / storm.radius).powi(2)).exp() * storm.intensity;
        let eye = distance.smoothstep(storm.radius * 0.08, storm.radius * 0.3);
        let tangent = inward.cross(up) * storm.center.z.signum();
        wind += (tangent * (100.0 * eye) + inward * 18.0) * envelope;
        convergence = convergence.max(envelope * eye);
        let basis = surface_frame(storm.center);
        let angle = up.dot(basis.y_axis).atan2(up.dot(basis.x_axis));
        let spiral = (angle * 3.0 - distance / storm.radius * 7.0 - storm.phase).cos() * 0.5 + 0.5;
        arms = arms.max(envelope * eye * spiral.powi(2));
    }
    (wind, convergence, arms)
}

fn formation(moisture: f32, convection: f32, arms: f32) -> f32 {
    (moisture + convection * 0.12 + arms * 0.18).smoothstep(0.42, 0.72)
}

fn regional_moisture(up: Vec3, seed: u32, storms: &[Cyclone]) -> f32 {
    let offset = vec3(
        random(seed ^ 0x47),
        random(seed ^ 0x85),
        random(seed ^ 0xbf),
    ) * 11.0;
    let broad = fbm(up * 0.65 + offset, 4, seed ^ 0x73);
    let fronts = fbm(up * 1.9 + offset, 4, seed ^ 0x117);
    let convergence = circulation(up, storms).1;
    (0.44
        + (1.0 - up.z.abs()) * 0.15
        + (broad - 0.5) * 1.5
        + (fronts - 0.5) * 0.35
        + convergence * 0.3)
        .clamp(0.04, 0.98)
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
            let reach = 8000.0 + strike * 20_000.0;
            Some(Lightning {
                id: id as u32,
                position: seed.up * (PLANET_RADIUS + reach * 0.5),
                up: seed.up,
                reach,
                radius: 6000.0 + seed.random * 10_000.0,
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
        let root = event.up * (PLANET_RADIUS - 900.0);
        view.aur_flashes[index] = [
            root.extend(event.radius.recip()),
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
    use skybound::scene::ViewUniform;
    use spirv_std::glam::Mat4;
    use spirv_std::glam::camera::rh::proj::directx::perspective_infinite_reverse;

    use super::*;

    #[test]
    fn cyclones_do_not_affect_the_opposite_side_of_the_planet() {
        let storm = Cyclone {
            center: vec3(0.8, 0.0, 0.6),
            radius: 200_000.0,
            intensity: 1.0,
            phase: 0.0,
        };
        let nearby = (storm.center + Vec3::Y * 0.03).normalize();
        assert!(circulation(nearby, &[storm]).1 > 0.1);
        let opposite = -nearby;
        assert_eq!(circulation(opposite, &[storm]), circulation(opposite, &[]));
    }

    #[test]
    fn regional_weather_creates_clear_zones_and_concentrates_cloud_births() {
        let planet = PlanetWeather::new(17, 0.0);
        let clear = planet
            .cells
            .iter()
            .filter(|cell| cell.moisture < 0.42)
            .count();
        let wet = planet
            .cells
            .iter()
            .filter(|cell| cell.moisture > 0.72)
            .count();
        assert!(clear > planet.cells.len() / 20, "missing dry climate zones");
        assert!(wet > planet.cells.len() / 50, "missing wet climate zones");
        let area_mean =
            planet.cells.iter().map(|cell| cell.moisture).sum::<f32>() / planet.cells.len() as f32;
        let cloud_mean = planet
            .clouds
            .iter()
            .map(|cloud| sample(&planet.cells, cloud.up.truncate()).moisture)
            .sum::<f32>()
            / planet.clouds.len() as f32;
        assert!(
            cloud_mean > area_mean + 0.07,
            "births must favour wet regions"
        );
        let opaque = planet
            .clouds
            .iter()
            .filter(|cloud| cloud.appearance.x > 0.001)
            .count();
        assert!(
            opaque > planet.clouds.len() / 5,
            "cloud cores must have meaningful optical thickness"
        );
    }
    #[test]
    fn planet_clouds_are_indexed_and_advect_without_radial_drift() {
        let mut planet = PlanetWeather::new(17, 0.0);
        let radii: Vec<_> = planet
            .clouds
            .iter()
            .map(|cloud| cloud.center.truncate().length())
            .collect();
        planet.tick();
        for (id, cloud) in planet.clouds.iter().enumerate().step_by(97) {
            assert!((cloud.center.truncate().length() - radii[id]).abs() < 1.0);
            let index = cloud_cell_index(cloud_cell(cloud.center.truncate())).unwrap();
            let start = CLOUD_GRID_CELLS + 1 + planet.grid[index] as usize;
            let end = CLOUD_GRID_CELLS + 1 + planet.grid[index + 1] as usize;
            assert!(
                planet.grid[start..end].contains(&(id as u32)),
                "cloud missing from spatial index"
            );
        }
        assert!(
            planet
                .cells
                .iter()
                .all(|cell| (0.0..=1.0).contains(&cell.moisture))
        );
        assert_eq!(planet.revision, 1);
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
            assert!((root.length() - PLANET_RADIUS + 900.0).abs() < 0.5);
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
