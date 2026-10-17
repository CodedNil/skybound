//! Slow, deterministic planetary moisture transport and individually advected clouds.
use crate::scene::noise::fbm;
use crate::scene::{
    PLANET_RADIUS,
    cloud::{Cloud, GRID_CELLS, cell, cell_index},
    planet::surface_frame,
};
use core::f32::consts::{PI, TAU};
use rayon::prelude::*;
use spirv_std::glam::{FloatExt, IVec3, Vec3, Vec4, vec3};

pub const TICK: f32 = 2.0;
const LATITUDES: usize = 32;
const LONGITUDES: usize = 64;
pub const CLOUD_COUNT: usize = 16_384;

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
        let delta = storm.center - up * up.dot(storm.center);
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
                    cell(cloud.center.truncate() - extent),
                    cell(cloud.center.truncate() + extent),
                )
            })
            .collect();
        let mut counts = vec![0u32; GRID_CELLS];
        let visit = |lower: IVec3, upper: IVec3, f: &mut dyn FnMut(usize)| {
            for z in lower.z..=upper.z {
                for y in lower.y..=upper.y {
                    for x in lower.x..=upper.x {
                        if let Some(index) = cell_index(IVec3::new(x, y, z)) {
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
        self.grid.resize(GRID_CELLS + 1, 0);
        for (index, count) in counts.iter().enumerate() {
            self.grid[index + 1] = self.grid[index] + count;
        }
        let total = self.grid[GRID_CELLS] as usize;
        let mut cursors = self.grid[..GRID_CELLS].to_vec();
        self.grid.resize(GRID_CELLS + 1 + total, 0);
        for (id, &(lower, upper)) in bounds.iter().enumerate() {
            visit(lower, upper, &mut |index| {
                self.grid[GRID_CELLS + 1 + cursors[index] as usize] = id as u32;
                cursors[index] += 1;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            let index = cell_index(cell(cloud.center.truncate())).unwrap();
            let start = GRID_CELLS + 1 + planet.grid[index] as usize;
            let end = GRID_CELLS + 1 + planet.grid[index + 1] as usize;
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
}
