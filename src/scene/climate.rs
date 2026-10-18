//! Compact cloud records and the single planet-wide spatial index.
use core::f32::consts::{FRAC_PI_2, TAU};

use spirv_std::glam::{FloatExt, IVec3, Vec2, Vec3, Vec3Swizzles, Vec4, ivec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::PLANET_RADIUS;
use crate::scene::math::{cube, intersect_sphere, lattice};

pub const WEATHER_SCALE: f32 = 1.0 / 256_000.0;

pub const CLOUD_BOTTOM: f32 = 2700.0;

pub const CLOUD_TOP: f32 = 78_000.0;

pub const WEATHER_SIZE: i32 = 64;

pub const CLOUD_CELL: f32 = 64_000.0;

pub const CLOUD_GRID_WIDTH: u32 = 52;

pub const CLOUD_GRID_CELLS: usize =
    (CLOUD_GRID_WIDTH * CLOUD_GRID_WIDTH * CLOUD_GRID_WIDTH) as usize;

pub const CLOUD_GRID_ORIGIN: f32 = -(CLOUD_GRID_WIDTH as f32 * CLOUD_CELL * 0.5);

#[derive(Clone, Copy, Debug)]
pub struct Climate {
    pub wind: Vec3,
    pub altitude: f32,
    pub pressure: f32,
    pub humidity: f32,
    pub light: f32,
    pub aur: f32,
    pub storm: f32,
    pub temperature: f32,
    pub precipitation: f32,
    pub ice: f32,
    pub low_cloud: f32,
    pub middle_cloud: f32,
    pub high_cloud: f32,
}

impl Climate {
    pub fn at(position: Vec3, seconds: f32, weather: Vec4) -> Self {
        Self::from_weather(
            position,
            seconds,
            FRAC_PI_2 - position.y / PLANET_RADIUS,
            -Vec3::Y,
            weather,
        )
    }

    pub fn on_planet(point: Vec3, seconds: f32, weather: Vec4) -> Self {
        let radius = point.length();
        let up = point / radius;
        let mut climate = Self::from_polar(
            Vec3::Z * (radius - PLANET_RADIUS),
            seconds,
            up.z,
            Vec3::Y,
            weather,
        );
        let east = vec3(-up.y, up.x, 0.0);
        let north = Vec3::Z - up * up.z;
        climate.wind = east * climate.wind.x + north * climate.wind.y + up * climate.wind.z;
        climate
    }

    pub fn weather_coordinates(point: Vec3, seconds: f32) -> Vec3 {
        (point.normalize() * PLANET_RADIUS - vec3(1.2, -0.7, 0.9) * seconds) * WEATHER_SCALE
    }

    pub fn weather_on_planet(point: Vec3, seconds: f32) -> Vec4 {
        let p = Self::weather_coordinates(point, seconds);
        weather_at(p)
    }

    pub fn weather_uv(position: Vec3, seconds: f32) -> Vec2 {
        (position.xy() - vec2(1.2, -0.7) * seconds) * WEATHER_SCALE
    }

    pub fn sample(position: Vec3, seconds: f32, latitude: f32) -> Self {
        Self::from_weather(
            position,
            seconds,
            latitude,
            Vec3::Y,
            weather_at(Self::weather_uv(position, seconds).extend(0.0)),
        )
    }

    pub fn from_weather(
        position: Vec3,
        seconds: f32,
        latitude: f32,
        north: Vec3,
        weather: Vec4,
    ) -> Self {
        Self::from_polar(position, seconds, latitude.sin(), north, weather)
    }

    fn from_polar(
        position: Vec3,
        seconds: f32,
        latitude_sine: f32,
        north: Vec3,
        weather: Vec4,
    ) -> Self {
        let activity = 0.1 + 0.9 * (0.5 + 0.5 * (seconds * TAU / 2400.0).cos());
        let altitude = position.z.max(0.0);
        let polar = latitude_sine.abs();
        let sinking = polar.smoothstep(0.94, 1.0);
        let moisture = (weather.x + (1.0 - polar) * 0.025).smoothstep(0.28, 0.72);
        let convection =
            (weather.y + (1.0 - polar * polar) * 0.08 - sinking * 0.15).smoothstep(0.32, 0.74);
        let humidity =
            (moisture * (0.7 + weather.w * 0.3) * (1.0 - sinking * 0.75)).clamp(0.02, 1.0);
        let storm = moisture.smoothstep(0.55, 0.95) * convection.smoothstep(0.65, 0.95);
        let upper = altitude.smoothstep(9000.0, 42_000.0);
        let basal_lift = (8.0 + convection * 4.0) * (-altitude / 1800.0).exp();
        let vertical = (basal_lift + (1.6 + convection * 3.0) * (1.0 - upper) + storm * 5.0)
            * (0.65 + activity * 0.35)
            - sinking * (4.0 + upper * 12.0);
        let direction = north * latitude_sine;
        let wind = direction * ((4.0 + polar * 45.0) * upper)
            + vec3((weather.z - 0.5) * 5.0, 0.0, vertical);
        let temperature = 284.0 + polar * 25.0 - altitude * 0.0055;
        let ice = 1.0 - temperature.smoothstep(248.0, 273.0);
        Self {
            wind,
            altitude: position.z,
            pressure: (-altitude / 11_000.0).exp(),
            humidity,
            light: activity * (0.08 + cube(polar) * 0.92) * (1.0 - moisture * 0.35),
            aur: activity * (-altitude / 1800.0).exp(),
            storm,
            temperature,
            precipitation: (storm + moisture.smoothstep(0.75, 1.0) * 0.2) * (1.0 - ice * 0.7),
            ice,
            low_cloud: humidity.smoothstep(0.25, 0.8) * (0.35 + convection * 0.65),
            middle_cloud: moisture.smoothstep(0.5, 0.95) * (1.0 - storm * 0.5),
            high_cloud: weather.z.smoothstep(0.43, 0.7) * (0.25 + moisture * 0.75),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cloud {
    pub center: Vec4,     // planet position, east radius
    pub up: Vec4,         // radial axis, vertical radius
    pub east: Vec4,       // horizontal axis, north radius
    pub appearance: Vec4, // extinction, erosion, noise scale, convection
    pub motion: Vec4,     // velocity, stable seed
}

impl Cloud {
    pub fn ray_interval(self, origin: Vec3, direction: Vec3, elapsed: f32) -> Vec2 {
        let local = self.coordinates(origin, elapsed);
        let north = self.up.truncate().cross(self.east.truncate());
        let ray = Vec3::new(
            direction.dot(self.east.truncate()) / self.center.w,
            direction.dot(north) / self.east.w,
            direction.dot(self.up.truncate()) / self.up.w,
        );
        intersect_sphere(local, ray, 1.0)
    }
    pub fn coordinates(self, point: Vec3, elapsed: f32) -> Vec3 {
        let p = point - self.center.truncate() - self.motion.truncate() * elapsed;
        Vec3::new(
            p.dot(self.east.truncate()) / self.center.w,
            p.dot(self.up.truncate().cross(self.east.truncate())) / self.east.w,
            p.dot(self.up.truncate()) / self.up.w,
        )
    }
    pub fn extent(self) -> Vec3 {
        self.east.truncate().abs() * self.center.w
            + self.up.truncate().cross(self.east.truncate()).abs() * self.east.w
            + self.up.truncate().abs() * self.up.w
    }
}

pub struct RayEvent {
    pub ids: [u32; 8],
    pub count: u32,
    pub end: f32,
    pub step: f32,
    pub overflow: bool,
}

pub fn weather_texel(cell: IVec3) -> Vec4 {
    let channel = |seed| (lattice(cell, WEATHER_SIZE, seed) * 255.0).round() / 255.0;
    vec4(
        channel(0x81),
        channel(0x81 + 1013),
        channel(0x81 + 2026),
        channel(0x81 + 3039),
    )
}

pub fn weather_at(point: Vec3) -> Vec4 {
    let cell = point.floor().as_ivec3();
    let fraction = point - point.floor();
    let mut weather = Vec4::ZERO;
    for index in 0..8 {
        let offset = ivec3(index & 1, (index >> 1) & 1, (index >> 2) & 1);
        let weight = Vec3::select(offset.cmpeq(IVec3::ZERO), Vec3::ONE - fraction, fraction);
        weather += weather_texel(cell + offset) * (weight.x * weight.y * weight.z);
    }
    weather
}

pub fn cloud_cell(point: Vec3) -> IVec3 {
    ((point - Vec3::splat(CLOUD_GRID_ORIGIN)) / CLOUD_CELL)
        .floor()
        .as_ivec3()
}

pub fn cloud_cell_index(cell: IVec3) -> Option<usize> {
    if cell.cmplt(IVec3::ZERO).any() || cell.cmpge(IVec3::splat(52)).any() {
        return None;
    }
    Some(
        ((cell.z as u32 * CLOUD_GRID_WIDTH + cell.y as u32) * CLOUD_GRID_WIDTH + cell.x as u32)
            as usize,
    )
}

/// Front-to-back ellipsoid events within one spatial cell. Overlaps contribute
/// together; dense cells fall back to the complete list instead of dropping clouds.
pub fn ray_event(
    origin: Vec3,
    direction: Vec3,
    start: f32,
    limit: f32,
    elapsed: f32,
    records: &[Cloud],
    grid: &[u32],
) -> RayEvent {
    let position = origin + direction * (start + 0.5);
    let coordinate = cloud_cell(position);
    let lower = coordinate.as_vec3() * CLOUD_CELL + Vec3::splat(CLOUD_GRID_ORIGIN);
    let face = Vec3::select(
        direction.cmpgt(Vec3::ZERO),
        lower + Vec3::splat(CLOUD_CELL),
        lower,
    );
    let distances = Vec3::select(
        direction.abs().cmpgt(Vec3::splat(0.000_000_1)),
        (face - position) / direction,
        Vec3::splat(f32::MAX),
    );
    let exit = distances.min_element().max(0.5) + start + 0.5;
    let mut event = RayEvent {
        ids: [0; 8],
        count: 0,
        end: exit.min(limit),
        step: 4000.0,
        overflow: false,
    };
    let Some(index) = cloud_cell_index(coordinate) else {
        return event;
    };
    for offset in grid[index]..grid[index + 1] {
        let id = grid[CLOUD_GRID_CELLS + 1 + offset as usize];
        let cloud = records[id as usize];
        if cloud.appearance.x <= 0.000_000_1 {
            continue;
        }
        let interval = cloud.ray_interval(origin, direction, elapsed);
        if interval.x >= interval.y || interval.y <= start + 0.25 {
            continue;
        }
        if interval.x > start + 0.5 {
            event.end = event.end.min(interval.x);
        } else {
            event.end = event.end.min(interval.y);
            if event.count < 8 {
                event.ids[event.count as usize] = id;
                event.count += 1;
            } else {
                event.overflow = true;
            }
            let local_direction = Vec3::new(
                direction.dot(cloud.east.truncate()) / cloud.center.w,
                direction.dot(cloud.up.truncate().cross(cloud.east.truncate())) / cloud.east.w,
                direction.dot(cloud.up.truncate()) / cloud.up.w,
            );
            event.step = event.step.min(0.08 / local_direction.length());
        }
    }
    event.end = event.end.max(start + 0.5).min(limit);
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basal_updraft_is_strongest_close_to_the_aur() {
        let wet = Vec4::splat(0.95);
        let surface = Climate::from_weather(Vec3::ZERO, 0.0, 0.0, Vec3::Y, wet);
        let upper = Climate::from_weather(Vec3::Z * 6000.0, 0.0, 0.0, Vec3::Y, wet);
        assert!(surface.wind.z > upper.wind.z + 8.0);
        let north = Climate::at(vec3(0.0, 800_000.0, 15_000.0), 0.0, wet);
        let south = Climate::at(
            vec3(0.0, PLANET_RADIUS * (FRAC_PI_2 * 2.0) - 800_000.0, 15_000.0),
            0.0,
            wet,
        );
        assert!(north.wind.y < 0.0 && south.wind.y > 0.0);
    }

    #[test]
    fn circulation_rises_at_equator_diverges_aloft_and_sinks_at_poles() {
        let weather = Vec4::splat(0.5);
        let at =
            |z, latitude| Climate::from_weather(vec3(0.0, 0.0, z), 0.0, latitude, Vec3::Y, weather);
        assert!(at(1000.0, 0.0).wind.z > 0.0);
        assert!(at(15_000.0, 0.8).wind.y > 0.0);
        assert!(at(15_000.0, -0.8).wind.y < 0.0);
        assert!(at(15_000.0, 1.57).wind.z < 0.0);
        assert!(at(15_000.0, 0.0).temperature < at(15_000.0, 1.57).temperature);
        let left = vec3(-2.0, 0.0, PLANET_RADIUS + 15_000.0);
        let right = vec3(2.0, 0.0, PLANET_RADIUS + 15_000.0);
        let a = Climate::weather_on_planet(left, 100.0);
        let b = Climate::weather_on_planet(right, 100.0);
        assert!(
            (a - b).abs().max_element() < 0.001,
            "weather seam at the pole"
        );
        let a = Climate::on_planet(left, 100.0, a);
        let b = Climate::on_planet(right, 100.0, b);
        assert!(
            a.wind.distance(b.wind) < 0.01,
            "wind discontinuity at the pole"
        );
    }

    #[test]
    fn moisture_and_instability_produce_distinct_vertical_cloud_regimes() {
        let at = |weather| Climate::from_weather(Vec3::Z * 2400.0, 0.0, 0.0, Vec3::Y, weather);
        let dry = at(Vec4::ZERO);
        assert!((dry.low_cloud + dry.middle_cloud + dry.high_cloud + dry.storm).abs() < 1e-6);
        let high = at(vec4(0.1, 0.1, 0.9, 0.5));
        assert!(high.high_cloud > 0.0 && high.low_cloud == 0.0);
        let wet = at(vec4(0.95, 0.95, 0.7, 0.95));
        assert!(wet.storm > 0.9);
    }

    #[test]
    fn ellipsoid_intervals_preserve_world_distance_and_motion() {
        let cloud = Cloud {
            center: Vec3::ZERO.extend(100.0),
            up: Vec3::Z.extend(20.0),
            east: Vec3::X.extend(60.0),
            motion: Vec3::X.extend(0.0),
            ..Default::default()
        };
        let bounds = cloud.ray_interval(Vec3::new(-200.0, 0.0, 0.0), Vec3::X, 10.0);
        assert!((bounds.x - 110.0).abs() < 0.001);
        assert!((bounds.y - 310.0).abs() < 0.001);
        let inside = cloud.ray_interval(Vec3::ZERO, Vec3::Z, 0.0);
        assert!((inside.x + 20.0).abs() < 0.001 && (inside.y - 20.0).abs() < 0.001);
        let miss = cloud.ray_interval(Vec3::new(-200.0, 0.0, 21.0), Vec3::X, 0.0);
        assert!(miss.x > miss.y);
    }

    #[test]
    fn traversal_orders_entries_exits_and_keeps_overlapping_clouds() {
        let origin = Vec3::new(100.0, 1000.0, 1000.0);
        let cloud = |x| Cloud {
            center: Vec3::new(x, 1000.0, 1000.0).extend(100.0),
            up: Vec3::Z.extend(100.0),
            east: Vec3::X.extend(100.0),
            appearance: Vec4::splat(0.001),
            ..Default::default()
        };
        let records = [cloud(1000.0), cloud(1150.0)];
        let index = cloud_cell_index(cloud_cell(origin)).unwrap();
        let mut grid = vec![0; CLOUD_GRID_CELLS + 1];
        grid[index + 1..].fill(2);
        grid.extend([0, 1]);
        let first = ray_event(origin, Vec3::X, 0.0, 100_000.0, 0.0, &records, &grid);
        assert_eq!(first.count, 0);
        assert!((first.end - 800.0).abs() < 0.01);
        let entry = ray_event(origin, Vec3::X, first.end, 100_000.0, 0.0, &records, &grid);
        assert_eq!(entry.count, 1);
        assert!((entry.end - 950.0).abs() < 0.01);
        let overlap = ray_event(origin, Vec3::X, entry.end, 100_000.0, 0.0, &records, &grid);
        assert_eq!(overlap.count, 2);
        assert!((overlap.end - 1000.0).abs() < 0.01);
        let empty = ray_event(origin, Vec3::X, 2000.0, 100_000.0, 0.0, &records, &grid);
        assert!(
            empty.end > 60_000.0,
            "parallel cell axes must not restrict traversal"
        );
        let dense = [cloud(1000.0); 9];
        grid[index + 1..=CLOUD_GRID_CELLS].fill(9);
        grid.truncate(CLOUD_GRID_CELLS + 1);
        grid.extend(0..9);
        let overlap = ray_event(origin, Vec3::X, 900.0, 100_000.0, 0.0, &dense, &grid);
        assert!(overlap.overflow);
        assert_eq!(overlap.count, 8);
    }
}
