use core::f32::consts::{FRAC_PI_2, TAU};
use spirv_std::glam::{FloatExt, IVec3, Vec2, Vec3, Vec3Swizzles, Vec4, ivec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::math::cube;
use super::{PLANET_RADIUS, noise::lattice};

pub mod clouds;
#[cfg(all(not(target_arch = "spirv"), feature = "desktop"))]
pub mod simulation;

pub const WEATHER_SCALE: f32 = 1.0 / 256_000.0;
pub const CLOUD_BOTTOM: f32 = 2700.0;
pub const CLOUD_TOP: f32 = 78_000.0;

pub const WEATHER_SIZE: i32 = 64;

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
}
