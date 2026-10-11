use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::noise::fbm;

pub const WEATHER_SCALE: f32 = 1.0 / 256_000.0;
pub const CLOUD_BOTTOM: f32 = 250.0;
pub const CLOUD_TOP: f32 = 26_000.0;

pub fn weather_map(uv: Vec2) -> Vec4 {
    let p = uv.extend(0.0);
    vec4(
        fbm(p, 4, 0x83),
        fbm(p, 4, 0xc5),
        fbm(p, 2, 0xf1),
        fbm(p, 4, 0x133),
    )
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
    pub condensation: f32,
    pub precipitation: f32,
    pub ice: f32,
    pub low_cloud: f32,
    pub middle_cloud: f32,
    pub high_cloud: f32,
}

impl Climate {
    pub fn weather_uv(position: Vec3, seconds: f32) -> Vec2 {
        (position.xy() - vec2(1.2, -0.7) * seconds) * WEATHER_SCALE
    }

    pub fn sample(position: Vec3, seconds: f32, latitude: f32) -> Self {
        Self::from_weather(
            position,
            seconds,
            latitude,
            Vec3::Y,
            weather_map(Self::weather_uv(position, seconds)),
        )
    }

    pub fn from_weather(
        position: Vec3,
        seconds: f32,
        latitude: f32,
        north: Vec3,
        weather: Vec4,
    ) -> Self {
        let activity = 0.1 + 0.9 * (0.5 + 0.5 * (seconds * TAU / 2400.0).cos());
        let altitude = position.z.max(0.0);
        let polar = latitude.sin().abs();
        let sinking = polar.smoothstep(0.94, 1.0);
        let moisture = (weather.x + (1.0 - polar) * 0.025).smoothstep(0.28, 0.72);
        let convection =
            (weather.y + (1.0 - polar * polar) * 0.08 - sinking * 0.15).smoothstep(0.32, 0.74);
        let humidity =
            (moisture * (0.7 + weather.w * 0.3) * (1.0 - sinking * 0.75)).clamp(0.02, 1.0);
        let storm = moisture.smoothstep(0.55, 0.95) * convection.smoothstep(0.65, 0.95);
        let upper = altitude.smoothstep(3000.0, 14_000.0);
        let vertical = ((1.6 + convection * 3.0) * (1.0 - upper) + storm * 5.0)
            * (0.65 + activity * 0.35)
            - sinking * (4.0 + upper * 12.0);
        let direction = if latitude < 0.0 { -north } else { north };
        let wind = direction * ((4.0 + polar * 45.0) * upper)
            + vec3((weather.z - 0.5) * 5.0, 0.0, vertical);
        let temperature = 284.0 + polar * 25.0 - altitude * 0.0055;
        let ice = 1.0 - temperature.smoothstep(248.0, 273.0);
        Self {
            wind,
            altitude: position.z,
            pressure: (-altitude / 11_000.0).exp(),
            humidity,
            light: activity * (0.08 + polar.powi(3) * 0.92) * (1.0 - moisture * 0.35),
            aur: activity * (-altitude / 1800.0).exp(),
            storm,
            temperature,
            precipitation: (storm + moisture.smoothstep(0.75, 1.0) * 0.2) * (1.0 - ice * 0.7),
            ice,
            condensation: 300.0 + (1.0 - humidity) * 3400.0,
            low_cloud: humidity.smoothstep(0.25, 0.8) * (0.35 + convection * 0.65),
            middle_cloud: moisture.smoothstep(0.5, 0.95) * (1.0 - storm * 0.5),
            high_cloud: weather.z.smoothstep(0.43, 0.7) * (0.25 + moisture * 0.75),
        }
    }

    pub fn layer(self, index: usize) -> Vec3 {
        match index {
            0 => vec3(
                self.condensation,
                1800.0 + self.humidity * 3200.0,
                self.low_cloud,
            ),
            1 => vec3(5500.0, 3200.0, self.middle_cloud),
            2 => vec3(
                11_000.0 + (1.0 - self.humidity) * 5000.0,
                4500.0,
                self.high_cloud,
            ),
            _ => vec3(
                self.condensation,
                19_000.0 + self.humidity * 3000.0,
                self.storm,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn moisture_and_instability_produce_distinct_vertical_cloud_regimes() {
        let at = |weather| Climate::from_weather(Vec3::Z * 2400.0, 0.0, 0.0, Vec3::Y, weather);
        let dry = at(Vec4::ZERO);
        assert!((dry.low_cloud + dry.middle_cloud + dry.high_cloud + dry.storm).abs() < 1e-6);
        let high = at(vec4(0.1, 0.1, 0.9, 0.5));
        assert!(high.high_cloud > 0.0 && high.low_cloud == 0.0);
        let wet = at(vec4(0.95, 0.95, 0.7, 0.95));
        assert!(wet.storm > 0.9 && wet.layer(3).x + wet.layer(3).y > 22_000.0);
        assert!(wet.condensation < high.condensation);
    }
}
