use spirv_std::glam::{FloatExt, Vec3};
use spirv_std::num_traits::Float;

use crate::scene::climate::Climate;
pub use crate::scene::climate::{
    CLOUD_BOTTOM as CLOUD_BOTTOM_HEIGHT, CLOUD_TOP as CLOUD_TOP_HEIGHT,
};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub fn sample_clouds(
    point: Vec3,
    view: &ViewUniform,
    simple: bool,
    footprint: f32,
    textures: &Textures,
) -> f32 {
    let altitude = point.length() - PLANET_RADIUS;
    if altitude < CLOUD_BOTTOM_HEIGHT || altitude > CLOUD_TOP_HEIGHT {
        return 0.0;
    }
    let weather = if view.weather.x >= 0.0 {
        view.weather
    } else {
        textures.weather_on_planet(point, view.time())
    };
    let climate = Climate::on_planet(point, view.time(), weather);
    let mut support: f32 = 0.0;
    for index in 0..4 {
        let layer = climate.layer(index);
        if altitude >= layer.x && altitude <= layer.x + layer.y {
            support = support.max(layer.z);
        }
    }
    if support <= 0.001 {
        return 0.0;
    }
    // Planet-space billows keep their shape when the camera or local frame moves.
    let flowing = point - climate.wind * view.time();
    let cirrus = altitude.smoothstep(28_500.0, 42_000.0) * (1.0 - climate.storm);
    let shape = textures.billows(flowing * 0.000035, footprint * 0.000035);
    let lobes = textures.billows(flowing * 0.00012, footprint * 0.00012).x;
    let surface = point.normalize() * PLANET_RADIUS;
    let towers = textures.billows(surface * 0.000012, footprint * 0.000012);
    let mut density: f32 = 0.0;
    for index in 0..4 {
        let layer = climate.layer(index);
        let h = (altitude - layer.x) / layer.y;
        if h < 0.0 || h > 1.0 || layer.z <= 0.001 {
            continue;
        }
        let fade = h.smoothstep(0.0, 0.08) * (1.0 - h.smoothstep(0.78, 1.0));
        let hull = if index == 3 {
            let anvil = h.smoothstep(0.65, 0.85) * 0.07;
            let taper = (h * core::f32::consts::PI).sin() * 0.025;
            let column = towers.y + (lobes - 0.65) * 0.24 + anvil + taper;
            (column - (0.69 - layer.z * 0.08)).smoothstep(-0.025, 0.075)
        } else {
            let coverage = if index == 2 { shape.z } else { shape.y };
            let mass = shape.x * 0.65 + lobes * 0.35;
            let edge = 0.79 - layer.z * 0.19 - (coverage - 0.5) * 0.24;
            (mass - edge).smoothstep(-0.04, 0.14)
        };
        density = density.max(hull * fade * layer.z * (1.0 - cirrus * 0.65));
    }
    if !simple && density > 0.0 {
        let detail = textures.details_filtered(flowing * 0.000_8, footprint * 0.000_8);
        let erosion = detail.x * 0.5 + detail.y * 0.3 + detail.z * 0.2;
        let billow = textures
            .base_filtered(flowing * 0.00035, footprint * 0.00035)
            .x;
        density = (density + (billow - 0.65) * 0.18 * density.smoothstep(0.0, 0.08)
            - erosion * 0.12 * (1.0 - density).powi(2))
        .max(0.0);
    }
    density
}
