use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use crate::scene::climate::Climate;
pub use crate::scene::climate::{
    CLOUD_BOTTOM as CLOUD_BOTTOM_HEIGHT, CLOUD_TOP as CLOUD_TOP_HEIGHT,
};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub fn sample_clouds(
    pos: Vec3,
    view: &ViewUniform,
    simple: bool,
    footprint: f32,
    textures: &Textures,
) -> f32 {
    if pos.z < CLOUD_BOTTOM_HEIGHT || pos.z > CLOUD_TOP_HEIGHT {
        return 0.0;
    }
    let weather = if view.weather.x >= 0.0 {
        view.weather
    } else {
        textures.weather(Climate::weather_uv(pos, view.time()))
    };
    let axis = view.planet_rotation.mul_vec3(Vec3::Z);
    let latitude = axis.z.clamp(-1.0, 1.0).asin();
    let north = axis.with_z(0.0) / axis.xy().length().max(0.0001);
    let climate = Climate::from_weather(pos, view.time(), latitude, north, weather);
    let mut support: f32 = 0.0;
    for index in 0..4 {
        let layer = climate.layer(index);
        if pos.z >= layer.x && pos.z <= layer.x + layer.y {
            support = support.max(layer.z);
        }
    }
    if support <= 0.001 {
        return 0.0;
    }
    // Advect each altitude with the circulation; stronger upper winds shear rising towers.
    let flowing = pos - climate.wind * view.time();
    let meridian = if north.length_squared() > 0.01 {
        north
    } else {
        Vec3::Y
    };
    let across = vec3(meridian.y, -meridian.x, 0.0);
    let coordinates = vec3(flowing.dot(across), flowing.dot(meridian), flowing.z);
    let cirrus = pos.z.smoothstep(9500.0, 14_000.0) * (1.0 - climate.storm);
    let scale = Vec3::splat(0.0001).lerp(vec3(0.00016, 0.000035, 0.00022), cirrus);
    let shape = textures.base_filtered(coordinates * scale, footprint * scale.max_element());
    let mut density: f32 = 0.0;
    let mut profile: f32 = 0.0;
    for index in 0..4 {
        let layer = climate.layer(index);
        let h = (pos.z - layer.x) / layer.y;
        if h < 0.0 || h > 1.0 || layer.z <= 0.001 {
            continue;
        }
        let fade = h.smoothstep(0.0, 0.12) * (1.0 - h.smoothstep(0.65, 1.0));
        let cover = if index == 3 {
            let tower = textures
                .base_filtered(
                    (flowing.xy() * 0.000_045).extend(0.17),
                    footprint * 0.000_045,
                )
                .y;
            layer.z * tower.smoothstep(0.52 - h.smoothstep(0.7, 0.95) * 0.08, 0.70)
        } else {
            let cells = if index == 2 { shape.z } else { shape.y };
            layer.z * cells.smoothstep(0.43, 0.65)
        };
        let edge = 0.83 - cover * 0.28 + (1.0 - fade) * 0.12;
        let hull = shape.x.smoothstep(edge, edge + 0.16) * cover * fade;
        density = density.max(hull);
        profile = profile.max(fade);
    }
    if !simple && density > 0.0 {
        let detail = textures.details_filtered(flowing * 0.000_8, footprint * 0.000_8);
        let erosion = detail.x * 0.35 + detail.y * 0.40 + detail.z * 0.25;
        let camera = (view.world_position.xy() + view.camera_offset().truncate())
            .extend(view.ro_relative().z - PLANET_RADIUS);
        let distant = pos.distance(camera).smoothstep(12_000.0, 80_000.0);
        density = (density - erosion.lerp(0.5, distant) * (0.07 + 0.05 * (1.0 - profile))).max(0.0);
    }
    density
}
