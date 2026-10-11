use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};

use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};

pub const CLOUD_BOTTOM_HEIGHT: f32 = 1800.0;
pub const CLOUD_TOP_HEIGHT: f32 = 10_200.0;

pub fn sample_clouds(pos: Vec3, view: &ViewUniform, simple: bool, textures: &Textures) -> f32 {
    if pos.z < CLOUD_BOTTOM_HEIGHT || pos.z > CLOUD_TOP_HEIGHT {
        return 0.0;
    }
    let flowing = pos + vec3(2.5, 0.625, 0.0) * view.time();
    let weather = if view.weather.x >= 0.0 {
        view.weather
    } else {
        textures.weather(flowing.xy() * 0.000_018)
    };
    if weather.x <= 0.40 && weather.y <= 0.70 {
        return 0.0;
    }
    let shape = textures.base(flowing * 0.000_16).x;
    let mut density = 0.0;
    let mut profile: f32 = 0.0;
    for layer in 0..2 {
        let (bottom, height, cover) = if layer == 0 {
            (
                1800.0 + weather.z * 1700.0,
                1000.0 + weather.w * 1900.0,
                (weather.y - 0.70) * 3.0,
            )
        } else {
            (
                6200.0 + weather.z * 700.0,
                1700.0 + weather.w * 1600.0,
                (weather.x - 0.40) * 2.0,
            )
        };
        if cover <= 0.0 {
            continue;
        }
        let h = (pos.z - bottom) / height;
        let fade = h.smoothstep(0.0, 0.18) * (1.0 - h).smoothstep(0.0, 0.35);
        density += ((shape - (1.0 - cover.min(0.95))) * 2.2 * fade).max(0.0);
        profile = profile.max(fade);
    }
    if !simple && density > 0.0 {
        let detail = textures.details(flowing * 0.001_3);
        let erosion = detail.x * 0.55 + detail.y * 0.30 + detail.z * 0.15;
        let camera = (view.world_position.xy() + view.camera_offset().truncate())
            .extend(view.ro_relative().z - PLANET_RADIUS);
        let distant = pos.distance(camera).smoothstep(8000.0, 40_000.0);
        density = (density - erosion.lerp(0.5, distant) * (0.12 + 0.12 * (1.0 - profile))).max(0.0);
    }
    density.min(1.0)
}
