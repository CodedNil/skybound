use super::VolumeSample;
use crate::scene::render::utils::{Textures, hash13};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4Swizzles, vec2, vec3};
use spirv_std::num_traits::{Euclid, Float};

pub const OCEAN_TOP_HEIGHT: f32 = 0.0;
const COLOR_A: Vec3 = vec3(0.6, 0.3, 0.8);
const COLOR_B: Vec3 = vec3(0.4, 0.1, 0.6);

fn flash_emission(pos: Vec2, time: f32) -> Vec3 {
    let cell = (pos / 10_000.0).floor();
    let mut emission = Vec3::ZERO;
    for x in -1..=1 {
        for y in -1..=1 {
            let cell = cell + vec2(x as f32, y as f32);
            let seed = hash13(cell.dot(vec2(127.1, 311.7)));
            let age = (time - seed.x * 20.0).rem_euclid(&20.0);
            let duration = 1.0 + seed.y * 3.0;
            if age >= duration {
                continue;
            }
            let life = age / duration;
            let center = (cell + seed.xy()) * 10_000.0;
            let radius = 700.0 + seed.z * 1800.0;
            let pulse = (life * PI).sin();
            emission += vec3(3.0, 3.0, 5.0) * (-pos.distance(center) / radius).exp() * pulse;
        }
    }
    emission
}

pub fn sample_ocean(pos: Vec3, time: f32, only_density: bool, textures: &Textures) -> VolumeSample {
    let mut sample = VolumeSample::default();
    if pos.z > OCEAN_TOP_HEIGHT {
        return sample;
    }
    let drift = vec2(time * 0.00002, time * 0.00001);
    let height = textures.details((pos.xy() * 0.00002 + drift).extend(0.0)).y * -1200.0;
    let altitude = pos.z - height;
    let mask = altitude.smoothstep(0.0, -500.0);
    if mask <= 0.0 {
        return sample;
    }
    let mut noise = 0.5;
    if mask >= 1.0 {
        sample.density = 1.0;
    } else {
        let warp = textures.weather(pos.xy() * 0.00008 + drift).xy() - 0.5;
        noise = textures
            .details((pos.xy() * 0.0002 + warp).extend(altitude * 0.001))
            .z;
        sample.density = noise.powi(2) * mask + altitude.smoothstep(-50.0, -1000.0);
    }
    if !only_density && sample.density > 0.0 {
        sample.color =
            COLOR_A.lerp(COLOR_B, noise) * (0.1 + 0.9 * (1.0 - altitude.smoothstep(-30.0, -500.0)));
        sample.emission = sample.color * altitude.smoothstep(-20.0, -1000.0) * sample.density
            + flash_emission(pos.xy(), time) * altitude.smoothstep(-100.0, -800.0) * sample.density;
    }
    sample
}
