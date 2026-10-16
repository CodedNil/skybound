use spirv_std::glam::{IVec3, Vec3, ivec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn cellular(position: Vec3, period: i32, seed: u32) -> f32 {
    let p = position * period as f32;
    let cell = p.floor().as_ivec3();
    let mut nearest = 3.0f32;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let c = cell + ivec3(x, y, z);
                let feature = c.as_vec3()
                    + vec3(
                        lattice(c, period, seed),
                        lattice(c, period, seed.wrapping_add(31)),
                        lattice(c, period, seed.wrapping_add(67)),
                    );
                nearest = nearest.min(p.distance_squared(feature));
            }
        }
    }
    nearest.sqrt().min(1.0)
}

pub fn rotated(p: Vec3) -> Vec3 {
    vec3(
        p.dot(vec3(0.36, 0.48, -0.8)),
        p.dot(vec3(-0.8, 0.6, 0.0)),
        p.dot(vec3(0.48, 0.64, 0.6)),
    ) * 0.754_877
        + vec3(0.37, 0.19, 0.63)
}

pub fn lattice(cell: IVec3, period: i32, seed: u32) -> f32 {
    let mut value = (cell.x.rem_euclid(period) as u32).wrapping_mul(73_856_093)
        ^ (cell.y.rem_euclid(period) as u32).wrapping_mul(19_349_663)
        ^ (cell.z.rem_euclid(period) as u32).wrapping_mul(83_492_791)
        ^ seed;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value >> 8) as f32 / 16_777_215.0
}

fn periodic_noise(position: Vec3, period: i32, seed: u32) -> f32 {
    let position = position * period as f32;
    let cell = position.floor().as_ivec3();
    let fraction = position - position.floor();
    let weight = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);
    let mut value = 0.0;
    for corner in 0..8 {
        let offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let blend = Vec3::select(offset.cmpeq(IVec3::ZERO), Vec3::ONE - weight, weight);
        let hash = (lattice(cell + offset, period, seed) * 255.0) as u32 & 15;
        let delta = fraction - offset.as_vec3();
        let u = if hash < 8 { delta.x } else { delta.y };
        let v = if hash < 4 {
            delta.y
        } else if hash == 12 || hash == 14 {
            delta.x
        } else {
            delta.z
        };
        let gradient = (if hash & 1 == 0 { u } else { -u }) + (if hash & 2 == 0 { v } else { -v });
        value += gradient * blend.x * blend.y * blend.z;
    }
    value * 0.5 + 0.5
}

pub fn fbm(position: Vec3, period: i32, seed: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    for octave in 0..4 {
        value += periodic_noise(position, period << octave, seed.wrapping_add(octave * 1013))
            * amplitude;
        amplitude *= 0.5;
    }
    ((value / 1.875 - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0)
}
