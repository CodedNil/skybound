use crate::utils::{hash12, hash13, mod1, smoothstep};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Mat2, Vec2, Vec3, Vec4, vec2, vec3};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

// Constants
const COLOR_A: Vec3 = vec3(0.6, 0.3, 0.8);
const COLOR_B: Vec3 = vec3(0.4, 0.1, 0.6);
const FLASH_COLOR: Vec3 = vec3(3.0, 3.0, 5.0); // (0.6, 0.6, 1.0) * 5.0
pub const OCEAN_TOP_HEIGHT: f32 = 0.0;

// Turbulence Configuration
const TURB_AMP: f32 = 0.6;
const TURB_SPEED: f32 = 0.5;
const TURB_FREQ: f32 = 0.4;
const TURB_EXP: f32 = 1.6;

const TURB_ROTS: [Mat2; 10] = [
    Mat2::from_cols_array(&[0.600, 0.800, -0.800, 0.600]),
    Mat2::from_cols_array(&[-0.280, 0.960, -0.960, -0.280]),
    Mat2::from_cols_array(&[-0.843, -0.538, 0.538, -0.843]),
    Mat2::from_cols_array(&[0.422, 0.907, -0.907, 0.422]),
    Mat2::from_cols_array(&[-0.644, 0.765, -0.765, -0.644]),
    Mat2::from_cols_array(&[-0.171, -0.985, 0.985, -0.171]),
    Mat2::from_cols_array(&[-0.942, 0.337, -0.337, -0.942]),
    Mat2::from_cols_array(&[0.773, -0.634, 0.634, 0.773]),
    Mat2::from_cols_array(&[0.196, -0.981, 0.981, 0.196]),
    Mat2::from_cols_array(&[-0.923, -0.384, 0.384, -0.923]),
];

// Lightning Configuration
const FLASH_GRID: f32 = 10000.0;
const FLASH_FREQUENCY: f32 = 0.05;
const FLASH_DURATION_MIN: f32 = 1.0;
const FLASH_DURATION_MAX: f32 = 4.0;
const FLASH_FLICKER: f32 = 120.0;
const FLASH_SCALE: f32 = 0.002;

const POISSON_OFFSETS: [Vec2; 16] = [
    vec2(0.0000, 0.5000),
    vec2(0.3621, 0.3536),
    vec2(0.4755, 0.1545),
    vec2(0.2939, -0.1545),
    vec2(0.0955, -0.4045),
    vec2(-0.0955, -0.4045),
    vec2(-0.2939, -0.1545),
    vec2(-0.4755, 0.1545),
    vec2(-0.3621, 0.3536),
    vec2(-0.0000, 0.0000),
    vec2(0.1545, 0.4755),
    vec2(0.4045, 0.0955),
    vec2(0.4045, -0.0955),
    vec2(0.1545, -0.4755),
    vec2(-0.1545, -0.4755),
    vec2(-0.4045, -0.0955),
];

const CELL_OFFSETS: [Vec2; 9] = [
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(-1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, -1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, -1.0),
];

fn compute_turbulence(initial_pos: Vec2, time: f32) -> Vec2 {
    let mut pos = initial_pos;
    let mut freq = TURB_FREQ;
    for i in 0..10 {
        let rot = TURB_ROTS[i];
        let phase = freq * (rot.transpose() * pos).y + TURB_SPEED * time + i as f32;
        pos += TURB_AMP * rot.col(0) * phase.sin() / freq;
        freq *= TURB_EXP;
    }
    pos
}

fn flash_emission(pos: Vec2, time: f32) -> Vec3 {
    let uv = pos / FLASH_GRID;
    let cell = uv.floor();
    let mut total = Vec3::ZERO;
    let max_effective_dist = 1.0 / (FLASH_SCALE * 0.1);

    for c in 0..9 {
        let cell_pos = cell + CELL_OFFSETS[c];
        let seed = cell_pos.dot(vec2(127.1, 311.7));

        let period = 1.0 / FLASH_FREQUENCY;
        let h_cell = hash12(seed);
        let start_time = h_cell.x * period;
        let tmod = mod1(time - start_time, period);
        let duration_jitter = FLASH_DURATION_MIN.lerp(FLASH_DURATION_MAX, h_cell.y);

        if tmod > duration_jitter {
            continue;
        }

        let life = (tmod / duration_jitter).clamp(0.0, 1.0);

        for i in 0..4 {
            // POISSON_SAMPLES
            let h = hash13(seed + i as f32 * 17.0);
            let offset = h.x;
            let rate = 0.8.lerp(1.2, h.y);

            let motion = vec2(
                (time * 0.5 + h.x * 10.0).sin(),
                (time * 0.4 + h.y * 9.0).cos(),
            ) * 0.3;

            let gp = (cell_pos + POISSON_OFFSETS[i] + motion) * FLASH_GRID;
            let d = pos.distance(gp);

            if d > max_effective_dist {
                continue;
            }

            let base_scale = 0.5.lerp(3.0, h.z);
            let pulse = 0.5 + 0.5 * (life * PI).sin();
            let scaled_distance = d * FLASH_SCALE * base_scale * pulse;

            let fade = smoothstep(0.0, 0.1, life) * (1.0 - smoothstep(0.9, 1.0, life));
            let flicker_time = time * FLASH_FLICKER * rate + offset * 100.0;
            let flicker = if (flicker_time.fract()) >= 0.5 {
                1.0
            } else {
                0.8
            };

            total += (-scaled_distance).exp() * FLASH_COLOR * fade * flicker;
        }
    }
    total
}

pub struct OceanSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

pub fn sample_ocean(
    pos: Vec3,
    time: f32,
    only_density: bool,
    details_texture: Image!(3D, type=f32, sampled=true),
    sampler: Sampler,
) -> OceanSample {
    let mut ocean_sample = OceanSample {
        density: 0.0,
        color: Vec3::ZERO,
        emission: Vec3::ZERO,
    };

    if pos.z > OCEAN_TOP_HEIGHT {
        return ocean_sample;
    }

    let noise_sample: Vec4 = details_texture.sample_by_lod(
        sampler,
        vec3(pos.x * 0.00002, pos.y * 0.00002, time * 0.04),
        0.0,
    );
    let height_noise = noise_sample.y * -1200.0;
    let altitude = pos.z - height_noise;
    let density_mask = smoothstep(0.0, -500.0, altitude);

    if density_mask <= 0.0 {
        return ocean_sample;
    }

    let mut fbm_value = -1.0;
    if density_mask >= 1.0 {
        ocean_sample.density = 1.0;
    } else {
        let turb_pos = compute_turbulence(vec2(pos.x, pos.y) * 0.001 + vec2(time, 0.0), time);
        let b_noise: Vec4 = details_texture.sample_by_lod(
            sampler,
            vec3(turb_pos.x * 0.2, turb_pos.y * 0.2, altitude * 0.001),
            0.0,
        );
        fbm_value = b_noise.z * 2.0 - 1.0;
        ocean_sample.density =
            fbm_value.powf(2.0) * density_mask + smoothstep(-50.0, -1000.0, altitude);
    }

    if only_density || ocean_sample.density <= 0.0 {
        return ocean_sample;
    }

    ocean_sample.color = Vec3::lerp(COLOR_A, COLOR_B, fbm_value);
    let shadow_factor = 1.0 - smoothstep(-30.0, -500.0, altitude);
    ocean_sample.color = Vec3::lerp(ocean_sample.color * 0.1, ocean_sample.color, shadow_factor);

    let emission_amount = smoothstep(-20.0, -1000.0, altitude);
    ocean_sample.emission = ocean_sample.color * emission_amount * ocean_sample.density;
    ocean_sample.emission += flash_emission(pos.truncate(), time)
        * smoothstep(-100.0, -800.0, altitude)
        * ocean_sample.density;

    ocean_sample
}
