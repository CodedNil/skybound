use core::f32::consts::PI;

use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles};
use spirv_std::num_traits::Float;

pub const COLOR_A: Vec3 = Vec3::new(0.6, 0.3, 0.8);
pub const COLOR_B: Vec3 = Vec3::new(0.4, 0.1, 0.6);
pub const FLASH_COLOR: Vec3 = Vec3::new(3.0, 3.0, 5.0); // 0.6, 0.6, 1.0 * 5.0
pub const OCEAN_TOP_HEIGHT: f32 = 0.0;

const TURB_ROTS: [(Vec2, Vec2); 10] = [
    (Vec2::new(0.600, 0.800), Vec2::new(-0.800, 0.600)),
    (Vec2::new(-0.280, 0.960), Vec2::new(-0.960, -0.280)),
    (Vec2::new(-0.843, -0.538), Vec2::new(0.538, -0.843)),
    (Vec2::new(0.422, 0.907), Vec2::new(-0.907, 0.422)),
    (Vec2::new(-0.644, 0.765), Vec2::new(-0.765, -0.644)),
    (Vec2::new(-0.171, -0.985), Vec2::new(0.985, -0.171)),
    (Vec2::new(-0.942, 0.337), Vec2::new(-0.337, -0.942)),
    (Vec2::new(0.773, -0.634), Vec2::new(0.634, 0.773)),
    (Vec2::new(0.196, -0.981), Vec2::new(0.981, 0.196)),
    (Vec2::new(-0.923, -0.384), Vec2::new(0.384, -0.923)),
];

const TURB_AMP: f32 = 0.6;
const TURB_SPEED: f32 = 0.5;
const TURB_FREQ: f32 = 0.4;
const TURB_EXP: f32 = 1.6;

fn compute_turbulence(initial_pos: Vec2, time: f32) -> Vec2 {
    let mut pos = initial_pos;
    let mut freq = TURB_FREQ;
    for i in 0..10 {
        let rot = TURB_ROTS[i];
        let phase = freq * (pos.x * rot.0.y + pos.y * rot.1.y) + TURB_SPEED * time + i as f32;
        pos += TURB_AMP * rot.0 * phase.sin() / freq;
        freq *= TURB_EXP;
    }
    pos
}

const FLASH_GRID: f32 = 10000.0;
const FLASH_FREQUENCY: f32 = 0.05;
const FLASH_DURATION_MIN: f32 = 1.0;
const FLASH_DURATION_MAX: f32 = 4.0;
const FLASH_FLICKER: f32 = 120.0;
const FLASH_SCALE: f32 = 0.002;

const POISSON_OFFSETS: [Vec2; 16] = [
    Vec2::new(0.0000, 0.5000),
    Vec2::new(0.3621, 0.3536),
    Vec2::new(0.4755, 0.1545),
    Vec2::new(0.2939, -0.1545),
    Vec2::new(0.0955, -0.4045),
    Vec2::new(-0.0955, -0.4045),
    Vec2::new(-0.2939, -0.1545),
    Vec2::new(-0.4755, 0.1545),
    Vec2::new(-0.3621, 0.3536),
    Vec2::new(-0.0000, 0.0000),
    Vec2::new(0.1545, 0.4755),
    Vec2::new(0.4045, 0.0955),
    Vec2::new(0.4045, -0.0955),
    Vec2::new(0.1545, -0.4755),
    Vec2::new(-0.1545, -0.4755),
    Vec2::new(-0.4045, -0.0955),
];

const CELL_OFFSETS: [Vec2; 9] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(1.0, 1.0),
    Vec2::new(-1.0, 1.0),
    Vec2::new(1.0, -1.0),
    Vec2::new(-1.0, -1.0),
];

fn hash12(p: f32) -> Vec2 {
    let mut v = (Vec2::splat(p) * Vec2::new(0.1031, 0.1030)).fract();
    v += v.dot(Vec2::new(v.y, v.x) + 33.33);
    ((v.x + v.y) * v).fract()
}

fn hash13(p: f32) -> Vec3 {
    let mut v = (Vec3::splat(p) * Vec3::new(0.1031, 0.1030, 0.1029)).fract();
    v += v.dot(Vec3::new(v.y, v.z, v.x) + 33.33);
    ((v.x + v.y + v.z) * v).fract()
}

fn flash_emission(pos: Vec2, time: f32) -> Vec3 {
    let uv = pos / FLASH_GRID;
    let cell = uv.floor();
    let mut total = Vec3::ZERO;
    let max_effective_dist = 1.0 / (FLASH_SCALE * 0.1);

    for c in 0..9 {
        let cell_pos = cell + CELL_OFFSETS[c];
        let seed = cell_pos.dot(Vec2::new(127.1, 311.7));

        let period = 1.0 / FLASH_FREQUENCY;
        let h_cell = hash12(seed);
        let start_time = h_cell.x * period;
        let tmod = (time - start_time) % period;
        let duration_jitter =
            FLASH_DURATION_MIN + (FLASH_DURATION_MAX - FLASH_DURATION_MIN) * h_cell.y;
        if tmod > duration_jitter || tmod < 0.0 {
            continue;
        }

        let life = (tmod / duration_jitter).clamp(0.0, 1.0);

        for i in 0..4 {
            let h = hash13(seed + i as f32 * 17.0);
            let offset = h.x;
            let rate = 0.8 + 0.4 * h.y;

            let motion = Vec2::new(
                (time * 0.5 + h.x * 10.0).sin(),
                (time * 0.4 + h.y * 9.0).cos(),
            ) * 0.3;
            let gp = (cell_pos + POISSON_OFFSETS[i] + motion) * FLASH_GRID;

            let d = pos.distance(gp);
            if d > max_effective_dist {
                continue;
            }

            let base_scale = 0.5 + 2.5 * h.z;
            let pulse = 0.5 + 0.5 * (life * PI).sin();
            let scaled_distance = d * FLASH_SCALE * base_scale * pulse;

            let fade = (life / 0.1).clamp(0.0, 1.0) * (1.0 - ((life - 0.9) / 0.1).clamp(0.0, 1.0));
            let flicker_time = time * FLASH_FLICKER * rate + offset * 100.0;
            let flicker = if (flicker_time % 1.0) > 0.5 { 1.0 } else { 0.8 };

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
    noise_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
) -> OceanSample {
    let mut sample = OceanSample {
        density: 0.0,
        color: Vec3::ZERO,
        emission: Vec3::ZERO,
    };

    if pos.z > OCEAN_TOP_HEIGHT {
        return sample;
    }

    let height_noise: f32 = noise_texture
        .sample_by_lod(*sampler, (pos.xy() * 0.00002).extend(time * 0.04), 0.0)
        .y
        * -1200.0;
    let altitude = pos.z - height_noise;
    let density_mask = (altitude / -500.0).clamp(0.0, 1.0);
    if density_mask <= 0.0 {
        return sample;
    }

    let mut fbm_value = -1.0;
    if density_mask >= 1.0 {
        sample.density = 1.0;
    } else {
        let turb_pos = compute_turbulence(pos.xy() * 0.001 + Vec2::new(time, 0.0), time);
        fbm_value = noise_texture
            .sample_by_lod(*sampler, (turb_pos * 0.2).extend(altitude * 0.001), 0.0)
            .z
            * 2.0
            - 1.0;
        sample.density = (fbm_value * fbm_value * density_mask
            + (altitude / -1000.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
    }

    if only_density || sample.density <= 0.0 {
        return sample;
    }

    sample.color = COLOR_A.lerp(COLOR_B, fbm_value * 0.5 + 0.5);
    let shadow_factor = 1.0 - (altitude / -500.0).clamp(0.0, 1.0);
    sample.color = (sample.color * 0.1).lerp(sample.color, shadow_factor);

    let emission_amount = (altitude / -1000.0).clamp(0.0, 1.0);
    sample.emission = sample.color * emission_amount * sample.density;
    sample.emission +=
        flash_emission(pos.xy(), time) * (altitude / -800.0).clamp(0.0, 1.0) * sample.density;

    sample
}
