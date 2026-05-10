use crate::utils::hash21;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAT_SPIKE: u32 = 2;

const SPIKE_GRID: f32 = 5000.0;
const MAX_SPIKE_ALT: f32 = 5000.0;

const SPIKE_COLOR_BASE: Vec3 = vec3(0.16, 0.09, 0.03);
pub const SPIKE_COLOR_MID: Vec3 = vec3(0.36, 0.20, 0.07);
const SPIKE_COLOR_WAX: Vec3 = vec3(0.50, 0.31, 0.12);

fn hash2(p: Vec2) -> Vec2 {
    vec2(hash21(p), hash21(p + Vec2::splat(41.3)))
}

/// Approximate SDF for a single spike: tapered cone with candle-drip bulges.
fn sdf_spike(local: Vec3, radius: f32, height: f32) -> f32 {
    let rxy = vec2(local.x, local.y).length();
    let tz = (local.z / height).saturate();

    let cone_r = radius * (1.0 - tz).powf(0.78);

    // Two drip harmonics, fading toward the tip.
    let zk = local.z * 0.009;
    let drip_t = (1.0 - tz).powf(1.8);
    let drip = (zk.sin().max(0.0).powf(2.0) * 0.13 + (zk * 1.85 + 1.15).sin().max(0.0) * 0.07)
        * radius
        * drip_t;

    let eff_r = (cone_r + drip).max(1.0);
    let d_lat = rxy - eff_r;

    if local.z < 0.0 {
        let d_base = -local.z;
        (d_lat.max(0.0) * d_lat.max(0.0) + d_base * d_base).sqrt()
    } else if local.z > height {
        // Above tip: distance to the small sphere cap at the apex.
        let d_top = local.z - height;
        (rxy * rxy + d_top * d_top).sqrt() - radius * 0.04
    } else {
        d_lat
    }
}

fn cell_sdf(p_xy: Vec2, p_z: f32, cell: Vec2) -> f32 {
    let h = hash2(cell);
    if h.x < 0.40 {
        return 1e9;
    }

    let cell_world = cell * SPIKE_GRID;
    let jitter = (hash2(cell + Vec2::splat(17.1)) - 0.5) * SPIKE_GRID * 0.42;
    let center = cell_world + jitter;

    let h2 = hash2(cell + Vec2::splat(53.7));
    let radius = 90.0 + h2.x * 260.0;
    let height = radius * (3.0 + h2.y * 3.0);

    let local = vec3(p_xy.x - center.x, p_xy.y - center.y, p_z);
    let mut d = sdf_spike(local, radius, height);

    // Satellite spike adjacent to the main one
    if h.y > 0.58 {
        let sh = hash2(cell + Vec2::splat(89.5));
        let s_off = (sh - 0.5) * radius * 2.6;
        let s_r = radius * (0.22 + sh.x * 0.28);
        let s_h = s_r * (2.5 + sh.y * 2.5);
        let s_local = vec3(
            p_xy.x - center.x - s_off.x,
            p_xy.y - center.y - s_off.y,
            p_z,
        );
        d = d.min(sdf_spike(s_local, s_r, s_h));
    }

    d
}

pub fn sdf_aur_spikes(p: Vec3) -> f32 {
    // Early out above the spike zone
    if p.z > MAX_SPIKE_ALT + 300.0 {
        return p.z - MAX_SPIKE_ALT;
    }

    let cell = (vec2(p.x, p.y) / SPIKE_GRID + Vec2::splat(0.5)).floor();
    let p_xy = vec2(p.x, p.y);

    let mut d = 1e9_f32;
    for ci in -1_i32..=1 {
        for cj in -1_i32..=1 {
            let nb = cell + vec2(ci as f32, cj as f32);
            let cd = cell_sdf(p_xy, p.z, nb);
            if cd < d {
                d = cd;
            }
        }
    }
    d
}

pub fn spike_albedo(p: Vec3) -> Vec3 {
    let tz = (p.z / 2500.0).saturate();
    let drip_hi = (p.z * 0.009).sin().max(0.0).powf(2.0) * 0.35;
    let h_var = hash21(vec2(p.x * 0.0018, p.y * 0.0018)) * 0.12;
    SPIKE_COLOR_BASE
        .lerp(SPIKE_COLOR_MID, (tz * 1.6).min(1.0))
        .lerp(SPIKE_COLOR_WAX, drip_hi + h_var)
}
