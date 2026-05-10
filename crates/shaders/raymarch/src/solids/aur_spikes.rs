use crate::utils::hash21;
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAT_SPIKE: u32 = 2;

const SPIKE_GRID: f32 = 5000.0;
/// Spikes rise from within the `aur_ocean` fog
pub const SPIKE_BASE_ALT: f32 = -3000.0;
/// Tips land between these altitudes
const SPIKE_MIN_TIP: f32 = -800.0;
const SPIKE_MAX_TIP: f32 = 100.0;
/// Plane distance above the tallest possible tip is always conservative
const SPIKE_CEILING: f32 = SPIKE_MAX_TIP;

const SPIKE_COLOR_BASE: Vec3 = vec3(0.16, 0.09, 0.03);
pub const SPIKE_COLOR_MID: Vec3 = vec3(0.36, 0.20, 0.07);
const SPIKE_COLOR_WAX: Vec3 = vec3(0.50, 0.31, 0.12);

fn hash2(p: Vec2) -> Vec2 {
    vec2(hash21(p), hash21(p + Vec2::splat(41.3)))
}

/// SDF for a single spike with candle-drip shaping.
fn sdf_spike(local: Vec3, radius: f32, height: f32, wax_seed: Vec2) -> f32 {
    let tz = (local.z / height).saturate();
    let inv_tz = 1.0 - tz;

    // ---- XY wobble: organic candle-drip offset that fades toward the tip ----
    let wobble_amp = radius * 0.17 * inv_tz.powf(0.9);
    let wobble_x = (local.z * 0.0055 + wax_seed.x * TAU).sin() * wobble_amp;
    let wobble_y = (local.z * 0.0073 + wax_seed.y * TAU).cos() * wobble_amp;
    let rxy = vec2(local.x - wobble_x, local.y - wobble_y).length();

    // ---- Tapered cone: variable power for shape variety ----
    let taper_pow = 0.55 + wax_seed.x * wax_seed.y * 0.40; // 0.55–0.95
    let cone_r = radius * inv_tz.powf(taper_pow);

    // ---- Drip harmonics: wax-like bulges along the shaft ----
    let zk1 = local.z * 0.013;
    let zk2 = local.z * 0.021 + 1.7;
    let zk3 = local.z * 0.031 + 3.1;
    // drip envelope fades out near tip
    let drip_env = inv_tz.powf(1.6);
    let drip = {
        let d1 = zk1.sin().max(0.0).powf(1.7) * 0.14;
        let d2 = zk2.sin().max(0.0).powf(2.5) * 0.10;
        let d3 = zk3.sin().max(0.0) * 0.06;
        (d1 + d2 + d3) * radius * drip_env
    };

    // ---- Bulbous tip: slight swelling at apex like a candle-flame base ----
    let tip_bulb = inv_tz.powf(16.0) * radius * 0.07;

    let eff_r = cone_r + drip + tip_bulb;
    let d_lat = rxy - eff_r;

    // ---- Handle regions ----
    if local.z < 0.0 {
        // Below base: smooth blend into ground plane
        let d_base = -local.z;
        let d_lat_clamped = d_lat.max(0.0);
        (d_lat_clamped * d_lat_clamped + d_base * d_base).sqrt() - 0.5
    } else if local.z > height {
        // Above tip: distance to sharp point
        rxy.max(local.z - height)
    } else {
        d_lat
    }
}

fn cell_sdf(p_xy: Vec2, p_z: f32, cell: Vec2) -> f32 {
    let h = hash2(cell);
    if h.x < 0.38 {
        return 1e9;
    }

    let cell_world = cell * SPIKE_GRID;
    let jitter = (hash2(cell + Vec2::splat(17.1)) - 0.5) * SPIKE_GRID * 0.40;
    let center = cell_world + jitter;

    let h2 = hash2(cell + Vec2::splat(53.7));
    // Radius drives the girth of the spike
    let radius = 100.0 + h2.x * 340.0; // 100–440
    // Height directly spans from SPIKE_BASE_ALT to tip in [SPIKE_MIN_TIP, SPIKE_MAX_TIP]
    let tip_alt = SPIKE_MIN_TIP.lerp(SPIKE_MAX_TIP, h2.y);
    let height = tip_alt - SPIKE_BASE_ALT; // 2200–3100

    // local z = 0 at base, = height at tip
    let local = vec3(p_xy.x - center.x, p_xy.y - center.y, p_z - SPIKE_BASE_ALT);
    let wax_seed = hash2(cell + Vec2::splat(71.3));
    let mut d = sdf_spike(local, radius, height, wax_seed);

    // Satellite spike adjacent to the main one
    if h.y > 0.52 {
        let sh = hash2(cell + Vec2::splat(89.5));
        let s_off = (sh - 0.5) * radius * 2.8;
        let s_r = radius * (0.18 + sh.x * 0.32); // 0.18–0.50 of main radius
        let s_tip = SPIKE_MIN_TIP.lerp(SPIKE_MAX_TIP, sh.y);
        let s_h = s_tip - SPIKE_BASE_ALT;
        let s_local = vec3(
            p_xy.x - center.x - s_off.x,
            p_xy.y - center.y - s_off.y,
            p_z - SPIKE_BASE_ALT,
        );
        let s_seed = hash2(cell + Vec2::splat(103.9));
        d = d.min(sdf_spike(s_local, s_r, s_h, s_seed));
    }

    d
}

pub fn sdf_aur_spikes(p: Vec3) -> f32 {
    // Early out above the spike zone.
    // p.z - SPIKE_CEILING is always ≤ true distance to nearest spike tip.
    if p.z > SPIKE_CEILING + 5000.0 {
        return p.z - SPIKE_CEILING;
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
    let alt = p.z - SPIKE_BASE_ALT;
    let tz = (alt / 2500.0).saturate();
    let drip_hi = (alt * 0.011).sin().max(0.0).powf(2.0) * 0.35;
    let h_var = hash21(vec2(p.x * 0.0018, p.y * 0.0018)) * 0.12;
    SPIKE_COLOR_BASE
        .lerp(SPIKE_COLOR_MID, (tz * 1.6).min(1.0))
        .lerp(SPIKE_COLOR_WAX, drip_hi + h_var)
}
