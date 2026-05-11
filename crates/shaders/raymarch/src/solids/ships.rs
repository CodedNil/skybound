use crate::{
    T_MAX,
    solids::{ShadeResult, estimate_normal},
    utils::{get_sun_position, quat_rotate},
};
use skybound_shared::{NODES_PER_STRING, NUM_STRINGS, ShipUniform, TOTAL_BEAD_NODES, ViewUniform};
use spirv_std::glam::{FloatExt, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

// ── Material IDs ──────────────────────────────────────────────────────────────

pub const MAT_CORE: u32 = 0;
pub const MAT_SHIELD: u32 = 1;
pub const MAT_TANK: u32 = 2;
pub const MAT_STRING: u32 = 3;

pub const fn mat_color(mat: u32) -> Vec3 {
    if mat == MAT_SHIELD {
        vec3(0.3, 0.55, 0.85)
    } else if mat == MAT_TANK {
        vec3(0.25, 0.55, 0.9)
    } else if mat == MAT_STRING {
        vec3(0.18, 0.18, 0.22) // dark cable
    } else {
        vec3(0.72, 0.68, 0.62) // metallic silver-gold core
    }
}

// ── SDF helpers ───────────────────────────────────────────────────────────────

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).saturate();
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

fn quat_conjugate(q: Vec4) -> Vec4 {
    Vec4::new(-q.x, -q.y, -q.z, q.w)
}

/// Capsule between endpoints `a` and `b` with radius `r`.
fn sdf_capsule(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let ab = b - a;
    let len_sq = ab.dot(ab);
    let t = if len_sq > 1e-8 {
        ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length() - r
}

// ── Core star SDF (in core-local space) ───────────────────────────────────────

fn sdf_core_star(p: Vec3) -> f32 {
    const SPHERE_R: f32 = 2.5;
    const SPIKE_LEN: f32 = 5.5;
    const SPIKE_W: f32 = 0.65;

    let base = p.length() - SPHERE_R;

    let spike = |dir: Vec3| -> f32 {
        let t = p.dot(dir).clamp(0.0, SPIKE_LEN);
        let w = SPIKE_W * (1.0 - t / SPIKE_LEN);
        (p - dir * t).length() - w
    };

    let d_spikes = spike(Vec3::X)
        .min(spike(-Vec3::X))
        .min(spike(Vec3::Y))
        .min(spike(-Vec3::Y))
        .min(spike(Vec3::Z))
        .min(spike(-Vec3::Z));

    smin(base, d_spikes, 0.9)
}

// ── Shield SDF (in core-local space) ─────────────────────────────────────────

fn sdf_shield(p: Vec3, phase: f32) -> f32 {
    const MAX_RADIUS: f32 = 18.0;
    const DEPTH: f32 = 9.0;
    const THICKNESS: f32 = 0.35;

    if phase < 0.01 {
        return 1e9;
    }

    let r = vec2(p.x, p.y).length();
    let r_max = MAX_RADIUS * phase;
    let r_norm = (r / r_max).min(1.0);
    let z_surf = DEPTH * r_norm * r_norm;

    let in_bounds = p.z >= 0.0 && r <= r_max;
    if in_bounds {
        (p.z - z_surf).abs() - THICKNESS
    } else {
        let dz = (p.z - z_surf.max(0.0)).abs() - THICKNESS;
        let dr = (r - r_max).max(0.0);
        vec2(dr, dz.max(0.0)).length() - THICKNESS.min(0.0)
    }
}

// ── Tank SDF (world / snap space) ────────────────────────────────────────────

const TANK_RADIUS: f32 = 0.9;
const STRING_RADIUS: f32 = 0.2;

fn sdf_tank(p: Vec3, pos: Vec3) -> f32 {
    (p - pos).length() - TANK_RADIUS
}

// ── Combined ship SDF ─────────────────────────────────────────────────────────

/// Returns `(signed_distance, material_id)`.  `p` is in camera-snap space.
pub fn sdf_ship(p: Vec3, ship: &ShipUniform) -> (f32, u32) {
    let phase = ship.core_position.w;
    if phase < 0.0 {
        return (1e9, MAT_CORE);
    }

    let core_pos = ship.core_position.xyz();
    let core_rot = ship.core_rotation;
    let p_local = quat_rotate(quat_conjugate(core_rot), p - core_pos);

    let d_core = sdf_core_star(p_local);
    let d_shield = sdf_shield(p_local, phase);

    let (mut best_d, mut best_mat) = if d_core < d_shield {
        (d_core, MAT_CORE)
    } else {
        (d_shield, MAT_SHIELD)
    };

    // Tank spheres (skip node 0 = attachment).
    for i in 0..TOTAL_BEAD_NODES {
        if i % NODES_PER_STRING == 0 {
            continue;
        }
        let d = sdf_tank(p, ship.bead_positions[i].xyz());
        if d < best_d {
            best_d = d;
            best_mat = MAT_TANK;
        }
    }

    // String capsules between every consecutive node pair in each string.
    for s in 0..NUM_STRINGS {
        let base = s * NODES_PER_STRING;
        for seg in 0..NODES_PER_STRING - 1 {
            let a = ship.bead_positions[base + seg].xyz();
            let b = ship.bead_positions[base + seg + 1].xyz();
            let d = sdf_capsule(p, a, b, STRING_RADIUS);
            if d < best_d {
                best_d = d;
                best_mat = MAT_STRING;
            }
        }
    }

    (best_d, best_mat)
}

pub fn raymarch_ship(ro: Vec3, rd: Vec3, view: &ViewUniform, ship: &ShipUniform) -> ShadeResult {
    const SHIP_MAX_T: f32 = 4000.0;
    const SHIP_MAX_STEPS: i32 = 128;
    const SHIP_EPSILON: f32 = 0.08;
    const SHIP_MIN_STEP: f32 = 0.05;

    let mut t = 0.0;
    let mut hit = false;
    let mut hit_mat = 0u32;

    for _ in 0..SHIP_MAX_STEPS {
        if t >= SHIP_MAX_T {
            break;
        }
        let p = ro + rd * t;
        let (d, mat) = sdf_ship(p, ship);
        if d < SHIP_EPSILON {
            hit = true;
            hit_mat = mat;
            break;
        }
        t += d.max(SHIP_MIN_STEP);
    }

    if !hit {
        return ShadeResult {
            color_depth: vec4(0.0, 0.0, 0.0, T_MAX),
        };
    }

    let p = ro + rd * t;
    let normal = estimate_normal(p, |p| sdf_ship(p, ship).0);

    let sun_pos = get_sun_position(
        view.planet_center(),
        view.planet_rotation,
        view.ro_relative(),
        view.latitude(),
    );
    let sun_dir = (sun_pos - ro).normalize();

    let dot_nl = normal.dot(sun_dir).max(0.0);
    let ambient = 0.06;
    let base_color = mat_color(hit_mat);
    let lit = base_color * (dot_nl * 0.9 + ambient);

    let view_dir = (ro - p).normalize();
    let fresnel = (1.0 - normal.dot(view_dir).abs()).powf(3.0) * 0.4;
    let color = lit + Vec3::splat(fresnel);

    ShadeResult {
        color_depth: color.extend(t),
    }
}
