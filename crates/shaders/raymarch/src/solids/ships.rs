use crate::{
    T_MAX,
    solids::{ShadeResult, estimate_normal},
    utils::{get_sun_position, quat_rotate},
};
use skybound_shared::{ShipUniform, ViewUniform};
use spirv_std::glam::{FloatExt, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAT_CORE: u32 = 0;
pub const MAT_SHIELD: u32 = 1;

pub const fn mat_color(mat: u32) -> Vec3 {
    if mat == MAT_SHIELD {
        vec3(0.3, 0.55, 0.85)
    } else {
        vec3(0.72, 0.68, 0.62)
    }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).saturate();
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

fn quat_conjugate(q: Vec4) -> Vec4 {
    Vec4::new(-q.x, -q.y, -q.z, q.w)
}

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

fn sdf_shield(p: Vec3) -> f32 {
    const MAX_RADIUS: f32 = 18.0;
    const DEPTH: f32 = 9.0;
    const THICKNESS: f32 = 0.35;

    let r = vec2(p.x, p.y).length();
    let r_norm = (r / MAX_RADIUS).min(1.0);
    let z_surf = DEPTH * r_norm * r_norm;

    let in_bounds = p.z >= 0.0 && r <= MAX_RADIUS;
    if in_bounds {
        (p.z - z_surf).abs() - THICKNESS
    } else {
        let dz = (p.z - z_surf.max(0.0)).abs() - THICKNESS;
        let dr = (r - MAX_RADIUS).max(0.0);
        vec2(dr, dz.max(0.0)).length() - THICKNESS.min(0.0)
    }
}

pub fn sdf_ship(p: Vec3, ship: &ShipUniform) -> (f32, u32) {
    let core_pos = ship.position.xyz();
    let core_rot = ship.rotation;
    let p_local = quat_rotate(quat_conjugate(core_rot), p - core_pos);

    let d_core = sdf_core_star(p_local);
    let d_shield = sdf_shield(p_local);

    let (best_d, best_mat) = if d_core < d_shield {
        (d_core, MAT_CORE)
    } else {
        (d_shield, MAT_SHIELD)
    };

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
