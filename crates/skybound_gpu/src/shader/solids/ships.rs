use crate::{
    ShipUniform, ViewUniform,
    shader::{
        T_MAX,
        solids::{ShadeResult, estimate_normal},
        utils::get_sun_position,
    },
};
use spirv_std::glam::{FloatExt, Quat, Vec3, Vec4Swizzles, vec2, vec3, vec4};
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

fn sdf_core_star(p: Vec3) -> f32 {
    const SPHERE_R: f32 = 2.5;
    const SPIKE_LEN: f32 = 5.5;
    const SPIKE_W: f32 = 0.65;

    let base = p.length() - SPHERE_R;
    let spike = |direction: Vec3| -> f32 {
        let t = p.dot(direction).clamp(0.0, SPIKE_LEN);
        let width = SPIKE_W * (1.0 - t / SPIKE_LEN);
        (p - direction * t).length() - width
    };
    let spikes = spike(Vec3::X)
        .min(spike(-Vec3::X))
        .min(spike(Vec3::Y))
        .min(spike(-Vec3::Y))
        .min(spike(Vec3::Z))
        .min(spike(-Vec3::Z));

    smin(base, spikes, 0.9)
}

fn sdf_shield(p: Vec3) -> f32 {
    const MAX_RADIUS: f32 = 18.0;
    const DEPTH: f32 = 9.0;
    const THICKNESS: f32 = 0.35;

    let radius = vec2(p.x, p.y).length();
    let z_surface = DEPTH * (radius / MAX_RADIUS).min(1.0).powi(2);
    if p.z >= 0.0 && radius <= MAX_RADIUS {
        (p.z - z_surface).abs() - THICKNESS
    } else {
        let dz = (p.z - z_surface.max(0.0)).abs() - THICKNESS;
        let dr = (radius - MAX_RADIUS).max(0.0);
        vec2(dr, dz.max(0.0)).length() - THICKNESS.min(0.0)
    }
}

pub fn sdf_ship(p: Vec3, ship: &ShipUniform) -> (f32, u32) {
    let rotation = Quat::from_xyzw(
        ship.rotation.x,
        ship.rotation.y,
        ship.rotation.z,
        ship.rotation.w,
    );
    let local_position = rotation.conjugate().mul_vec3(p - ship.position.xyz());
    let core_distance = sdf_core_star(local_position);
    let shield_distance = sdf_shield(local_position);

    if core_distance < shield_distance {
        (core_distance, MAT_CORE)
    } else {
        (shield_distance, MAT_SHIELD)
    }
}

pub fn raymarch_ship(ro: Vec3, rd: Vec3, view: &ViewUniform, ship: &ShipUniform) -> ShadeResult {
    const SHIP_MAX_T: f32 = 4000.0;
    const SHIP_MAX_STEPS: i32 = 128;
    const SHIP_EPSILON: f32 = 0.08;
    const SHIP_MIN_STEP: f32 = 0.05;

    let mut t = 0.0;
    let mut hit_mat = MAT_CORE;
    let mut hit = false;

    for _ in 0..SHIP_MAX_STEPS {
        if t >= SHIP_MAX_T {
            break;
        }
        let (distance, material) = sdf_ship(ro + rd * t, ship);
        if distance < SHIP_EPSILON {
            hit = true;
            hit_mat = material;
            break;
        }
        t += distance.max(SHIP_MIN_STEP);
    }

    if !hit {
        return ShadeResult {
            color_depth: vec4(0.0, 0.0, 0.0, T_MAX),
            normal: Vec3::ZERO,
        };
    }

    let position = ro + rd * t;
    let normal = estimate_normal(position, |p| sdf_ship(p, ship).0);
    let sun_position = get_sun_position(
        view.planet_center(),
        view.planet_rotation,
        view.ro_relative(),
        view.latitude(),
    );
    let sun_direction = (sun_position - ro).normalize();
    let diffuse = normal.dot(sun_direction).max(0.0);
    let base_color = mat_color(hit_mat);
    let lit = base_color * (diffuse * 0.9 + 0.06);
    let view_direction = (ro - position).normalize();
    let fresnel = (1.0 - normal.dot(view_direction).abs()).powf(3.0) * 0.4;

    ShadeResult {
        color_depth: (lit + Vec3::splat(fresnel)).extend(t),
        normal,
    }
}
