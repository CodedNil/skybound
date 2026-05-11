mod aur_spikes;
pub mod ships;

use crate::solids::aur_spikes::{MAT_SPIKE, SPIKE_COLOR, sdf_aur_spikes};
use crate::utils::Textures;
use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.1;
const NORMAL_EPS: f32 = 0.5;
const MIN_STEP: f32 = 0.1;
const REFLECTION_MAX_DIST: f32 = 24000.0;
const REFLECTION_STEPS: u32 = 64;

/// Surface shading result
pub struct ShadeResult {
    pub color: Vec3,
    pub specular: f32,
    pub depth: f32,
    pub normal: Vec3,
    pub hit: f32,
    pub refl_color: Vec3,
    pub refl_weight: f32,
    pub refl_hit: f32,
    pub refl_depth: f32,
}

fn world_to_curved(p_raw: Vec3, planet_center: Vec3, camera_offset_z: f32) -> Vec3 {
    let dx = p_raw.x - planet_center.x;
    let dy = p_raw.y - planet_center.y;
    let altitude = p_raw.z + camera_offset_z + (dx * dx + dy * dy) / (2.0 * PLANET_RADIUS);
    vec3(p_raw.x, p_raw.y, altitude)
}

fn sdf_combined(p: Vec3, camera_offset: Vec2, textures: &Textures) -> (f32, u32) {
    let d_spikes = sdf_aur_spikes(p, camera_offset, textures);
    (d_spikes, MAT_SPIKE)
}

fn sdf_dist(p: Vec3, camera_offset: Vec2, textures: &Textures) -> f32 {
    sdf_combined(p, camera_offset, textures).0
}

fn estimate_normal(p: Vec3, camera_offset: Vec2, textures: &Textures) -> Vec3 {
    let dx = vec3(NORMAL_EPS, 0.0, 0.0);
    let dy = vec3(0.0, NORMAL_EPS, 0.0);
    let dz = vec3(0.0, 0.0, NORMAL_EPS);
    let nx = sdf_dist(p + dx, camera_offset, textures) - sdf_dist(p - dx, camera_offset, textures);
    let ny = sdf_dist(p + dy, camera_offset, textures) - sdf_dist(p - dy, camera_offset, textures);
    let nz = sdf_dist(p + dz, camera_offset, textures) - sdf_dist(p - dz, camera_offset, textures);
    let n = vec3(nx, ny, nz);
    if n.length_squared() > 0.0 {
        n.normalize()
    } else {
        vec3(0.0, 0.0, 1.0)
    }
}

fn trace_reflection(
    pos: Vec3,
    refl_dir: Vec3,
    camera_offset: Vec2,
    textures: &Textures,
) -> (Vec3, f32, f32) {
    let mut t = 0.1;
    for _ in 0..REFLECTION_STEPS {
        let p = pos + refl_dir * t;
        let (d, _mat) = sdf_combined(p, camera_offset, textures);
        if d < 0.05 {
            return (SPIKE_COLOR, t, 1.0);
        }
        t += d.max(0.05);
        if t > REFLECTION_MAX_DIST {
            break;
        }
    }
    (Vec3::ZERO, REFLECTION_MAX_DIST, 0.0)
}

fn trace_shadow(
    pos: Vec3,
    light_dir: Vec3,
    camera_offset: Vec2,
    textures: &Textures,
    max_dist: f32,
) -> f32 {
    let mut t = 1.0;
    for _ in 0..32 {
        let p = pos + light_dir * t;
        let d = sdf_dist(p, camera_offset, textures);
        if d < 0.1 {
            return 0.0;
        }
        t += d.max(0.5);
        if t > max_dist {
            break;
        }
    }
    1.0
}

fn shade_sdfs(
    p: Vec3,
    n: Vec3,
    camera_offset: Vec2,
    view: &ViewUniform,
    textures: &Textures,
) -> ShadeResult {
    let sun_pos = crate::utils::get_sun_position(
        view.planet_center(),
        view.planet_rotation,
        view.ro_relative(),
        view.latitude(),
    );
    let planet_center = view.planet_center();
    let light_dir = (sun_pos - (p + planet_center)).normalize();
    let dot_nl = n.dot(light_dir).max(0.0);
    let shadow = if dot_nl > 0.0 {
        trace_shadow(p, light_dir, camera_offset, textures, 15000.0)
    } else {
        0.0
    };
    let (refl_color, refl_depth, refl_hit) = trace_reflection(
        p,
        {
            let vd = (view.world_position.xyz() - p).normalize();
            2.0 * n.dot(vd) * n - vd
        },
        camera_offset,
        textures,
    );
    ShadeResult {
        color: SPIKE_COLOR * (dot_nl * shadow + 0.05),
        specular: 0.0,
        depth: 0.0,
        normal: n,
        hit: 1.0,
        refl_color,
        refl_weight: 0.0,
        refl_hit,
        refl_depth,
    }
}

pub fn raymarch_solids(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    t_max: f32,
    dither: f32,
    textures: &Textures,
) -> ShadeResult {
    let planet_center = view.planet_center();
    let camera_offset = view.camera_offset();
    let camera_offset_xy = camera_offset.xy();
    let mut t = dither * 1.5;

    let mut out_color = Vec3::ZERO;
    let mut out_refl_color = Vec3::ZERO;
    let mut out_refl_weight = 0.0;
    let mut out_refl_hit = 0.0;
    let mut out_spec = 0.0;
    let mut out_normal = Vec3::ZERO;
    let mut out_depth = t_max;
    let mut out_refl_depth = REFLECTION_MAX_DIST;
    let mut hit = 0.0;

    for _ in 0..MAX_STEPS {
        if t >= t_max {
            break;
        }
        let p_raw = ro + rd * t;
        let p = world_to_curved(p_raw, planet_center, camera_offset.z);
        let (dist, _mat) = sdf_combined(p, camera_offset_xy, textures);
        if dist < EPSILON {
            let normal = estimate_normal(p, camera_offset_xy, textures);
            out_depth = t;
            hit = 1.0;
            if rd.dot(normal) <= 0.0 {
                let shade = shade_sdfs(p, normal, camera_offset_xy, view, textures);
                out_color = shade.color;
                out_refl_color = shade.refl_color;
                out_refl_weight = shade.refl_weight;
                out_refl_hit = shade.refl_hit;
                out_spec = shade.specular;
                out_normal = normal;
                out_refl_depth = shade.refl_depth;
            }
            break;
        }
        t += dist.max(MIN_STEP);
    }

    ShadeResult {
        color: out_color.saturate(),
        refl_color: out_refl_color,
        refl_weight: out_refl_weight,
        refl_hit: out_refl_hit,
        depth: out_depth,
        specular: out_spec,
        normal: out_normal,
        hit,
        refl_depth: out_refl_depth,
    }
}
