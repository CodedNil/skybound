mod aur_spikes;
mod ships;

use crate::solids::aur_spikes::{MAT_SPIKE, SPIKE_COLOR, sdf_aur_spikes, sdf_ground};
use crate::solids::ships::{MAT_SPHERE, MAT_WING, mat_color, sdf_ships, shade_ship};
use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 512;
const EPSILON: f32 = 0.01;
const NORMAL_EPS: f32 = 0.02;
const MIN_STEP: f32 = 0.03;

const REFLECTION_MAX_DIST: f32 = 24000.0;
const REFLECTION_STEPS: u32 = 256;

/// Surface shading result
pub struct ShadeResult {
    /// Diffuse surface color
    pub color: Vec3,
    /// Surface specular intensity (0.0 for non-specular, 1.0 for perfect reflection).
    pub specular: f32,
    /// Travel distance to the solid hit in the ray direction.
    pub depth: f32,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// 1.0 if the ray hit a solid, 0.0 if it hit sky.
    pub hit: f32,
    /// Shaded solid hit in reflection direction, ZERO if sky
    pub refl_color: Vec3,
    /// Fresnel blend weight: how much of the pixel is reflection (0 for non-specular).
    pub refl_weight: f32,
    /// 1.0 if the reflection hit a solid, 0.0 if it hit sky.
    pub refl_hit: f32,
    /// Travel distance to the solid hit in the reflection direction
    pub refl_depth: f32,
}

/// SDF evaluation frame: xy is camera-local, z is true altitude (`camera_offset.z` + r²/2R).
/// xy must stay camera-local so normal finite-differences are never below the f32 ULP.
fn world_to_curved(p_raw: Vec3, planet_center: Vec3, camera_offset_z: f32) -> Vec3 {
    let dx = p_raw.x - planet_center.x;
    let dy = p_raw.y - planet_center.y;
    let altitude = p_raw.z + camera_offset_z + (dx * dx + dy * dy) / (2.0 * PLANET_RADIUS);
    vec3(p_raw.x, p_raw.y, altitude)
}

fn sdf_combined(p: Vec3, camera_offset: Vec2, time: f32) -> (f32, u32) {
    let (d_creatures, mat_creatures) = sdf_ships(p, time);

    let d_spikes = sdf_aur_spikes(p, camera_offset);
    let d_ground = sdf_ground(p);
    let (d_aur, mat_aur) = if d_spikes < d_ground {
        (d_spikes, MAT_SPIKE)
    } else {
        (d_ground, MAT_SPIKE)
    };

    if d_aur < d_creatures {
        (d_aur, mat_aur)
    } else {
        (d_creatures, mat_creatures)
    }
}

fn sdf_dist(p: Vec3, camera_offset: Vec2, time: f32) -> f32 {
    sdf_combined(p, camera_offset, time).0
}

fn estimate_normal(p: Vec3, camera_offset: Vec2, time: f32) -> Vec3 {
    let dx = vec3(NORMAL_EPS, 0.0, 0.0);
    let dy = vec3(0.0, NORMAL_EPS, 0.0);
    let dz = vec3(0.0, 0.0, NORMAL_EPS);
    let nx = sdf_dist(p + dx, camera_offset, time) - sdf_dist(p - dx, camera_offset, time);
    let ny = sdf_dist(p + dy, camera_offset, time) - sdf_dist(p - dy, camera_offset, time);
    let nz = sdf_dist(p + dz, camera_offset, time) - sdf_dist(p - dz, camera_offset, time);
    vec3(nx, ny, nz).normalize()
}

/// Returns (color, depth, hit): color = shaded solid or ZERO if sky;
fn trace_reflection(pos: Vec3, refl_dir: Vec3, camera_offset: Vec2, time: f32) -> (Vec3, f32, f32) {
    let mut t = 0.1;

    for _ in 0..REFLECTION_STEPS {
        let p = pos + refl_dir * t;
        let (d, mat) = sdf_combined(p, camera_offset, time);
        if d < 0.05 {
            let albedo = if mat == MAT_SPIKE {
                SPIKE_COLOR
            } else {
                mat_color(mat)
            };
            return (albedo, t, 1.0);
        }
        t += d.max(0.05);
        if t > REFLECTION_MAX_DIST {
            break;
        }
    }

    (Vec3::ZERO, REFLECTION_MAX_DIST, 0.0)
}

fn shade_basic(
    p: Vec3,
    n: Vec3,
    mat: u32,
    camera_offset: Vec2,
    view: &ViewUniform,
    time: f32,
) -> ShadeResult {
    if mat == MAT_SPHERE || mat == MAT_WING {
        return shade_ship(p, n, mat, view, |pos, refl_dir| {
            trace_reflection(pos, refl_dir, camera_offset, time)
        });
    }

    // Spikes and ground
    ShadeResult {
        color: SPIKE_COLOR,
        specular: 0.0,
        depth: 0.0,
        normal: Vec3::ZERO,
        hit: 0.0,
        refl_color: Vec3::ZERO,
        refl_weight: 0.0,
        refl_hit: 0.0,
        refl_depth: REFLECTION_MAX_DIST,
    }
}

pub fn raymarch_solids(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    t_max: f32,
    dither: f32,
) -> ShadeResult {
    let time = view.time();
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
        let (dist, mat) = sdf_combined(p, camera_offset_xy, time);
        if dist < EPSILON {
            let normal = estimate_normal(p, camera_offset_xy, time);
            out_depth = t;
            hit = 1.0;
            if rd.dot(normal) <= 0.0 {
                let shade = shade_basic(p, normal, mat, camera_offset_xy, view, time);
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
