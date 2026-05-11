mod aur_spikes;
mod ships;

use crate::solids::aur_spikes::{MAT_SPIKE, SPIKE_COLOR, sdf_aur_spikes};
use crate::solids::ships::{MAT_SPHERE, MAT_WING, mat_color, sdf_ships, shade_ship};
use crate::utils::Textures;
use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, vec3};

const MAX_STEPS: i32 = 256;
const EPSILON: f32 = 0.1;
const NORMAL_EPS: f32 = 0.5;
const MIN_STEP: f32 = 0.1;

const REFLECTION_MAX_DIST: f32 = 24000.0;
const REFLECTION_STEPS: u32 = 64;

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

fn sdf_combined(p: Vec3, camera_offset: Vec2, time: f32, textures: &Textures) -> (f32, u32) {
    let (d_ships, mat_ships) = sdf_ships(p, time);
    let d_spikes = sdf_aur_spikes(p, camera_offset, textures);
    if d_spikes < d_ships {
        (d_spikes, MAT_SPIKE)
    } else {
        (d_ships, mat_ships)
    }
}

fn sdf_dist(p: Vec3, camera_offset: Vec2, time: f32, textures: &Textures) -> f32 {
    sdf_combined(p, camera_offset, time, textures).0
}

fn estimate_normal(p: Vec3, camera_offset: Vec2, time: f32, textures: &Textures) -> Vec3 {
    let dx = vec3(NORMAL_EPS, 0.0, 0.0);
    let dy = vec3(0.0, NORMAL_EPS, 0.0);
    let dz = vec3(0.0, 0.0, NORMAL_EPS);

    let nx = sdf_dist(p + dx, camera_offset, time, textures)
        - sdf_dist(p - dx, camera_offset, time, textures);
    let ny = sdf_dist(p + dy, camera_offset, time, textures)
        - sdf_dist(p - dy, camera_offset, time, textures);
    let nz = sdf_dist(p + dz, camera_offset, time, textures)
        - sdf_dist(p - dz, camera_offset, time, textures);

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
    time: f32,
    textures: &Textures,
) -> (Vec3, f32, f32) {
    let mut t = 0.1;

    for _ in 0..REFLECTION_STEPS {
        let p = pos + refl_dir * t;
        let (d, mat) = sdf_combined(p, camera_offset, time, textures);
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

fn trace_shadow(
    pos: Vec3,
    light_dir: Vec3,
    camera_offset: Vec2,
    time: f32,
    textures: &Textures,
    max_dist: f32,
) -> f32 {
    let mut t = 1.0; // Start offset to avoid self-intersection
    for _ in 0..32 {
        let p = pos + light_dir * t;
        let d = sdf_dist(p, camera_offset, time, textures);
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
    mat: u32,
    camera_offset: Vec2,
    view: &ViewUniform,
    time: f32,
    textures: &Textures,
) -> ShadeResult {
    let sun_pos = crate::utils::get_sun_position(view);
    let planet_center = view.planet_center();

    // Approximate light direction in the curved space
    let light_dir = (sun_pos - (p + planet_center)).normalize();
    let dot_nl = n.dot(light_dir).max(0.0);

    let shadow = if dot_nl > 0.0 {
        trace_shadow(p, light_dir, camera_offset, time, textures, 15000.0)
    } else {
        0.0
    };

    if mat == MAT_SPHERE || mat == MAT_WING {
        let mut shade = shade_ship(p, n, mat, view, |pos, refl_dir| {
            trace_reflection(pos, refl_dir, camera_offset, time, textures)
        });
        shade.color *= dot_nl * shadow + 0.05;
        return shade;
    }

    ShadeResult {
        color: SPIKE_COLOR * (dot_nl * shadow + 0.05),
        specular: 0.0,
        depth: 0.0,
        normal: n,
        hit: 1.0,
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
    textures: &Textures,
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
        let (dist, mat) = sdf_combined(p, camera_offset_xy, time, textures);
        if dist < EPSILON {
            let normal = estimate_normal(p, camera_offset_xy, time, textures);
            out_depth = t;
            hit = 1.0;
            if rd.dot(normal) <= 0.0 {
                let shade = shade_sdfs(p, normal, mat, camera_offset_xy, view, time, textures);
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
