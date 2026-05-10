use crate::lighting::{compute_surface_light, trace_sun_visibility};
use crate::utils::{AtmosphereData, get_sun_position};
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec3, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

/// Maximum raymarch steps before giving up.
const MAX_STEPS: i32 = 256;
/// Surface intersection threshold in meters.
const EPSILON: f32 = 0.01;
/// Minimum safe step to avoid getting stuck.
const MIN_STEP: f32 = 0.03;

/// Distance between sphere centres in the XY tiling grid.
const SPACING: f32 = 250.0;
/// Sphere radius in meters.
const RADIUS: f32 = 5.0;

/// Half-extent along Y — total span per side.
const WING_HALF_SPAN: f32 = 4.0;
/// Half-extent along Z — wing profile thickness.
const WING_HALF_THICKNESS: f32 = 0.3;
/// Half-extent along X — front-to-back chord.
const WING_HALF_CHORD: f32 = 2.5;

const MAT_SPHERE: u32 = 0;
const MAT_WING: u32 = 1;

const SUN_VISIBILITY_DIST: f32 = 50.0;
const SUN_VISIBILITY_STEPS: u32 = 12;
const REFLECTION_MAX_DIST: f32 = 40.0;
const REFLECTION_STEPS: u32 = 24;

/// Tile `p` in the XY plane with spacing `s`. Z passes through unchanged.
fn repeat_to_cell_xy(p: Vec3, s: f32) -> Vec3 {
    let cell_center = (p / s + Vec3::splat(0.5)).floor() * s;
    vec3(p.x - cell_center.x, p.y - cell_center.y, p.z)
}

/// Signed distance to a sphere centred at the origin.
fn sdf_sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

/// Signed distance to an axis-aligned box centred at the origin.
fn sdf_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Signed distance for a single wing on the +Y or -Y side of the sphere.
fn sdf_wing_single(p: Vec3, sign: f32, flap_angle: f32) -> f32 {
    let center = vec3(0.0, sign * (RADIUS + WING_HALF_SPAN), 0.0);
    let local = p - center;
    // Quadratic bend along the wing span.
    let bend = local.y * local.y * flap_angle * 0.015;
    let bent = vec3(local.x, local.y, local.z - bend);
    sdf_box(
        bent,
        vec3(WING_HALF_CHORD, WING_HALF_SPAN, WING_HALF_THICKNESS),
    )
}

/// Union of both wings, driven by time.
fn sdf_wings(p: Vec3, time: f32) -> f32 {
    let flap_angle = (time * 3.0).sin() * 0.8;
    sdf_wing_single(p, 1.0, flap_angle).min(sdf_wing_single(p, -1.0, flap_angle))
}

/// Combined signed distance field returning `(distance, material_id)`.
fn sdf_combined(p: Vec3, time: f32) -> (f32, u32) {
    let local = repeat_to_cell_xy(p, SPACING);

    let d_sphere = sdf_sphere(local, RADIUS);
    let d_wings = sdf_wings(local, time);

    if d_sphere < d_wings {
        (d_sphere, MAT_SPHERE)
    } else {
        (d_wings, MAT_WING)
    }
}

/// Distance-only convenience wrapper.
fn sdf_dist(p: Vec3, time: f32) -> f32 {
    sdf_combined(p, time).0
}

/// Estimate the surface normal at `p` via central differences of `sdf_dist`.
fn estimate_normal(p: Vec3, time: f32) -> Vec3 {
    let dx = vec3(EPSILON, 0.0, 0.0);
    let dy = vec3(0.0, EPSILON, 0.0);
    let dz = vec3(0.0, 0.0, EPSILON);

    let nx = sdf_dist(p + dx, time) - sdf_dist(p - dx, time);
    let ny = sdf_dist(p + dy, time) - sdf_dist(p - dy, time);
    let nz = sdf_dist(p + dz, time) - sdf_dist(p - dz, time);
    vec3(nx, ny, nz).normalize()
}

/// Light-blue wing colour.
const WING_COLOR: Vec3 = vec3(0.4, 0.7, 1.0);
/// Neutral warm grey sphere colour.
const SPHERE_COLOR: Vec3 = vec3(0.55, 0.52, 0.48);

/// Result of a reflection trace within the solids SDF.
struct ReflectionResult {
    color: Vec3,
    ao: f32,
    hit: bool,
}

/// March a reflected ray through the solids SDF.
///
/// Returns the colour of whatever surface is hit (if any) and accumulated
/// ambient occlusion along the ray.
fn trace_reflection(
    pos: Vec3,
    refl_dir: Vec3,
    max_dist: f32,
    steps: u32,
    time: f32,
) -> ReflectionResult {
    let step_size = max_dist / steps as f32;
    let mut t = 0.1; // nudge off surface to avoid self-hit
    let mut ao = 1.0;

    for _ in 0..steps {
        let p = pos + refl_dir * t;
        let (d, mat) = sdf_combined(p, time);

        // Accumulate ambient occlusion from nearby surfaces.
        let proximity = (1.0 - (d / t).saturate()).powf(2.0);
        ao = ao.min(1.0 - proximity * 0.6);

        if d < 0.02 {
            let color = if mat == MAT_WING {
                WING_COLOR
            } else {
                Vec3::ZERO
            };
            return ReflectionResult {
                color,
                ao,
                hit: true,
            };
        }

        t += d.max(step_size * 0.3);
    }

    ReflectionResult {
        color: Vec3::ZERO,
        ao,
        hit: false,
    }
}

/// Result of shading a single surface hit.
struct ShadeResult {
    color: Vec3,
    specular: f32,
}

/// Shade a surface hit point.
fn shade_basic(
    p: Vec3,
    n: Vec3,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    time: f32,
) -> ShadeResult {
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - p).normalize();

    // --- material lookup ---
    let (_, mat) = sdf_combined(p, time);
    let is_specular = mat == MAT_SPHERE;
    let base_color = if mat == MAT_WING {
        WING_COLOR
    } else {
        SPHERE_COLOR
    };

    // --- self-shadowing toward sun ---
    let sun_vis =
        trace_sun_visibility(p, sun_dir, SUN_VISIBILITY_DIST, SUN_VISIBILITY_STEPS, |q| {
            sdf_dist(q, time)
        });

    // --- incident light ---
    let light = compute_surface_light(n, sun_dir, sun_vis, atmosphere);

    // --- diffuse ---
    let mut color = base_color * (light.sun + light.sky * 0.5 + light.ambient * 0.8);

    // --- specular & mirror reflection (sphere only) ---
    if is_specular {
        let view_dir = (view.world_position.xyz() - p).normalize();
        let refl_dir = (view_dir - 2.0 * n.dot(view_dir) * n).normalize();

        let ReflectionResult {
            color: refl_color,
            ao,
            hit,
        } = trace_reflection(p, refl_dir, REFLECTION_MAX_DIST, REFLECTION_STEPS, time);

        if hit {
            let fresnel = 1.0 - n.dot(view_dir).abs();
            let fresnel = fresnel.powf(2.5);
            let mirror_weight = 0.75 * fresnel + 0.05;
            color = color.lerp(refl_color, mirror_weight);
        }

        // Blinn-Phong sun highlight.
        let half = (sun_dir + view_dir).normalize();
        let spec = n.dot(half).max(0.0).powf(256.0);
        color += Vec3::splat(spec * sun_vis) * light.sun * 0.7;

        // Ambient occlusion from reflection trace.
        color *= 0.3 + 0.7 * ao;
    }

    let specular = if is_specular { 1.0 } else { 0.0 };

    ShadeResult { color, specular }
}

/// Result of the solids raymarch pass.
pub struct SolidsResult {
    pub color: Vec3,
    pub depth: f32,
    pub specular: f32,
    pub normal: Vec3,
    pub hit: f32,
}

/// Raymarch the repeating sphere+wing scene.
pub fn raymarch_solids(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    t_max: f32,
) -> SolidsResult {
    let time = view.time();
    let mut t = 0.0;

    let mut out_color = Vec3::ZERO;
    let mut out_spec = 0.0;
    let mut out_normal = Vec3::ZERO;
    let mut out_depth = t_max;
    let mut hit = 0.0;

    for _ in 0..MAX_STEPS {
        if t >= t_max {
            break;
        }

        let p = ro + rd * t;
        let dist = sdf_dist(p, time);

        if dist < EPSILON {
            out_depth = t;
            let normal = estimate_normal(p, time);
            let shade = shade_basic(p, normal, view, atmosphere, time);
            out_color = shade.color;
            out_spec = shade.specular;
            out_normal = normal;
            hit = 1.0;
            break;
        }

        t += dist.max(MIN_STEP);
    }

    SolidsResult {
        color: out_color.saturate(),
        depth: out_depth,
        specular: out_spec,
        normal: out_normal,
        hit,
    }
}
