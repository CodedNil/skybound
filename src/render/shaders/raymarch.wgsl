#define_import_path skybound::raymarch
#import skybound::utils::{View, AtmosphereData, get_sun_position}

// Raymarch parameters
const MAX_STEPS: i32 = 64;
const EPSILON: f32 = 0.01;
const MIN_STEP: f32 = 0.03;

// Shared shape parameters
const SPACING: f32 = 200.0;
const RADIUS: f32 = 2.0;

// Repeating cell helper: move 'p' into the nearest cell center for spacing 's'
fn repeat_to_cell(p: vec3<f32>, s: f32) -> vec3<f32> {
    let cell_center = floor(p / s + vec3<f32>(0.5)) * s;
    return p - cell_center;
}

// SDF for a single sphere of radius r (centered at origin)
fn sdf_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// SDF for an infinite grid of spheres
fn sdf_repeating_spheres_world(p_world: vec3<f32>) -> f32 {
    let local = repeat_to_cell(p_world, SPACING);
    return sdf_sphere(local, RADIUS);
}

// Numerical normal via central differences
fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let dx = vec3<f32>(EPSILON, 0.0, 0.0);
    let dy = vec3<f32>(0.0, EPSILON, 0.0);
    let dz = vec3<f32>(0.0, 0.0, EPSILON);

    // Approximate local cell once and evaluate the cheap sphere SDF directly.
    // This avoids calling `repeat_to_cell` repeatedly inside sdf calls.
    let local = repeat_to_cell(p, SPACING);
    let nx = sdf_sphere(local + dx, RADIUS) - sdf_sphere(local - dx, RADIUS);
    let ny = sdf_sphere(local + dy, RADIUS) - sdf_sphere(local - dy, RADIUS);
    let nz = sdf_sphere(local + dz, RADIUS) - sdf_sphere(local - dz, RADIUS);
    return normalize(vec3<f32>(nx, ny, nz));
}

// Basic lambert shading with a single directional light (sun).
// Also computes a simple specular contribution when requested.
struct ShadeResult {
    color: vec3<f32>,
    spec: f32,
}
fn shade_basic(p: vec3<f32>, n: vec3<f32>, rd: vec3<f32>, view: View) -> ShadeResult {
    // Green base color for spheres
    let base_color = vec3<f32>(0.1, 0.8, 0.2);

    // Sun direction from view; fall back to a reasonable default if identical position
    let sun_pos = get_sun_position(view);
    let sun_dir = normalize(sun_pos - p);

    // Lambert
    let lam = max(dot(n, sun_dir), 0.0);

    // Ambient + lambert
    let ambient = 0.12;
    var color = base_color * (ambient + lam * 0.9);

    // View direction
    let view_dir = normalize(-rd);

    // Specular controlled by hemisphere: top half only (world +Z)
    // Here "top" is defined as normal.z > 0.
    let hemi_mask = select(0.0, 1.0, n.z > 0.0);
    var spec = f32(hemi_mask > 0.0);

    color += vec3<f32>(spec);

    var out: ShadeResult;
    out.color = color;
    out.spec = spec * hemi_mask;
    return out;
}

// Main solids raymarcher
struct SolidsResult {
    color: vec3<f32>,
    depth: f32,
    specular: f32,
    normal: vec3<f32>,
    hit: f32,
}
fn raymarch_solids(ro: vec3<f32>, rd: vec3<f32>, view: View, t_max: f32, time: f32) -> SolidsResult {
    var t = 0.0;

    var out_color = vec3<f32>(0.0);
    var out_spec = 0.0;
    var out_normal = vec3<f32>(0.0);
    var out_depth = t_max;

    for (var i: i32 = 0; i < MAX_STEPS; i = i + 1) {
        if t >= t_max {
            break;
        }

        let p = ro + rd * t;
        let dist = sdf_repeating_spheres_world(p);

        if dist < EPSILON {
            // Hit the surface
            out_depth = t;
            out_normal = estimate_normal(p);

            // Shade
            let shade = shade_basic(p, out_normal, rd, view);
            out_color = shade.color;
            out_spec = shade.spec;

            break;
        }

        // advance by the signed distance (clamp to min step)
        t += max(dist, MIN_STEP);
    }

    var res: SolidsResult;
    res.color = clamp(out_color, vec3<f32>(0.0), vec3<f32>(1.0));
    res.depth = out_depth;
    res.specular = out_spec;
    res.normal = out_normal;
    return res;
}
