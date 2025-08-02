#define_import_path skybound::poles

const POLE_WIDTH: f32 = 10000.0; // Cylinder radius
const POLE_COLOR: vec3<f32> = vec3(0.0, 0.5, 1.0); // Blue

// Rotate vector v by quaternion q = (xyz, w)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let uv = cross(u, v);
    return v + 2.0 * (q.w * uv + cross(u, uv));
}

// Ray–infinite‐cylinder intersection
// returns t > 0 if hit, else -1
fn intersect_cylinder(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, axis: vec3<f32>, radius: f32) -> f32 {
    let oc = ro - center;
    let ad = dot(axis, rd);
    let ao = dot(axis, oc);
    let a = 1.0 - ad * ad;
    let b = 2.0 * (dot(oc, rd) - ao * ad);
    let c = dot(oc, oc) - ao * ao - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return -1.0;
    }
    let t0 = (-b - sqrt(disc)) / (2.0 * a);
    if t0 <= 0.0 {
        return -1.0;
    }
    return t0;
}

fn render_poles(ro: vec3<f32>, rd: vec3<f32>, planet_rot: vec4<f32>, planet_radius: f32) -> vec4<f32> {
    let center = ro - vec3<f32>(0.0, planet_radius, 0.0);

    // Rotate the Y‐axis by the planet’s quaternion to get the pole‐axis
    let axis = normalize(quat_rotate(planet_rot, vec3<f32>(0.0, 1.0, 0.0)));

    let t = intersect_cylinder(ro, rd, center, axis, POLE_WIDTH);
    if t < 0.0 {
        return vec4<f32>(0.0);
    }

    // Clamp to above horizon only
    let p = ro + rd * t;
    if p.y < 0.0 {
        return vec4<f32>(0.0);
    }

    // Compute a soft‐edge alpha via an SDF & smoothstep
    let proj = center + axis * dot(p - center, axis);
    let sdf = length(p - proj) - POLE_WIDTH;
    let alpha = 1.0 - smoothstep(0.0, POLE_WIDTH * 0.5, sdf);

    return vec4<f32>(POLE_COLOR, alpha);
}
