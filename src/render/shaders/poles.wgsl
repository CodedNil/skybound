#define_import_path skybound::poles
#import skybound::utils::{ATMOSPHERE_HEIGHT, View, quat_rotate}

const POLE_WIDTH: f32 = 10000.0;

struct PolesSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_poles(pos: vec3<f32>, time: f32, linear_sampler: sampler) -> PolesSample {
    var sample: PolesSample;
    sample.density = 1.0;
    sample.color = vec3(0.0, 0.5, 1.0);

    // Make it more intense at the top of the atmosphere
    let atmosphere_dist = smoothstep(5000.0, 100.0, abs(pos.z - ATMOSPHERE_HEIGHT));
    if atmosphere_dist > 0.0 {
        sample.color += atmosphere_dist;
    }

    sample.emission = sample.color;
    return sample;
}

// Returns vec2(entry_t, exit_t), or vec2(max, 0.0) if no hit
fn poles_raymarch_entry(ro: vec3<f32>, rd: vec3<f32>, view: View, t_max: f32) -> vec2<f32> {
    let axis = normalize(quat_rotate(view.planet_rotation, vec3<f32>(0.0, 0.0, 1.0)));

    let oc = ro - view.planet_center;
    let ad = dot(axis, rd);
    let ao = dot(axis, oc);
    let a = 1.0 - ad * ad;
    let b = 2.0 * (dot(oc, rd) - ao * ad);
    let c = dot(oc, oc) - ao * ao - POLE_WIDTH * POLE_WIDTH;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 || a == 0.0 { return vec2<f32>(t_max, 0.0); }

    let s = sqrt(disc);
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);

    // Ensure t0 ≤ t1
    let entry = min(t0, t1);
    let exit  = max(t0, t1);

    if exit <= 0.0 { return vec2<f32>(t_max, 0.0); }

    return vec2<f32>(entry, exit);
}
