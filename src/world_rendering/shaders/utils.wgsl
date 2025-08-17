#define_import_path skybound::utils

struct View {
    time: f32,
    frame_count: u32,

    clip_from_world: mat4x4<f32>,
    world_from_clip: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    view_from_world: mat4x4<f32>,

    clip_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    camera_offset: vec2<f32>,

    // Previous frame matrices for motion vectors
    prev_clip_from_world: mat4x4<f32>,
    prev_world_from_clip: mat4x4<f32>,
    prev_world_position: vec3<f32>,

    planet_rotation: vec4<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
    sun_direction: vec3<f32>,
};

struct AtmosphereData {
    sky: vec3<f32>,
    sun: vec3<f32>,
    ambient: vec3<f32>,
}


// Many hash functions https://www.shadertoy.com/view/XlGcRh
// https://github.com/johanhelsing/noisy_bevy/blob/main/assets/noisy_bevy.wgsl

// Remap a range
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return (((x - a) / (b - a)) * (d - c)) + c;
}

// Modulo functions
fn mod1(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}
fn mod3(x: vec3<f32>, y: f32) -> vec3<f32> {
    return x - floor(x / y) * y;
}

// White noise hash: f32 → f32 [0,1]
fn hash11(p: f32) -> f32 {
    var v: f32 = fract(p * 0.1031);
    v *= v + 33.33;
    v *= v + v;
    return fract(v);
}

// White noise hash: f32 → vec2 [0,1]
fn hash12(p: f32) -> vec2<f32> {
    var v: vec2<f32> = fract(vec2<f32>(p) * vec2<f32>(0.1031, 0.1030));
    v += dot(v, v.yx + 33.33);
    return fract((v.x + v.y) * v);
}

// White noise hash: f32 → vec3 [0,1]
fn hash13(p: f32) -> vec3<f32> {
    var v: vec3<f32> = fract(vec3<f32>(p) * vec3<f32>(0.1031, 0.1030, 0.1029));
    v += dot(v, v.yxz + 33.33);
    return fract((v.x + v.y + v.z) * v);
}

// White noise hash: vec2 → f32 [0,1]
fn hash21(p: vec2<f32>) -> f32 {
    var v3: vec3<f32> = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    v3 += dot(v3, v3.yzx + 33.33);
    return fract((v3.x + v3.y) * v3.z);
}

// Blue noise approx.: vec2 → f32 [0,1]
fn blue_noise(uv: vec2<f32>) -> f32 {
    var s0: f32 = hash21(uv + vec2<f32>(-1.0, 0.0));
    var s1: f32 = hash21(uv + vec2<f32>(1.0, 0.0));
    var s2: f32 = hash21(uv + vec2<f32>(0.0, 1.0));
    var s3: f32 = hash21(uv + vec2<f32>(0.0, -1.0));
    var s: f32 = s0 + s1 + s2 + s3;
    return hash21(uv) - s * 0.25 + 0.5;
}

// Returns the near (x) and far (y) intersection distances
// If the ray misses, returns vec2(1.0, -1.0)
fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(rd, rd);
    let b = 2.0 * dot(rd, ro);
    let c = dot(ro, ro) - (radius * radius);
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return vec2<f32>(1.0, -1.0); // No real intersection
    }

    let sqrt_disc = sqrt(disc);
    return vec2<f32>(
        (-b - sqrt_disc) / (2.0 * a), // Near intersection (t_near)
        (-b + sqrt_disc) / (2.0 * a)  // Far intersection (t_far)
    );
}

// Calculate intersection with a horizontal plane at given height
fn intersect_plane(ro: vec3<f32>, rd: vec3<f32>, plane_height: f32) -> f32 {
    if abs(rd.z) < 0.001 {
        return -1.0; // Ray is parallel to plane
    }

    let t = (plane_height - ro.z) / rd.z;
    return select(-1.0, t, t > 0.0);
}

// Calculates the entry and exit distances for a ray intersecting a spherical shell, including if either are behind the camera
fn ray_shell_intersect(ro: vec3<f32>, rd: vec3<f32>, view: View, bottom_altitude: f32, top_altitude: f32) -> vec2<f32> {
    let local_ro = ro - view.planet_center;

    // The entry point is the nearest intersection with the top sphere
    let top_radius = view.planet_radius + top_altitude;
    let top_interval = intersect_sphere(local_ro, rd, top_radius);
    if top_interval.x > top_interval.y {
        return vec2<f32>(1.0, 0.0); // If we miss the top sphere, we miss the shell entirely
    }

    // The exit point is the nearest intersection with the bottom sphere
    let bottom_radius = view.planet_radius + bottom_altitude;
    let bottom_interval = intersect_sphere(local_ro, rd, bottom_radius);
    if bottom_interval.x > bottom_interval.y {
        return top_interval; // Glancing shot that hits the top layer but misses the bottom, exit is the far side of the top layer
    }

    // Check the near-side segment first.
    let seg1 = vec2<f32>(top_interval.x, bottom_interval.x);
    if seg1.x < seg1.y && seg1.y > 0.0 {
        return seg1;
    }

    // If the near-side segment is invalid or entirely behind us, check the far-side segment.
    let seg2 = vec2<f32>(bottom_interval.y, top_interval.y);
    if seg2.x < seg2.y && seg2.y > 0.0 {
        return seg2;
    }

    // No valid intersection segment is in front of the camera.
    return vec2<f32>(1.0, 0.0);
}
