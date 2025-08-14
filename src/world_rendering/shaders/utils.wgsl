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
    camera_offset: vec3<f32>,

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
    ground: vec3<f32>,
    phase: f32,
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

// Ray intersection of a sphere, outputs distance to the intersection point or -1.0
fn intersect_sphere(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let a = dot(rd, rd);
    let b = 2.0 * dot(rd, ro);
    let c = dot(ro, ro) - (radius * radius);
    let disc = (b * b) - 4.0 * a * c;
    if disc < 0.0 { return -1.0; }
    let d = sqrt(disc);
    let t0 = (-b - d) / (2.0 * a);
    let t1 = (-b + d) / (2.0 * a);
    // Return nearest positive intersection (entry), or the positive exit if inside
    if t0 > 0.0 { return t0; }
    if t1 > 0.0 { return t1; }
    return -1.0;
}
