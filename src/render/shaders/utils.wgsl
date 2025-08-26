#define_import_path skybound::utils

const MAGNETOSPHERE_HEIGHT: f32 = 400000;

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

    planet_rotation: vec4<f32>,
    planet_center: vec3<f32>,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
};

struct AtmosphereData {
    sun_pos: vec3<f32>,
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

// Rotate vector v by quaternion q = (xyz, w)
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let uv = cross(u, v);
    return v + 2.0 * (q.w * uv + cross(u, uv));
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

// Calculates up to two intersection segments (entry/exit pairs) for a ray intersecting a spherical shell
// x, y = first intersection segment (entry, exit)
// z, w = second intersection segment (entry, exit)
// An invalid segment is represented by entry > exit (e.g., 1.0, 0.0)
fn ray_shell_intersect(ro: vec3<f32>, rd: vec3<f32>, view: View, bottom_altitude: f32, top_altitude: f32) -> vec4<f32> {
    let local_ro = ro - view.planet_center;

    // The entry point is the nearest intersection with the top sphere
    let top_radius = view.planet_radius + top_altitude;
    let top_interval = intersect_sphere(local_ro, rd, top_radius);
    if top_interval.x > top_interval.y {
        return vec4<f32>(1.0, 0.0, 1.0, 0.0); // If we miss the top sphere, we miss the shell entirely
    }

    // The exit point is the nearest intersection with the bottom sphere
    let bottom_radius = view.planet_radius + bottom_altitude;
    let bottom_interval = intersect_sphere(local_ro, rd, bottom_radius);
    if bottom_interval.x > bottom_interval.y {
        return vec4<f32>(top_interval.x, top_interval.y, 1.0, 0.0); // Glancing shot that hits the top layer but misses the bottom, exit is the far side of the top layer
    }

    // The ray hits both spheres, creating two segments through the shell
    // Segment 1: Enters top sphere (near), exits bottom sphere (near)
    // Segment 2: Enters bottom sphere (far), exits top sphere (far)
    return vec4<f32>(top_interval.x, bottom_interval.x, bottom_interval.y, top_interval.y);
}


/// Calculates the world position of the two polar suns and returns the one highest in the sky
fn get_sun_position(view: View) -> vec3<f32> {
    // Determine the planet's current north pole axis based on its rotation
    let north_axis = normalize(quat_rotate(view.planet_rotation, vec3<f32>(0.0, 0.0, 1.0)));

    // The "up" direction is the vector from the planet's center to the camera
    let up_vector = normalize(view.world_position - view.planet_center);

    // The sign of the dot product tells us which hemisphere the camera is in
    let is_in_northern_hemisphere = dot(north_axis, up_vector) > 0.0;

    // Select the correct axis (north or south) based on the hemisphere
    let sun_axis = select(-north_axis, north_axis, is_in_northern_hemisphere);

    // Calculate the sun's position at a fixed altitude above the relevant pole
    let sun_altitude = view.planet_radius + MAGNETOSPHERE_HEIGHT;
    var sun_pos = view.planet_center + sun_axis * sun_altitude;

    // To smooth harsh switch on equator we progressively move the sun down in world-space Z as latitude approaches 0
    let blend = clamp(abs(view.latitude) / 0.35, 0.0, 1.0);
    sun_pos.z += mix(sun_altitude * -2.0, 0.0, blend);

    return sun_pos;
}
