use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{
    FloatExt, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3, vec4,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const MAGNETOSPHERE_HEIGHT: f32 = 400_000.0;

pub struct AtmosphereData {
    pub sun_pos: Vec3,
    pub sky: Vec3,
    pub sun: Vec3,
    pub ambient: Vec3,
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).saturate();
    t * t * (3.0 - 2.0 * t)
}

// Modulo functions
pub fn mod1(x: f32, y: f32) -> f32 {
    x - y * (x / y).floor()
}

pub fn mod3(x: Vec3, y: f32) -> Vec3 {
    x - (x / y).floor() * y
}

// White noise hash: f32 → f32 [0,1]
pub fn hash11(p: f32) -> f32 {
    let mut v = (p * 0.1031).fract();
    v *= v + 33.33;
    v *= v + v;
    v.fract()
}

// White noise hash: f32 → vec2 [0,1]
pub fn hash12(p: f32) -> Vec2 {
    let mut v = (Vec2::splat(p) * vec2(0.1031, 0.1030)).fract();
    v += v.dot(v.yx() + 33.33);
    ((v.x + v.y) * v).fract()
}

// White noise hash: f32 → vec3 [0,1]
pub fn hash13(p: f32) -> Vec3 {
    let mut v = (Vec3::splat(p) * vec3(0.1031, 0.1030, 0.1029)).fract();
    v += v.dot(v.yxz() + 33.33);
    ((v.x + v.y + v.z) * v).fract()
}

// White noise hash: vec2 → f32 [0,1]
pub fn hash21(p: Vec2) -> f32 {
    let mut v3 = (p.xyx() * 0.1031).fract();
    v3 += v3.dot(v3.yzx() + 33.33);
    ((v3.x + v3.y) * v3.z).fract()
}

// Blue noise approx.: vec2 → f32 [0,1]
pub fn blue_noise(uv: Vec2) -> f32 {
    let s0 = hash21(uv + vec2(-1.0, 0.0));
    let s1 = hash21(uv + vec2(1.0, 0.0));
    let s2 = hash21(uv + vec2(0.0, 1.0));
    let s3 = hash21(uv + vec2(0.0, -1.0));
    let s = s0 + s1 + s2 + s3;
    hash21(uv) - s * 0.25 + 0.5
}

// Rotate vector v by quaternion q = (xyz, w)
pub fn quat_rotate(q: Vec4, v: Vec3) -> Vec3 {
    let u = q.xyz();
    let uv = u.cross(v);
    v + 2.0 * (q.w * uv + u.cross(uv))
}

// Returns the near (x) and far (y) intersection distances
// If the ray misses, returns vec2(1.0, -1.0)
pub fn intersect_sphere(ro: Vec3, rd: Vec3, radius: f32) -> Vec2 {
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(ro);
    let c = ro.dot(ro) - (radius * radius);
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return vec2(1.0, -1.0);
    }

    let sqrt_disc = disc.sqrt();
    vec2(-b - sqrt_disc, -b + sqrt_disc) / (2.0 * a)
}

// Calculate intersection with a horizontal plane at given height
pub fn intersect_plane(ro: Vec3, rd: Vec3, plane_height: f32) -> f32 {
    if rd.z.abs() < 0.001 {
        return -1.0; // Ray is parallel to plane
    }
    let t = (plane_height - ro.z) / rd.z;
    if t > 0.0 { t } else { -1.0 }
}

// Calculates up to two intersection segments (entry/exit pairs) for a ray intersecting a spherical shell
// x, y = first intersection segment (entry, exit)
// z, w = second intersection segment (entry, exit)
// An invalid segment is represented by entry > exit (e.g., 1.0, 0.0)
pub fn ray_shell_intersect(
    ro: Vec3,
    rd: Vec3,
    view: &ViewUniform,
    bottom_altitude: f32,
    top_altitude: f32,
) -> Vec4 {
    let local_ro = ro - view.planet_center();

    // The entry point is the nearest intersection with the top sphere
    let top_radius = PLANET_RADIUS + top_altitude;
    let top_interval = intersect_sphere(local_ro, rd, top_radius);
    if top_interval.x > top_interval.y {
        return vec4(1.0, 0.0, 1.0, 0.0); // If we miss the top sphere, we miss the shell entirely
    }

    // The exit point is the nearest intersection with the bottom sphere
    let bottom_radius = PLANET_RADIUS + bottom_altitude;
    let bottom_interval = intersect_sphere(local_ro, rd, bottom_radius);
    if bottom_interval.x > bottom_interval.y {
        return vec4(top_interval.x, top_interval.y, 1.0, 0.0); // Glancing shot that hits the top layer but misses the bottom, exit is the far side of the top layer
    }

    // The ray hits both spheres, creating two segments through the shell
    // Segment 1: Enters top sphere (near), exits bottom sphere (near)
    // Segment 2: Enters bottom sphere (far), exits top sphere (far)
    vec4(
        top_interval.x,
        bottom_interval.x,
        bottom_interval.y,
        top_interval.y,
    )
}

/// Calculates the world position of the two polar suns and returns the one highest in the sky
pub fn get_sun_position(view: &ViewUniform) -> Vec3 {
    let north_axis = quat_rotate(view.planet_rotation, vec3(0.0, 0.0, 1.0)).normalize();
    let up_vector = view.ro_relative().normalize();
    let is_in_northern_hemisphere = north_axis.dot(up_vector) > 0.0;
    let sun_axis = if is_in_northern_hemisphere {
        north_axis
    } else {
        -north_axis
    };

    let sun_altitude = PLANET_RADIUS + MAGNETOSPHERE_HEIGHT;
    let mut sun_pos = view.planet_center() + sun_axis * sun_altitude;

    let blend = (view.latitude().abs() / 0.35).saturate();
    sun_pos.z += 0.0.lerp(sun_altitude * -2.0, 1.0 - blend);

    sun_pos
}
