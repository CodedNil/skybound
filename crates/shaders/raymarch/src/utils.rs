pub use skybound_shared::CloudsViewUniform as View;
use spirv_std::glam::{Mat4, Quat, Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::num_traits::Float;

pub const MAGNETOSPHERE_HEIGHT: f32 = 400_000.0;

pub struct AtmosphereData {
    pub sun_pos: Vec3,
    pub sky: Vec3,
    pub sun: Vec3,
    pub ambient: Vec3,
}

pub fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    (((x - a) / (b - a)) * (d - c)) + c
}

pub fn hash21(p: Vec2) -> f32 {
    let mut v3 = (Vec3::new(p.x, p.y, p.x) * 0.1031).fract();
    v3 += v3.dot(Vec3::new(v3.y, v3.z, v3.x) + 33.33);
    ((v3.x + v3.y) * v3.z).fract()
}

pub fn blue_noise(uv: Vec2) -> f32 {
    let s0 = hash21(uv + Vec2::new(-1.0, 0.0));
    let s1 = hash21(uv + Vec2::new(1.0, 0.0));
    let s2 = hash21(uv + Vec2::new(0.0, 1.0));
    let s3 = hash21(uv + Vec2::new(0.0, -1.0));
    let s = s0 + s1 + s2 + s3;
    hash21(uv) - s * 0.25 + 0.5
}

pub fn quat_rotate(q: Vec4, v: Vec3) -> Vec3 {
    let quat = Quat::from_vec4(q);
    quat * v
}

pub fn intersect_sphere(ro: Vec3, rd: Vec3, radius: f32) -> Vec2 {
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(ro);
    let c = ro.dot(ro) - (radius * radius);
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return Vec2::new(1.0, -1.0);
    }

    let sqrt_disc = disc.sqrt();
    Vec2::new((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
}

pub fn intersect_plane(ro: Vec3, rd: Vec3, plane_height: f32) -> f32 {
    if rd.z.abs() < 0.001 {
        return -1.0;
    }
    let t = (plane_height - ro.z) / rd.z;
    if t > 0.0 { t } else { -1.0 }
}

pub fn ray_shell_intersect(
    ro: Vec3,
    rd: Vec3,
    view: &View,
    bottom_altitude: f32,
    top_altitude: f32,
) -> Vec4 {
    let local_ro = ro - view.planet_center;
    let top_radius = view.planet_radius + top_altitude;
    let top_interval = intersect_sphere(local_ro, rd, top_radius);
    if top_interval.x > top_interval.y {
        return Vec4::new(1.0, 0.0, 1.0, 0.0);
    }
    let bottom_radius = view.planet_radius + bottom_altitude;
    let bottom_interval = intersect_sphere(local_ro, rd, bottom_radius);
    if bottom_interval.x > bottom_interval.y {
        return Vec4::new(top_interval.x, top_interval.y, 1.0, 0.0);
    }
    Vec4::new(
        top_interval.x,
        bottom_interval.x,
        bottom_interval.y,
        top_interval.y,
    )
}

pub fn get_sun_position(view: &View) -> Vec3 {
    let north_axis = quat_rotate(view.planet_rotation, Vec3::new(0.0, 0.0, 1.0)).normalize();
    let up_vector = (view.world_position - view.planet_center).normalize();
    let is_in_northern_hemisphere = north_axis.dot(up_vector) > 0.0;
    let sun_axis = if is_in_northern_hemisphere {
        north_axis
    } else {
        -north_axis
    };

    let sun_altitude = view.planet_radius + MAGNETOSPHERE_HEIGHT;
    let mut sun_pos = view.planet_center + sun_axis * sun_altitude;

    let blend = (view.latitude.abs() / 0.35).clamp(0.0, 1.0);
    sun_pos.z += (sun_altitude * -2.0) * (1.0 - blend);

    sun_pos
}

pub fn position_ndc_to_world(ndc_pos: Vec3, world_from_clip: Mat4) -> Vec3 {
    let world_pos = world_from_clip * ndc_pos.extend(1.0);
    world_pos.xyz() / world_pos.w
}

pub fn uv_to_ndc(uv: Vec2) -> Vec2 {
    uv * Vec2::new(2.0, -2.0) + Vec2::new(-1.0, 1.0)
}
