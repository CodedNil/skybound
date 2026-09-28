use crate::PLANET_RADIUS;
use spirv_std::glam::{
    FloatExt, Quat, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, vec2, vec3, vec4,
};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

pub const MAGNETOSPHERE_HEIGHT: f32 = 400_000.0;

pub struct AtmosphereData {
    pub sun_pos: Vec3,
    pub sky: Vec3,
    pub sun: Vec3,
    pub ambient: Vec3,
}

pub struct Textures<'a> {
    pub base: &'a Image!(3D, type=f32, sampled=true),
    pub details: &'a Image!(3D, type=f32, sampled=true),
    pub weather: &'a Image!(2D, type=f32, sampled=true),
    pub extra: &'a Image!(2D, type=f32, sampled=true),
    pub sampler: &'a Sampler,
}

impl Textures<'_> {
    pub fn base(&self, p: Vec3) -> Vec4 {
        self.base.sample(*self.sampler, p)
    }

    pub fn details(&self, p: Vec3) -> Vec4 {
        self.details.sample(*self.sampler, p)
    }

    pub fn weather(&self, p: Vec2) -> Vec4 {
        self.weather.sample(*self.sampler, p)
    }

    pub fn extra(&self, p: Vec2) -> Vec4 {
        self.extra.sample(*self.sampler, p)
    }
}

pub fn hash12(p: f32) -> Vec2 {
    let mut v = (Vec2::splat(p) * vec2(0.1031, 0.1030)).fract_gl();
    v += v.dot(v.yx() + 33.33);
    ((v.x + v.y) * v).fract_gl()
}

pub fn hash13(p: f32) -> Vec3 {
    let mut v = (Vec3::splat(p) * vec3(0.1031, 0.1030, 0.1029)).fract_gl();
    v += v.dot(v.yxz() + 33.33);
    ((v.x + v.y + v.z) * v).fract_gl()
}

pub fn hash21(p: Vec2) -> f32 {
    let mut v3 = (p.xyx() * 0.1031).fract_gl();
    v3 += v3.dot(v3.yzx() + 33.33);
    ((v3.x + v3.y) * v3.z).fract()
}

pub fn blue_noise(uv: Vec2) -> f32 {
    let s0 = hash21(uv + vec2(-1.0, 0.0));
    let s1 = hash21(uv + vec2(1.0, 0.0));
    let s2 = hash21(uv + vec2(0.0, 1.0));
    let s3 = hash21(uv + vec2(0.0, -1.0));
    let s = s0 + s1 + s2 + s3;
    hash21(uv) - s * 0.25 + 0.5
}

/// Ray-sphere intersection. Returns `(near, far)` where `near > far` signals a miss.
pub fn intersect_sphere(ro: Vec3, rd: Vec3, radius: f32) -> Vec2 {
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(ro);
    let c = ro.dot(ro) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2(1.0, -1.0);
    }
    let sqrt_disc = disc.sqrt();
    vec2(-b - sqrt_disc, -b + sqrt_disc) / (2.0 * a)
}

/// Returns up to two entry/exit segments through a spherical shell.
/// `xy` = first segment, `zw` = second; `entry > exit` signals an invalid segment.
pub fn ray_shell_intersect(
    ro: Vec3,
    rd: Vec3,
    planet_center: Vec3,
    bottom_altitude: f32,
    top_altitude: f32,
) -> Vec4 {
    let local_ro = ro - planet_center;
    let top_radius = PLANET_RADIUS + top_altitude;
    let top_interval = intersect_sphere(local_ro, rd, top_radius);
    if top_interval.x > top_interval.y {
        return vec4(1.0, 0.0, 1.0, 0.0);
    }
    let bottom_radius = PLANET_RADIUS + bottom_altitude;
    let bottom_interval = intersect_sphere(local_ro, rd, bottom_radius);
    if bottom_interval.x > bottom_interval.y {
        return vec4(top_interval.x, top_interval.y, 1.0, 0.0);
    }
    vec4(
        top_interval.x,
        bottom_interval.x,
        bottom_interval.y,
        top_interval.y,
    )
}

/// Compute the sun's world-space position from view uniforms.
pub fn get_sun_position(
    planet_center: Vec3,
    planet_rotation: Quat,
    ro_relative: Vec3,
    latitude: f32,
) -> Vec3 {
    let north_axis = planet_rotation.mul_vec3(Vec3::Z).normalize();
    let up_vector = ro_relative.normalize();
    let sun_axis = if north_axis.dot(up_vector) > 0.0 {
        north_axis
    } else {
        -north_axis
    };
    let sun_altitude = PLANET_RADIUS + MAGNETOSPHERE_HEIGHT;
    let mut sun_pos = planet_center + sun_axis * sun_altitude;
    let blend = (latitude.abs() / 0.35).saturate();
    sun_pos.z += 0.0.lerp(sun_altitude * -2.0, 1.0 - blend);
    sun_pos
}
