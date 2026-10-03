use crate::scene::PLANET_RADIUS;
use spirv_std::glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, vec2, vec3, vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

pub use crate::scene::geometry::{MAGNETOSPHERE_HEIGHT, intersect_sphere};

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
    pub sampler: &'a Sampler,
}

impl Textures<'_> {
    pub fn base(&self, p: Vec3) -> Vec4 {
        self.base.sample_by_lod(*self.sampler, p, 0.0)
    }

    pub fn details(&self, p: Vec3) -> Vec4 {
        self.details.sample_by_lod(*self.sampler, p, 0.0)
    }

    pub fn weather(&self, p: Vec2) -> Vec4 {
        self.weather.sample_by_lod(*self.sampler, p, 0.0)
    }
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
