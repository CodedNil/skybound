use spirv_std::glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, vec2, vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

use crate::scene::PLANET_RADIUS;
pub use crate::scene::geometry::intersect_sphere;

pub struct AtmosphereData {
    pub sky: Vec3,
    pub ambient: Vec3,
}

pub struct Textures<'a> {
    pub base: &'a Image!(3D, type=f32, sampled=true),
    pub details: &'a Image!(3D, type=f32, sampled=true),
    pub weather: &'a Image!(2D, type=f32, sampled=true),
    pub sampler: &'a Sampler,
}

impl Textures<'_> {
    pub fn billows(&self, p: Vec3, footprint: f32) -> Vec4 {
        let rotated = Vec3::new(
            p.dot(Vec3::new(0.36, 0.48, -0.8)),
            p.dot(Vec3::new(-0.8, 0.6, 0.0)),
            p.dot(Vec3::new(0.48, 0.64, 0.6)),
        ) * 0.754_877
            + Vec3::new(0.37, 0.19, 0.63);
        (self.base_filtered(p, footprint) + self.base_filtered(rotated, footprint * 0.754_877))
            * 0.5
    }

    pub fn base_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.base
            .sample_by_lod(*self.sampler, p, (footprint * 128.0).max(1.0).log2())
    }

    pub fn details_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.details
            .sample_by_lod(*self.sampler, p, (footprint * 64.0).max(1.0).log2())
    }

    pub fn weather(&self, p: Vec2) -> Vec4 {
        self.weather.sample_by_lod(*self.sampler, p, 0.0)
    }

    pub fn weather_on_planet(&self, point: Vec3, time: f32) -> Vec4 {
        let p = crate::scene::climate::Climate::weather_coordinates(point, time);
        let weights = point.normalize() * point.normalize();
        self.weather(p.yz()) * weights.x
            + self.weather(p.zx()) * weights.y
            + self.weather(p.xy()) * weights.z
    }
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
