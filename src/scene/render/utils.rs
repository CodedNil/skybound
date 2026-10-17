use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

use crate::scene::PLANET_RADIUS;
use crate::scene::atmosphere::TRANSMITTANCE_SIZE;
pub use crate::scene::geometry::intersect_sphere;
use crate::scene::noise::rotated;

pub struct AtmosphereData {
    pub sky: Vec3,
}

pub struct Textures<'a> {
    pub cache_shape: &'a Image!(2D, type=f32, sampled=true),
    pub cache_columns: &'a Image!(2D, type=f32, sampled=true),
    pub cached: bool,
    pub base: &'a Image!(3D, type=f32, sampled=true),
    pub details: &'a Image!(3D, type=f32, sampled=true),
    pub weather: &'a Image!(3D, type=f32, sampled=true),
    pub transmittance: &'a Image!(2D, type=f32, sampled=true),
    pub ejections: &'a Image!(2D, type=f32, sampled=true),
    pub sampler: &'a Sampler,
}

pub struct FilteredNoise<'a, 'b> {
    textures: &'a Textures<'b>,
    lod: f32,
}

impl FilteredNoise<'_, '_> {
    pub fn billows(&self, p: Vec3, scale: f32) -> Vec4 {
        let (a, b) = self.pair(p, scale);
        (a + b) * 0.5
    }

    fn pair(&self, p: Vec3, scale: f32) -> (Vec4, Vec4) {
        let rotated = rotated(p);
        let lod = self.lod + scale.log2();
        (
            self.textures
                .base
                .sample_by_lod(*self.textures.sampler, p, lod.max(0.0)),
            self.textures.base.sample_by_lod(
                *self.textures.sampler,
                rotated,
                (lod - 0.405_686_5).max(0.0),
            ),
        )
    }
}

impl Textures<'_> {
    pub fn air_transmittance(&self, origin: Vec3, direction: Vec3, length: f32) -> f32 {
        crate::scene::atmosphere::air::cached_transmittance(
            origin,
            direction,
            length,
            |point, ray| {
                let margin = 0.5 / TRANSMITTANCE_SIZE as f32;
                let uv = crate::scene::atmosphere::transmittance_coordinates(point, ray)
                    .clamp(Vec2::splat(margin), Vec2::splat(1.0 - margin));
                self.transmittance.sample_by_lod(*self.sampler, uv, 0.0).w
            },
        )
    }
    pub fn ejection_field(&self, position: Vec3, limit: f32) -> Vec3 {
        use crate::scene::aur::{
            EJECTION_CELL, EJECTION_GRID_HALF, EJECTION_GRID_WIDTH, ejection_field_with,
        };
        let cell = (position.xy() / EJECTION_CELL).floor().as_ivec2();
        let mask = self
            .ejections
            .fetch(cell + spirv_std::glam::ivec2(EJECTION_GRID_HALF, 1))
            .w as u32
            >> 1;
        ejection_field_with(position, mask, limit, |cell| {
            let pixel = cell.as_ivec2() + spirv_std::glam::ivec2(EJECTION_GRID_HALF, 1);
            let mut event = self.ejections.fetch(pixel);
            event.w = (event.w as u32 & 1) as f32;
            [
                event,
                self.ejections
                    .fetch(pixel + spirv_std::glam::ivec2(EJECTION_GRID_WIDTH as i32, 0)),
            ]
        })
    }
    pub fn noise(&self, footprint: f32) -> FilteredNoise<'_, '_> {
        FilteredNoise {
            textures: self,
            lod: (footprint * 128.0).max(1e-20).log2(),
        }
    }
    pub fn pole_light(&self, point: Vec3, source: Vec3) -> Vec3 {
        if point.length() > crate::scene::polar::REACTION_RADIUS {
            return crate::scene::atmosphere::get_pole_light_color(point, source);
        }
        let direction = (source - point).normalize();
        let margin = 0.5 / TRANSMITTANCE_SIZE as f32;
        let uv = crate::scene::atmosphere::transmittance_coordinates(point, direction)
            .clamp(Vec2::splat(margin), Vec2::splat(1.0 - margin));
        self.transmittance
            .sample_by_lod(*self.sampler, uv, 0.0)
            .xyz()
            * crate::scene::atmosphere::POLE_RADIANCE
    }
    pub fn billows(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.noise(footprint).billows(p, 1.0)
    }

    pub fn base_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.base
            .sample_by_lod(*self.sampler, p, (footprint * 128.0).max(1.0).log2())
    }

    pub fn details_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.details
            .sample_by_lod(*self.sampler, p, (footprint * 64.0).max(1.0).log2())
    }

    pub fn weather(&self, p: Vec3) -> Vec4 {
        self.weather.sample_by_lod(
            *self.sampler,
            (p + 0.5) / crate::scene::climate::WEATHER_SIZE as f32,
            0.0,
        )
    }

    pub fn weather_on_planet(&self, point: Vec3, time: f32) -> Vec4 {
        let p = crate::scene::climate::Climate::weather_coordinates(point, time);
        self.weather(p)
    }
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
