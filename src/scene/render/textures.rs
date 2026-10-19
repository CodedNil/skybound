use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

use crate::scene::PLANET_RADIUS;
use crate::scene::atmosphere::{
    POLE_RADIANCE, TRANSMITTANCE_SIZE, cached_air_transmittance, get_pole_light_color,
    transmittance_coordinates,
};
#[cfg(doc)]
use crate::scene::aur::ejection_clearance;
use crate::scene::aur::{
    EJECTION_CELL, EJECTION_GRID_HALF, EJECTION_GRID_WIDTH, ejection_field_with,
};
#[cfg(doc)]
use crate::scene::climate::CloudCover;
use crate::scene::climate::{
    CLOUD_FACE, CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CLOUD_TOP, LIGHT_FACE, LIGHT_HEIGHT,
    LIGHT_LAYERS, LIGHT_WIDTH, atlas_texel,
};
use crate::scene::math::rotated;
use crate::scene::planet::REACTION_RADIUS;

pub struct AtmosphereData {
    /// Sky radiance along the pixel ray.
    pub sky: Vec3,
    /// Light scattered into the air along the ray. Looking down, the sky
    /// texture holds below-horizon radiance that must never show through.
    pub haze: Vec3,
}

pub struct Textures<'a> {
    pub cache_shape: &'a Image!(2D, type=f32, sampled=true),
    pub cloud_cover: &'a Image!(2D, type=f32, sampled=true),
    pub cloud_layers: &'a Image!(2D, type=f32, sampled=true),
    pub cloud_light: &'a Image!(3D, type=f32, sampled=true),
    pub cloud_bounds: &'a Image!(2D, type=f32, sampled=true),
    pub cloud_towers: &'a Image!(2D, type=f32, sampled=true),
    pub cached: bool,
    pub base: &'a Image!(3D, type=f32, sampled=true),
    pub details: &'a Image!(3D, type=f32, sampled=true),
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
        cached_air_transmittance(origin, direction, length, |point, ray| {
            let margin = 0.5 / TRANSMITTANCE_SIZE as f32;
            let uv = transmittance_coordinates(point, ray)
                .clamp(Vec2::splat(margin), Vec2::splat(1.0 - margin));
            self.transmittance.sample_by_lod(*self.sampler, uv, 0.0).w
        })
    }
    pub fn ejection_field(&self, position: Vec3) -> Vec4 {
        let cell = (position.xy() / EJECTION_CELL).floor().as_ivec2();
        let mask = self
            .ejections
            .fetch(cell + spirv_std::glam::ivec2(EJECTION_GRID_HALF, 1))
            .w as u32
            >> 1;
        ejection_field_with(position, mask, |cell| {
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
    /// See [`ejection_clearance`]; valid anywhere in the position's cell.
    pub fn ejection_clearance(&self, position: Vec3) -> f32 {
        let cell = (position.xy() / EJECTION_CELL).floor().as_ivec2();
        self.ejections
            .fetch(
                cell + spirv_std::glam::ivec2(
                    EJECTION_GRID_HALF + EJECTION_GRID_WIDTH as i32 * 2,
                    1,
                ),
            )
            .x
    }
    /// The cloud map over a column: see [`CloudCover::texels`].
    pub fn cloud_column(&self, up: Vec3) -> [Vec4; 2] {
        let uv = atlas_texel(up, CLOUD_FACE)
            / Vec2::new(CLOUD_MAP_WIDTH as f32, CLOUD_MAP_HEIGHT as f32);
        [
            self.cloud_cover.sample_by_lod(*self.sampler, uv, 0.0),
            self.cloud_layers.sample_by_lod(*self.sampler, uv, 0.0),
        ]
    }
    /// How far convective cloud near a column can reach, and the storm tower
    /// over it, if any.
    pub fn cloud_reach(&self, up: Vec3) -> [Vec4; 2] {
        let texel = atlas_texel(up, CLOUD_FACE).as_ivec2();
        [
            self.cloud_bounds.fetch(texel),
            self.cloud_towers.fetch(texel),
        ]
    }
    /// Cloud optical depth from a point towards its pole light, beyond the
    /// near shadow each sample traces itself and in full; and down to the aur.
    pub fn light_depth(&self, point: Vec3) -> Vec3 {
        let radius = point.length();
        let texel = atlas_texel(point / radius, LIGHT_FACE);
        let layer = ((radius - PLANET_RADIUS) / CLOUD_TOP * (LIGHT_LAYERS - 1) as f32)
            .clamp(0.0, (LIGHT_LAYERS - 1) as f32);
        let uvw = Vec3::new(
            texel.x / LIGHT_WIDTH as f32,
            texel.y / LIGHT_HEIGHT as f32,
            (layer + 0.5) / LIGHT_LAYERS as f32,
        );
        self.cloud_light
            .sample_by_lod(*self.sampler, uvw, 0.0)
            .xyz()
    }
    pub fn noise(&self, footprint: f32) -> FilteredNoise<'_, '_> {
        FilteredNoise {
            textures: self,
            lod: (footprint * 128.0).max(1e-20).log2(),
        }
    }
    pub fn pole_light(&self, point: Vec3, source: Vec3) -> Vec3 {
        if point.length() > REACTION_RADIUS {
            return get_pole_light_color(point, source);
        }
        let direction = (source - point).normalize();
        let margin = 0.5 / TRANSMITTANCE_SIZE as f32;
        let uv = transmittance_coordinates(point, direction)
            .clamp(Vec2::splat(margin), Vec2::splat(1.0 - margin));
        self.transmittance
            .sample_by_lod(*self.sampler, uv, 0.0)
            .xyz()
            * POLE_RADIANCE
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
}
