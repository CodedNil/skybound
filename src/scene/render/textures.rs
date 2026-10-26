use crate::scene::{
    PLANET_RADIUS,
    atmosphere::{
        POLE_RADIANCE, TRANSMITTANCE_SIZE, cached_air_transmittance, get_pole_light_color,
        transmittance_coordinates,
    },
    aur::{EJECTION_WINDOW, Place, ejection_field_with},
    climate::{
        CLOUD_FACE, CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CLOUD_TOP, LIGHT_BOTTOM, LIGHT_FACE,
        LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH, atlas_texel,
    },
    math::{intersect_sphere, rotated},
    planet::{REACTION_RADIUS, light_towards},
};
use spirv_std::{
    Image, Sampler,
    glam::{IVec2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles},
    num_traits::Float,
};

pub struct AtmosphereData {
    /// Sky radiance along the pixel ray.
    pub sky: Vec3,
    /// Light scattered into the air along the ray. Looking down, the sky texture holds
    /// below-horizon radiance that must never show through.
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
    pub shapes: &'a Image!(3D, type=f32, sampled=true),
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
    /// See [`ejection_field_with`], for a place near the camera.
    pub fn ejection_field(&self, place: Place) -> (Vec4, f32) {
        let (cell, position) = place.arc();
        let mask = self.ejections.fetch(window_texel(cell)).w as u32 >> 1;
        ejection_field_with(cell, position, mask, |cell| {
            let pixel = window_texel(cell);
            let mut event = self.ejections.fetch(pixel);
            event.w = (event.w as u32 & 1) as f32;
            let window = EJECTION_WINDOW as i32;
            [
                event,
                self.ejections.fetch(pixel + IVec2::new(window, 0)),
                self.ejections.fetch(pixel + IVec2::new(window * 2, 0)),
            ]
        })
    }
    /// See [`ejection_clearance`](crate::scene::aur::ejection_clearance); valid anywhere in the
    /// place's arc cell.
    pub fn ejection_clearance(&self, place: Place) -> f32 {
        let (cell, _) = place.arc();
        self.ejections
            .fetch(window_texel(cell) + IVec2::new(EJECTION_WINDOW as i32 * 2, 0))
            .x
    }
    /// The cloud map over a column, see
    /// [`CloudCover::texels`](crate::scene::climate::CloudCover::texels);
    /// then how far convective cloud near it can reach, and the storm tower
    /// over it, if any.
    pub fn cloud_column(&self, up: Vec3) -> [Vec4; 4] {
        let texel = atlas_texel(up, CLOUD_FACE);
        let uv = texel / Vec2::new(CLOUD_MAP_WIDTH as f32, CLOUD_MAP_HEIGHT as f32);
        [
            self.cloud_cover.sample_by_lod(*self.sampler, uv, 0.0),
            self.cloud_layers.sample_by_lod(*self.sampler, uv, 0.0),
            self.cloud_bounds.fetch(texel.as_ivec2()),
            self.cloud_towers.fetch(texel.as_ivec2()),
        ]
    }
    /// Cloud optical depth from a point towards its pole light, beyond the near shadow each sample
    /// traces itself and in full; and down to the aur.
    pub fn light_depth(&self, point: Vec3) -> Vec3 {
        let mut point = point;
        let mut radius = point.length();
        if radius < PLANET_RADIUS + LIGHT_BOTTOM {
            // Beneath every cloud the shadows only run on along the light: read them where this
            // point's ray towards the light rises through the volume's floor, with nothing below
            // to shade.
            let light = light_towards(point);
            let rise = intersect_sphere(point, light, PLANET_RADIUS + LIGHT_BOTTOM).y;
            if rise <= 0.0 {
                return Vec3::ZERO;
            }
            point += light * rise;
            radius = PLANET_RADIUS + LIGHT_BOTTOM;
        }
        let texel = atlas_texel(point / radius, LIGHT_FACE);
        let layer = ((radius - PLANET_RADIUS - LIGHT_BOTTOM) / (CLOUD_TOP - LIGHT_BOTTOM)
            * (LIGHT_LAYERS - 1) as f32)
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

    /// The clouds' own noise: cells, puffs, billows and a warp.
    pub fn shapes_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.shapes
            .sample_by_lod(*self.sampler, p, (footprint * 128.0).max(1.0).log2())
    }

    pub fn details_filtered(&self, p: Vec3, footprint: f32) -> Vec4 {
        self.details
            .sample_by_lod(*self.sampler, p, (footprint * 64.0).max(1.0).log2())
    }
}

/// The lookup texel holding an arc cell: its coordinate modulo the window.
fn window_texel(cell: Vec2) -> IVec2 {
    let window = EJECTION_WINDOW as f32;
    (cell - (cell / window).floor() * window).as_ivec2()
}
