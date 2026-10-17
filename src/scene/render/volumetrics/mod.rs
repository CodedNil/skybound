pub(super) mod aur_ocean;
pub(super) mod clouds;
pub(super) mod native;
pub(super) mod poles;

use super::{
    cache,
    utils::{AtmosphereData, Textures},
};
use crate::scene::{PLANET_RADIUS, ViewUniform, volume::simpson};
use spirv_std::glam::{Vec3, Vec4, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

#[derive(Default)]
pub(super) struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

#[derive(Copy, Clone)]
pub struct RaymarchResult {
    pub color: Vec4,
    pub depth: f32,
}
impl RaymarchResult {
    pub(super) fn empty(limit: f32) -> Self {
        Self {
            color: Vec4::W,
            depth: limit,
        }
    }
    pub(super) fn absorb(&mut self, radiance: Vec3, optical_depth: f32, distance: f32) {
        let transmission = (-optical_depth.max(0.0)).exp();
        let weight = self.color.w * (1.0 - transmission);
        let opacity = 1.0 - self.color.w;
        if opacity + weight > 0.00001 {
            self.depth = (self.depth * opacity + distance * weight) / (opacity + weight);
        }
        self.color = (self.color.xyz() + radiance * weight).extend(self.color.w * transmission);
        self.color = crate::scene::volume::finish_opaque(self.color, radiance);
    }
}

pub(super) fn cloud_lighting(
    position: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    density: f32,
    textures: &Textures,
    records: &[crate::scene::cloud::Cloud],
    grid: &[u32],
) -> Vec3 {
    let source = crate::scene::polar::light_position(position);
    let light = (source - position).normalize();
    let mut optical = density * 100.0;
    let mut distance = 150.0;
    let mut step = 300.0;
    for _ in 0..4 {
        optical += clouds::extinction(
            position + light * distance,
            view,
            step * 0.3,
            textures,
            records,
            grid,
        ) * step;
        distance += step;
        step *= 2.5;
    }
    let cosine = direction.dot(light);
    let denominator = (1.0 + 0.6 * 0.6 - 1.2 * cosine).max(0.01);
    let phase = (1.0 - 0.6 * 0.6) / (denominator * denominator.sqrt());
    let sunlight = textures.pole_light(position, source) * view.sky_horizon.y;
    let direct = (-optical).exp() * (0.12 * phase + 0.04);
    let multiple = (-optical * 0.12).exp() * 0.18;
    let altitude = position.length() - PLANET_RADIUS;
    let mut aur_depth = density * 160.0;
    if altitude < 25_000.0 {
        aur_depth += clouds::extinction(
            position - position.normalize() * 400.0,
            view,
            100.0,
            textures,
            records,
            grid,
        ) * 640.0;
    }
    let (_, palette) = cache::sample(position, view, 4000.0, textures);
    let source = if palette.w > 0.0 {
        palette.xyz()
    } else {
        vec3(0.40, 0.34, 0.52) * view.sky_horizon.y
    };
    let upward = source
        * (-altitude.max(0.0) / 16_000.0).exp()
        * ((-aur_depth).exp() * 0.7 + (-aur_depth * 0.2).exp() * 0.3);
    sunlight * (direct + multiple)
        + upward
        + view.ambient_color.xyz() * 0.45
        + vec3(0.025, 0.035, 0.055)
}

pub fn short_volume(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    length: f32,
    textures: &Textures,
    records: &[crate::scene::cloud::Cloud],
    grid: &[u32],
) -> RaymarchResult {
    let origin = view.to_planet(ro);
    let direction = view.planet_rotation.conjugate() * rd;
    let middle = origin + direction * (length * 0.5);
    let footprint = (length / 8.0).max(4.0);
    let a = clouds::extinction(origin, view, footprint, textures, records, grid);
    let m = clouds::extinction(middle, view, footprint, textures, records, grid);
    let b = clouds::extinction(
        origin + direction * length,
        view,
        footprint,
        textures,
        records,
        grid,
    );
    let optical = simpson(a, m, b, length);
    let air = textures.air_transmittance(origin, direction, length);
    let transmission = (-optical).exp() * air;
    let lighting = if optical > 0.0001 {
        cloud_lighting(
            middle,
            direction,
            view,
            a.max(m).max(b),
            textures,
            records,
            grid,
        )
    } else {
        atmosphere.sky
    };
    RaymarchResult {
        color: (lighting * (1.0 - transmission)).extend(transmission),
        depth: length,
    }
}
