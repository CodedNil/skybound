use crate::scene::{
    atmosphere::{get_pole_light_color, render_sky},
    aur::ambient,
    life::CreatureInstance,
    planet::{get_pole_position, surface_coordinates},
};
use spirv_std::glam::{Mat4, Quat, Vec2, Vec3, Vec4, Vec4Swizzles};

pub mod atmosphere;
pub mod aur;
pub mod cache;
pub mod climate;
pub mod life;
pub mod math;
pub mod planet;
#[cfg(target_arch = "spirv")]
pub mod render;

pub const PLANET_RADIUS: f32 = 750_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    /// The projection without this frame's jitter, which motion is measured in, as
    /// upscalers expect.
    pub unjittered_clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub direction_from_clip_unjittered: Mat4,
    pub direction_from_clip: Mat4,
    pub world_position: Vec4,
    pub camera_position: Vec4,
    pub planet_center_value: Vec4,
    pub planet_rotation: Quat,
    pub times: Vec4,
    pub weather: Vec4,
    pub pole_position: Vec4,
    pub pole_color: Vec4,
    pub ambient_color: Vec4,
    pub sky_horizon: Vec4,
    pub cache_origin: Vec4,
    pub cache_east: Vec4,
    pub cache_north: Vec4,
    /// The camera's spire cells and offsets on each face, from `aur::spire_origins`.
    pub spire_origins: [Vec4; 6],
    pub aur_flashes: [[Vec4; 2]; 32],
    pub aur_bolts: [[Vec4; 2]; 32],
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub view: ViewUniform,
    pub players: [CreatureInstance; 2],
    pub viewport: Vec4,
}

impl ViewUniform {
    /// Reconstruct a ray without subtracting large world positions near the camera.
    pub fn ray_direction(&self, uv: Vec2, jittered: bool) -> Vec3 {
        let matrix = if jittered {
            self.direction_from_clip
        } else {
            self.direction_from_clip_unjittered
        };
        let clip = (uv * Vec2::new(2.0, -2.0) + Vec2::new(-1.0, 1.0))
            .extend(0.01)
            .extend(1.0);
        (matrix * clip).xyz().normalize()
    }

    pub fn up(&self, local: Vec3) -> Vec3 {
        (local - self.planet_center()).normalize()
    }

    pub fn down(&self, local: Vec3) -> Vec3 {
        -self.up(local)
    }

    pub fn prepare_atmosphere(&mut self) {
        let pole = get_pole_position(
            self.planet_center(),
            self.planet_rotation,
            self.ro_relative(),
        );
        let up = (self.world_position.xyz() - self.planet_center()).normalize();
        let source = pole - self.planet_center();
        let ro = self.ro_relative();
        self.pole_position = pole.extend(0.0);
        self.pole_color = get_pole_light_color(ro, source).extend(0.0);
        self.ambient_color =
            (render_sky(up, ro, source) * 0.7 + render_sky(-up, ro, source) * 0.15).extend(0.0);
        let position = self.surface_position(self.world_position.xyz());
        self.ambient_color += ambient(position, self.time()).extend(0.0);
    }

    pub fn planet_center(&self) -> Vec3 {
        self.planet_center_value.xyz()
    }

    pub fn ro_relative(&self) -> Vec3 {
        self.world_position.xyz() - self.planet_center()
    }

    /// The camera's position in the planet, which turns under the scene.
    pub fn camera(&self) -> Vec3 {
        self.to_planet(self.world_position.xyz())
    }

    pub fn to_planet(&self, point: Vec3) -> Vec3 {
        self.planet_rotation.conjugate() * (point - self.planet_center())
    }

    pub fn from_planet(&self, point: Vec3) -> Vec3 {
        self.planet_center() + self.planet_rotation * point
    }

    pub fn surface_position(&self, point: Vec3) -> Vec3 {
        surface_coordinates(self.to_planet(point))
    }

    pub fn altitude(&self) -> f32 {
        self.camera_position.z
    }

    /// The place of a point given by its offset from the camera along the planet's axes: how
    /// everything placed on the face grid is found near the camera.
    pub fn place(&self, offset: Vec3) -> aur::Place {
        aur::place_near(&self.spire_origins, self.camera(), self.altitude(), offset)
    }

    pub fn time(&self) -> f32 {
        self.times.x
    }

    pub fn pole_direction(&self) -> Vec3 {
        let offset = self.pole_position.truncate() - self.world_position.truncate();
        if offset.length_squared() < 0.0001 {
            Vec3::Z
        } else {
            offset.normalize()
        }
    }

    pub fn frame_count(&self) -> f32 {
        self.times.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::planet::REACTION_RADIUS;
    use core::f32::consts::FRAC_PI_2;
    use spirv_std::glam::{camera::rh::proj::directx::perspective_infinite_reverse, vec3};

    #[test]
    fn adjacent_pixel_rays_remain_distinct_after_large_world_translations() {
        let matrix = Mat4::from_quat(Quat::from_rotation_x(0.7))
            * perspective_infinite_reverse(1.0, 16.0 / 9.0, 0.1).inverse();
        let mut view = ViewUniform {
            direction_from_clip: matrix,
            direction_from_clip_unjittered: matrix,
            ..Default::default()
        };
        let uv = Vec2::new(0.7, 0.8);
        let expected = view.ray_direction(uv, false);
        for translation in [
            Vec3::ZERO,
            Vec3::splat(1_500_000.0),
            Vec3::splat(100_000_000.0),
        ] {
            view.world_position = translation.extend(0.0);
            let direction = view.ray_direction(uv, false);
            let adjacent = view.ray_direction(uv + Vec2::X / 3840.0, false);
            assert!(direction.distance(expected) < 1e-7);
            assert!(direction.distance(adjacent) > 0.0001);
        }
    }

    #[test]
    fn polar_light_rises_symmetrically_from_dark_equatorial_conditions() {
        let light = |latitude: f32| {
            let mut view = ViewUniform {
                world_position: vec3(0.0, 0.0, 2400.0).extend(0.0),
                planet_center_value: vec3(0.0, 0.0, -PLANET_RADIUS).extend(0.0),
                planet_rotation: Quat::from_rotation_x(FRAC_PI_2 - latitude),
                ..ViewUniform::default()
            };
            view.prepare_atmosphere();
            view.pole_color.truncate().length()
        };
        let sunrise = (PLANET_RADIUS / REACTION_RADIUS).asin() + 0.05;
        assert!(light(0.0) < 0.001);
        assert!(light(sunrise) > light(0.0) + 0.1);
        assert!(light(1.4) > light(sunrise));
        assert!((light(sunrise) - light(-sunrise)).abs() < 0.01);
    }
}
