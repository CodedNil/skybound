use crate::scene::{
    atmosphere::{get_sun_light_color, render_sky},
    geometry::get_sun_position,
    life::CreatureInstance,
};
use spirv_std::glam::{Mat4, Quat, Vec3, Vec4, Vec4Swizzles, vec3};

pub const PLANET_RADIUS: f32 = 1_000_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub world_from_clip_unjittered: Mat4,
    pub world_position: Vec4,
    pub camera_position: Vec4,
    pub planet_rotation: Quat,
    pub times: Vec4,
    pub sun_position: Vec4,
    pub sun_color: Vec4,
    pub ambient_color: Vec4,
    pub sky_horizon: Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub view: ViewUniform,
    pub player: CreatureInstance,
}

impl ViewUniform {
    pub fn prepare_atmosphere(&mut self) {
        let sun = get_sun_position(
            self.planet_center(),
            self.planet_rotation,
            self.ro_relative(),
            self.latitude(),
        );
        let sun_dir = (sun - self.world_position.xyz()).normalize();
        let up = (self.world_position.xyz() - self.planet_center()).normalize();
        let ro = self.ro_relative();
        self.sun_position = sun.extend(0.0);
        self.sun_color = get_sun_light_color(ro, sun_dir).extend(0.0);
        self.ambient_color =
            (render_sky(up, ro, sun_dir) * 0.7 + render_sky(-up, ro, sun_dir) * 0.15).extend(0.0);
    }

    pub fn planet_center(&self) -> Vec3 {
        self.world_position
            .xy()
            .extend(-PLANET_RADIUS - self.world_position.w)
    }

    pub fn ro_relative(&self) -> Vec3 {
        vec3(
            0.0,
            0.0,
            self.world_position.z + PLANET_RADIUS + self.world_position.w,
        )
    }

    pub fn latitude(&self) -> f32 {
        self.camera_position.x
    }

    pub fn longitude(&self) -> f32 {
        self.camera_position.y
    }

    pub fn camera_offset(&self) -> Vec3 {
        vec3(
            self.camera_position.z,
            self.camera_position.w,
            self.world_position.w,
        )
    }

    pub fn time(&self) -> f32 {
        self.times.x
    }

    pub fn frame_count(&self) -> f32 {
        self.times.y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atmosphere_stays_finite_across_altitudes_and_latitudes() {
        for altitude in [-1000.0, 0.0, 20_000.0, 400_000.0] {
            for latitude in [-1.0, 0.0, 1.0] {
                let mut view = ViewUniform {
                    world_position: vec3(0.0, 0.0, altitude).extend(0.0),
                    camera_position: Vec4::new(latitude, 0.0, 0.0, 0.0),
                    planet_rotation: Quat::from_rotation_x(latitude),
                    ..ViewUniform::default()
                };
                view.prepare_atmosphere();
                assert!(view.sun_position.is_finite(), "invalid sun position");
                assert!(view.sun_color.is_finite(), "invalid sun color");
                assert!(view.ambient_color.is_finite(), "invalid ambient color");
            }
        }
    }
}
