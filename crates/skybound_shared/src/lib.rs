#![no_std]

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles, vec3};

pub const PLANET_RADIUS: f32 = 1_000_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default)]
#[cfg_attr(
    feature = "cpu",
    derive(bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)
)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub world_from_view: Mat4,
    pub view_from_world: Mat4,
    pub clip_from_view: Mat4,
    pub view_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub world_from_clip_unjittered: Mat4,
    pub world_position: Vec4, // xyz = ray origin, w = altitude camera-snap accumulator
    pub camera_position: Vec4, // latitude, longitude, xy camera-snap accumulators
    pub planet_rotation: Vec4,
    pub times: Vec4, // time, frame_count
}

impl ViewUniform {
    /// Planet centre in camera-local space, accounting for the accumulated altitude offset.
    pub fn planet_center(&self) -> Vec3 {
        self.world_position
            .xy()
            .extend(-PLANET_RADIUS - self.world_position.w)
    }

    /// Camera position relative to planet centre, used to derive the local "up" direction.
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

    /// Accumulated camera-snap offset: xy = surface position, z = altitude.
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
