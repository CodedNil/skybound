#![no_std]

use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};

pub const PLANET_RADIUS: f32 = 1_000_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default)]
#[cfg_attr(feature = "cpu", derive(bevy::render::render_resource::ShaderType))]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub world_from_view: Mat4,
    pub view_from_world: Mat4,
    pub clip_from_view: Mat4,
    pub view_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub world_from_clip_unjittered: Mat4,
    pub world_position: Vec4, // Ray origin

    pub camera_position: Vec4, // latitude, longitude, offset_x, offset_y
    pub planet_rotation: Vec4,
    pub times: Vec4, // time, frame_count
}

// Add function planet_center to viewuniform
impl ViewUniform {
    pub fn planet_center(&self) -> Vec3 {
        self.world_position.xy().extend(-PLANET_RADIUS)
    }

    pub fn ro_relative(&self) -> Vec3 {
        self.world_position.xyz() - self.planet_center()
    }

    pub fn latitude(&self) -> f32 {
        self.camera_position.x
    }

    pub fn longitude(&self) -> f32 {
        self.camera_position.y
    }

    pub fn camera_offset(&self) -> Vec2 {
        self.camera_position.zy()
    }

    pub fn time(&self) -> f32 {
        self.times.x
    }

    pub fn frame_count(&self) -> f32 {
        self.times.y
    }
}
