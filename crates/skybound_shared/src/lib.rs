#![no_std]

use glam::{Mat4, Vec2, Vec3, Vec4};

#[cfg_attr(
    feature = "cpu",
    derive(bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub world_from_view: Mat4,
    pub view_from_world: Mat4,
    pub clip_from_view: Mat4,
    pub view_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub world_from_clip_unjittered: Mat4,

    pub time: f32,
    pub frame_count: u32,
    pub camera_offset: Vec2,

    pub world_position: Vec3,
    pub padding1: u32,

    pub planet_center: Vec3,
    pub planet_radius: f32,
    pub planet_rotation: Vec4,

    pub latitude: f32,
    pub longitude: f32,
    pub padding2: Vec2,
}
