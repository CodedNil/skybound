#![no_std]

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles, vec3};

pub const PLANET_RADIUS: f32 = 1_000_000.0;

pub const NUM_STRINGS: usize = 5;
pub const TANKS_PER_STRING: usize = 3;
pub const NODES_PER_STRING: usize = TANKS_PER_STRING + 1; // attachment + 3 tanks
pub const TOTAL_BEAD_NODES: usize = NUM_STRINGS * NODES_PER_STRING; // 20

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

/// Per-ship uniform passed to the ship render pass.
///
/// All positions are in camera-snap-space (same coordinate frame as `ViewUniform.world_position`).
#[repr(C)]
#[derive(Copy, Clone, Default)]
#[cfg_attr(
    feature = "cpu",
    derive(bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType)
)]
pub struct ShipUniform {
    /// xyz = core world position, w = shield phase [0..1] (negative = ship invalid/hidden)
    pub core_position: Vec4,
    /// Core orientation quaternion (x, y, z, w)
    pub core_rotation: Vec4,
    /// Flattened bead nodes: [string * NODES_PER_STRING + node]
    /// node 0 = attachment point on core, nodes 1..NODES_PER_STRING = tank positions.
    /// xyz = world position, w = unused
    pub bead_positions: [Vec4; TOTAL_BEAD_NODES],
}
