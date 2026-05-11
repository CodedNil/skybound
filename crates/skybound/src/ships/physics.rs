use bevy::{prelude::*, render::extract_resource::ExtractResource};
use skybound_shared::{
    NODES_PER_STRING, NUM_STRINGS, ShipUniform, TANKS_PER_STRING, TOTAL_BEAD_NODES,
};
use std::f32::consts::PI;

use crate::ships::player::PlayerShip;

// ── Physics constants ────────────────────────────────────────────────────────

const LINK_LENGTH: f32 = 14.0;
/// Radius of the ring on the core's back face where strings attach.
const ATTACH_RADIUS: f32 = 3.0;
/// How far behind the core (+Z in core-local) the attachment ring sits.
const CORE_BACK: f32 = 2.5;
/// Per-frame velocity damping (≈ 3% per frame ≈ ~1.8 s half-life at 60 fps).
const DAMPING: f32 = 0.97;
/// Gauss-Seidel relaxation passes per update.
const RELAX_ITERS: usize = 5;

// ── Types ────────────────────────────────────────────────────────────────────

/// Verlet particle chain for one string.
pub struct BeadChain {
    /// Attachment point in core-LOCAL space (fixed, follows core rigidly).
    pub local_attach: Vec3,
    /// World-space positions of the free bead nodes (one per tank).
    pub positions: [Vec3; TANKS_PER_STRING],
    pub prev_positions: [Vec3; TANKS_PER_STRING],
}

impl BeadChain {
    fn new(string_idx: usize, core_world_pos: Vec3, core_world_rot: Quat) -> Self {
        let angle = string_idx as f32 * 2.0 * PI / NUM_STRINGS as f32;
        // Attachment in core-local: ring in XY plane, at +Z (behind the ship).
        // In core-local space: forward = -Z, behind = +Z, up = +Y.
        let local_attach = Vec3::new(
            ATTACH_RADIUS * angle.cos(),
            ATTACH_RADIUS * angle.sin(),
            CORE_BACK,
        );
        let attach_world = core_world_pos + core_world_rot.mul_vec3(local_attach);
        // Trail direction: +Z in core-local = behind the ship in world space.
        let trail = core_world_rot.mul_vec3(Vec3::Z);
        let positions =
            std::array::from_fn(|i| attach_world + trail * LINK_LENGTH * (i + 1) as f32);
        Self {
            local_attach,
            positions,
            prev_positions: positions,
        }
    }

    fn world_attach(&self, core_pos: Vec3, core_rot: Quat) -> Vec3 {
        core_pos + core_rot.mul_vec3(self.local_attach)
    }

    /// Verlet integration + distance constraint relaxation.
    fn update(&mut self, attach: Vec3, dt: f32) {
        // Verlet step with damping.
        for i in 0..TANKS_PER_STRING {
            let vel = (self.positions[i] - self.prev_positions[i]) * DAMPING;
            let new = self.positions[i] + vel * (dt / (1.0 / 60.0)); // normalised to 60 fps
            self.prev_positions[i] = self.positions[i];
            self.positions[i] = new;
        }

        // Gauss-Seidel distance constraints.
        for _ in 0..RELAX_ITERS {
            // First bead must be LINK_LENGTH from attach.
            let d = self.positions[0] - attach;
            let len = d.length();
            self.positions[0] = attach
                + if len > 0.001 {
                    d / len * LINK_LENGTH
                } else {
                    Vec3::Z * LINK_LENGTH
                };

            // Each subsequent bead must be LINK_LENGTH from the one before it.
            for i in 1..TANKS_PER_STRING {
                let prev = self.positions[i - 1];
                let d = self.positions[i] - prev;
                let len = d.length();
                self.positions[i] = prev
                    + if len > 0.001 {
                        d / len * LINK_LENGTH
                    } else {
                        Vec3::Z * LINK_LENGTH
                    };
            }
        }
    }

    /// Apply a global offset to all bead positions (used for coordinate-snap correction).
    pub fn apply_offset(&mut self, offset: Vec3) {
        for p in self.positions.iter_mut() {
            *p += offset;
        }
        for p in self.prev_positions.iter_mut() {
            *p += offset;
        }
    }
}

// ── Resource ─────────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct ShipPhysics {
    pub chains: Option<[BeadChain; NUM_STRINGS]>,
}

// ── Extracted data ────────────────────────────────────────────────────────────

/// Built each frame from the physics state, then extracted into the render world.
#[derive(Resource, Clone, Default, ExtractResource)]
pub struct ExtractedShipData {
    pub uniform: ShipUniform,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

pub struct ShipPhysicsPlugin;

impl Plugin for ShipPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShipPhysics>()
            .init_resource::<ExtractedShipData>()
            .add_systems(PreUpdate, update_ship_physics);
    }
}

// ── Systems ───────────────────────────────────────────────────────────────────

pub fn update_ship_physics(
    time: Res<Time>,
    mut physics: ResMut<ShipPhysics>,
    mut extracted: ResMut<ExtractedShipData>,
    ship_query: Query<&Transform, With<PlayerShip>>,
) {
    let Ok(ship_transform) = ship_query.single() else {
        extracted.uniform.core_position.w = -1.0; // mark invalid
        return;
    };

    let core_pos = ship_transform.translation;
    let core_rot = ship_transform.rotation;
    let dt = time.delta_secs().min(0.05);

    // Initialise chains on first valid frame.
    if physics.chains.is_none() {
        physics.chains = Some(std::array::from_fn(|i| {
            BeadChain::new(i, core_pos, core_rot)
        }));
    }

    let chains = physics.chains.as_mut().unwrap();

    for chain in chains.iter_mut() {
        let attach = chain.world_attach(core_pos, core_rot);
        chain.update(attach, dt);
    }

    // Build the ShipUniform.
    let mut bead_positions = [Vec4::ZERO; TOTAL_BEAD_NODES];
    for (s, chain) in chains.iter().enumerate() {
        let base = s * NODES_PER_STRING;
        // Node 0: world-space attachment point.
        bead_positions[base] = chain.world_attach(core_pos, core_rot).extend(0.0);
        // Nodes 1..NODES_PER_STRING: tank positions.
        for t in 0..TANKS_PER_STRING {
            bead_positions[base + t + 1] = chain.positions[t].extend(0.0);
        }
    }

    extracted.uniform = ShipUniform {
        core_position: core_pos.extend(1.0), // w=1 = valid/shield open
        core_rotation: Vec4::from(core_rot),
        bead_positions,
    };
}
