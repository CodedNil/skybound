use bevy::{
    anti_alias::dlss::{Dlss, DlssPerfQualityMode, DlssSuperResolutionFeature},
    camera::Hdr,
    core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
    post_process::bloom::Bloom,
    prelude::*,
};
use skybound_shared::PLANET_RADIUS;
use std::f32::consts::FRAC_PI_4;

use crate::ships::{physics::ShipPhysics, player::PlayerShip};

// --- Constants ---
const CAMERA_RESET_THRESHOLD: f32 = 50_000.0;

// --- Components ---
#[derive(Resource)]
pub struct WorldData {
    pub camera_offset: Vec3,
}
impl Default for WorldData {
    /// Initialize world data with a quantized camera offset based on planet radius.
    fn default() -> Self {
        let offset = Quat::from_rotation_x(FRAC_PI_4).mul_vec3(Vec3::Z * PLANET_RADIUS);
        Self {
            camera_offset: Vec3::new(
                offset.x - (offset.x % CAMERA_RESET_THRESHOLD),
                offset.y - (offset.y % CAMERA_RESET_THRESHOLD),
                0.0,
            ),
        }
    }
}
impl WorldData {
    /// Calculates the latitude at a given position.
    pub fn latitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Z);
        v_local.z.clamp(-1.0, 1.0).asin()
    }

    /// Calculates the longitude at a given position.
    pub fn longitude(&self, pos: Vec3) -> f32 {
        let v_local = self.planet_rotation(pos).conjugate().mul_vec3(Vec3::Z);
        if v_local.x.abs() < f32::EPSILON && v_local.y.abs() < f32::EPSILON {
            0.0
        } else {
            v_local.x.atan2(-v_local.y)
        }
    }

    /// Compute a rotation quaternion from an X/Y translation on the planet surface.
    fn rotation_from_translation(translation: Vec3) -> Quat {
        let delta_xy = translation.xy();
        if delta_xy.length_squared() > f32::EPSILON {
            Quat::from_axis_angle(
                Vec3::new(delta_xy.y, -delta_xy.x, 0.0).normalize(),
                delta_xy.length() / PLANET_RADIUS,
            )
        } else {
            Quat::IDENTITY
        }
    }

    /// Return the planet rotation quaternion at the given local position.
    pub fn planet_rotation(&self, pos: Vec3) -> Quat {
        let flat_offset = Vec3::new(self.camera_offset.x, self.camera_offset.y, 0.0);
        Self::rotation_from_translation(flat_offset + pos)
    }
}

// --- Plugin ---
pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    /// Initialize world resources and systems.
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldData>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

/// Spawns the main camera. Transform is set each frame by the ship follow system.
fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Projection::default(),
        NormalPrepass,
        DepthPrepass,
        MotionVectorPrepass,
        Msaa::Off,
        Dlss::<DlssSuperResolutionFeature> {
            perf_quality_mode: DlssPerfQualityMode::UltraPerformance,
            reset: false,
            _phantom_data: std::marker::PhantomData,
        },
        Hdr,
        Bloom::NATURAL,
        // Initial position behind the ship's spawn point; the follow system overwrites this.
        Transform::from_xyz(0.0, -26.0, 20.0).looking_at(Vec3::new(0.0, 4.0, 12.0), Vec3::Z),
    ));
}

/// Snaps the ship into the world offset grid to prevent float precision loss.
fn update(
    mut world_coords: ResMut<WorldData>,
    mut ship_query: Query<&mut Transform, With<PlayerShip>>,
    mut camera_query: Query<&mut Transform, (With<Camera>, Without<PlayerShip>)>,
    mut physics: ResMut<ShipPhysics>,
) {
    let Ok(mut ship) = ship_query.single_mut() else {
        return;
    };

    let mut snap = Vec3::ZERO;
    {
        let snap_axis = |coord: &mut f32, off: &mut f32, delta: &mut f32| {
            if coord.abs() > CAMERA_RESET_THRESHOLD {
                let s = CAMERA_RESET_THRESHOLD * coord.signum();
                *off += s;
                *coord -= s;
                *delta -= s;
            }
        };
        snap_axis(
            &mut ship.translation.x,
            &mut world_coords.camera_offset.x,
            &mut snap.x,
        );
        snap_axis(
            &mut ship.translation.y,
            &mut world_coords.camera_offset.y,
            &mut snap.y,
        );
        snap_axis(
            &mut ship.translation.z,
            &mut world_coords.camera_offset.z,
            &mut snap.z,
        );
    }

    if snap != Vec3::ZERO {
        if let Ok(mut cam) = camera_query.single_mut() {
            cam.translation += snap;
        }
        if let Some(chains) = physics.chains.as_mut() {
            for chain in chains.iter_mut() {
                chain.apply_offset(snap);
            }
        }
    }
}
