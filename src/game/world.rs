use bevy::render::{RenderApp, extract_resource::ExtractResource};
use bevy::{
    anti_alias::{
        contrast_adaptive_sharpening::ContrastAdaptiveSharpening, taa::TemporalAntiAliasing,
    },
    camera::Hdr,
    core_pipeline::prepass::{
        DepthPrepass, MotionVectorPrepass, NoBackgroundMotionVectors, NormalPrepass,
    },
    post_process::bloom::Bloom,
    prelude::*,
};
use skybound::scene::PLANET_RADIUS;
use std::f32::consts::FRAC_PI_4;

use crate::game::flight::{Player, SPAWN};

/// Every world-space entity moves together when the floating origin changes.
#[derive(Component)]
pub struct WorldAnchor;

const CAMERA_RESET_THRESHOLD: f32 = 50_000.0;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WorldData {
    pub camera_offset: Vec3,
}
impl Default for WorldData {
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
    pub fn planet_frame(&self, pos: Vec3) -> (Quat, f32, f32) {
        let rotation = self.planet_rotation(pos);
        let north = rotation.conjugate().mul_vec3(Vec3::Z);
        let latitude = north.z.clamp(-1.0, 1.0).asin();
        let longitude = if north.x.abs() < f32::EPSILON && north.y.abs() < f32::EPSILON {
            0.0
        } else {
            north.x.atan2(-north.y)
        };
        (rotation, latitude, longitude)
    }

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

    pub fn planet_rotation(&self, pos: Vec3) -> Quat {
        let flat_offset = Vec3::new(self.camera_offset.x, self.camera_offset.y, 0.0);
        Self::rotation_from_translation(flat_offset + pos)
    }
}

pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldData>()
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        WorldAnchor,
        Camera::default(),
        Projection::default(),
        NormalPrepass,
        DepthPrepass,
        MotionVectorPrepass,
        NoBackgroundMotionVectors,
        Msaa::Off,
        Hdr,
        Bloom::NATURAL,
        TemporalAntiAliasing::default(),
        ContrastAdaptiveSharpening {
            sharpening_strength: 0.3,
            ..default()
        },
        Transform::from_translation(SPAWN + vec3(0.0, -56.0, 13.0)).looking_at(SPAWN, Vec3::Z),
    ));
}

fn update(
    mut world_coords: ResMut<WorldData>,
    player: Single<Entity, With<Player>>,
    mut anchors: Query<&mut Transform, With<WorldAnchor>>,
    mut taa: Single<&mut TemporalAntiAliasing>,
) {
    let position = anchors
        .get(*player)
        .expect("player world anchor")
        .translation;

    let snap = Vec3::select(
        position.abs().cmpgt(Vec3::splat(CAMERA_RESET_THRESHOLD)),
        position.signum() * CAMERA_RESET_THRESHOLD,
        Vec3::ZERO,
    );
    if snap != Vec3::ZERO {
        taa.reset = true;
        world_coords.camera_offset += snap;
        for mut transform in &mut anchors {
            transform.translation -= snap;
        }
    }
}
