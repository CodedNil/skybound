use avian3d::prelude::*;
use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    pbr::{CascadeShadowConfigBuilder, NotShadowCaster, light_consts::lux},
    prelude::*,
    render::camera::Exposure,
};
use smooth_bevy_cameras::{
    LookTransformPlugin,
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
};

mod clouds;
mod wind;
use crate::clouds::{CloudsPlugin, VolumetricClouds};
use crate::wind::apply_wind_force;

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            brightness: lux::AMBIENT_DAYLIGHT,
            ..default()
        })
        .insert_resource(Gravity(Vec3::NEG_Y * 4.0))
        .add_plugins((
            DefaultPlugins,
            CloudsPlugin,
            PhysicsPlugins::default(),
            LookTransformPlugin,
            UnrealCameraPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, clouds::update_settings)
        .add_systems(FixedUpdate, apply_wind_force)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                hdr: true,
                ..default()
            },
            Transform::from_xyz(-1.2, 0.15, 0.0).looking_at(Vec3::Y * 0.1, Vec3::Y),
            Exposure::SUNLIGHT,
            Tonemapping::AcesFitted,
            Bloom::NATURAL,
            DistanceFog {
                color: Color::srgba(0.35, 0.48, 0.66, 1.0),
                directional_light_color: Color::srgba(1.0, 0.95, 0.85, 0.5),
                directional_light_exponent: 30.0,
                falloff: FogFalloff::from_visibility_colors(
                    500.0,
                    Color::srgb(0.35, 0.5, 0.66),
                    Color::srgb(0.8, 0.844, 1.0),
                ),
            },
            VolumetricClouds {
                time: 0.0,
                intensity: 2.0,
            },
        ))
        .insert(UnrealCameraBundle::new(
            UnrealCameraController::default(),
            Vec3::new(-15.0, 8.0, 18.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::Y,
        ));

    // Circular base
    commands.spawn((
        RigidBody::Static,
        Collider::cylinder(8.0, 0.1),
        Mesh3d(meshes.add(Cylinder::new(8.0, 0.1))),
        MeshMaterial3d(materials.add(Color::WHITE)),
    ));

    // Cube
    commands.spawn((
        RigidBody::Dynamic,
        LinearDamping(0.8),
        AngularDamping(1.6),
        Collider::cuboid(1.0, 1.0, 1.0),
        AngularVelocity(Vec3::new(2.5, 3.5, 1.5)),
        Mesh3d(meshes.add(Cuboid::from_length(1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 4.0, 0.0),
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: lux::DIRECT_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 0.3,
            maximum_distance: 3.0,
            ..default()
        }
        .build(),
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::Y, Vec3::Y),
    ));
    // Sky
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Srgba::hex("888888").unwrap().into(),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(1_000_000.0)),
        NotShadowCaster,
    ));
}
