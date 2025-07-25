#![feature(portable_simd, default_field_values)]
#![allow(dead_code)]

use bevy::{
    core_pipeline::{bloom::Bloom, prepass::DepthPrepass},
    pbr::{
        Atmosphere, AtmosphereSettings, CascadeShadowConfigBuilder, NotShadowCaster,
        light_consts::lux,
    },
    prelude::*,
    render::camera::Exposure,
};

mod clouds;
// mod wind;
use crate::clouds::CloudsPlugin;
// use crate::wind::apply_wind_force;

mod fpscounter;
use crate::fpscounter::FpsCounterPlugin;
mod camera;
use crate::camera::{CameraController, CameraPlugin};
mod world;
use crate::world::{WorldCoordinates, WorldPlugin};

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            brightness: lux::AMBIENT_DAYLIGHT,
            ..default()
        })
        .add_plugins((
            DefaultPlugins,
            CloudsPlugin,
            CameraPlugin,
            FpsCounterPlugin,
            WorldPlugin,
        ))
        .add_systems(Startup, setup)
        // .add_systems(FixedUpdate, apply_wind_force)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Transform::from_xyz(0.0, 4.0, 12.0).looking_at(Vec3::Y * 4.0, Vec3::Y),
        Atmosphere::EARTH,
        AtmosphereSettings {
            aerial_view_lut_max_distance: 3.2e5,
            scene_units_to_m: 1.0,
            ..Default::default()
        },
        Exposure::SUNLIGHT,
        Bloom::NATURAL,
        DepthPrepass,
        WorldCoordinates::default(),
        CameraController {
            speed: 40.0,
            sensitivity: 0.005,
            yaw: 0.0,
            pitch: 0.0,
        },
    ));

    // Circular base
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(8.0, 0.1))),
        MeshMaterial3d(materials.add(Color::WHITE)),
    ));

    // Cube
    commands.spawn((
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
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(-Vec3::Y, Vec3::Y),
    ));
    // Aur Light
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
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Srgba::hex("000000").unwrap().into(),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(1_000_000.0)),
        NotShadowCaster,
    ));
}
