#![feature(portable_simd, default_field_values)]
#![allow(dead_code)]

use bevy::{
    pbr::{NotShadowCaster, light_consts::lux},
    prelude::*,
};

mod clouds;
use crate::clouds::CloudsPlugin;
// mod wind;
// use crate::wind::apply_wind_force;

mod debugtext;
use crate::debugtext::DebugTextPlugin;
mod camera;
use crate::camera::CameraPlugin;
mod world;
use crate::world::WorldPlugin;

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            brightness: lux::AMBIENT_DAYLIGHT,
            ..default()
        })
        .add_plugins((
            DefaultPlugins,
            // CloudsPlugin,
            CameraPlugin,
            DebugTextPlugin,
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

    // Sky
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Srgba::hex("000000").unwrap().into(),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(100_000_000.0)),
        NotShadowCaster,
    ));
}
