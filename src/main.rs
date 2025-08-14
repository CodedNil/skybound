#![feature(portable_simd, default_field_values)]

use bevy::light::NotShadowCaster;
use bevy::prelude::*;

mod world_rendering;
use crate::world_rendering::WorldRenderingPlugin;
// mod wind;
// use crate::wind::apply_wind_force;

mod debugtext;
use crate::debugtext::DebugTextPlugin;
mod camera;
use crate::camera::CameraPlugin;
pub mod world;
use crate::world::WorldPlugin;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            WorldRenderingPlugin,
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
