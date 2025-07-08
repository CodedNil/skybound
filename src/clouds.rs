use bevy::{
    pbr::MaterialPlugin,
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderRef},
};

pub struct CloudPlugin;

impl Plugin for CloudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CloudMaterial>::default())
            .add_systems(Startup, setup_clouds);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CloudMaterial {}

impl Material for CloudMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/clouds.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

fn setup_clouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<CloudMaterial>>,
) {
    // Create a large cube or sphere to represent the cloud volume
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(20.0, 20.0, 20.0))),
        MeshMaterial3d(materials.add(CloudMaterial {})),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}
