use bevy::anti_alias::taa::temporal_anti_alias;
use bevy::core_pipeline::core_3d::main_opaque_pass_3d;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::Shader;

use crate::game::camera::DebugView;
use crate::game::flight::PlayerModel;
use crate::game::weather::CloudField;
use crate::game::world::WorldData;
use crate::graphics::pipeline::{
    SceneBuffer, ScenePipeline, composite_pass, prepare_bindings, prepare_scene_view, raymarch_pass,
};
use crate::graphics::textures::{VolumeTextures, prepare_volume_textures};

mod light;
pub mod pipeline;
mod targets;
mod textures;

#[derive(Resource, Clone, bevy::render::extract_resource::ExtractResource)]
#[extract_app(RenderApp)]
pub struct AtmosphereSettings(pub Vec4);
impl Default for AtmosphereSettings {
    fn default() -> Self {
        Self(Vec4::splat(-1.0))
    }
}

pub struct GraphicsPlugin;

#[derive(Resource)]
pub struct SceneShader(pub Handle<Shader>);

#[derive(Resource)]
pub struct VolumeShaders {
    pub filter: Handle<Shader>,
    pub composite: Handle<Shader>,
}

impl Plugin for GraphicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<targets::VolumeScale>()
            .init_resource::<AtmosphereSettings>()
            .init_resource::<PlayerModel>()
            .add_plugins((
                ExtractResourcePlugin::<VolumeTextures>::default(),
                ExtractResourcePlugin::<AtmosphereSettings>::default(),
                ExtractResourcePlugin::<PlayerModel>::default(),
                ExtractResourcePlugin::<WorldData>::default(),
                ExtractResourcePlugin::<CloudField>::default(),
                ExtractResourcePlugin::<DebugView>::default(),
                ExtractResourcePlugin::<targets::VolumeScale>::default(),
            ))
            .add_systems(Startup, prepare_volume_textures);

        let shader = Shader::from_spirv(
            include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv")).as_slice(),
            "scene.spv",
        );
        let shader = app.world_mut().resource_mut::<Assets<Shader>>().add(shader);
        let mut assets = app.world_mut().resource_mut::<Assets<Shader>>();
        let filter = assets.add(Shader::from_wgsl(
            include_str!("../../assets/volume_resolve.wgsl"),
            "volume_resolve.wgsl",
        ));
        let composite = assets.add(Shader::from_wgsl(
            include_str!("../../assets/volume_composite.wgsl"),
            "volume_composite.wgsl",
        ));
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should exist");
        render_app.insert_resource(SceneShader(shader));
        render_app.insert_resource(VolumeShaders { filter, composite });
        render_app
            .add_systems(RenderStartup, init_resources)
            .add_systems(
                Render,
                prepare_scene_view.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Render,
                prepare_bindings.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Core3d,
                composite_pass
                    .after(temporal_anti_alias)
                    .in_set(Core3dSystems::EarlyPostProcess),
            )
            .add_systems(
                Core3d,
                raymarch_pass
                    .before(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

fn init_resources(world: &mut World) {
    world.init_resource::<SceneBuffer>();
    world.init_resource::<ScenePipeline>();
    world.init_resource::<textures::SkyLookup>();
    world.init_resource::<targets::SceneTargets>();
}
