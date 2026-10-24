use crate::{
    game::{camera::DebugView, flight::PlayerModel, weather::CloudField, world::WorldData},
    graphics::{
        pipeline::{
            SceneBuffer, ScenePipeline, composite_pass, prepare_bindings, prepare_scene_view,
            raymarch_pass,
        },
        textures::{VolumeTextures, prepare_volume_textures},
    },
};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems, core_3d::main_opaque_pass_3d},
    post_process::auto_exposure::AutoExposurePlugin,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::ExtractResourcePlugin,
    },
    shader::Shader,
};

mod binding;
mod light;
pub mod pipeline;
mod targets;
mod textures;
pub mod upscale;

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

impl Plugin for GraphicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AutoExposurePlugin)
            .init_resource::<AtmosphereSettings>()
            .init_resource::<PlayerModel>()
            .add_plugins((
                ExtractResourcePlugin::<VolumeTextures>::default(),
                ExtractResourcePlugin::<AtmosphereSettings>::default(),
                ExtractResourcePlugin::<PlayerModel>::default(),
                ExtractResourcePlugin::<WorldData>::default(),
                ExtractResourcePlugin::<CloudField>::default(),
                ExtractResourcePlugin::<DebugView>::default(),
            ))
            .add_systems(Startup, prepare_volume_textures);

        let shader = Shader::from_spirv(
            include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv")).as_slice(),
            "scene.spv",
        );
        let shader = app.world_mut().resource_mut::<Assets<Shader>>().add(shader);
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should exist");
        render_app.insert_resource(SceneShader(shader));
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
            // The whole scene is composited before any post-process, so an
            // upscaler or anti-aliasing works on all of it.
            .add_systems(
                Core3d,
                composite_pass
                    .after(main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
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
