mod noise;
mod raymarch;

use crate::render::{
    noise::{NoiseTextures, setup_noise_textures},
    raymarch::{
        PreviousViewData, RaymarchPipeline, ViewUniforms, extract_clouds_view_uniform,
        prepare_clouds_view_uniforms, raymarch_pass,
    },
};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::ExtractResourcePlugin,
    },
};

pub struct WorldRenderingPlugin;

impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<NoiseTextures>::default())
            .add_systems(Startup, setup_noise_textures);

        app.get_sub_app_mut(RenderApp)
            .expect("RenderApp should already exist in App")
            .init_resource::<PreviousViewData>()
            .add_systems(RenderStartup, init_resources)
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                Render,
                prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                raymarch_pass
                    .before(bevy::core_pipeline::core_3d::main_opaque_pass_3d)
                    .in_set(Core3dSystems::MainPass),
            );
    }
}

fn init_resources(mut commands: Commands) {
    commands.init_resource::<ViewUniforms>();
    commands.init_resource::<RaymarchPipeline>();
}
