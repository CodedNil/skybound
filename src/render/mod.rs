mod noise;
mod raymarch;

use crate::render::{
    noise::{NoiseTextures, setup_noise_textures},
    raymarch::{
        CloudsViewUniforms, PreviousViewData, RaymarchPipeline, extract_clouds_view_uniform,
        prepare_clouds_view_uniforms, raymarch_pass,
    },
};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::ExtractResourcePlugin,
        renderer::RenderDevice,
    },
    shader::load_shader_library,
};

pub struct WorldRenderingPlugin;

impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/rendering.wgsl");
        load_shader_library!(app, "shaders/utils.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/raymarch.wgsl");
        load_shader_library!(app, "shaders/volumetrics.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_ocean.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        app.add_plugins(ExtractResourcePlugin::<NoiseTextures>::default())
            .add_systems(Startup, setup_noise_textures);

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should already exist in App");

        render_app
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

fn init_resources(mut commands: Commands, _: Res<RenderDevice>) {
    commands.init_resource::<CloudsViewUniforms>();
    commands.init_resource::<RaymarchPipeline>();
}
