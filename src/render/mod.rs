mod clouds;
mod noise;
mod raymarch;

use crate::render::{
    clouds::{CloudsBufferData, setup_clouds, update_clouds, update_clouds_buffer},
    noise::{NoiseTextures, setup_noise_textures},
    raymarch::{
        CloudsViewUniforms, PreviousViewData, RaymarchLabel, RaymarchNode, RaymarchPipeline,
        extract_clouds_view_uniform, prepare_clouds_view_uniforms,
    },
};
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::ExtractResourcePlugin,
        render_graph::{RenderGraphExt, ViewNodeRunner},
        renderer::RenderDevice,
    },
    shader::load_shader_library,
};

/// Plugin that sets up shaders, extraction, and render graph nodes for world rendering.
pub struct WorldRenderingPlugin;
impl Plugin for WorldRenderingPlugin {
    /// Register shaders, resources, and render graph nodes for cloud rendering.
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
            .add_plugins(ExtractResourcePlugin::<CloudsBufferData>::default())
            .add_systems(Startup, (setup_clouds, setup_noise_textures))
            .add_systems(Update, update_clouds);

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should already exist in App");

        render_app
            .init_resource::<PreviousViewData>()
            .add_systems(RenderStartup, init_resources)
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                Render,
                (
                    update_clouds_buffer.in_set(RenderSystems::Prepare),
                    prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<RaymarchNode>>(Core3d, RaymarchLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::StartMainPass, RaymarchLabel, Node3d::Bloom),
            );
    }
}

fn init_resources(mut commands: Commands, _: Res<RenderDevice>) {
    commands.init_resource::<CloudsViewUniforms>();
    commands.init_resource::<RaymarchPipeline>();
}
