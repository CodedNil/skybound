mod composite;
mod noise;
mod volumetrics;

use crate::world_rendering::{
    composite::{CompositeLabel, CompositeNode, CompositePipeline},
    noise::{NoiseTextures, setup_noise_textures},
    volumetrics::{
        CloudRenderTexture, CloudsViewUniforms, PreviousViewData, VolumetricsLabel,
        VolumetricsNode, VolumetricsPipeline, extract_clouds_view_uniform, manage_textures,
        prepare_clouds_view_uniforms, update_previous_view_data,
    },
};
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::ExtractResourcePlugin,
        render_graph::{RenderGraphExt, ViewNodeRunner},
    },
    shader::load_shader_library,
};

pub struct WorldRenderingPlugin;
impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/world_rendering.wgsl");
        load_shader_library!(app, "shaders/world_rendering_composite.wgsl");
        load_shader_library!(app, "shaders/utils.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/raymarch.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_fog.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        app.add_plugins(ExtractResourcePlugin::<NoiseTextures>::default())
            .add_systems(Startup, setup_noise_textures);

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should already exist in App");

        render_app
            .init_resource::<CloudRenderTexture>()
            .init_resource::<PreviousViewData>()
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                Render,
                (
                    manage_textures.in_set(RenderSystems::Queue),
                    (
                        prepare_clouds_view_uniforms,
                        update_previous_view_data.after(prepare_clouds_view_uniforms),
                    )
                        .in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<VolumetricsNode>>(Core3d, VolumetricsLabel)
            .add_render_graph_node::<ViewNodeRunner<CompositeNode>>(Core3d, CompositeLabel)
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricsLabel))
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPass,
                    VolumetricsLabel,
                    CompositeLabel,
                    Node3d::Bloom,
                    Node3d::EndMainPassPostProcessing,
                    Node3d::Upscaling,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<CloudsViewUniforms>()
                .init_resource::<VolumetricsPipeline>()
                .init_resource::<CompositePipeline>();
        }
    }
}
