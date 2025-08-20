mod composite;
mod noise;
mod raymarch;

use crate::render::{
    composite::{CompositeLabel, CompositeNode, CompositePipeline},
    noise::{NoiseTextures, setup_noise_textures},
    raymarch::{
        CloudRenderTexture, CloudsViewUniforms, PreviousViewData, RaymarchLabel, RaymarchNode,
        RaymarchPipeline, extract_clouds_view_uniform, manage_textures,
        prepare_clouds_view_uniforms,
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
        load_shader_library!(app, "shaders/rendering.wgsl");
        load_shader_library!(app, "shaders/utils.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/raymarch.wgsl");
        load_shader_library!(app, "shaders/volumetrics.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_fog.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        load_shader_library!(app, "shaders/composite.wgsl");

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
                    prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<RaymarchNode>>(Core3d, RaymarchLabel)
            .add_render_graph_node::<ViewNodeRunner<CompositeNode>>(Core3d, CompositeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPass,
                    RaymarchLabel,
                    CompositeLabel,
                    Node3d::Bloom,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<CloudsViewUniforms>()
                .init_resource::<RaymarchPipeline>()
                .init_resource::<CompositePipeline>();
        }
    }
}
