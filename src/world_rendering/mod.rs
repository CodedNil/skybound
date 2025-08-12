mod composite;
mod froxels;
mod noise;
mod volumetrics;

use crate::world_rendering::{
    composite::{CompositeLabel, CompositeNode, setup_composite_pipeline},
    froxels::{
        FroxelsLabel, FroxelsNode, FroxelsTexture, setup_froxels_pipeline, setup_froxels_texture,
    },
    noise::{NoiseTextures, setup_noise_textures},
    volumetrics::{
        CloudRenderTexture, CloudsViewUniforms, VolumetricsLabel, VolumetricsNode,
        extract_clouds_view_uniform, manage_textures, prepare_clouds_view_uniforms,
        setup_volumetrics_pipeline,
    },
};
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
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
        load_shader_library!(app, "shaders/world_rendering_froxels.wgsl");
        load_shader_library!(app, "shaders/utils.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/froxels.wgsl");
        load_shader_library!(app, "shaders/raymarch.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_fog.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        app.add_plugins((
            ExtractResourcePlugin::<NoiseTextures>::default(),
            ExtractResourcePlugin::<FroxelsTexture>::default(),
        ))
        .add_systems(Startup, (setup_noise_textures, setup_froxels_texture));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<CloudRenderTexture>()
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                RenderStartup,
                (
                    setup_froxels_pipeline,
                    setup_volumetrics_pipeline,
                    setup_composite_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    manage_textures.in_set(RenderSystems::Queue),
                    prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            );

        render_app
            .add_render_graph_node::<ViewNodeRunner<FroxelsNode>>(Core3d, FroxelsLabel)
            .add_render_graph_node::<ViewNodeRunner<VolumetricsNode>>(Core3d, VolumetricsLabel)
            .add_render_graph_node::<ViewNodeRunner<CompositeNode>>(Core3d, CompositeLabel)
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricsLabel))
            .add_render_graph_edges(
                Core3d,
                (
                    FroxelsLabel,
                    VolumetricsLabel,
                    CompositeLabel,
                    Node3d::Bloom,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<CloudsViewUniforms>();
        }
    }
}
