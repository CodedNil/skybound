mod composite;
mod froxels;
mod noise;
mod volumetrics;

use crate::world_rendering::{
    composite::{
        VolumetricCloudsCompositeLabel, VolumetricCloudsCompositeNode,
        setup_volumetric_clouds_composite_pipeline,
    },
    volumetrics::{
        CloudRenderTexture, CloudsViewUniforms, VolumetricCloudsLabel, VolumetricCloudsNode,
        extract_clouds_view_uniform, manage_textures, prepare_clouds_view_uniforms,
        setup_volumetric_clouds_pipeline,
    },
};
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::ExtractResourcePlugin,
        load_shader_library,
        render_graph::{RenderGraphExt, ViewNodeRunner},
    },
};

pub struct WorldRenderingPlugin;
impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/functions.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/raymarch.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_fog.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        app.add_plugins((ExtractResourcePlugin::<noise::NoiseTextures>::default(),))
            .add_systems(Startup, noise::setup_noise_textures);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<CloudRenderTexture>()
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(RenderStartup, setup_volumetric_clouds_pipeline)
            .add_systems(RenderStartup, setup_volumetric_clouds_composite_pipeline)
            .add_systems(Render, manage_textures.in_set(RenderSystems::Queue))
            .add_systems(
                Render,
                prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
            );

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsCompositeNode>>(
                Core3d,
                VolumetricCloudsCompositeLabel,
            )
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricCloudsLabel))
            .add_render_graph_edges(
                Core3d,
                (
                    VolumetricCloudsLabel,
                    VolumetricCloudsCompositeLabel,
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
