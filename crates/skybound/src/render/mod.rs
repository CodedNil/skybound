mod noise;
pub mod raymarch;

use crate::{
    render::{
        noise::{NoiseTextures, setup_noise_textures},
        raymarch::{
            PreviousViewData, RaymarchPipeline, ViewUniforms, extract_clouds_view_uniform,
            prepare_clouds_view_uniforms, raymarch_pass,
        },
    },
    ships::{
        physics::ExtractedShipData,
        render_pass::{
            init_ship_resources, prepare_ship_render_targets, prepare_ship_uniforms,
            ship_pass as ship_raymarch_pass,
        },
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
        app.add_plugins((
            ExtractResourcePlugin::<NoiseTextures>::default(),
            ExtractResourcePlugin::<ExtractedShipData>::default(),
        ))
        .add_systems(Startup, setup_noise_textures);

        app.get_sub_app_mut(RenderApp)
            .expect("RenderApp should exist")
            .init_resource::<PreviousViewData>()
            .add_systems(RenderStartup, |world: &mut World| {
                init_resources(world);
                init_ship_resources(world);
            })
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                Render,
                (
                    prepare_clouds_view_uniforms,
                    prepare_ship_uniforms,
                    prepare_ship_render_targets,
                )
                    .in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                (
                    ship_raymarch_pass
                        .before(bevy::core_pipeline::core_3d::main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                    raymarch_pass
                        .after(ship_raymarch_pass)
                        .before(bevy::core_pipeline::core_3d::main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }
}

fn init_resources(world: &mut World) {
    world.init_resource::<ViewUniforms>();
    world.init_resource::<RaymarchPipeline>();
}
