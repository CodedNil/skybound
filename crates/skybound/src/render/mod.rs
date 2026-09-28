mod noise;
pub mod raymarch;

use crate::{
    render::{
        noise::{NoiseTextures, setup_noise_textures},
        raymarch::{
            PreviousViewData, RaymarchPipeline, ShipUniforms, ViewUniforms,
            extract_clouds_view_uniform, prepare_clouds_view_uniforms, prepare_ship_uniforms,
            raymarch_pass,
        },
    },
    ships::player::ExtractedShipData,
};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems, core_3d::main_opaque_pass_3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::ExtractResourcePlugin,
    },
    shader::Shader,
};

pub struct WorldRenderingPlugin;

#[derive(Resource)]
pub struct SkyboundGpuShader(pub Handle<Shader>);

impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ExtractedShipData>()
            .add_plugins((
                ExtractResourcePlugin::<NoiseTextures>::default(),
                ExtractResourcePlugin::<ExtractedShipData>::default(),
            ))
            .add_systems(Startup, setup_noise_textures);

        let shader = Shader::from_spirv(
            include_bytes!(concat!(env!("OUT_DIR"), "/skybound_gpu.spv")).as_slice(),
            "skybound_gpu.spv",
        );
        let shader = app.world_mut().resource_mut::<Assets<Shader>>().add(shader);
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should exist");
        render_app.insert_resource(SkyboundGpuShader(shader));
        render_app
            .init_resource::<PreviousViewData>()
            .add_systems(RenderStartup, init_resources)
            .add_systems(ExtractSchedule, extract_clouds_view_uniform)
            .add_systems(
                Render,
                (prepare_clouds_view_uniforms, prepare_ship_uniforms)
                    .in_set(RenderSystems::PrepareResources),
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
    world.init_resource::<ShipUniforms>();
    world.init_resource::<ViewUniforms>();
    world.init_resource::<RaymarchPipeline>();
}
