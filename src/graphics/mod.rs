pub mod pipeline;
mod prepare;
mod targets;

use crate::{
    game::{camera::DebugView, flight::PlayerModel, world::WorldData},
    graphics::{
        pipeline::{SceneBuffer, ScenePipeline, prepare_scene_view, raymarch_pass},
        prepare::textures::{VolumeTextures, prepare_volume_textures},
    },
};
use bevy::{
    core_pipeline::{Core3d, Core3dSystems, core_3d::main_opaque_pass_3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems, extract_resource::ExtractResourcePlugin,
    },
    shader::Shader,
};

pub struct GraphicsPlugin;

#[derive(Resource)]
pub struct SceneShader(pub Handle<Shader>);

#[derive(Resource)]
pub struct ResolveShader(pub Handle<Shader>);

impl Plugin for GraphicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<targets::RenderScale>()
            .init_resource::<PlayerModel>()
            .add_plugins((
                ExtractResourcePlugin::<VolumeTextures>::default(),
                ExtractResourcePlugin::<PlayerModel>::default(),
                ExtractResourcePlugin::<WorldData>::default(),
                ExtractResourcePlugin::<DebugView>::default(),
                ExtractResourcePlugin::<targets::RenderScale>::default(),
            ))
            .add_systems(Startup, prepare_volume_textures);

        let shader = Shader::from_spirv(
            include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv")).as_slice(),
            "scene.spv",
        );
        let shader = app.world_mut().resource_mut::<Assets<Shader>>().add(shader);
        let resolve_shader =
            app.world_mut()
                .resource_mut::<Assets<Shader>>()
                .add(Shader::from_wgsl(
                    include_str!("../../assets/resolve.wgsl"),
                    "resolve.wgsl",
                ));
        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("RenderApp should exist");
        render_app.insert_resource(SceneShader(shader));
        render_app.insert_resource(ResolveShader(resolve_shader));
        render_app
            .add_systems(RenderStartup, init_resources)
            .add_systems(
                Render,
                prepare_scene_view.in_set(RenderSystems::PrepareResources),
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
    world.init_resource::<prepare::sky::SkyLookup>();
    world.init_resource::<targets::SceneTargets>();
}

#[cfg(test)]
mod tests {
    use naga::{
        front::spv::{Options, parse_u8_slice},
        valid::{Capabilities, ValidationFlags, Validator},
    };
    use skybound::scene::FrameUniform;
    use std::mem::size_of;

    #[test]
    fn shader_accepts_uniform_layout_and_matches_host_frame() {
        let module = parse_u8_slice(
            include_bytes!(concat!(env!("OUT_DIR"), "/scene.spv")),
            &Options::default(),
        )
        .expect("parse shader");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("validate shader");
        let (_, frame) = module
            .global_variables
            .iter()
            .find(|(_, global)| {
                global
                    .binding
                    .as_ref()
                    .is_some_and(|binding| binding.group == 0 && binding.binding == 0)
            })
            .expect("frame uniform");
        let naga::TypeInner::Struct { span, .. } = module.types[frame.ty].inner else {
            panic!("frame must be a struct")
        };
        assert_eq!(
            span as usize,
            size_of::<FrameUniform>(),
            "shader and host frame layouts differ"
        );
    }
}
