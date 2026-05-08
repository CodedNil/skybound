use bevy::{
    core_pipeline::{Core3d, Core3dSystems},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::*,
        renderer::RenderContext,
        view::{ExtractedView, ViewTarget},
        Render, RenderApp, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use spirv_builder::SpirvBuilder;
use std::path::PathBuf;

#[derive(Copy, Clone, Pod, Zeroable, Default, Resource, ExtractResource)]
#[repr(C)]
pub struct ShaderConstants {
    pub width: u32,
    pub height: u32,
    pub time: f32,
}

fn main() {
    // Build the shader at startup
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let crate_path = [manifest_dir, "shaders", "sky"]
        .iter()
        .copied()
        .collect::<PathBuf>();

    let builder = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.1");
    builder.build().expect("Failed to build shader");

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(SkyRenderPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update_constants)
        .run();
}

fn setup(mut commands: Commands<'_, '_>) {
    commands.spawn(Camera3d::default());
}

fn update_constants(
    mut constants: ResMut<'_, ShaderConstants>,
    time: Res<'_, Time>,
    windows: Query<'_, '_, &Window>,
) {
    if let Ok(window) = windows.single() {
        constants.time = time.elapsed_secs();
        constants.width = window.width() as u32;
        constants.height = window.height() as u32;
    }
}

struct SkyRenderPlugin;

impl Plugin for SkyRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShaderConstants>()
            .add_plugins(ExtractResourcePlugin::<ShaderConstants>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(
                Render,
                prepare_sky_pipeline.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                sky_pass
                    .in_set(Core3dSystems::MainPass)
                    .before(bevy::core_pipeline::core_3d::main_opaque_pass_3d),
            );
    }
}

#[derive(Resource)]
struct SkyPipeline {
    pipeline_id: CachedRenderPipelineId,
}

fn prepare_sky_pipeline(
    mut commands: Commands<'_, '_>,
    pipeline_cache: Res<'_, PipelineCache>,
    sky_pipeline: Option<Res<'_, SkyPipeline>>,
    asset_server: Res<'_, AssetServer>,
    view_targets: Query<'_, '_, &ViewTarget>,
) {
    let Ok(view_target) = view_targets.single() else {
        return;
    };

    if sky_pipeline.is_none() {
        let shader_handle = asset_server.load("shaders/sky.spv");

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("Sky Pipeline".into()),
            layout: vec![],
            vertex: VertexState {
                shader: shader_handle.clone(),
                entry_point: Some("main_vs".into()),
                buffers: vec![],
                shader_defs: vec![],
            },
            fragment: Some(FragmentState {
                shader: shader_handle.clone(),
                entry_point: Some("main_fs".into()),
                targets: vec![Some(ColorTargetState {
                    format: view_target.main_texture_format(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                shader_defs: vec![],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: 4,
                ..default()
            },
            zero_initialize_workgroup_memory: false,
            immediate_size: std::mem::size_of::<ShaderConstants>() as u32,
        });

        commands.insert_resource(SkyPipeline { pipeline_id });
    }
}

fn sky_pass(
    world: &World,
    mut render_context: RenderContext<'_, '_>,
    view_query: Query<'_, '_, (Entity, &ExtractedView, &ViewTarget)>,
) {
    let pipeline_cache = world.resource::<PipelineCache>();
    let sky_pipeline = match world.get_resource::<SkyPipeline>() {
        Some(p) => p,
        None => return,
    };
    let constants = world.resource::<ShaderConstants>();

    for (entity, view, view_target) in &view_query {
        if let Some(pipeline) = pipeline_cache.get_render_pipeline(sky_pipeline.pipeline_id) {
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("Sky Pass"),
                color_attachments: &[Some(view_target.get_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_render_pipeline(pipeline);
            render_pass.set_immediates(0, bytemuck::bytes_of(constants));
            render_pass.draw(0..3, 0..1);
        }
    }
}
