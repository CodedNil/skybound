use bevy::{
    asset::embedded_asset,
    core_pipeline::{Core3d, Core3dSystems, FullscreenShader, prepass::ViewPrepassTextures},
    material::descriptor::BindGroupLayoutDescriptor,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, IntoBinding,
            Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderStages, ShaderType, TextureFormat, TextureSampleType,
            binding_types::{texture_2d, texture_depth_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
    },
};

/// Visualizes the camera's depth, normal, or motion-vector attachment.
#[derive(Debug, Default)]
pub struct ShowPrepassPlugin;

impl Plugin for ShowPrepassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "show_prepass.wgsl");
        app.add_plugins((
            ExtractComponentPlugin::<ShowPrepass>::default(),
            ExtractComponentPlugin::<ShowPrepassDepthPower>::default(),
            UniformComponentPlugin::<ShowPrepassUniform>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_uniforms
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareResources),
                    prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                show_prepass_render_system.in_set(Core3dSystems::PostProcess),
            );
    }
}

/// Select which prepass attachment to display on a camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Component, ExtractComponent)]
#[extract_app(RenderApp)]
pub enum ShowPrepass {
    Depth,
    Normals,
    MotionVectors,
}

/// Optional exponent for depth visualization.
#[derive(Debug, Clone, Copy, PartialEq, Component, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct ShowPrepassDepthPower(pub f32);

#[derive(Component, Clone, ShaderType)]
struct ShowPrepassUniform {
    depth_power: f32,
    delta_time: f32,
    mode: u32,
}

type ShowPrepassViewQuery = (
    &'static ViewTarget,
    &'static ExtractedCamera,
    &'static ShowPrepassBindGroup,
    &'static DynamicUniformIndex<ShowPrepassUniform>,
);

fn show_prepass_render_system(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<ShowPrepassPipeline>,
    views: Query<ShowPrepassViewQuery, With<ShowPrepass>>,
) {
    for (view_target, camera, bind_group, uniform_index) in &views {
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline_id) else {
            continue;
        };

        let post_process = view_target.post_process_write();
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("show_prepass_render_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        render_pass.set_render_pipeline(render_pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[uniform_index.index()]);
        render_pass.draw(0..3, 0..1);
    }
}

fn prepare_uniforms(
    mut commands: Commands,
    views: Query<(Entity, &ShowPrepass, Option<&ShowPrepassDepthPower>)>,
    time: Res<Time>,
) {
    for (entity, show_prepass, depth_power) in &views {
        commands.entity(entity).insert(ShowPrepassUniform {
            depth_power: depth_power.map_or(1.0, |depth| depth.0),
            delta_time: time.delta_secs(),
            mode: match show_prepass {
                ShowPrepass::Depth => 0,
                ShowPrepass::Normals => 1,
                ShowPrepass::MotionVectors => 2,
            },
        });
    }
}

#[derive(Resource)]
struct ShowPrepassPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout_entries = BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            uniform_buffer::<ShowPrepassUniform>(true),
            texture_depth_2d(),
            texture_2d(TextureSampleType::Float { filterable: false }),
            texture_2d(TextureSampleType::Float { filterable: false }),
        ),
    );
    let layout_descriptor =
        BindGroupLayoutDescriptor::new("show_prepass_bind_group_layout", &layout_entries);
    let layout =
        render_device.create_bind_group_layout("show_prepass_bind_group_layout", &layout_entries);
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("show prepass pipeline".into()),
        layout: vec![layout_descriptor],
        immediate_size: 0,
        vertex: fullscreen_shader.to_vertex_state(),
        primitive: default(),
        depth_stencil: None,
        multisample: default(),
        fragment: Some(FragmentState {
            shader: asset_server.load("embedded://skybound/show_prepass.wgsl"),
            entry_point: Some("fragment".into()),
            constants: Vec::new(),
            shader_defs: Vec::new(),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(ShowPrepassPipeline {
        layout,
        pipeline_id,
    });
}

#[derive(Component)]
struct ShowPrepassBindGroup(BindGroup);

fn prepare_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, Option<&ViewPrepassTextures>), With<ShowPrepass>>,
    uniforms: Res<ComponentUniforms<ShowPrepassUniform>>,
    render_device: Res<RenderDevice>,
    pipeline: Res<ShowPrepassPipeline>,
) {
    for (entity, prepass_textures) in &views {
        let Some(uniform) = uniforms.uniforms().binding() else {
            continue;
        };
        let Some(prepass_textures) = prepass_textures else {
            continue;
        };
        let (Some(depth), Some(normal), Some(motion)) = (
            prepass_textures
                .depth_only_view()
                .map(IntoBinding::into_binding),
            prepass_textures
                .normal_view()
                .map(IntoBinding::into_binding),
            prepass_textures
                .motion_vectors_view()
                .map(IntoBinding::into_binding),
        ) else {
            continue;
        };

        let bind_group = render_device.create_bind_group(
            "show_prepass_bind_group",
            &pipeline.layout,
            &BindGroupEntries::sequential((uniform, depth, normal, motion)),
        );
        commands
            .entity(entity)
            .insert(ShowPrepassBindGroup(bind_group));
    }
}
