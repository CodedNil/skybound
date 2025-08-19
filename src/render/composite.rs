use crate::render::volumetrics::CloudRenderTexture;
use bevy::{
    asset::load_embedded_asset,
    core_pipeline::FullscreenShader,
    diagnostic::FrameCount,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FilterMode, FragmentState, LoadOp, MultisampleState,
            Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
            StoreOp, TextureFormat, TextureSampleType,
            binding_types::{sampler, texture_2d},
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
    },
};

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CompositeLabel;

#[derive(Default)]
pub struct CompositeNode;

impl ViewNode for CompositeNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let composite_pipeline = world.resource::<CompositePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();
        let frame_count = world.resource::<FrameCount>();
        let diagnostics = render_context.diagnostic_recorder();

        // Ensure the intermediate resources are ready
        let (Some(pipeline), Some(color_view), Some(motion_view), Some(depth_view)) = (
            pipeline_cache.get_render_pipeline(composite_pipeline.pipeline_id),
            cloud_render_texture.color_view.as_ref(),
            cloud_render_texture.motion_view.as_ref(),
            cloud_render_texture.depth_view.as_ref(),
        ) else {
            return Ok(());
        };

        // Ping-pong between which history buffer is our source and target
        let (source_history_view, target_history_view) = if frame_count.0.is_multiple_of(2) {
            (
                cloud_render_texture.history_a_view.as_ref(),
                cloud_render_texture.history_b_view.as_ref(),
            )
        } else {
            (
                cloud_render_texture.history_b_view.as_ref(),
                cloud_render_texture.history_a_view.as_ref(),
            )
        };
        let (Some(source_history_view), Some(target_history_view)) =
            (source_history_view, target_history_view)
        else {
            return Ok(());
        };

        // Create the bind group for the composite shader
        let bind_group = render_context.render_device().create_bind_group(
            "composite_bind_group",
            &composite_pipeline.layout,
            &BindGroupEntries::sequential((
                color_view,
                source_history_view,
                motion_view,
                depth_view,
                &composite_pipeline.nearest_sampler,
                &composite_pipeline.linear_sampler,
            )),
        );

        // Begin the render pass to composite clouds onto the main view
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("composite_pass"),
            color_attachments: &[
                // The main screen
                Some(RenderPassColorAttachment {
                    view: view_target.main_texture_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu_types::Color::BLACK),
                        store: StoreOp::Store,
                    },
                }),
                // The history buffer for the next frame
                Some(RenderPassColorAttachment {
                    view: target_history_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu_types::Color::BLACK),
                        store: StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set the render pipeline, bind group, and draw a full-screen triangle
        let pass_span = diagnostics.pass_span(&mut render_pass, "composite");
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        pass_span.end(&mut render_pass);

        Ok(())
    }
}

#[derive(Resource)]
pub struct CompositePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    linear_sampler: Sampler,
    nearest_sampler: Sampler,
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader =
            load_embedded_asset!(world.resource::<AssetServer>(), "shaders/composite.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

        let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("composite_nearest_sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        });
        let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("composite_linear_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        let layout = render_device.create_bind_group_layout(
            "composite_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }), // Color
                    texture_2d(TextureSampleType::Float { filterable: true }), // History
                    texture_2d(TextureSampleType::Float { filterable: false }), // Motion
                    texture_2d(TextureSampleType::Float { filterable: false }), // Depth
                    sampler(SamplerBindingType::NonFiltering),                 // Nearest sampler
                    sampler(SamplerBindingType::Filtering),                    // Linear sampler
                ),
            ),
        );

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("composite_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                targets: vec![
                    // View Target
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    // History Buffer
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
                ..default()
            }),
            ..default()
        });

        Self {
            layout,
            pipeline_id,
            linear_sampler,
            nearest_sampler,
        }
    }
}
