use crate::world_rendering::volumetrics::CloudRenderTexture;
use bevy::{
    asset::load_embedded_asset,
    core_pipeline::FullscreenShader,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendState,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, LoadOp,
            MultisampleState, Operations, PipelineCache, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType, ShaderStages,
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

#[derive(Resource)]
pub struct CompositePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

impl ViewNode for CompositeNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Get necessary resources from the render world
        let volumetric_clouds_composite_pipeline = world.resource::<CompositePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();

        // Ensure the intermediate resources are ready
        let (Some(cloud_texture_view), Some(cloud_sampler), Some(pipeline)) = (
            cloud_render_texture.view.as_ref(),
            cloud_render_texture.sampler.as_ref(),
            pipeline_cache.get_render_pipeline(volumetric_clouds_composite_pipeline.pipeline_id),
        ) else {
            return Ok(());
        };

        // Create the bind group for the composite shader
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_composite_bind_group",
            &volumetric_clouds_composite_pipeline.layout,
            &BindGroupEntries::sequential((cloud_texture_view, cloud_sampler)),
        );

        // Begin the render pass to composite clouds onto the main view
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_composite_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu_types::Color::WHITE),
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });

        // Set the render pipeline, bind group, and draw a full-screen triangle
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Draw 3 vertices for a full-screen triangle

        Ok(())
    }
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(
            world.resource::<AssetServer>(),
            "shaders/world_rendering_composite.wgsl"
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

        let layout = render_device.create_bind_group_layout(
            "volumetric_clouds_composite_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("volumetric_clouds_composite_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        });

        Self {
            layout,
            pipeline_id,
        }
    }
}
