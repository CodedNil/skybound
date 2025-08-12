use bevy::{
    core_pipeline::FullscreenShader,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendState,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, LoadOp,
            Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, SamplerBindingType, ShaderStages, StoreOp, TextureFormat,
            TextureSampleType,
            binding_types::{sampler, texture_2d},
        },
        renderer::{RenderContext, RenderDevice},
        view::ViewTarget,
    },
};

use crate::world_rendering::volumetrics::CloudRenderTexture;

// --- Volumetric Clouds Composite Render Pipeline (Pass 2: Composites clouds onto main scene) ---

/// Label for the volumetric clouds composite render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VolumetricCloudsCompositeLabel;

/// Render graph node for compositing volumetric clouds onto the main view.
#[derive(Default)]
pub struct VolumetricCloudsCompositeNode;

impl ViewNode for VolumetricCloudsCompositeNode {
    // Query for the main view target
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Get necessary resources from the render world
        let volumetric_clouds_composite_pipeline =
            world.resource::<VolumetricCloudsCompositePipeline>();
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

        // Get the main view's post-process write target (source is current scene, destination is where we write)
        let post_process = view_target.post_process_write();

        // Create the bind group for the composite shader
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_composite_bind_group",
            &volumetric_clouds_composite_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source, // The current scene's color texture
                cloud_texture_view,  // Our rendered clouds texture
                cloud_sampler,       // Sampler for the clouds texture
            )),
        );

        // Begin the render pass to composite clouds onto the main view
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_composite_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination, // Render to the main view target
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
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

/// Resource holding the ID and layout for the volumetric clouds composite render pipeline.
#[derive(Resource)]
struct VolumetricCloudsCompositePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

/// Render world system: Sets up the pipeline for compositing volumetric clouds.
pub fn setup_volumetric_clouds_composite_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Define the bind group layout for the composite shader
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_composite_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT, // Bindings primarily for the fragment shader
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // Original scene color texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Clouds texture
                sampler(SamplerBindingType::Filtering), // Sampler for both textures
            ),
        ),
    );

    // Queue the render pipeline for creation
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_composite_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/world_rendering_composite.wgsl"),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(VolumetricCloudsCompositePipeline {
        layout,
        pipeline_id,
    });
}
