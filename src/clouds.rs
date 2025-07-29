use bevy::{
    core_pipeline::{
        FullscreenShader,
        core_3d::graph::{Core3d, Node3d},
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        globals::{GlobalsBuffer, GlobalsUniform},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_depth_2d, uniform_buffer, uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedWindows, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};

const MAX_VISIBLE: usize = 2048;

// --- Plugin Definition ---
pub struct CloudsPlugin;
impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<CloudRenderTexture>()
            .add_systems(RenderStartup, setup_volumetric_clouds_pipeline)
            .add_systems(RenderStartup, setup_volumetric_clouds_composite_pipeline)
            .add_systems(Render, manage_render_target.in_set(RenderSystems::Queue));

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsCompositeNode>>(
                Core3d,
                VolumetricCloudsCompositeLabel,
            )
            // Clouds are rendered after the main 3D pass
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricCloudsLabel))
            // Compositing happens after clouds are rendered, before bloom
            .add_render_graph_edges(
                Core3d,
                (
                    VolumetricCloudsLabel,
                    VolumetricCloudsCompositeLabel,
                    Node3d::Bloom,
                ),
            );
    }
}

// --- Systems (Render World) ---

/// Resource to manage the intermediate render target for clouds.
#[derive(Resource, Default)]
struct CloudRenderTexture {
    texture: Option<Texture>,
    view: Option<TextureView>,
    sampler: Option<Sampler>,
}

/// Render world system: Manages the creation and resizing of the intermediate cloud render target.
fn manage_render_target(
    mut cloud_render_texture: ResMut<CloudRenderTexture>,
    render_device: Res<RenderDevice>,
    windows: Res<ExtractedWindows>,
) {
    let Some(primary_window) = windows
        .primary
        .and_then(|entity| windows.windows.get(&entity))
    else {
        return;
    };

    // Define the desired size for the intermediate texture
    let new_size = Extent3d {
        width: primary_window.physical_width / 2,
        height: primary_window.physical_height / 2,
        depth_or_array_layers: 1,
    };

    let current_texture_size = cloud_render_texture.texture.as_ref().map(|t| t.size());
    if current_texture_size != Some(new_size) {
        // Create the new texture, view, and sampler
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_render_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        // Update the resource with the newly created assets
        cloud_render_texture.texture = Some(texture);
        cloud_render_texture.view = Some(view);
        cloud_render_texture.sampler = Some(sampler);
    }
}

// --- Volumetric Clouds Render Pipeline (Pass 1: Renders clouds to intermediate texture) ---

/// Label for the volumetric clouds render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsLabel;

/// Render graph node for drawing volumetric clouds.
#[derive(Default)]
struct VolumetricCloudsNode;

impl ViewNode for VolumetricCloudsNode {
    type ViewQuery = (&'static ViewUniformOffset, &'static ViewPrepassTextures);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_uniform_offset, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();

        // Ensure the intermediate data is ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(texture),
            Some(view),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<ViewUniforms>().uniforms.binding(),
            world.resource::<GlobalsBuffer>().buffer.binding(),
            prepass_textures.depth_view(),
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
        )
        else {
            return Ok(());
        };

        // Create the bind group for the clouds shader
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                globals_binding.clone(),
                depth_view,
            )),
        );

        // Begin the render pass to draw clouds to the intermediate texture
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view, // Render to our intermediate cloud texture
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()), // Clear the texture before drawing
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });

        // Set the viewport to match the intermediate texture's size
        render_pass.set_viewport(
            0.0,
            0.0,
            texture.width() as f32,
            texture.height() as f32,
            0.0,
            1.0,
        );

        // Set the render pipeline, bind group, and draw a full-screen triangle
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1); // Draw 3 vertices for a full-screen triangle

        Ok(())
    }
}

/// Resource holding the ID and layout for the volumetric clouds render pipeline.
#[derive(Resource)]
struct VolumetricCloudsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

/// Render world system: Sets up the pipeline for rendering volumetric clouds.
fn setup_volumetric_clouds_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Define the bind group layout for the cloud rendering shader
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true), // View uniforms (camera projection, etc.)
                uniform_buffer_sized(false, Some(GlobalsUniform::min_size())), // Global uniforms (time, etc.)
                texture_depth_2d(), // Depth texture from prepass
            ),
        ),
    );

    // Queue the render pipeline for creation
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/clouds.wgsl"), // Shader for cloud rendering
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(VolumetricCloudsPipeline {
        layout,
        pipeline_id,
    });
}

// --- Volumetric Clouds Composite Render Pipeline (Pass 2: Composites clouds onto main scene) ---

/// Label for the volumetric clouds composite render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsCompositeLabel;

/// Render graph node for compositing volumetric clouds onto the main view.
#[derive(Default)]
struct VolumetricCloudsCompositeNode;

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
fn setup_volumetric_clouds_composite_pipeline(
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
            shader: asset_server.load("shaders/clouds_composite.wgsl"),
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
