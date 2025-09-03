use crate::{
    render::{
        clouds::{CloudsBuffer, CloudsBufferData},
        noise::NoiseTextures,
    },
    world::{PLANET_RADIUS, WorldData},
};
use bevy::{
    asset::load_embedded_asset,
    core_pipeline::prepass::ViewPrepassTextures,
    diagnostic::FrameCount,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Extract,
        camera::TemporalJitter,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
            DepthStencilState, DynamicUniformBuffer, FilterMode, FragmentState, MultisampleState,
            PipelineCache, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, ShaderType, TextureFormat, TextureSampleType,
            binding_types::{sampler, storage_buffer_read_only, texture_3d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::ExtractedView,
    },
};
use wgpu_types::{LoadOp, Operations, StoreOp};

#[derive(Clone, ShaderType)]
pub struct CloudsViewUniform {
    time: f32,
    frame_count: u32,

    clip_from_world: Mat4,
    world_from_clip: Mat4,
    world_from_view: Mat4,
    view_from_world: Mat4,

    clip_from_view: Mat4,
    view_from_clip: Mat4,
    world_position: Vec3,
    camera_offset: Vec2,

    // Previous frame matrices for motion vectors
    prev_clip_from_world: Mat4,

    planet_rotation: Vec4,
    planet_center: Vec3,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
}

#[derive(Resource, Default)]
pub struct PreviousViewData {
    clip_from_world: Mat4,
}

#[derive(Resource)]
pub struct CloudsViewUniforms {
    pub uniforms: DynamicUniformBuffer<CloudsViewUniform>,
}

impl FromWorld for CloudsViewUniforms {
    /// Create dynamic uniform storage for per-view cloud uniforms.
    fn from_world(world: &mut World) -> Self {
        let mut uniforms = DynamicUniformBuffer::default();
        uniforms.set_label(Some("view_uniforms_buffer"));

        let render_device = world.resource::<RenderDevice>();
        if render_device.limits().max_storage_buffers_per_shader_stage > 0 {
            uniforms.add_usages(BufferUsages::STORAGE);
        }

        Self { uniforms }
    }
}

#[derive(Component)]
pub struct CloudsViewUniformOffset {
    pub offset: u32,
}

#[derive(Resource, Default, ExtractResource, Clone)]
pub struct ExtractedViewData {
    planet_rotation: Vec4,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
    camera_offset: Vec2,
}

/// Extract per-frame view and world data for cloud rendering into the render world.
pub fn extract_clouds_view_uniform(
    mut commands: Commands,
    time: Extract<Res<Time>>,
    world_coords: Extract<Res<WorldData>>,
    camera_query: Extract<Query<&Transform, With<Camera>>>,
) {
    commands.insert_resource(**time);
    if let Ok(camera_transform) = camera_query.single() {
        commands.insert_resource(ExtractedViewData {
            planet_rotation: Vec4::from(world_coords.planet_rotation(camera_transform.translation)),
            planet_radius: PLANET_RADIUS,
            latitude: world_coords.latitude(camera_transform.translation),
            longitude: world_coords.longitude(camera_transform.translation),
            camera_offset: world_coords.camera_offset,
        });
    }
}

/// Prepare and write per-view cloud uniform blocks (matrices, jitter, TAA data).
pub fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<CloudsViewUniforms>,
    views: Query<(Entity, &ExtractedView, &TemporalJitter), With<Camera3d>>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
    data: Res<ExtractedViewData>,
    mut prev_view_data: ResMut<PreviousViewData>,
) {
    let view_count = views.iter().len();
    let Some(mut writer) =
        view_uniforms
            .uniforms
            .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, extracted_view, temporal_jitter) in &views {
        let viewport = extracted_view.viewport.as_vec4();
        let view_size = viewport.zw();

        let mut clip_from_view = extracted_view.clip_from_view;
        temporal_jitter.jitter_projection(&mut clip_from_view, view_size);

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = clip_from_view * view_from_world;
        let world_from_clip = world_from_view * view_from_clip;
        let world_position = extracted_view.world_from_view.translation();

        let offset = writer.write(&CloudsViewUniform {
            time: time.elapsed_secs_wrapped(),
            frame_count: frame_count.0,

            clip_from_world,
            world_from_clip,
            world_from_view,
            view_from_world,

            clip_from_view,
            view_from_clip,
            world_position,
            camera_offset: data.camera_offset,

            prev_clip_from_world: prev_view_data.clip_from_world,

            planet_rotation: data.planet_rotation,
            planet_center: Vec3::new(world_position.x, world_position.y, -data.planet_radius),
            planet_radius: data.planet_radius,
            latitude: data.latitude,
            longitude: data.longitude,
        });

        commands
            .entity(entity)
            .insert(CloudsViewUniformOffset { offset });

        prev_view_data.clip_from_world = extracted_view
            .clip_from_world
            .unwrap_or_else(|| extracted_view.clip_from_view * view_from_world);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct RaymarchLabel;

#[derive(Default)]
pub struct RaymarchNode;

#[derive(Resource)]
pub struct RaymarchPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    linear_sampler: Sampler,
}

impl ViewNode for RaymarchNode {
    type ViewQuery = (
        &'static CloudsViewUniformOffset,
        &'static bevy::render::view::ViewTarget,
        &'static ViewPrepassTextures,
    );

    /// Render the volumetric clouds in a fragment shader.
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_uniform_offset, view_target, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<RaymarchPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let noise_texture_handle = world.resource::<NoiseTextures>();

        // Ensure required resources are ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(clouds_buffer),
            Some(base_noise),
            Some(detail_noise),
            Some(depth_view),
            Some(motion_view),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            world.get_resource::<CloudsBuffer>(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            prepass_textures.depth_view(),
            prepass_textures.motion_vectors_view(),
        )
        else {
            return Ok(());
        };

        // Create bind group for the fragment shader
        let device = render_context.render_device();
        let bind_group = device.create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &volumetric_clouds_pipeline.linear_sampler,
                clouds_buffer.buffer.as_entire_binding(),
                &base_noise.texture_view,
                &detail_noise.texture_view,
            )),
        );

        // Build color attachments: main view + prepass motion. Use the prepass depth texture as
        // the depth-stencil attachment (it's a depth format and cannot be used as a color target).
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_fragment_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: view_target.main_texture_view(),
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu_types::Color::BLACK),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(RenderPassColorAttachment {
                    view: motion_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu_types::Color::BLACK),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

impl FromWorld for RaymarchPipeline {
    /// Create the render pipeline, bind group layout and samplers used for raymarching.
    fn from_world(world: &mut World) -> Self {
        let shader =
            load_embedded_asset!(world.resource::<AssetServer>(), "shaders/rendering.wgsl");
        let render_device = world.resource::<RenderDevice>();
        let fullscreen_shader = world.resource::<bevy::core_pipeline::FullscreenShader>();

        let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        let layout = render_device.create_bind_group_layout(
            "volumetric_clouds_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<CloudsViewUniform>(true), // View uniforms
                    sampler(SamplerBindingType::Filtering),    // Linear sampler
                    storage_buffer_read_only::<CloudsBufferData>(false), // Clouds data buffer
                    texture_3d(TextureSampleType::Float { filterable: true }), // Base noise
                    texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise
                ),
            ),
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("volumetric_clouds_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rg16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
                ..default()
            }),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::LessEqual,
                stencil: default(),
                bias: default(),
            }),
            ..default()
        });

        Self {
            layout,
            pipeline_id,
            linear_sampler,
        }
    }
}
