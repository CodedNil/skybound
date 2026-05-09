use crate::{render::noise::NoiseTextures, world::WorldData};
use bevy::{
    camera::MainPassResolutionOverride,
    core_pipeline::prepass::ViewPrepassTextures,
    diagnostic::FrameCount,
    prelude::*,
    render::{
        Extract,
        camera::TemporalJitter,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, BufferUsages, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, DynamicUniformBuffer,
            FilterMode, FragmentState, LoadOp, MultisampleState, Operations, PipelineCache,
            PrimitiveState, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
            RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, StencilState, StoreOp, TextureFormat,
            TextureSampleType,
            binding_types::{sampler, texture_2d, texture_3d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};
pub use skybound_shared::ViewUniform;

#[derive(Resource, Default)]
pub struct PreviousViewData {
    clip_from_world: Mat4,
}

#[derive(Resource)]
pub struct ViewUniforms {
    pub uniforms: DynamicUniformBuffer<ViewUniform>,
}

impl FromWorld for ViewUniforms {
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
    latitude: f32,
    longitude: f32,
    camera_offset: Vec2,
}

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
            latitude: world_coords.latitude(camera_transform.translation),
            longitude: world_coords.longitude(camera_transform.translation),
            camera_offset: world_coords.camera_offset,
        });
    }
}

type ViewQuery = (
    Entity,
    &'static ExtractedView,
    Option<&'static TemporalJitter>,
    Option<&'static MainPassResolutionOverride>,
);

pub fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<ViewUniforms>,
    views: Query<ViewQuery, With<Camera3d>>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
    data: Res<ExtractedViewData>,
    mut prev_view_data: ResMut<PreviousViewData>,
) {
    view_uniforms.uniforms.clear();

    for (entity, extracted_view, temporal_jitter, resolution_override) in &views {
        let viewport = extracted_view.viewport.as_vec4();
        let view_size = resolution_override.map_or_else(|| viewport.zw(), |r| r.as_vec2());

        let mut clip_from_view = extracted_view.clip_from_view;
        if let Some(jitter) = temporal_jitter {
            jitter.jitter_projection(&mut clip_from_view, view_size);
        }

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = clip_from_view * view_from_world;
        let world_from_clip = world_from_view * view_from_clip;
        let world_position = extracted_view.world_from_view.translation();

        // Unjittered inverse-projection used for motion vector ray reconstruction only
        let world_from_clip_unjittered = world_from_view * extracted_view.clip_from_view.inverse();

        let offset = view_uniforms.uniforms.push(&ViewUniform {
            clip_from_world,
            world_from_clip,
            world_from_view,
            view_from_world,
            clip_from_view,
            view_from_clip,
            prev_clip_from_world: prev_view_data.clip_from_world,
            world_from_clip_unjittered,
            world_position: world_position.extend(0.0),
            camera_position: vec4(
                data.latitude,
                data.longitude,
                data.camera_offset.x,
                data.camera_offset.y,
            ),
            planet_rotation: data.planet_rotation,
            times: vec4(time.elapsed_secs_wrapped(), frame_count.0 as f32, 0.0, 0.0),
        });

        commands
            .entity(entity)
            .insert(CloudsViewUniformOffset { offset });

        prev_view_data.clip_from_world = extracted_view
            .clip_from_world
            .unwrap_or_else(|| extracted_view.clip_from_view * view_from_world);
    }

    view_uniforms
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

#[derive(Resource)]
pub struct RaymarchPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    linear_sampler: Sampler,
}

pub fn raymarch_pass(
    world: &World,
    mut render_context: RenderContext,
    view_query: Query<(
        &ExtractedView,
        &CloudsViewUniformOffset,
        &ViewTarget,
        &ViewPrepassTextures,
        Option<&MainPassResolutionOverride>,
    )>,
) {
    for (view, view_uniform_offset, view_target, prepass_textures, resolution_override) in
        &view_query
    {
        let volumetric_clouds_pipeline = world.resource::<RaymarchPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let noise_texture_handle = world.resource::<NoiseTextures>();

        // Ensure required resources are ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(base_noise),
            Some(detail_noise),
            Some(depth_view),
            Some(motion_view),
            Some(weather_noise),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<ViewUniforms>().uniforms.binding(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            prepass_textures.depth_view(),
            prepass_textures.motion_vectors_view(),
            gpu_images.get(&noise_texture_handle.weather),
        )
        else {
            continue;
        };

        // Create bind group for the fragment shader
        let device = render_context.render_device();
        let bind_group = device.create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &volumetric_clouds_pipeline.linear_sampler,
                &base_noise.texture_view,
                &detail_noise.texture_view,
                &weather_noise.texture_view,
            )),
        );

        // Build color attachments: main view + prepass motion. Use the prepass depth texture as
        // the depth-stencil attachment (it's a depth format and cannot be used as a color target).
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_fragment_pass"),
            color_attachments: &[
                Some(view_target.get_color_attachment()),
                Some(RenderPassColorAttachment {
                    view: motion_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let vp = view.viewport;
        let (vp_x, vp_y, vp_w, vp_h) = resolution_override
            .map_or((vp.x, vp.y, vp.z, vp.w), |override_size| {
                (0u32, 0u32, override_size.x, override_size.y)
            });
        render_pass.set_viewport(vp_x as f32, vp_y as f32, vp_w as f32, vp_h as f32, 0.0, 1.0);

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);
    }
}

impl FromWorld for RaymarchPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        // let shader = asset_server.load("shaders/raymarch.spv");
        let shader = bevy::asset::load_embedded_asset!(asset_server, "shaders/rendering.wgsl");
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

        let layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),    // View uniforms
                sampler(SamplerBindingType::Filtering), // Linear sampler
                texture_3d(TextureSampleType::Float { filterable: true }), // Base noise
                texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise
                texture_2d(TextureSampleType::Float { filterable: true }), // Weather noise texture
            ),
        );
        let layout_descriptor =
            BindGroupLayoutDescriptor::new("volumetric_clouds_bind_group_layout", &layout_entries);
        let layout = render_device
            .create_bind_group_layout("volumetric_clouds_bind_group_layout", &layout_entries);

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("volumetric_clouds_pipeline".into()),
            layout: vec![layout_descriptor],
            immediate_size: 0,
            vertex: fullscreen_shader.to_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                entry_point: Some("main".into()),
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
                shader_defs: Vec::new(),
            }),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            layout,
            pipeline_id,
            linear_sampler,
        }
    }
}
