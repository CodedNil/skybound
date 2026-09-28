use crate::{render::noise::NoiseTextures, ships::player::ExtractedShipData, world::WorldData};
use bevy::{
    camera::MainPassResolutionOverride,
    core_pipeline::FullscreenShader,
    core_pipeline::prepass::ViewPrepassTextures,
    diagnostic::FrameCount,
    ecs::system::SystemParam,
    prelude::*,
    render::{
        Extract,
        camera::TemporalJitter,
        render_asset::RenderAssets,
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, BindingResource, Buffer, BufferBinding, BufferDescriptor,
            BufferUsages, CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction,
            DepthBiasState, DepthStencilState, FilterMode, FragmentState, LoadOp, MultisampleState,
            Operations, PipelineCache, PrimitiveState, RawBufferVec, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, StencilState, StoreOp,
            TextureFormat, TextureSampleType,
            binding_types::{sampler, texture_2d, texture_3d, uniform_buffer_sized},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};
use skybound_gpu::{ShipUniform, ViewUniform};
use std::{mem::size_of, num::NonZeroU64};

#[derive(Resource, Default)]
pub struct PreviousViewData {
    clip_from_world: Mat4,
}

#[derive(Resource)]
pub struct ViewUniforms {
    buffer: Option<Buffer>,
    buffer_size: u64,
    stride: usize,
    values: Vec<ViewUniform>,
    staging: Vec<u8>,
}

impl ViewUniforms {
    fn clear(&mut self) {
        self.values.clear();
    }

    fn push(&mut self, value: ViewUniform) -> u32 {
        let offset = u32::try_from(self.values.len() * self.stride)
            .expect("view uniform buffer exceeds dynamic-offset range");
        self.values.push(value);
        offset
    }

    fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        if self.values.is_empty() {
            return;
        }

        let size = self.stride * self.values.len();
        self.staging.resize(size, 0);
        self.staging.fill(0);
        for (index, value) in self.values.iter().enumerate() {
            let offset = index * self.stride;
            self.staging[offset..offset + size_of::<ViewUniform>()]
                .copy_from_slice(bytemuck::bytes_of(value));
        }

        if size as u64 > self.buffer_size {
            self.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: Some("view_uniforms_buffer"),
                size: size as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.buffer_size = size as u64;
        }

        queue.write_buffer(
            self.buffer
                .as_ref()
                .expect("view uniform buffer was created"),
            0,
            &self.staging,
        );
    }

    pub(crate) fn binding(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer.as_ref()?,
            offset: 0,
            size: NonZeroU64::new(size_of::<ViewUniform>() as u64),
        }))
    }
}

#[derive(Resource)]
pub struct ShipUniforms(RawBufferVec<ShipUniform>);

impl Default for ShipUniforms {
    fn default() -> Self {
        Self(RawBufferVec::new(BufferUsages::UNIFORM))
    }
}

pub fn prepare_ship_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ship_uniforms: ResMut<ShipUniforms>,
    data: Res<ExtractedShipData>,
) {
    ship_uniforms.0.clear();
    ship_uniforms.0.push(data.uniform);
    ship_uniforms.0.write_buffer(&render_device, &render_queue);
}

impl FromWorld for ViewUniforms {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let offset_alignment = render_device.limits().min_uniform_buffer_offset_alignment as usize;
        let stride = size_of::<ViewUniform>().next_multiple_of(offset_alignment.max(1));

        Self {
            buffer: None,
            buffer_size: 0,
            stride,
            values: Vec::new(),
            staging: Vec::new(),
        }
    }
}

#[derive(Component)]
pub struct CloudsViewUniformOffset {
    pub offset: u32,
}

#[derive(Resource, Default, Clone)]
pub struct ExtractedViewData {
    planet_rotation: Quat,
    latitude: f32,
    longitude: f32,
    camera_offset: Vec3,
}

pub fn extract_clouds_view_uniform(
    mut commands: Commands,
    time: Extract<Res<Time>>,
    world_coords: Extract<Res<WorldData>>,
    camera_query: Extract<Query<&Transform, With<Camera>>>,
) {
    commands.insert_resource(**time);
    if let Ok(camera_transform) = camera_query.single() {
        let (planet_rotation, latitude, longitude) =
            world_coords.planet_frame(camera_transform.translation);
        commands.insert_resource(ExtractedViewData {
            planet_rotation,
            latitude,
            longitude,
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

#[derive(SystemParam)]
pub(super) struct PrepareCloudsViewUniforms<'w, 's> {
    commands: Commands<'w, 's>,
    render_device: Res<'w, RenderDevice>,
    render_queue: Res<'w, RenderQueue>,
    view_uniforms: ResMut<'w, ViewUniforms>,
    views: Query<'w, 's, ViewQuery, With<Camera3d>>,
    time: Res<'w, Time>,
    frame_count: Res<'w, FrameCount>,
    data: Res<'w, ExtractedViewData>,
    prev_view_data: ResMut<'w, PreviousViewData>,
}

pub(super) fn prepare_clouds_view_uniforms(
    PrepareCloudsViewUniforms {
        mut commands,
        render_device,
        render_queue,
        mut view_uniforms,
        views,
        time,
        frame_count,
        data,
        mut prev_view_data,
    }: PrepareCloudsViewUniforms,
) {
    view_uniforms.clear();

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

        let offset = view_uniforms.push(ViewUniform {
            clip_from_world,
            world_from_clip,
            world_from_view,
            view_from_world,
            clip_from_view,
            view_from_clip,
            prev_clip_from_world: prev_view_data.clip_from_world,
            world_from_clip_unjittered,
            world_position: world_position.extend(data.camera_offset.z),
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

    view_uniforms.write_buffer(&render_device, &render_queue);
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

        let ship_uniforms = world.resource::<ShipUniforms>();

        // Ensure required resources are ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(depth_view),
            Some(motion_view),
            Some(normal_view),
            Some(base_noise),
            Some(detail_noise),
            Some(weather_noise),
            Some(extra_noise),
            Some(ship_uniform_binding),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<ViewUniforms>().binding(),
            prepass_textures.depth_only_view(),
            prepass_textures.motion_vectors_view(),
            prepass_textures.normal_view(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            gpu_images.get(&noise_texture_handle.weather),
            gpu_images.get(&noise_texture_handle.extra),
            ship_uniforms.0.binding(),
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
                &extra_noise.texture_view,
                ship_uniform_binding,
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
                Some(RenderPassColorAttachment {
                    view: normal_view,
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
        let shader = world.resource::<super::SkyboundGpuShader>().0.clone();
        let render_device = world.resource::<RenderDevice>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

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
                uniform_buffer_sized(true, NonZeroU64::new(size_of::<ViewUniform>() as u64)), // 0: View uniforms
                sampler(SamplerBindingType::Filtering), // 1: Linear sampler
                texture_3d(TextureSampleType::Float { filterable: true }), // 2: Base noise
                texture_3d(TextureSampleType::Float { filterable: true }), // 3: Detail noise
                texture_2d(TextureSampleType::Float { filterable: true }), // 4: Weather noise
                texture_2d(TextureSampleType::Float { filterable: true }), // 5: Extra noise
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<ShipUniform>() as u64)), // 6: Ship uniform
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
                constants: Vec::new(),
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
                    Some(ColorTargetState {
                        format: TextureFormat::Rgb10a2Unorm,
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
