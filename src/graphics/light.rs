use std::num::NonZeroU64;

use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_sized, texture_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntries, BindingResource, Buffer, BufferDescriptor, BufferUsages,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
    PipelineCache, Sampler, SamplerBindingType, ShaderStages, TexelCopyBufferInfo,
    TexelCopyBufferLayout, Texture, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use skybound::scene::FrameUniform;
use skybound::scene::climate::{LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH};

/// The cloud's shadow from the pole lights and the aur at every altitude,
/// traced by a compute pass into a buffer and copied into a 3D texture.
pub struct LightVolume {
    pub view: TextureView,
    texture: Texture,
    buffer: Buffer,
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    group: Option<BindGroup>,
}

impl LightVolume {
    pub fn new(world: &World) -> Self {
        let device = world.resource::<RenderDevice>();
        let map = || texture_2d(TextureSampleType::Float { filterable: true });
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<FrameUniform>() as u64)),
                sampler(SamplerBindingType::Filtering),
                map(),
                map(),
                storage_buffer_sized(false, None),
            ),
        );
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("cloud_light".into()),
                    layout: vec![BindGroupLayoutDescriptor::new("cloud_light", &entries)],
                    shader: world.resource::<super::SceneShader>().0.clone(),
                    entry_point: Some("cloud_light".into()),
                    ..default()
                });
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("cloud_light"),
            size: Self::extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&TextureViewDescriptor::default()),
            texture,
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some("cloud_light_staging"),
                size: u64::from(LIGHT_WIDTH * LIGHT_HEIGHT * LIGHT_LAYERS) * 8,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            layout: device.create_bind_group_layout("cloud_light", &entries),
            pipeline,
            group: None,
        }
    }

    pub fn prepare(
        &mut self,
        device: &RenderDevice,
        frame: BindingResource,
        sampler: &Sampler,
        maps: [&TextureView; 2],
    ) {
        self.group = Some(device.create_bind_group(
            "cloud_light",
            &self.layout,
            &BindGroupEntries::sequential((
                frame,
                sampler,
                maps[0],
                maps[1],
                self.buffer.as_entire_binding(),
            )),
        ));
    }

    pub fn draw(&self, context: &mut RenderContext, cache: &PipelineCache) -> bool {
        let (Some(pipeline), Some(group)) =
            (cache.get_compute_pipeline(self.pipeline), &self.group)
        else {
            return false;
        };
        let recorder = context.diagnostic_recorder();
        let span = recorder
            .as_ref()
            .map(|r| r.time_span(context.command_encoder(), "cloud_light"));
        {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloud_light"),
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                LIGHT_WIDTH.div_ceil(8),
                LIGHT_HEIGHT.div_ceil(8),
                LIGHT_LAYERS.div_ceil(2),
            );
        }
        context.command_encoder().copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(LIGHT_WIDTH * 8),
                    rows_per_image: Some(LIGHT_HEIGHT),
                },
            },
            self.texture.as_image_copy(),
            Self::extent(),
        );
        if let Some(span) = span {
            span.end(context.command_encoder());
        }
        true
    }

    const fn extent() -> Extent3d {
        Extent3d {
            width: LIGHT_WIDTH,
            height: LIGHT_HEIGHT,
            depth_or_array_layers: LIGHT_LAYERS,
        }
    }
}
