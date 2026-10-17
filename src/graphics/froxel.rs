use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d, texture_3d,
    uniform_buffer_sized,
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
use skybound::scene::{FrameUniform, froxel as grid};
use std::{mem::size_of, num::NonZeroU64};

pub struct FroxelVolume {
    pub view: TextureView,
    texture: Texture,
    buffer: Buffer,
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
    group: Option<BindGroup>,
}
pub struct FroxelInputs<'a> {
    pub frame: BindingResource<'a>,
    pub sampler: &'a Sampler,
    pub images: [&'a TextureView; 8],
    pub clouds: BindingResource<'a>,
    pub grid: BindingResource<'a>,
}
impl FroxelVolume {
    pub fn new(world: &World) -> Self {
        let device = world.resource::<RenderDevice>();
        let float2 = || texture_2d(TextureSampleType::Float { filterable: true });
        let data2 = || texture_2d(TextureSampleType::Float { filterable: false });
        let float3 = || texture_3d(TextureSampleType::Float { filterable: true });
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<FrameUniform>() as u64)),
                sampler(SamplerBindingType::Filtering),
                float3(),
                float3(),
                float3(),
                float2(),
                data2(),
                float2(),
                float2(),
                data2(),
                storage_buffer_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        );
        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("froxel_medium".into()),
                    layout: vec![BindGroupLayoutDescriptor::new("froxel_medium", &entries)],
                    shader: world.resource::<super::SceneShader>().0.clone(),
                    entry_point: Some("scene::render::froxel::froxel_medium".into()),
                    ..default()
                });
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("froxel_medium"),
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
                label: Some("froxel_medium_staging"),
                size: (grid::COUNT * 8) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            layout: device.create_bind_group_layout("froxel_medium", &entries),
            pipeline,
            group: None,
        }
    }
    pub fn prepare(&mut self, device: &RenderDevice, inputs: FroxelInputs) {
        self.group = Some(device.create_bind_group(
            "froxel_medium",
            &self.layout,
            &BindGroupEntries::sequential((
                inputs.frame,
                inputs.sampler,
                inputs.images[0],
                inputs.images[1],
                inputs.images[2],
                inputs.images[3],
                inputs.images[4],
                inputs.images[5],
                inputs.images[6],
                inputs.images[7],
                self.buffer.as_entire_binding(),
                inputs.clouds,
                inputs.grid,
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
            .map(|r| r.time_span(context.command_encoder(), "froxel_medium"));
        {
            let mut pass = context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("froxel_medium"),
                    ..default()
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                grid::WIDTH.div_ceil(4),
                grid::HEIGHT.div_ceil(4),
                grid::DEPTH.div_ceil(4),
            );
        }
        context.command_encoder().copy_buffer_to_texture(
            TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(grid::WIDTH * 8),
                    rows_per_image: Some(grid::HEIGHT),
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
            width: grid::WIDTH,
            height: grid::HEIGHT,
            depth_or_array_layers: grid::DEPTH,
        }
    }
}
