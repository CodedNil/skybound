use crate::graphics::binding::{Bind, Binder, Compute};
use bevy::{
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_resource::{
            BindGroup, BindingResource, Buffer, BufferDescriptor, BufferUsages,
            ComputePassDescriptor, Extent3d, PipelineCache, Sampler, TexelCopyBufferInfo,
            TexelCopyBufferLayout, Texture, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureView, TextureViewDescriptor,
        },
        renderer::{RenderContext, RenderDevice},
    },
};
use skybound::scene::climate::{LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH};

/// The cloud's shadow from the pole lights and the aur at every altitude,
/// traced by a compute pass into a buffer and copied into a 3D texture.
pub struct LightVolume {
    pub view: TextureView,
    texture: Texture,
    buffer: Buffer,
    pass: Compute,
    group: Option<BindGroup>,
}

impl LightVolume {
    pub fn new(device: &RenderDevice) -> Self {
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
            pass: Compute::new("cloud_light"),
            group: None,
        }
    }

    pub fn prepare(
        &mut self,
        binder: &Binder,
        frame: BindingResource,
        sampler: &Sampler,
        maps: [&TextureView; 2],
    ) {
        self.group = Some(self.pass.group(
            binder,
            &[
                Bind::Frame(frame),
                Bind::Sampler(sampler),
                Bind::Float(maps[0]),
                Bind::Float(maps[1]),
                Bind::Output(self.buffer.as_entire_binding()),
            ],
        ));
    }

    pub fn draw(&self, context: &mut RenderContext, cache: &PipelineCache) -> bool {
        let (Some(pipeline), Some(group)) = (self.pass.get(cache), &self.group) else {
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
                LIGHT_LAYERS.div_ceil(4),
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
