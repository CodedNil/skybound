use bevy::{
    core_pipeline::FullscreenShader,
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingResource, CachedComputePipelineId, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, CompareFunction, ComputePipeline, ComputePipelineDescriptor,
            DepthBiasState, DepthStencilState, FragmentState, MultisampleState, PipelineCache,
            PrimitiveState, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerBindingType,
            ShaderStages, StencilState, TextureFormat, TextureSampleType, TextureView,
            binding_types::{
                sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
                texture_3d, texture_depth_2d, uniform_buffer_sized,
            },
        },
        renderer::RenderDevice,
    },
};
use skybound::scene::FrameUniform;
use std::num::NonZeroU64;

/// One resource a scene pass binds, by the kind of binding it takes. A
/// pass's bindings are numbered in the order they are listed.
pub enum Bind<'a> {
    Frame(BindingResource<'a>),
    Sampler(&'a Sampler),
    /// A filterable 2D texture.
    Float(&'a TextureView),
    /// A 2D texture that is only fetched from.
    Data(&'a TextureView),
    /// A filterable 3D texture.
    Volume(&'a TextureView),
    Depth(&'a TextureView),
    Storage(BindingResource<'a>),
    /// A storage buffer the pass writes.
    Output(BindingResource<'a>),
}

/// Everything needed to build scene bind groups and queue their pipelines.
pub struct Binder<'a> {
    pub device: &'a RenderDevice,
    pub cache: &'a PipelineCache,
    pub fullscreen: &'a FullscreenShader,
    pub shader: Handle<Shader>,
}

/// A fullscreen scene pass: its shader entry point and targets, and the
/// pipeline, queued once its bindings are first known.
pub struct Pass {
    entry: &'static str,
    formats: &'static [TextureFormat],
    depth: bool,
    pipeline: Option<CachedRenderPipelineId>,
}

/// A compute pass, queued like a [`Pass`].
pub struct Compute {
    entry: &'static str,
    pipeline: Option<CachedComputePipelineId>,
}

impl Bind<'_> {
    fn layout(&self, binding: u32) -> BindGroupLayoutEntry {
        let builder = match self {
            Bind::Frame(_) => {
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<FrameUniform>() as u64))
            }
            Bind::Sampler(_) => sampler(SamplerBindingType::Filtering),
            Bind::Float(_) => texture_2d(TextureSampleType::Float { filterable: true }),
            Bind::Data(_) => texture_2d(TextureSampleType::Float { filterable: false }),
            Bind::Volume(_) => texture_3d(TextureSampleType::Float { filterable: true }),
            Bind::Depth(_) => texture_depth_2d(),
            Bind::Storage(_) => storage_buffer_read_only_sized(false, None),
            Bind::Output(_) => storage_buffer_sized(false, None),
        };
        builder.build(binding, ShaderStages::FRAGMENT | ShaderStages::COMPUTE)
    }

    fn resource(&self) -> BindingResource<'_> {
        match self {
            Bind::Frame(resource) | Bind::Storage(resource) | Bind::Output(resource) => {
                resource.clone()
            }
            Bind::Sampler(sampler) => BindingResource::Sampler(sampler),
            Bind::Float(view) | Bind::Data(view) | Bind::Volume(view) | Bind::Depth(view) => {
                BindingResource::TextureView(view)
            }
        }
    }
}

impl Binder<'_> {
    /// A bind group of these resources, and the layout they imply.
    pub fn group(
        &self,
        label: &'static str,
        binds: &[Bind],
    ) -> (BindGroupLayoutDescriptor, BindGroup) {
        let entries: Vec<_> = (0..)
            .zip(binds)
            .map(|(index, bind)| bind.layout(index))
            .collect();
        let descriptor = BindGroupLayoutDescriptor::new(label, &entries);
        let resources: Vec<_> = (0..)
            .zip(binds)
            .map(|(binding, bind)| BindGroupEntry {
                binding,
                resource: bind.resource(),
            })
            .collect();
        let group = self.device.create_bind_group(
            label,
            &self.cache.get_bind_group_layout(&descriptor),
            &resources,
        );
        (descriptor, group)
    }
}

impl Pass {
    pub const fn new(entry: &'static str, formats: &'static [TextureFormat], depth: bool) -> Self {
        Self {
            entry,
            formats,
            depth,
            pipeline: None,
        }
    }

    /// Queue the pipeline for a layout, the first time one is known.
    pub fn queue(&mut self, binder: &Binder, layout: &BindGroupLayoutDescriptor) {
        if self.pipeline.is_some() {
            return;
        }
        self.pipeline = Some(
            binder
                .cache
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(self.entry.into()),
                    layout: vec![layout.clone()],
                    immediate_size: 0,
                    vertex: binder.fullscreen.to_vertex_state(),
                    primitive: PrimitiveState::default(),
                    depth_stencil: self.depth.then_some(DepthStencilState {
                        format: TextureFormat::Depth32Float,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(CompareFunction::Always),
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: binder.shader.clone(),
                        entry_point: Some(self.entry.into()),
                        constants: Vec::new(),
                        targets: self
                            .formats
                            .iter()
                            .map(|&format| {
                                Some(ColorTargetState {
                                    format,
                                    blend: None,
                                    write_mask: ColorWrites::ALL,
                                })
                            })
                            .collect(),
                        shader_defs: Vec::new(),
                    }),
                    zero_initialize_workgroup_memory: false,
                }),
        );
    }

    /// A bind group for this pass, queueing its pipeline on the first call.
    pub fn group(&mut self, binder: &Binder, binds: &[Bind]) -> BindGroup {
        let (layout, group) = binder.group(self.entry, binds);
        self.queue(binder, &layout);
        group
    }

    pub fn get<'a>(&self, cache: &'a PipelineCache) -> Option<&'a RenderPipeline> {
        cache.get_render_pipeline(self.pipeline?)
    }
}

impl Compute {
    pub const fn new(entry: &'static str) -> Self {
        Self {
            entry,
            pipeline: None,
        }
    }

    pub fn group(&mut self, binder: &Binder, binds: &[Bind]) -> BindGroup {
        let (layout, group) = binder.group(self.entry, binds);
        self.pipeline.get_or_insert_with(|| {
            binder
                .cache
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(self.entry.into()),
                    layout: vec![layout],
                    shader: binder.shader.clone(),
                    entry_point: Some(self.entry.into()),
                    ..default()
                })
        });
        group
    }

    pub fn get<'a>(&self, cache: &'a PipelineCache) -> Option<&'a ComputePipeline> {
        cache.get_compute_pipeline(self.pipeline?)
    }
}
