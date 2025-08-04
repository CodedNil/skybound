use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        load_shader_library,
        render_asset::RenderAssets,
        render_graph::{self, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
            BindingType, CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, ShaderStages, StorageTextureAccess,
            TextureDimension, TextureFormat, TextureUsages, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
    shader::PipelineCacheError,
};

const RESOLUTION: u32 = 128;
const SHADER_ASSET_PATH: &str = "shaders/perlin_worley_compute.wgsl";

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct PerlinWorleyLabel;

pub struct PerlinWorleyPlugin;
impl Plugin for PerlinWorleyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_perlinworley_texture)
            .add_plugins(ExtractResourcePlugin::<PerlinWorleyTexture>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(RenderStartup, init_perlinworley_pipeline)
            .add_systems(
                Render,
                prepare_perlinworley_bind_group.in_set(RenderSystems::PrepareBindGroups),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(PerlinWorleyLabel, PerlinWorleyComputeNode::default());
    }
}

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct PerlinWorleyTexture {
    pub handle: Handle<Image>,
}

#[derive(Resource)]
struct PerlinWorleyPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

#[derive(Resource)]
struct PerlinWorleyBindGroup(BindGroup);

fn setup_perlinworley_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let bytes_per_pixel = 16;
    let total_bytes = (RESOLUTION * RESOLUTION * RESOLUTION * bytes_per_pixel) as usize;
    let data = vec![0u8; total_bytes];

    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION,
            height: RESOLUTION,
            depth_or_array_layers: RESOLUTION,
        },
        TextureDimension::D3,
        data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    let handle = images.add(image);
    commands.insert_resource(PerlinWorleyTexture { handle });
}

fn init_perlinworley_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = render_device.create_bind_group_layout(
        Some("perlin_worley_bind_group_layout"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba32Float,
                view_dimension: TextureViewDimension::D3,
            },
            count: None,
        }],
    );
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("perlin_worley_compute".into()),
        layout: vec![layout.clone()],
        shader: asset_server.load(SHADER_ASSET_PATH),
        entry_point: Some("generate_cloud_base".into()),
        ..default()
    });

    commands.insert_resource(PerlinWorleyPipeline {
        layout,
        pipeline_id,
    });
}

fn prepare_perlinworley_bind_group(
    mut commands: Commands,
    pipeline: Res<PerlinWorleyPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    tex_handle: Res<PerlinWorleyTexture>,
    render_device: Res<RenderDevice>,
) {
    if let Some(gpu_image) = gpu_images.get(&tex_handle.handle) {
        let bind_group = render_device.create_bind_group(
            Some("perlin_worley_bind_group"),
            &pipeline.layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&gpu_image.texture_view),
            }],
        );
        commands.insert_resource(PerlinWorleyBindGroup(bind_group));
    }
}

#[derive(Default)]
struct PerlinWorleyComputeNode {
    state: PerlinWorleyComputeState,
}
#[derive(Default, PartialEq)]
enum PerlinWorleyComputeState {
    #[default]
    Loading,
    Init,
    Ran,
}

impl render_graph::Node for PerlinWorleyComputeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<PerlinWorleyPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        match self.state {
            PerlinWorleyComputeState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id) {
                    CachedPipelineState::Ok(_) => {
                        self.state = PerlinWorleyComputeState::Init;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            PerlinWorleyComputeState::Init => {
                self.state = PerlinWorleyComputeState::Ran;
            }
            PerlinWorleyComputeState::Ran => {}
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if self.state != PerlinWorleyComputeState::Init {
            return Ok(());
        }

        let bind_group = &world.resource::<PerlinWorleyBindGroup>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<PerlinWorleyPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // Check if the pipeline is ready. If not, just return Ok and wait for the next frame.
        let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
            return Ok(());
        };

        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.set_pipeline(init_pipeline);
        let groups = RESOLUTION / 4;
        pass.dispatch_workgroups(groups, groups, groups);

        Ok(())
    }
}
