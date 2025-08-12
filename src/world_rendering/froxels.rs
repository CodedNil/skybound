use bevy::{
    asset::{RenderAssetUsages, load_embedded_asset},
    ecs::query::QueryItem,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedComputePipelineId,
            CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            PipelineCache, ShaderStages, StorageTextureAccess, TextureDimension, TextureFormat,
            TextureUsages,
            binding_types::{texture_storage_3d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
    shader::PipelineCacheError,
};

use crate::world_rendering::volumetrics::{
    CloudsViewUniform, CloudsViewUniformOffset, CloudsViewUniforms,
};

const RESOLUTION: [u32; 3] = [128, 80, 512];

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct FroxelsTexture {
    pub handle: Handle<Image>,
}

pub fn setup_froxels_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let total_bytes = (RESOLUTION[0] * RESOLUTION[1] * RESOLUTION[2] * 4) as usize;
    let data = vec![0u8; total_bytes];

    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION[0],
            height: RESOLUTION[1],
            depth_or_array_layers: RESOLUTION[2],
        },
        TextureDimension::D3,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });
    let handle = images.add(image);
    commands.insert_resource(FroxelsTexture { handle });
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FroxelsLabel;

#[derive(Default)]
pub struct FroxelsNode {
    state: FroxelsState,
}

#[derive(Default, PartialEq)]
enum FroxelsState {
    #[default]
    Loading,
    Running,
}

#[derive(Resource)]
pub struct FroxelsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

pub fn setup_froxels_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = render_device.create_bind_group_layout(
        Some("froxels_bind_group_layout"),
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_3d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly), // Froxels 3D texture
                uniform_buffer::<CloudsViewUniform>(true), // View uniforms
            ),
        ),
    );
    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("froxels_compute".into()),
        layout: vec![layout.clone()],
        shader: load_embedded_asset!(
            asset_server.as_ref(),
            "shaders/world_rendering_froxels.wgsl"
        ),
        entry_point: Some("generate".into()),
        ..default()
    });

    commands.insert_resource(FroxelsPipeline {
        layout,
        pipeline_id,
    });
}

impl ViewNode for FroxelsNode {
    type ViewQuery = &'static CloudsViewUniformOffset;

    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<FroxelsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        match self.state {
            FroxelsState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id) {
                    CachedPipelineState::Ok(_) => {
                        self.state = FroxelsState::Running;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing froxel shader:\n{err}")
                    }
                    _ => {}
                }
            }
            FroxelsState::Running => {}
        }
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_uniform_offset: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if self.state != FroxelsState::Running {
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FroxelsPipeline>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let tex_handle = world.resource::<FroxelsTexture>();

        let (Some(compute_pipeline), Some(gpu_image), Some(view_binding)) = (
            pipeline_cache.get_compute_pipeline(pipeline.pipeline_id),
            gpu_images.get(&tex_handle.handle),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
        ) else {
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "froxels_bind_group",
            &pipeline.layout,
            &BindGroupEntries::sequential((&gpu_image.texture_view, view_binding.clone())),
        );

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.set_pipeline(compute_pipeline);
        pass.dispatch_workgroups(RESOLUTION[0] / 8, RESOLUTION[1] / 8, RESOLUTION[2] / 8);

        Ok(())
    }
}
