use crate::world_rendering::{
    noise::NoiseTextures,
    volumetrics::{CloudsViewUniform, CloudsViewUniformOffset, CloudsViewUniforms},
};
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
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            FilterMode, PipelineCache, Sampler, SamplerBindingType, SamplerDescriptor,
            ShaderStages, StorageTextureAccess, TextureDimension, TextureFormat, TextureSampleType,
            TextureUsages,
            binding_types::{sampler, texture_2d, texture_3d, texture_storage_3d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
    },
};

const RESOLUTION: UVec3 = UVec3::new(128, 80, 1024);

#[derive(Resource, Component, ExtractResource, Clone)]
pub struct FroxelsTexture {
    pub handle: Handle<Image>,
}

pub fn setup_froxels_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let total_bytes = (RESOLUTION.x * RESOLUTION.y * RESOLUTION.z * 4) as usize;
    let data = vec![0u8; total_bytes];

    let mut image = Image::new(
        Extent3d {
            width: RESOLUTION.x,
            height: RESOLUTION.y,
            depth_or_array_layers: RESOLUTION.z,
        },
        TextureDimension::D3,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToBorder,
        address_mode_v: ImageAddressMode::ClampToBorder,
        address_mode_w: ImageAddressMode::ClampToBorder,
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
pub struct FroxelsNode {}

#[derive(Resource)]
pub struct FroxelsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
    linear_sampler: Sampler,
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
                sampler(SamplerBindingType::Filtering),    // Linear sampler
                texture_3d(TextureSampleType::Float { filterable: true }), // Base noise texture
                texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Turbulence noise texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Weather noise texture
                texture_3d(TextureSampleType::Float { filterable: true }), // Fog noise texture
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

    let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    commands.insert_resource(FroxelsPipeline {
        layout,
        pipeline_id,
        linear_sampler,
    });
}

impl ViewNode for FroxelsNode {
    type ViewQuery = &'static CloudsViewUniformOffset;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_uniform_offset: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FroxelsPipeline>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let tex_handle = world.resource::<FroxelsTexture>();
        let noise_texture_handle = world.resource::<NoiseTextures>();

        let (
            Some(compute_pipeline),
            Some(view_binding),
            Some(froxels_texture),
            Some(base_noise),
            Some(detail_noise),
            Some(turbulence_noise),
            Some(weather_noise),
            Some(fog_noise),
        ) = (
            pipeline_cache.get_compute_pipeline(pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            gpu_images.get(&tex_handle.handle),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            gpu_images.get(&noise_texture_handle.turbulence),
            gpu_images.get(&noise_texture_handle.weather),
            gpu_images.get(&noise_texture_handle.fog),
        )
        else {
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "froxels_bind_group",
            &pipeline.layout,
            &BindGroupEntries::sequential((
                &froxels_texture.texture_view,
                view_binding.clone(),
                &pipeline.linear_sampler,
                &base_noise.texture_view,
                &detail_noise.texture_view,
                &turbulence_noise.texture_view,
                &weather_noise.texture_view,
                &fog_noise.texture_view,
            )),
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
