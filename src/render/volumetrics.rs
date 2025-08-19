use crate::{
    render::noise::NoiseTextures,
    world::{PLANET_RADIUS, WorldData},
};
use bevy::{
    asset::load_embedded_asset,
    diagnostic::FrameCount,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Extract,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            DynamicUniformBuffer, Extent3d, FilterMode, PipelineCache, Sampler, SamplerBindingType,
            SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess, TextureDescriptor,
            TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
            TextureViewDescriptor,
            binding_types::{sampler, texture_2d, texture_3d, texture_storage_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ExtractedWindows},
    },
};

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

pub fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<CloudsViewUniforms>,
    views: Query<(Entity, &ExtractedView), With<Camera3d>>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
    data: Res<ExtractedViewData>,
    mut prev_view_data: ResMut<PreviousViewData>,
) {
    // Halton sequence (2, 3) - 0.5
    const HALTON_SEQUENCE: [Vec2; 8] = [
        vec2(0.0, 0.0),
        vec2(0.0, -0.166_666_66),
        vec2(-0.25, 0.166_666_69),
        vec2(0.25, -0.388_888_9),
        vec2(-0.375, -0.055_555_552),
        vec2(0.125, 0.277_777_8),
        vec2(-0.125, -0.277_777_8),
        vec2(0.375, 0.055_555_582),
    ];

    let view_iter = views.iter();
    let view_count = view_iter.len();
    let Some(mut writer) =
        view_uniforms
            .uniforms
            .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, extracted_view) in &views {
        let viewport = extracted_view.viewport.as_vec4();
        let view_size = viewport.zw();

        let mut clip_from_view = extracted_view.clip_from_view;

        // Jitter the current frame from the Halton sequence
        let offset = HALTON_SEQUENCE[frame_count.0 as usize % HALTON_SEQUENCE.len()];
        let jitter = (offset * vec2(2.0, -2.0)) / view_size;
        clip_from_view.z_axis.x += jitter.x;
        clip_from_view.z_axis.y += jitter.y;

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = clip_from_view * view_from_world;
        let world_from_clip = world_from_view * view_from_clip;
        let world_position = extracted_view.world_from_view.translation();

        commands.entity(entity).insert(CloudsViewUniformOffset {
            offset: writer.write(&CloudsViewUniform {
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

                // Previous frame matrices for motion vectors
                prev_clip_from_world: prev_view_data.clip_from_world,

                planet_rotation: data.planet_rotation,
                planet_center: Vec3::new(world_position.x, world_position.y, -data.planet_radius),
                planet_radius: data.planet_radius,
                latitude: data.latitude,
                longitude: data.longitude,
            }),
        });

        prev_view_data.clip_from_world = extracted_view
            .clip_from_world
            .unwrap_or_else(|| extracted_view.clip_from_view * view_from_world);
    }
}

#[derive(Resource, Default)]
pub struct CloudRenderTexture {
    pub color_view: Option<TextureView>,
    pub motion_view: Option<TextureView>,
    pub depth_view: Option<TextureView>,
    pub sampler: Option<Sampler>,

    // Full-resolution history textures for TAA
    pub history_a_view: Option<TextureView>,
    pub history_b_view: Option<TextureView>,
}

pub fn manage_textures(
    mut cloud_render_texture: ResMut<CloudRenderTexture>,
    render_device: Res<RenderDevice>,
    windows: Res<ExtractedWindows>,
) {
    let Some(primary_window) = windows
        .primary
        .and_then(|entity| windows.windows.get(&entity))
    else {
        return;
    };

    // Low res textures
    let low_res_size = Extent3d {
        width: primary_window.physical_width / 3,
        height: primary_window.physical_height / 3,
        depth_or_array_layers: 1,
    };
    let current_low_res_size = cloud_render_texture
        .color_view
        .as_ref()
        .map(|t| t.texture().size());
    if current_low_res_size != Some(low_res_size) {
        let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;

        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_render_texture"),
            size: low_res_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: texture_usage,
            view_formats: &[],
        });
        cloud_render_texture.color_view =
            Some(texture.create_view(&TextureViewDescriptor::default()));

        let motion_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_motion_texture"),
            size: low_res_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: texture_usage,
            view_formats: &[],
        });
        cloud_render_texture.motion_view =
            Some(motion_texture.create_view(&TextureViewDescriptor::default()));

        let depth_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_depth_texture"),
            size: low_res_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: texture_usage,
            view_formats: &[],
        });
        cloud_render_texture.depth_view =
            Some(depth_texture.create_view(&TextureViewDescriptor::default()));

        cloud_render_texture.sampler = Some(render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        }));
    }

    // Full-resolution history textures
    let full_res_size = Extent3d {
        width: primary_window.physical_width,
        height: primary_window.physical_height,
        depth_or_array_layers: 1,
    };
    let current_history_size = cloud_render_texture
        .history_a_view
        .as_ref()
        .map(|t| t.texture().size());
    if current_history_size != Some(full_res_size) {
        let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
        let texture_format = TextureFormat::Rgba16Float;

        // Create History Texture A
        let history_a_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("taa_history_a_texture"),
            size: full_res_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: texture_format,
            usage: texture_usage,
            view_formats: &[],
        });
        cloud_render_texture.history_a_view = Some(history_a_texture.create_view(&default()));

        // Create History Texture B
        let history_b_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("taa_history_b_texture"),
            size: full_res_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: texture_format,
            usage: texture_usage,
            view_formats: &[],
        });
        cloud_render_texture.history_b_view = Some(history_b_texture.create_view(&default()));
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VolumetricsLabel;

#[derive(Default)]
pub struct VolumetricsNode;

#[derive(Resource)]
pub struct VolumetricsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
    linear_sampler: Sampler,
}

impl ViewNode for VolumetricsNode {
    type ViewQuery = &'static CloudsViewUniformOffset;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_uniform_offset: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let noise_texture_handle = world.resource::<NoiseTextures>();

        // Ensure the intermediate data is ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(color_view),
            Some(motion_view),
            Some(depth_view),
            Some(base_noise),
            Some(detail_noise),
            Some(weather_noise),
        ) = (
            pipeline_cache.get_compute_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            cloud_render_texture.color_view.as_ref(),
            cloud_render_texture.motion_view.as_ref(),
            cloud_render_texture.depth_view.as_ref(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            gpu_images.get(&noise_texture_handle.weather),
        )
        else {
            return Ok(());
        };

        // Create the bind group for the clouds shader
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                &volumetric_clouds_pipeline.linear_sampler,
                &base_noise.texture_view,
                &detail_noise.texture_view,
                &weather_noise.texture_view,
                color_view,
                motion_view,
                depth_view,
            )),
        );

        // Dispatch the compute pass
        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("volumetric_clouds_compute_pass"),
                    timestamp_writes: None,
                });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);

        // Get texture dimensions for workgroup calculation
        let size = color_view.texture().size();
        let workgroup_count_x = size.width.div_ceil(8);
        let workgroup_count_y = size.height.div_ceil(8);
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

        Ok(())
    }
}

impl FromWorld for VolumetricsPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader =
            load_embedded_asset!(world.resource::<AssetServer>(), "shaders/rendering.wgsl");
        let render_device = world.resource::<RenderDevice>();

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
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<CloudsViewUniform>(true), // View uniforms
                    sampler(SamplerBindingType::Filtering),    // Linear sampler
                    texture_3d(TextureSampleType::Float { filterable: true }), // Base noise texture
                    texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise texture
                    texture_2d(TextureSampleType::Float { filterable: true }), // Weather noise texture
                    // --- Bind output textures for writing ---
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly), // Output Color
                    texture_storage_2d(TextureFormat::Rg16Float, StorageTextureAccess::WriteOnly), // Output Motion
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly), // Output Depth
                ),
            ),
        );

        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("volumetric_clouds_pipeline".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Some("main".into()),
            zero_initialize_workgroup_memory: true,
        });

        Self {
            layout,
            pipeline_id,
            linear_sampler,
        }
    }
}
