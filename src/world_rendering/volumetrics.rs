use crate::{
    world::{PLANET_RADIUS, WorldData},
    world_rendering::noise,
};
use bevy::{
    core_pipeline::{FullscreenShader, prepass::ViewPrepassTextures},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Extract,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{
            AddressMode, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, DynamicUniformBuffer, Extent3d,
            FilterMode, FragmentState, LoadOp, Operations, PipelineCache,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, StoreOp, Texture,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureView, TextureViewDescriptor,
            binding_types::{sampler, texture_2d, texture_3d, texture_depth_2d, uniform_buffer},
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ExtractedWindows},
    },
};

// --- Uniform Definition ---
#[derive(Clone, ShaderType)]
struct CloudsViewUniform {
    time: f32, // Time since startup
    world_from_clip: Mat4,
    world_position: Vec3,
    planet_rotation: Vec4,
    planet_radius: f32,
    camera_offset: Vec3,
    latitude: f32,
    longitude: f32,
    sun_direction: Vec3,
}

#[derive(Resource)]
pub struct CloudsViewUniforms {
    uniforms: DynamicUniformBuffer<CloudsViewUniform>,
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
    offset: u32,
}

#[derive(Resource, Default, ExtractResource, Clone)]
pub struct ExtractedViewData {
    planet_rotation: Vec4,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
    camera_offset: Vec3,
    sun_direction: Vec3,
}

// --- Systems (Render World) ---
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
            sun_direction: (world_coords.sun_rotation * Vec3::Z).normalize(),
        });
    }
}

pub fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<CloudsViewUniforms>,
    views: Query<(Entity, &ExtractedView)>,
    time: Res<Time>,
    data: Res<ExtractedViewData>,
) {
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
        let view_from_clip = extracted_view.clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        commands.entity(entity).insert(CloudsViewUniformOffset {
            offset: writer.write(&CloudsViewUniform {
                time: time.elapsed_secs_wrapped(),
                world_from_clip: world_from_view * view_from_clip,
                world_position: extracted_view.world_from_view.translation(),
                planet_rotation: data.planet_rotation,
                planet_radius: data.planet_radius,
                camera_offset: data.camera_offset,
                latitude: data.latitude,
                longitude: data.longitude,
                sun_direction: data.sun_direction,
            }),
        });
    }
}

#[derive(Resource, Default)]
pub struct CloudRenderTexture {
    pub texture: Option<Texture>,
    pub view: Option<TextureView>,
    pub sampler: Option<Sampler>,
}

/// Render world system: Manages the creation and resizing of the intermediate cloud render target.
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

    // Define the desired size for the intermediate texture
    let new_size = Extent3d {
        width: primary_window.physical_width / 4,
        height: primary_window.physical_height / 4,
        depth_or_array_layers: 1,
    };

    // Update CloudRenderTexture
    let current_texture_size = cloud_render_texture.texture.as_ref().map(|t| t.size());
    if current_texture_size != Some(new_size) {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_render_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        // Update the resource with the newly created assets
        cloud_render_texture.texture = Some(texture);
        cloud_render_texture.view = Some(view);
        cloud_render_texture.sampler = Some(sampler);
    }
}

// --- Volumetric Clouds Render Pipeline (Pass 1: Renders clouds to intermediate texture) ---

/// Label for the volumetric clouds render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VolumetricCloudsLabel;

/// Render graph node for drawing volumetric clouds.
#[derive(Default)]
pub struct VolumetricCloudsNode;

impl ViewNode for VolumetricCloudsNode {
    type ViewQuery = (
        &'static CloudsViewUniformOffset,
        &'static ViewPrepassTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_uniform_offset, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let noise_texture_handle = world.resource::<noise::NoiseTextures>();

        // Ensure the intermediate data is ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(depth_view),
            Some(texture),
            Some(view),
            Some(base_noise),
            Some(detail_noise),
            Some(turbulence_noise),
            Some(weather_noise),
            Some(fog_noise),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            prepass_textures.depth_view(),
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            gpu_images.get(&noise_texture_handle.turbulence),
            gpu_images.get(&noise_texture_handle.weather),
            gpu_images.get(&noise_texture_handle.fog),
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
                depth_view,
                &base_noise.texture_view,
                &detail_noise.texture_view,
                &turbulence_noise.texture_view,
                &weather_noise.texture_view,
                &fog_noise.texture_view,
            )),
        );

        // Begin the render pass to draw clouds to the intermediate texture
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view, // Render to our intermediate cloud texture
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(default()), // Clear the texture before drawing
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });

        // Set the viewport to match the intermediate texture's size
        render_pass.set_viewport(
            0.0,
            0.0,
            texture.width() as f32,
            texture.height() as f32,
            0.0,
            1.0,
        );

        // Set the render pipeline, bind group, and draw a full-screen triangle
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1); // Draw 3 vertices for a full-screen triangle

        Ok(())
    }
}

/// Resource holding the ID and layout for the volumetric clouds render pipeline.
#[derive(Resource)]
struct VolumetricCloudsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    linear_sampler: Sampler,
}

/// Render world system: Sets up the pipeline for rendering volumetric clouds.
pub fn setup_volumetric_clouds_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Create a linear sampler for the volumetric clouds
    let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    // Define the bind group layout for the cloud rendering shader
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<CloudsViewUniform>(true), // View uniforms
                sampler(SamplerBindingType::Filtering),    // Linear sampler
                texture_depth_2d(),                        // Depth texture from prepass
                texture_3d(TextureSampleType::Float { filterable: true }), // Base noise texture
                texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Turbulence noise texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Weather noise texture
                texture_3d(TextureSampleType::Float { filterable: true }), // Fog noise texture
            ),
        ),
    );

    // Queue the render pipeline for creation
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/world_rendering.wgsl"), // Shader for cloud rendering
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(VolumetricCloudsPipeline {
        layout,
        pipeline_id,
        linear_sampler,
    });
}
