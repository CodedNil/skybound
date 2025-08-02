mod perlinworley;

use crate::{
    world::{CameraCoordinates, PLANET_RADIUS, SunLight, WorldCoordinates},
    world_rendering::perlinworley::{PerlinWorleyTextureHandle, setup_perlinworley_texture},
};
use bevy::{
    core_pipeline::{
        FullscreenShader,
        core_3d::graph::{Core3d, Node3d},
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    pbr::light_consts::lux,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        load_shader_library,
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_3d, texture_depth_2d, uniform_buffer,
                uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ExtractedWindows, ViewTarget},
    },
};

// --- Plugin Definition ---
pub struct WorldRenderingPlugin;
impl Plugin for WorldRenderingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/functions.wgsl");
        load_shader_library!(app, "shaders/sky.wgsl");
        load_shader_library!(app, "shaders/clouds.wgsl");
        load_shader_library!(app, "shaders/aur_fog.wgsl");
        load_shader_library!(app, "shaders/poles.wgsl");

        app.add_plugins((ExtractResourcePlugin::<PerlinWorleyTextureHandle>::default(),))
            .add_systems(Startup, setup_perlinworley_texture);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<CloudRenderTexture>()
            .init_resource::<CloudsGlobalUniforms>()
            .add_systems(
                ExtractSchedule,
                (extract_clouds_global_uniform, extract_clouds_view_uniform),
            )
            .add_systems(RenderStartup, setup_volumetric_clouds_pipeline)
            .add_systems(RenderStartup, setup_volumetric_clouds_composite_pipeline)
            .add_systems(Render, manage_textures.in_set(RenderSystems::Queue))
            .add_systems(
                Render,
                (
                    prepare_clouds_view_uniforms.in_set(RenderSystems::PrepareResources),
                    prepare_clouds_global_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            );

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsCompositeNode>>(
                Core3d,
                VolumetricCloudsCompositeLabel,
            )
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricCloudsLabel))
            .add_render_graph_edges(
                Core3d,
                (
                    VolumetricCloudsLabel,
                    VolumetricCloudsCompositeLabel,
                    Node3d::Bloom,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<CloudsViewUniforms>();
        }
    }
}

// --- Uniform Definition ---
#[derive(Default, Clone, Resource, ExtractResource, Reflect, ShaderType)]
struct CloudsGlobalUniform {
    time: f32, // Time since startup
    planet_radius: f32,
    sun_direction: Vec3,
    sun_intensity: f32,
}

#[derive(Resource, Default)]
struct CloudsGlobalUniforms {
    uniforms: UniformBuffer<CloudsGlobalUniform>,
}

#[derive(Clone, ShaderType)]
struct CloudsViewUniform {
    world_from_clip: Mat4,
    world_position: Vec3,
    planet_rotation: Vec4,
    latitude: f32,
    longitude: f32,
    latitude_meters: f32,
    longitude_meters: f32,
    altitude: f32,
}

#[derive(Resource)]
struct CloudsViewUniforms {
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
struct CloudsViewUniformOffset {
    offset: u32,
}

#[derive(Resource, Default, ExtractResource, Clone)]
struct ExtractedGlobalData {
    direction: Vec3,
    intensity: f32,
}

#[derive(Resource, Default, ExtractResource, Clone)]
struct ExtractedViewData {
    planet_rotation: Vec4,
    latitude: f32,
    longitude: f32,
    latitude_meters: f32,
    longitude_meters: f32,
    altitude: f32,
}

// --- Systems (Render World) ---
fn extract_clouds_global_uniform(
    mut commands: Commands,
    time: Extract<Res<Time>>,
    sun_query: Extract<Query<(&Transform, &DirectionalLight), With<SunLight>>>,
) {
    commands.insert_resource(**time);
    if let Ok((sun_transform, sun_light)) = sun_query.single() {
        commands.insert_resource(ExtractedGlobalData {
            direction: -sun_transform.forward().normalize(),
            intensity: sun_light.illuminance / lux::DIRECT_SUNLIGHT,
        });
    }
}

fn extract_clouds_view_uniform(
    mut commands: Commands,
    world_coords: Extract<Res<WorldCoordinates>>,
    camera_query: Extract<Query<(&Transform, &CameraCoordinates), With<Camera>>>,
) {
    if let Ok((camera_transform, camera_coordinates)) = camera_query.single() {
        let planet_rotation = camera_coordinates.planet_rotation(&world_coords, camera_transform);
        let latitude = camera_coordinates.latitude(planet_rotation, camera_transform);
        let longitude = camera_coordinates.longitude(planet_rotation, camera_transform);
        let latitude_meters = camera_coordinates.latitude_meters(latitude);
        let longitude_meters = camera_coordinates.longitude_meters(longitude, latitude);
        commands.insert_resource(ExtractedViewData {
            planet_rotation: planet_rotation.into(),
            latitude: latitude,
            longitude: longitude,
            latitude_meters: latitude_meters,
            longitude_meters: longitude_meters,
            altitude: camera_coordinates.altitude(camera_transform),
        });
    }
}

fn prepare_clouds_global_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut clouds_buffer: ResMut<CloudsGlobalUniforms>,
    time: Res<Time>,
    data: Res<ExtractedGlobalData>,
) {
    let buffer = clouds_buffer.uniforms.get_mut();
    buffer.planet_radius = PLANET_RADIUS;
    buffer.time = time.elapsed_secs_wrapped();
    buffer.sun_direction = data.direction;
    buffer.sun_intensity = data.intensity;
    clouds_buffer
        .uniforms
        .write_buffer(&render_device, &render_queue);
}

fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<CloudsViewUniforms>,
    views: Query<(Entity, &ExtractedView)>,
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
                world_from_clip: world_from_view * view_from_clip,
                world_position: extracted_view.world_from_view.translation(),
                planet_rotation: data.planet_rotation,
                latitude: data.latitude,
                longitude: data.longitude,
                latitude_meters: data.latitude_meters,
                longitude_meters: data.longitude_meters,
                altitude: data.altitude,
            }),
        });
    }
}

#[derive(Resource, Default)]
struct CloudRenderTexture {
    texture: Option<Texture>,
    view: Option<TextureView>,
    sampler: Option<Sampler>,
}

/// Render world system: Manages the creation and resizing of the intermediate cloud render target.
fn manage_textures(
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
        width: primary_window.physical_width / 2,
        height: primary_window.physical_height / 2,
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
struct VolumetricCloudsLabel;

/// Render graph node for drawing volumetric clouds.
#[derive(Default)]
struct VolumetricCloudsNode;

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
        let noise_texture_handle = world.resource::<PerlinWorleyTextureHandle>();

        // Ensure the intermediate data is ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(uniforms_binding),
            Some(depth_view),
            Some(texture),
            Some(view),
            Some(noise_gpu_image),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            world.resource::<CloudsGlobalUniforms>().uniforms.binding(),
            prepass_textures.depth_view(),
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
            gpu_images.get(&noise_texture_handle.handle),
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
                uniforms_binding,
                &volumetric_clouds_pipeline.linear_sampler,
                depth_view,
                &noise_gpu_image.texture_view,
                &noise_gpu_image.sampler,
            )),
        );

        // Begin the render pass to draw clouds to the intermediate texture
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view, // Render to our intermediate cloud texture
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()), // Clear the texture before drawing
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
fn setup_volumetric_clouds_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Create a linear sampler for the volumetric clouds
    let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
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
                uniform_buffer::<CloudsViewUniform>(true), // View uniforms (camera projection, etc.)
                uniform_buffer_sized(false, Some(CloudsGlobalUniform::min_size())),
                sampler(SamplerBindingType::Filtering), // Linear sampler
                texture_depth_2d(),                     // Depth texture from prepass
                texture_3d(TextureSampleType::Float { filterable: true }), // Noise texture
                sampler(SamplerBindingType::Filtering), // Noise sampler
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

// --- Volumetric Clouds Composite Render Pipeline (Pass 2: Composites clouds onto main scene) ---

/// Label for the volumetric clouds composite render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsCompositeLabel;

/// Render graph node for compositing volumetric clouds onto the main view.
#[derive(Default)]
struct VolumetricCloudsCompositeNode;

impl ViewNode for VolumetricCloudsCompositeNode {
    // Query for the main view target
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Get necessary resources from the render world
        let volumetric_clouds_composite_pipeline =
            world.resource::<VolumetricCloudsCompositePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();

        // Ensure the intermediate resources are ready
        let (Some(cloud_texture_view), Some(cloud_sampler), Some(pipeline)) = (
            cloud_render_texture.view.as_ref(),
            cloud_render_texture.sampler.as_ref(),
            pipeline_cache.get_render_pipeline(volumetric_clouds_composite_pipeline.pipeline_id),
        ) else {
            return Ok(());
        };

        // Get the main view's post-process write target (source is current scene, destination is where we write)
        let post_process = view_target.post_process_write();

        // Create the bind group for the composite shader
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_composite_bind_group",
            &volumetric_clouds_composite_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source, // The current scene's color texture
                cloud_texture_view,  // Our rendered clouds texture
                cloud_sampler,       // Sampler for the clouds texture
            )),
        );

        // Begin the render pass to composite clouds onto the main view
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_composite_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination, // Render to the main view target
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            ..default()
        });

        // Set the render pipeline, bind group, and draw a full-screen triangle
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Draw 3 vertices for a full-screen triangle

        Ok(())
    }
}

/// Resource holding the ID and layout for the volumetric clouds composite render pipeline.
#[derive(Resource)]
struct VolumetricCloudsCompositePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

/// Render world system: Sets up the pipeline for compositing volumetric clouds.
fn setup_volumetric_clouds_composite_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Define the bind group layout for the composite shader
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_composite_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT, // Bindings primarily for the fragment shader
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // Original scene color texture
                texture_2d(TextureSampleType::Float { filterable: true }), // Clouds texture
                sampler(SamplerBindingType::Filtering), // Sampler for both textures
            ),
        ),
    );

    // Queue the render pipeline for creation
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_composite_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/world_rendering_composite.wgsl"),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            ..default()
        }),
        ..default()
    });
    commands.insert_resource(VolumetricCloudsCompositePipeline {
        layout,
        pipeline_id,
    });
}
