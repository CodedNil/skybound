use crate::{
    world::{PLANET_RADIUS, WorldData},
    world_rendering::noise::NoiseTextures,
};
use bevy::{
    asset::load_embedded_asset,
    core_pipeline::FullscreenShader,
    diagnostic::FrameCount,
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Extract,
        camera::TemporalJitter,
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
            binding_types::{sampler, texture_2d, texture_3d, uniform_buffer},
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
    prev_world_from_clip: Mat4,
    prev_world_position: Vec3,

    planet_rotation: Vec4,
    planet_center: Vec3,
    planet_radius: f32,
    latitude: f32,
    longitude: f32,
    sun_direction: Vec3,
}

#[derive(Resource, Default)]
pub struct PreviousViewData {
    clip_from_world: Mat4,
    world_from_clip: Mat4,
    world_position: Vec3,
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
    sun_direction: Vec3,
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
            sun_direction: (world_coords.sun_rotation * Vec3::Y).normalize(),
        });
    }
}

pub fn prepare_clouds_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut view_uniforms: ResMut<CloudsViewUniforms>,
    views: Query<(Entity, &ExtractedView, Option<&TemporalJitter>)>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
    data: Res<ExtractedViewData>,
    prev_view_data: Res<PreviousViewData>,
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
    for (entity, extracted_view, temporal_jitter) in &views {
        let viewport = extracted_view.viewport.as_vec4();
        let main_pass_viewport = viewport;

        let unjittered_projection = extracted_view.clip_from_view;
        let mut clip_from_view = unjittered_projection;

        if let Some(temporal_jitter) = temporal_jitter {
            temporal_jitter.jitter_projection(&mut clip_from_view, main_pass_viewport.zw());
        }

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();

        let clip_from_world = if temporal_jitter.is_some() {
            clip_from_view * view_from_world
        } else {
            extracted_view
                .clip_from_world
                .unwrap_or_else(|| clip_from_view * view_from_world)
        };

        let world_position = extracted_view.world_from_view.translation();
        commands.entity(entity).insert(CloudsViewUniformOffset {
            offset: writer.write(&CloudsViewUniform {
                time: time.elapsed_secs_wrapped(),
                frame_count: frame_count.0,

                clip_from_world,
                world_from_clip: world_from_view * view_from_clip,
                world_from_view,
                view_from_world,

                clip_from_view,
                view_from_clip,
                world_position,
                camera_offset: data.camera_offset,

                // Previous frame matrices for motion vectors
                prev_clip_from_world: prev_view_data.clip_from_world,
                prev_world_from_clip: prev_view_data.world_from_clip,
                prev_world_position: prev_view_data.world_position,

                planet_rotation: data.planet_rotation,
                planet_center: Vec3::new(world_position.x, world_position.y, -data.planet_radius),
                planet_radius: data.planet_radius,
                latitude: data.latitude,
                longitude: data.longitude,
                sun_direction: data.sun_direction,
            }),
        });
    }
}

/// System that runs late in the frame to update the `PreviousViewData` resource
pub fn update_previous_view_data(
    mut prev_view_data: ResMut<PreviousViewData>,
    views: Query<(&ExtractedView, Option<&TemporalJitter>)>,
) {
    // Get the main view
    if let Some((extracted_view, temporal_jitter)) = views.iter().next() {
        // Recalculate the main view's matrices exactly as before
        let viewport = extracted_view.viewport.as_vec4();
        let mut clip_from_view = extracted_view.clip_from_view;
        if let Some(jitter) = temporal_jitter {
            jitter.jitter_projection(&mut clip_from_view, viewport.zw());
        }
        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();

        // Store these as the "previous" for the next frame
        prev_view_data.clip_from_world = clip_from_view * view_from_world;
        prev_view_data.world_from_clip = world_from_view * view_from_clip;
        prev_view_data.world_position = extracted_view.world_from_view.translation();
    }
}

#[derive(Resource, Default)]
pub struct CloudRenderTexture {
    pub texture: Option<Texture>,
    pub view: Option<TextureView>,
    pub motion_texture: Option<Texture>,
    pub motion_view: Option<TextureView>,
    pub depth_texture: Option<Texture>,
    pub depth_view: Option<TextureView>,
    pub sampler: Option<Sampler>,
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

    // Define the desired size for the intermediate texture
    let new_size = Extent3d {
        width: primary_window.physical_width / 3,
        height: primary_window.physical_height / 3,
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

        let motion_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_motion_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let motion_view = motion_texture.create_view(&TextureViewDescriptor::default());

        let depth_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_depth_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        // Update the resource with the newly created assets
        cloud_render_texture.texture = Some(texture);
        cloud_render_texture.view = Some(view);
        cloud_render_texture.motion_texture = Some(motion_texture);
        cloud_render_texture.motion_view = Some(motion_view);
        cloud_render_texture.depth_texture = Some(depth_texture);
        cloud_render_texture.depth_view = Some(depth_view);
        cloud_render_texture.sampler = Some(sampler);
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct VolumetricsLabel;

#[derive(Default)]
pub struct VolumetricsNode;

#[derive(Resource)]
pub struct VolumetricsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
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
            Some(texture),
            Some(view),
            Some(motion_view),
            Some(volumetric_depth_view),
            Some(base_noise),
            Some(detail_noise),
            Some(turbulence_noise),
            Some(weather_noise),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<CloudsViewUniforms>().uniforms.binding(),
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
            cloud_render_texture.motion_view.as_ref(),
            cloud_render_texture.depth_view.as_ref(),
            gpu_images.get(&noise_texture_handle.base),
            gpu_images.get(&noise_texture_handle.detail),
            gpu_images.get(&noise_texture_handle.turbulence),
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
                &turbulence_noise.texture_view,
                &weather_noise.texture_view,
            )),
        );

        // Begin the render pass to draw clouds to the intermediate texture with motion vectors
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view, // Render to our intermediate cloud texture
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(default()), // Clear the texture before drawing
                        store: StoreOp::Store,
                    },
                }),
                Some(RenderPassColorAttachment {
                    view: motion_view, // Render motion vectors to second attachment
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(default()),
                        store: StoreOp::Store,
                    },
                }),
                Some(RenderPassColorAttachment {
                    view: volumetric_depth_view, // Render volumetric depth to third attachment
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(default()),
                        store: StoreOp::Store,
                    },
                }),
            ],
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

impl FromWorld for VolumetricsPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(
            world.resource::<AssetServer>(),
            "shaders/world_rendering.wgsl"
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let fullscreen_shader = world.resource::<FullscreenShader>();

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
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<CloudsViewUniform>(true), // View uniforms
                    sampler(SamplerBindingType::Filtering),    // Linear sampler
                    texture_3d(TextureSampleType::Float { filterable: true }), // Base noise texture
                    texture_3d(TextureSampleType::Float { filterable: true }), // Detail noise texture
                    texture_2d(TextureSampleType::Float { filterable: true }), // Turbulence noise texture
                    texture_2d(TextureSampleType::Float { filterable: true }), // Weather noise texture
                ),
            ),
        );

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("volumetric_clouds_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader,
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rg16Float, // Motion vectors
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::R32Float, // Volumetric depth
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
                ..default()
            }),
            ..default()
        });

        Self {
            layout,
            pipeline_id,
            linear_sampler,
        }
    }
}
