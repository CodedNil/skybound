use bevy::{
    core_pipeline::{
        FullscreenShader,
        core_3d::graph::{Core3d, Node3d},
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
        primitives::{Frustum, Sphere},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, storage_buffer_read_only, texture_2d, texture_depth_2d_multisampled,
                uniform_buffer, uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{ExtractedWindows, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};
use rand::{Rng, rng};
use std::cmp::Ordering;

const MAX_VISIBLE: usize = 2048;

// --- Plugin Definition ---
pub struct CloudsPlugin;
impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<CloudsBufferData>::default())
            .add_systems(Startup, setup)
            .add_systems(Update, update);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<CloudRenderTexture>()
            .add_systems(RenderStartup, setup_volumetric_clouds_pipeline)
            .add_systems(RenderStartup, setup_volumetric_clouds_composite_pipeline)
            .add_systems(Render, update_buffer.in_set(RenderSystems::Prepare))
            .add_systems(Render, manage_render_target.in_set(RenderSystems::Queue));

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsCompositeNode>>(
                Core3d,
                VolumetricCloudsCompositeLabel,
            )
            // Clouds are rendered after the main 3D pass
            .add_render_graph_edges(Core3d, (Node3d::EndMainPass, VolumetricCloudsLabel))
            // Compositing happens after clouds are rendered, before bloom
            .add_render_graph_edges(
                Core3d,
                (
                    VolumetricCloudsLabel,
                    VolumetricCloudsCompositeLabel,
                    Node3d::Bloom,
                ),
            );
    }
}

// --- Data Structures ---

/// Stores the current state of all clouds in the main world.
#[derive(Resource)]
struct CloudsState {
    clouds: Vec<Cloud>,
}

/// Buffer for visible cloud data, extracted to the render world and sent to the GPU.
#[derive(Resource, ExtractResource, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CloudsBufferData {
    num_clouds: u32,
    _padding: [u32; 3], // Padding to align clouds array to 16 bytes (Vec4 alignment)
    clouds: [Cloud; MAX_VISIBLE],
}

/// Represents a cloud with its properties
#[derive(Default, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Cloud {
    // 16 bytes
    pos: Vec3, // Position of the cloud (12 bytes)
    data: u32, // Packed data (4 bytes)
    // 6 bits for seed, as a number 0-64
    // 4 bits for density (Overall fill, 0=almost empty mist, 1=solid cloud mass), 0-1 in 16 steps
    // 4 bits for detail (Noise detail power, 0=smooth blob, 1=lots of little puffs), 0-1 in 16 steps
    // 4 bits for brightness, 0-1 in 16 steps
    // 4 bits for white-blue-purple (to be implemented)
    // 2 bits for form, cumulus status or cirrus, then vertical level determines cloud shape determined from both

    // 16 bytes
    scale: Vec3,    // x=width, y=height, z=length (12 bytes)
    _padding0: u32, // Padding for alignment
}

enum CloudForm {
    Cumulus,
    Stratus,
    Cirrus,
}

impl Cloud {
    // Bit masks and shifts as constants
    const FORM_MASK: u32 = 0b11; // Bits 0-1
    const FORM_SHIFT: u32 = 0;
    const DENSITY_MASK: u32 = 0b1111 << 2; // Bits 2-5
    const DENSITY_SHIFT: u32 = 2;
    const DETAIL_MASK: u32 = 0b1111 << 6; // Bits 6-9
    const DETAIL_SHIFT: u32 = 6;
    const BRIGHTNESS_MASK: u32 = 0b1111 << 10; // Bits 10-13
    const BRIGHTNESS_SHIFT: u32 = 10;
    const COLOR_MASK: u32 = 0b1111 << 14; // Bits 14-17
    const COLOR_SHIFT: u32 = 14;
    const SEED_MASK: u32 = 0b111111 << 18; // Bits 18-23
    const SEED_SHIFT: u32 = 18;

    /// Creates a new Cloud instance with the specified position and scale.
    fn new(position: Vec3, scale: Vec3) -> Self {
        Self {
            pos: position,
            data: 0,
            scale,
            _padding0: 0,
        }
    }

    // Helpers for setting and getting packed data fields
    fn set_data_field(data: u32, value: u32, mask: u32, shift: u32) -> u32 {
        (data & !mask) | ((value << shift) & mask)
    }
    fn get_data_field(data: u32, mask: u32, shift: u32) -> u32 {
        (data & mask) >> shift
    }

    // Form: 0 = cumulus, 1 = stratus, 2 = cirrus
    fn set_form(mut self, form: CloudForm) -> Self {
        let val = match form {
            CloudForm::Cumulus => 0,
            CloudForm::Stratus => 1,
            CloudForm::Cirrus => 2,
        };
        self.data = Self::set_data_field(self.data, val, Self::FORM_MASK, Self::FORM_SHIFT);
        self
    }
    fn get_form(&self) -> CloudForm {
        let val = Self::get_data_field(self.data, Self::FORM_MASK, Self::FORM_SHIFT);
        match val {
            0 => CloudForm::Cumulus,
            1 => CloudForm::Stratus,
            2 => CloudForm::Cirrus,
            _ => unreachable!(),
        }
    }

    // Density: 0.0 to 1.0
    fn set_density(mut self, density: f32) -> Self {
        let raw = (density.clamp(0.0, 1.0) * 15.0).round() as u32;
        self.data = Self::set_data_field(self.data, raw, Self::DENSITY_MASK, Self::DENSITY_SHIFT);
        self
    }
    fn get_density(&self) -> f32 {
        let raw = Self::get_data_field(self.data, Self::DENSITY_MASK, Self::DENSITY_SHIFT);
        raw as f32 / 15.0
    }

    // Detail: 0.0 to 1.0
    fn set_detail(mut self, detail: f32) -> Self {
        let raw = (detail.clamp(0.0, 1.0) * 15.0).round() as u32;
        self.data = Self::set_data_field(self.data, raw, Self::DETAIL_MASK, Self::DETAIL_SHIFT);
        self
    }
    fn get_detail(&self) -> f32 {
        let raw = Self::get_data_field(self.data, Self::DETAIL_MASK, Self::DETAIL_SHIFT);
        raw as f32 / 15.0
    }

    // Brightness: 0.0 to 1.0
    fn set_brightness(mut self, brightness: f32) -> Self {
        let raw = (brightness.clamp(0.0, 1.0) * 15.0).round() as u32;
        self.data = Self::set_data_field(
            self.data,
            raw,
            Self::BRIGHTNESS_MASK,
            Self::BRIGHTNESS_SHIFT,
        );
        self
    }
    fn get_brightness(&self) -> f32 {
        let raw = Self::get_data_field(self.data, Self::BRIGHTNESS_MASK, Self::BRIGHTNESS_SHIFT);
        raw as f32 / 15.0
    }

    // Color: 0-15, to be interpreted as white-blue-purple in shader
    fn set_color(mut self, color: u32) -> Self {
        self.data = Self::set_data_field(self.data, color, Self::COLOR_MASK, Self::COLOR_SHIFT);
        self
    }
    fn get_color(&self) -> u32 {
        Self::get_data_field(self.data, Self::COLOR_MASK, Self::COLOR_SHIFT)
    }

    // Seed: 0-63
    fn set_seed(mut self, seed: u32) -> Self {
        self.data = Self::set_data_field(self.data, seed, Self::SEED_MASK, Self::SEED_SHIFT);
        self
    }
    fn get_seed(&self) -> u32 {
        Self::get_data_field(self.data, Self::SEED_MASK, Self::SEED_SHIFT)
    }
}

// --- Systems (Main World) ---

/// Sets up the initial state of clouds in the world.
fn setup(mut commands: Commands) {
    let mut rng = rng();

    // --- Configurable Weather Variables ---
    let stratiform_chance = 0.15; // Chance for stratus clouds
    let cloudiness = 1.0f32; // Overall cloud coverage (0.0 to 1.0)
    let turbulence = 0.5f32; // Affects detail level of clouds (0.0 = calm, 1.0 = very turbulent)
    let field_extent = 6000.0; // Distance to cover in meters for x/z plane

    // Calculate approximate number of clouds based on coverage and field size
    let num_clouds = ((field_extent * field_extent) / 50_000.0 * cloudiness).round() as usize;

    let mut clouds = Vec::with_capacity(num_clouds);
    for _ in 0..num_clouds {
        // Determine cloud form
        let form = if rng.random::<f32>() < stratiform_chance {
            CloudForm::Stratus
        } else if rng.random::<f32>() < 0.9 {
            CloudForm::Cumulus
        } else {
            CloudForm::Cirrus
        };

        // Determine height based on cloud form
        let height = match form {
            CloudForm::Cumulus => rng.random_range(200.0..=5000.0),
            CloudForm::Stratus => rng.random_range(2000.0..=8000.0),
            CloudForm::Cirrus => rng.random_range(7000.0..=8000.0),
        };

        // Determine size and proportion based on cloud form
        let (width, length, depth) = match form {
            CloudForm::Cumulus => {
                let width: f32 = rng.random_range(400.0..=2000.0);
                let length = width * rng.random_range(0.8..=1.5);
                let depth = rng.random_range(300.0..=width.min(1200.0));
                (width, length, depth)
            }
            CloudForm::Stratus => (
                rng.random_range(800.0..=3000.0),
                rng.random_range(1500.0..=3500.0),
                rng.random_range(60.0..=140.0),
            ),
            CloudForm::Cirrus => (
                rng.random_range(200.0..=1000.0),
                rng.random_range(2000.0..=4000.0),
                rng.random_range(40.0..=80.0),
            ),
        };

        // Determine position within the defined field extent
        let x = rng.random_range(-field_extent..=field_extent);
        let z = rng.random_range(-field_extent..=field_extent);
        let position = Vec3::new(x, height, z);
        let scale = Vec3::new(length, depth, width);

        // Determine density and detail based on form and turbulence
        let density = match form {
            CloudForm::Cumulus => rng.random_range(0.8..=1.0),
            CloudForm::Stratus => rng.random_range(0.5..=0.8),
            CloudForm::Cirrus => rng.random_range(0.4..=0.7),
        };
        let detail = match form {
            CloudForm::Cumulus => rng.random_range(0.7..=1.0),
            CloudForm::Stratus => rng.random_range(0.3..=0.6),
            CloudForm::Cirrus => rng.random_range(0.1..=0.2),
        } * turbulence; // Apply turbulence factor to detail

        // Create and configure the cloud, then add to the list
        clouds.push(
            Cloud::new(position, scale)
                .set_form(form)
                .set_density(density)
                .set_detail(detail)
                .set_seed(rng.random_range(0..64))
                .set_brightness(1.0)
                .set_color(0),
        );
    }

    // Insert resources into the main world
    commands.insert_resource(CloudsState { clouds });
    commands.insert_resource(CloudsBufferData {
        num_clouds: 0,
        _padding: [0; 3],
        clouds: [Cloud::default(); MAX_VISIBLE], // Initialize with default clouds
    });
}

/// Main world system: Updates cloud positions and identifies visible clouds for rendering.
fn update(
    time: Res<Time>,
    mut state: ResMut<CloudsState>,
    mut buffer: ResMut<CloudsBufferData>,
    camera_query: Query<(&GlobalTransform, &Frustum), With<Camera>>,
) {
    // Get camera data for frustum culling and sorting
    let Ok((camera_transform, camera_frustum)) = camera_query.single() else {
        buffer.num_clouds = 0;
        error!("No camera found, clearing clouds buffer");
        return;
    };

    let mut visible_cloud_count = 0;
    let cam_pos = camera_transform.translation();

    // Iterate through all clouds, update position, and check visibility
    for cloud in &mut state.clouds {
        cloud.pos.x += time.delta_secs() * 10.0;

        // Check if the cloud's bounding sphere intersects the camera frustum
        if visible_cloud_count < MAX_VISIBLE
            && camera_frustum.intersects_sphere(
                &Sphere {
                    center: cloud.pos.into(),
                    radius: cloud.scale.max_element() / 2.0, // Use the largest dimension as radius
                },
                false,
            )
        {
            // If visible and space available, add to the buffer
            buffer.clouds[visible_cloud_count] = *cloud;
            visible_cloud_count += 1;
        }
    }

    // Sort visible clouds by distance from the camera (for proper alpha blending/overdraw)
    buffer.clouds[..visible_cloud_count].sort_unstable_by(|a, b| {
        let dist_a_sq = (a.pos - cam_pos).length_squared();
        let dist_b_sq = (b.pos - cam_pos).length_squared();
        // Compare squared distances to avoid sqrt, then unwrap or default to Equal
        dist_a_sq.partial_cmp(&dist_b_sq).unwrap_or(Ordering::Equal)
    });

    // Update the number of clouds to be rendered
    buffer.num_clouds = u32::try_from(visible_cloud_count).unwrap_or_else(|e| {
        error!("Failed to convert visible cloud count to u32: {}", e);
        0
    });
}

// --- Systems (Render World) ---

/// Resource holding the GPU buffer for cloud data.
#[derive(Resource)]
struct CloudsBuffer {
    buffer: Buffer,
}

/// Render world system: Uploads the `CloudsBufferData` to the GPU.
fn update_buffer(
    mut commands: Commands,
    clouds_data: Res<CloudsBufferData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    clouds_buffer: Option<Res<CloudsBuffer>>,
) {
    // Convert the CloudsBufferData resource into a byte slice for GPU upload
    let bytes = bytemuck::bytes_of(&*clouds_data);

    if let Some(clouds_buffer) = clouds_buffer {
        // If the buffer already exists, simply write the new data to it
        render_queue.write_buffer(&clouds_buffer.buffer, 0, bytes);
    } else {
        // If the buffer doesn't exist, create a new one with the data
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("clouds_buffer"),
            contents: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, // Storage for shader, CopyDst for updates
        });
        commands.insert_resource(CloudsBuffer { buffer });
    }
}

/// Resource to manage the intermediate render target for clouds.
#[derive(Resource, Default)]
struct CloudRenderTexture {
    texture: Option<Texture>,
    view: Option<TextureView>,
    sampler: Option<Sampler>,
}

/// Render world system: Manages the creation and resizing of the intermediate cloud render target.
fn manage_render_target(
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

    let current_texture_size = cloud_render_texture.texture.as_ref().map(|t| t.size());
    if current_texture_size != Some(new_size) {
        // Create the new texture, view, and sampler
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_render_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear, // Linear filtering for smooth scaling
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
    type ViewQuery = (&'static ViewUniformOffset, &'static ViewPrepassTextures);

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

        // Ensure the intermediate data is ready
        let (
            Some(pipeline),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(clouds_buffer),
            Some(texture),
            Some(view),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<ViewUniforms>().uniforms.binding(),
            world.resource::<GlobalsBuffer>().buffer.binding(),
            prepass_textures.depth_view(),
            world.get_resource::<CloudsBuffer>(),
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
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
                globals_binding.clone(),
                depth_view,
                clouds_buffer.buffer.as_entire_binding(),
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
}

/// Render world system: Sets up the pipeline for rendering volumetric clouds.
fn setup_volumetric_clouds_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Define the bind group layout for the cloud rendering shader
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true), // View uniforms (camera projection, etc.)
                uniform_buffer_sized(false, Some(GlobalsUniform::min_size())), // Global uniforms (time, etc.)
                texture_depth_2d_multisampled(), // Depth texture from prepass
                storage_buffer_read_only::<CloudsBufferData>(false), // Our clouds data buffer
            ),
        ),
    );

    // Queue the render pipeline for creation
    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/clouds.wgsl"), // Shader for cloud rendering
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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
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
            shader: asset_server.load("shaders/clouds_composite.wgsl"),
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
