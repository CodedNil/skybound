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
        render_app.init_resource::<CloudRenderTexture>();
        render_app
            .add_systems(RenderStartup, setup_volumetric_clouds_pipeline)
            .add_systems(RenderStartup, setup_volumetric_clouds_composite_pipeline)
            .add_systems(Render, update_buffer.in_set(RenderSystems::Prepare))
            .add_systems(
                Render,
                manage_cloud_render_target.in_set(RenderSystems::Queue),
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
}

// --- Data Structures ---

/// Stores the state of all clouds, updated every frame
#[derive(Resource)]
struct CloudsState {
    clouds: Vec<Cloud>,
}

/// Temporary buffer for visible clouds sent to the GPU
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
    scale: Vec3, // x=width, y=height, z=length (12 bytes)
    _padding0: u32,
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

    // Form: 0 = cumulus, 1 = stratus, 2 = cirrus
    fn set_form(mut self, form: CloudForm) -> Self {
        let val = match form {
            CloudForm::Cumulus => 0,
            CloudForm::Stratus => 1,
            CloudForm::Cirrus => 2,
        };
        self.data = (self.data & !Self::FORM_MASK) | ((val << Self::FORM_SHIFT) & Self::FORM_MASK);
        self
    }
    fn get_form(&self) -> CloudForm {
        let val = (self.data & Self::FORM_MASK) >> Self::FORM_SHIFT;
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
        self.data =
            (self.data & !Self::DENSITY_MASK) | ((raw << Self::DENSITY_SHIFT) & Self::DENSITY_MASK);
        self
    }
    fn get_density(&self) -> f32 {
        let raw = (self.data & Self::DENSITY_MASK) >> Self::DENSITY_SHIFT;
        raw as f32 / 15.0
    }

    // Detail: 0.0 to 1.0
    fn set_detail(mut self, detail: f32) -> Self {
        let raw = (detail.clamp(0.0, 1.0) * 15.0).round() as u32;
        self.data =
            (self.data & !Self::DETAIL_MASK) | ((raw << Self::DETAIL_SHIFT) & Self::DETAIL_MASK);
        self
    }
    fn get_detail(&self) -> f32 {
        let raw = (self.data & Self::DETAIL_MASK) >> Self::DETAIL_SHIFT;
        raw as f32 / 15.0
    }

    // Brightness: 0.0 to 1.0
    fn set_brightness(mut self, brightness: f32) -> Self {
        let raw = (brightness.clamp(0.0, 1.0) * 15.0).round() as u32;
        self.data = (self.data & !Self::BRIGHTNESS_MASK)
            | ((raw << Self::BRIGHTNESS_SHIFT) & Self::BRIGHTNESS_MASK);
        self
    }
    fn get_brightness(&self) -> f32 {
        let raw = (self.data & Self::BRIGHTNESS_MASK) >> Self::BRIGHTNESS_SHIFT;
        raw as f32 / 15.0
    }

    // Color: 0-15, to be interpreted as white-blue-purple in shader
    fn set_color(mut self, color: u32) -> Self {
        assert!(color <= 15, "Color must be 0-15");
        self.data =
            (self.data & !Self::COLOR_MASK) | ((color << Self::COLOR_SHIFT) & Self::COLOR_MASK);
        self
    }
    fn get_color(&self) -> u32 {
        (self.data & Self::COLOR_MASK) >> Self::COLOR_SHIFT
    }

    // Seed: 0-63
    fn set_seed(mut self, seed: u32) -> Self {
        assert!(seed <= 63, "Seed must be between 0 and 63");
        self.data = (self.data & !Self::SEED_MASK) | ((seed << Self::SEED_SHIFT) & Self::SEED_MASK);
        self
    }
    fn get_seed(&self) -> u32 {
        (self.data & Self::SEED_MASK) >> Self::SEED_SHIFT
    }
}

// --- Systems ---

fn setup(mut commands: Commands) {
    let mut rng = rng();

    // --- Configurable Weather Variables ---
    let stratiform_chance = 0.15;
    let cloudiness = 1.0f32;
    let turbulence = 0.5f32; // 0.0 = calm, 1.0 = very turbulent
    let field_extent = 6000.0; // Distance to cover in meters for x/z
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

        // Height based on form
        let height = match form {
            CloudForm::Cumulus => rng.random_range(200.0..=5000.0),
            CloudForm::Stratus => rng.random_range(2000.0..=8000.0),
            CloudForm::Cirrus => rng.random_range(7000.0..=8000.0),
        };

        // Size and proportion based on form
        let (width, length, depth) = match form {
            CloudForm::Cumulus => {
                let width: f32 = rng.random_range(400.0..=2000.0);
                let length = width * rng.random_range(0.8..=1.5);
                let depth = rng.random_range(100.0..=width.min(800.0));
                (width, length, depth)
            }
            CloudForm::Stratus => (
                rng.random_range(800.0..=3000.0),
                rng.random_range(1500.0..=3500.0),
                rng.random_range(20.0..=100.0),
            ),
            CloudForm::Cirrus => (
                rng.random_range(200.0..=1000.0),
                rng.random_range(2000.0..=4000.0),
                rng.random_range(10.0..=40.0),
            ),
        };

        // Position in field
        let x = rng.random_range(-field_extent..=field_extent);
        let z = rng.random_range(-field_extent..=field_extent);
        let position = Vec3::new(x, height, z);
        let scale = Vec3::new(length, depth, width);

        let density = match form {
            CloudForm::Cumulus => rng.random_range(0.8..=1.0),
            CloudForm::Stratus => rng.random_range(0.5..=0.8),
            CloudForm::Cirrus => rng.random_range(0.4..=0.7),
        };
        let detail = match form {
            CloudForm::Cumulus => rng.random_range(0.7..=1.0),
            CloudForm::Stratus => rng.random_range(0.3..=0.6),
            CloudForm::Cirrus => rng.random_range(0.1..=0.2),
        } * turbulence;

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

    commands.insert_resource(CloudsState { clouds });
    commands.insert_resource(CloudsBufferData {
        num_clouds: 0,
        _padding: [0; 3],
        clouds: [Cloud::default(); MAX_VISIBLE],
    });
}

// Main world system: Update cloud positions and compute visible subset
fn update(
    time: Res<Time>,
    mut state: ResMut<CloudsState>,
    mut buffer: ResMut<CloudsBufferData>,
    camera_query: Query<(&GlobalTransform, &Frustum), With<Camera>>,
) {
    // Get camera data
    let Ok((transform, frustum)) = camera_query.single() else {
        buffer.num_clouds = 0;
        error!("No camera found, clearing clouds buffer");
        return;
    };

    // Update all clouds' positions, and gather ones that are visible
    let mut count = 0;
    for cloud in &mut state.clouds {
        cloud.pos.x += time.delta_secs() * 10.0;
        if count < MAX_VISIBLE
            && frustum.intersects_sphere(
                &Sphere {
                    center: cloud.pos.into(),
                    radius: cloud.scale.max_element() / 2.0,
                },
                false,
            )
        {
            buffer.clouds[count] = *cloud;
            count += 1;
        }
    }

    // Sort clouds by distance from camera
    let cam_pos = transform.translation();
    buffer.clouds[..count].sort_unstable_by(|a, b| {
        let da = (a.pos - cam_pos).length_squared();
        let db = (b.pos - cam_pos).length_squared();
        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
    });
    buffer.num_clouds = u32::try_from(count).unwrap();
}

// Render world system: Upload visible clouds to GPU
#[derive(Resource)]
struct CloudsBuffer {
    buffer: Buffer,
}

fn update_buffer(
    mut commands: Commands,
    clouds_data: Res<CloudsBufferData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    clouds_buffer: Option<Res<CloudsBuffer>>,
) {
    let binding = vec![*clouds_data].into_boxed_slice();
    let bytes = bytemuck::cast_slice(&binding);
    if let Some(clouds_buffer) = clouds_buffer {
        render_queue.write_buffer(&clouds_buffer.buffer, 0, bytes);
    } else {
        // Create new buffer
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("clouds_buffer"),
            contents: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        commands.insert_resource(CloudsBuffer { buffer });
    }
}

// --- Intermediate Cloud Render Target ---
#[derive(Resource, Default)]
struct CloudRenderTexture {
    texture: Option<Texture>,
    view: Option<TextureView>,
    sampler: Option<Sampler>,
}

// Combined system to manage cloud render target: create and resize
fn manage_cloud_render_target(
    mut cloud_render_texture: ResMut<CloudRenderTexture>, // Now directly the resource
    render_device: Res<RenderDevice>,
    windows: Res<ExtractedWindows>,
) {
    // Safely get the texture, primary window's entity and then its data
    let Some(primary_window_entity) = windows.primary else {
        return;
    };
    let Some(primary_window) = windows.windows.get(&primary_window_entity) else {
        return;
    };

    let new_size = Extent3d {
        width: primary_window.physical_width / 2, // Quarter resolution (half width, half height)
        height: primary_window.physical_height / 2,
        depth_or_array_layers: 1,
    };

    let mut needs_recreation = false;

    // Check if the texture needs to be created or resized
    if cloud_render_texture.texture.is_none() {
        // Texture not yet created
        needs_recreation = true;
    } else if let Some(current_texture) = &cloud_render_texture.texture {
        // Texture exists, check if size has changed
        if current_texture.size() != new_size {
            needs_recreation = true;
        }
    }

    if needs_recreation {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_render_texture"),
            size: new_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float, // Needs alpha for compositing
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..default()
        });

        // Update the resource with the new texture, view, and sampler
        cloud_render_texture.texture = Some(texture);
        cloud_render_texture.view = Some(view);
        cloud_render_texture.sampler = Some(sampler);

        info!(
            "{} cloud render target to {}x{}",
            if cloud_render_texture.texture.is_none() {
                "Created"
            } else {
                "Resized"
            },
            new_size.width,
            new_size.height
        );
    }
}

// --- Volumetric Clouds Render Pipeline (renders clouds to intermediate texture) ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsLabel;

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

        // If the texture hasn't been initialized yet, return early
        let (Some(texture), Some(view), Some(_sampler)) = (
            cloud_render_texture.texture.as_ref(),
            cloud_render_texture.view.as_ref(),
            cloud_render_texture.sampler.as_ref(),
        ) else {
            return Ok(());
        };

        // Fetch the data safely.
        let (
            Some(pipeline),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(clouds_buffer),
        ) = (
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world.resource::<ViewUniforms>().uniforms.binding(),
            world.resource::<GlobalsBuffer>().buffer.binding(),
            prepass_textures.depth_view(),
            world.get_resource::<CloudsBuffer>(),
        )
        else {
            return Ok(());
        };

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

        // Begin the render pass
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view, // Render to our intermediate texture
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Set viewport to match the quarter resolution texture
        render_pass.set_viewport(
            0.0,
            0.0,
            texture.width() as f32,
            texture.height() as f32,
            0.0,
            1.0,
        );

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct VolumetricCloudsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

fn setup_volumetric_clouds_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(false, Some(GlobalsUniform::min_size())),
                texture_depth_2d_multisampled(),
                storage_buffer_read_only::<CloudsBufferData>(false),
            ),
        ),
    );

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/clouds.wgsl"),
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

// --- Volumetric Clouds Composite Render Pipeline ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsCompositeLabel;

#[derive(Default)]
struct VolumetricCloudsCompositeNode;

impl ViewNode for VolumetricCloudsCompositeNode {
    type ViewQuery = &'static ViewTarget;

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        view_target: QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_composite_pipeline =
            world.resource::<VolumetricCloudsCompositePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let cloud_render_texture = world.resource::<CloudRenderTexture>();

        // If the texture hasn't been initialized yet, return early
        let (Some(cloud_texture_view), Some(cloud_sampler)) = (
            cloud_render_texture.view.as_ref(),
            cloud_render_texture.sampler.as_ref(),
        ) else {
            return Ok(());
        };

        // Fetch the data safely.
        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(volumetric_clouds_composite_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_composite_bind_group",
            &volumetric_clouds_composite_pipeline.layout,
            &BindGroupEntries::sequential((post_process.source, cloud_texture_view, cloud_sampler)),
        );

        // Begin the render pass
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_composite_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination, // Render to main view target
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

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct VolumetricCloudsCompositePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

fn setup_volumetric_clouds_composite_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = render_device.create_bind_group_layout(
        "volumetric_clouds_composite_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }), // Original scene color
                texture_2d(TextureSampleType::Float { filterable: true }), // Clouds texture
                sampler(SamplerBindingType::Filtering),                    // Sampler for clouds
            ),
        ),
    );

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_composite_pipeline".into()),
        layout: vec![layout.clone()],
        push_constant_ranges: Vec::new(),
        vertex: fullscreen_shader.to_vertex_state(),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            shader: asset_server.load("shaders/clouds_composite.wgsl"),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float, // Output to default scene format
                blend: Some(BlendState::ALPHA_BLENDING), // Alpha blend the clouds
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
