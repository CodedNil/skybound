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
                storage_buffer_read_only, texture_2d, texture_depth_2d_multisampled,
                uniform_buffer, uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
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
        render_app.add_systems(RenderStartup, setup_pipeline);
        render_app.add_systems(Render, update_buffer.in_set(RenderSystems::Prepare));

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndMainPass, VolumetricCloudsLabel, Node3d::Bloom),
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
    fn set_form(mut self, form: u32) -> Self {
        assert!(form <= 3, "Form must be 0, 1, or 2");
        self.data = (self.data & !Self::FORM_MASK) | ((form << Self::FORM_SHIFT) & Self::FORM_MASK);
        self
    }
    fn get_form(&self) -> u32 {
        (self.data & Self::FORM_MASK) >> Self::FORM_SHIFT
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
    let stratiform_chance = 0.25;
    let cloudiness = 1.0f32;
    let turbulence = 0.5f32; // 0.0 = calm, 1.0 = very turbulent
    let field_extent = 2000.0; // Distance to cover in meters for x/z
    let num_clouds =
        ((field_extent * 2.0 * field_extent * 2.0) / 100_000.0 * cloudiness).round() as usize;

    let mut clouds = Vec::with_capacity(num_clouds);
    for _ in 0..num_clouds {
        // Determine cloud form
        let form = if rng.random::<f32>() < stratiform_chance {
            1 // Stratus
        } else if rng.random::<f32>() < 0.75 {
            0 // Cumulus
        } else {
            2 // Cirrus
        };

        // Height based on form
        let height = match form {
            1 => rng.random_range(10.0..=3000.0),   // Stratus
            0 => rng.random_range(10.0..=1800.0),   // Cumulus
            2 => rng.random_range(2600.0..=3000.0), // Cirrus
            _ => unreachable!(),
        };

        // Size and proportion based on form
        let (width, length, depth) = match form {
            1 => {
                // Stratus
                let width = rng.random_range(500.0..=3000.0);
                let length = rng.random_range(2000.0..=field_extent * 2.0);
                let depth = rng.random_range(10.0..=100.0);
                (width, length, depth)
            }
            0 => {
                // Cumulus
                let width: f32 = rng.random_range(150.0..=1000.0);
                let length = width * rng.random_range(0.8..=1.2);
                let depth = rng.random_range(100.0..=width.min(800.0));
                (width, length, depth)
            }
            2 => {
                // Cirrus
                let width = rng.random_range(40.0..=100.0);
                let length = width * rng.random_range(1.2..=2.2);
                let depth = rng.random_range(10.0..=40.0);
                (width, length, depth)
            }
            _ => unreachable!(),
        };

        // Position in field
        let x = rng.random_range(-field_extent..=field_extent);
        let z = rng.random_range(-field_extent..=field_extent);
        let position = Vec3::new(x, height, z);
        let scale = Vec3::new(length, depth, width);

        let density = match form {
            0 => rng.random_range(0.8..=1.0), // Cumulus
            1 => rng.random_range(0.3..=0.6), // Stratus
            2 => rng.random_range(0.1..=0.3), // Cirrus
            _ => unreachable!(),
        };
        let base_detail = match form {
            1 => rng.random_range(0.1..=0.4), // Stratus
            0 => rng.random_range(0.6..=0.9), // Cumulus
            2 => rng.random_range(0.7..=1.0), // Cirrus
            _ => unreachable!(),
        };
        let detail = base_detail + turbulence * 0.2;

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

// --- Render Pipeline ---

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsLabel;

#[derive(Default)]
struct VolumetricCloudsNode;

impl ViewNode for VolumetricCloudsNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static ViewPrepassTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

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

        let post_process = view_target.post_process_write();
        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                view_binding.clone(),
                globals_binding.clone(),
                post_process.source,
                depth_view,
                clouds_buffer.buffer.as_entire_binding(),
            )),
        );

        // Begin the render pass
        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("volumetric_clouds_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

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

fn setup_pipeline(
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
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d_multisampled(),
                storage_buffer_read_only::<CloudsBufferData>(false),
            ),
        ),
    );

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("volumetric_clouds_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader.to_vertex_state(),
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
