use bevy::{
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    ecs::{query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
        primitives::{Frustum, Sphere},
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
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
        app.add_plugins((ExtractResourcePlugin::<CloudsBufferData>::default(),))
            .add_systems(Startup, setup)
            .add_systems(Update, update);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<VolumetricCloudsPipeline>()
            .add_systems(Render, update_buffer.in_set(RenderSet::Prepare))
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
    // Anvil of cumulonimbus clouds can be automatically generated based on scale, really tall and wide clouds get an anvil
    // Cirrus is potentially smooth low detail, yet stretched out?, cirrocumulus and stratocumulus could simply be clouds with very low height but high detail so they are broken up a lot
    // Cirrostratus clouds which are more faded, if its a really short cloud then reduce its density

    // 16 bytes
    pos: Vec3, // Position of the cloud
    seed: f32, // Unique identifier for noise

    // 16 bytes
    scale: Vec3,  // x=width, y=height, z=length
    density: f32, // Overall fill (0=almost empty mist, 1=solid cloud mass)

    // 16 bytes
    detail: f32, // Fractal/noise detail power (0=smooth blob, 1=lots of little puffs)
    color: f32,  // 0 = white, 1 = black
    _padding0: f32,
    _padding1: f32,
}
impl Cloud {
    fn new(position: Vec3, scale: Vec3) -> Self {
        let mut cloud = Cloud::default();
        cloud.seed = rng().random::<f32>() * 10000.0;
        cloud.density = 1.0;
        cloud.color = rng().random();
        cloud.pos = position;
        cloud.scale = scale;
        cloud
    }

    fn set_detail(mut self, detail: f32) -> Self {
        self.detail = detail;
        self
    }
}

// --- Systems ---

fn setup(mut commands: Commands) {
    let mut rng = rng();

    // --- Configurable Weather Variables ---
    let stratiform_chance = 0.25;
    let cloudiness = 1.0f32;
    let (detail_lower, detail_upper) = (0.4f32, 0.9f32);
    let field_extent = 2000.0; // Distance to cover in meters for x/z
    let num_clouds =
        ((field_extent * 2.0 * field_extent * 2.0) / 100_000.0 * cloudiness).round() as usize;

    let mut clouds = Vec::with_capacity(num_clouds);
    for _ in 0..num_clouds {
        // Stratiform (layer) vs cumuliform (heap-shaped)
        let is_stratus = rng.random::<f32>() < stratiform_chance;
        let height = if is_stratus {
            // Stratus/Altostratus/Cirrostratus layers
            rng.random_range(10.0..=3000.0)
        } else {
            // Heap clouds: bias heights for more cumulus at low, more altocumulus at middle, rare cirrocumulus at top
            let r = rng.random::<f32>();
            if r < 0.75 {
                // Cumulus: 10–1800m
                10.0 + rng.random::<f32>().powf(2.5) * 1790.0
            } else if r < 0.97 {
                // Altocumulus: 1800–2600m
                1800.0 + rng.random::<f32>() * 800.0
            } else {
                // Cirrocumulus (rare at 2600–3000m, scaled for effect)
                2600.0 + rng.random::<f32>() * 400.0
            }
        };

        // Size and proportion based on cloud type
        let (width, length, depth) = if is_stratus {
            // Wide, thin sheets: sideways dominant, very shallow
            let width = rng.random_range(500.0..=3000.0);
            let length = rng.random_range(2000.0..=field_extent * 2.0);
            let depth = rng.random_range(10.0..=100.0);
            (width, length, depth)
        } else if height < 1800.0 {
            // Cumulus (round, puffy)
            let width: f32 = rng.random_range(150.0..=1000.0);
            let length = width * rng.random_range(0.8..=1.2);
            let depth = rng.random_range(100.0..=width.min(800.0));
            (width, length, depth)
        } else if height < 2600.0 {
            // Altocumulus (smaller, more fragments)
            let width = rng.random_range(80.0..=300.0);
            let length = width * rng.random_range(1.0..=2.0);
            let depth = rng.random_range(40.0..=150.0);
            (width, length, depth)
        } else {
            // Cirrocumulus (small, sheet-like, upper air)
            let width = rng.random_range(40.0..=100.0);
            let length = width * rng.random_range(1.2..=2.2);
            let depth = rng.random_range(10.0..=40.0);
            (width, length, depth)
        };

        // Position in field
        let x = rng.random_range(-field_extent..=field_extent);
        let z = rng.random_range(-field_extent..=field_extent);

        // Noise/detail: more "ragged" at higher altitudes or for less stratiform
        let detail =
            rng.random_range(detail_lower..=detail_upper) + if is_stratus { -0.2 } else { 0.0 };

        clouds.push(
            Cloud::new(Vec3::new(x, height, z), Vec3::new(length, depth, width))
                .set_detail(detail.clamp(0.1, 1.0)),
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
    buffer.num_clouds = count as u32;
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
    let binding = [*clouds_data];
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

#[derive(Resource)]
struct VolumetricCloudsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

#[derive(Default)]
struct VolumetricCloudsNode;

impl ViewNode for VolumetricCloudsNode {
    type ViewQuery = (
        Read<ViewTarget>,
        Read<ViewUniformOffset>,
        Read<ViewPrepassTextures>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();

        // Fetch the data safely.
        let (
            Some(pipeline),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(clouds_buffer),
        ) = (
            world
                .resource::<PipelineCache>()
                .get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
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

impl FromWorld for VolumetricCloudsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

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

        let pipeline_descriptor = RenderPipelineDescriptor {
            label: Some("volumetric_clouds_pipeline".into()),
            layout: vec![layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: world.load_asset("shaders/clouds.wgsl"),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            .queue_render_pipeline(pipeline_descriptor);

        Self {
            layout,
            pipeline_id,
        }
    }
}
