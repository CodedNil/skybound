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
use std::{cmp::Ordering, f32::consts::TAU};

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
    // 2× 16 bytes
    transform: Vec4, // xyz=centre, w used for rotation yaw
    scale: Vec4,     // xyz=scale, w is dynamically used for squared radius

    // 4 floats → 16 bytes
    seed: f32,    // Unique identifier for noise
    density: f32, // Overall fill (0=almost empty mist, 1=solid cloud mass)
    detail: f32,  // Fractal/noise detail power (0=smooth blob, 1=lots of little puffs)
    form: f32, // 0 = linear streaks like cirrus, 0.5 = solid like cumulus, 1 = anvil like cumulonimbus
}
impl Cloud {
    fn new(position: Vec3, scale: Vec3, rotation: f32) -> Self {
        let mut cloud = Cloud::default();
        cloud.seed = rng().random();
        cloud.density = 1.0;
        cloud.detail = 0.5;
        cloud.update_transforms(position, rotation, scale);
        cloud
    }

    fn update_transforms(&mut self, position: Vec3, rotation: f32, scale: Vec3) {
        self.transform = position.extend(rotation);
        self.scale = scale.extend(0.0);
        self.calc_dynamics();
    }

    fn calc_dynamics(&mut self) {
        // Use the largest extent as radius
        let r = self.scale.xyz().max_element() / 2.0;
        self.scale.w = r * r; // Add squared radius in the w component
    }

    fn set_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    fn set_detail(mut self, detail: f32) -> Self {
        self.detail = detail;
        self
    }
}

// --- Systems ---

fn setup(mut commands: Commands) {
    let mut rng = rng();
    let num_clouds = 500;
    let mut clouds = Vec::with_capacity(num_clouds + 1);
    for _ in 1..=num_clouds {
        clouds.push(
            Cloud::new(
                Vec3::new(
                    rng.random_range(-500.0..500.0),
                    rng.random_range(10.0..1000.0),
                    rng.random_range(-500.0..500.0),
                ),
                Vec3::new(
                    rng.random_range(15.0..50.0),
                    rng.random_range(5.0..20.0),
                    rng.random_range(15.0..50.0),
                ),
                rng.random_range(0.0..=TAU),
            )
            .set_density(rng.random_range(0.5..=1.0))
            .set_detail(rng.random_range(0.0..=1.0)),
        );
    }
    clouds.push(Cloud::new(
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(40.0, 10.0, 20.0),
        0.0,
    ));

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
    mut clouds_state: ResMut<CloudsState>,
    mut clouds_buffer: ResMut<CloudsBufferData>,
    camera_query: Query<(&GlobalTransform, &Frustum), With<Camera>>,
) {
    // Get camera data
    let Ok((transform, frustum)) = camera_query.single() else {
        clouds_buffer.num_clouds = 0;
        info!("No camera found, clearing clouds buffer");
        return;
    };

    // Update all clouds' positions, and gather ones that are visible
    let mut visible_clouds = Vec::new();
    for cloud in &mut clouds_state.clouds {
        cloud.transform.x += time.delta_secs() * 10.0;
        cloud.calc_dynamics();
        if frustum.intersects_sphere(
            &Sphere {
                center: cloud.transform.truncate().into(),
                radius: cloud.scale.w.sqrt(),
            },
            true,
        ) {
            visible_clouds.push(cloud);
        }
    }

    // Sort clouds by distance from camera
    let cam_pos = transform.translation();
    visible_clouds.sort_unstable_by(|a, b| {
        let da = (a.transform.xyz() - cam_pos).length_squared() - a.scale.w;
        let db = (b.transform.xyz() - cam_pos).length_squared() - b.scale.w;
        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
    });

    // Update the temporary buffer for rendering
    clouds_buffer.num_clouds = visible_clouds.len().min(MAX_VISIBLE) as u32;
    for (i, cloud) in visible_clouds.iter().enumerate().take(MAX_VISIBLE) {
        clouds_buffer.clouds[i] = **cloud;
    }
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
