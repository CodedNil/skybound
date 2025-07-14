use bevy::{
    core_pipeline::{
        bloom::Bloom,
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::{DepthPrepass, ViewPrepassTextures},
    },
    ecs::{query::QueryItem, system::lifetimeless::Read},
    pbr::{Atmosphere, AtmosphereSettings},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        camera::Exposure,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
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
use smooth_bevy_cameras::controllers::unreal::{UnrealCameraBundle, UnrealCameraController};
use std::f32::consts::TAU;

/// Component that will get passed to the shader
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
struct VolumetricClouds {
    num_clouds: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, ShaderType)]
#[repr(C)]
struct Cloud {
    // 2× Vec4 (position, scale)
    position: Vec4, // xyz=centre, w unused
    scale: Vec4,    // xyz=scale, w unused

    // 4 floats → Vec4
    rotation: f32, // Yaw
    radius2: f32,  // For sphere testing, squared radius
    seed: f32,     // Unique identifier for noise
    density: f32,  // Overall fill (0=almost empty mist, 1=solid cloud mass)

    // 4 floats → Vec4
    detail: f32,   // Fractal/noise detail power (0=smooth blob, 1=lots of little puffs)
    flatness: f32, // 0 = fully 3D (cumulus), 1 = totally squashed pancake (stratus)
    streakiness: f32, // 0 = no banding, 1 = strong linear streaks (cirrus style)
    anvil: f32,    // 0 = none, 1 = full anvil cap (cumulonimbus top shear)
}

pub struct CloudsPlugin;
impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<CloudsBufferData>::default(),
            ExtractComponentPlugin::<VolumetricClouds>::default(),
            UniformComponentPlugin::<VolumetricClouds>::default(),
        ))
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

impl Cloud {
    fn new(position: Vec3, scale: Vec3, rotation: f32) -> Self {
        let mut cloud = Cloud {
            position: position.extend(0.0),
            scale: scale.extend(0.0),
            rotation: rotation,
            radius2: 0.0,
            seed: rng().random(),
            density: 1.0,
            detail: 0.5,
            flatness: 0.0,
            streakiness: 0.0,
            anvil: 0.0,
        };
        cloud.compute_radius(scale, rotation);
        cloud
    }

    fn update_transforms(&mut self, position: Vec3, scale: Vec3, rotation: f32) {
        self.position = position.extend(0.0);
        self.scale = scale.extend(0.0);
        self.rotation = rotation;
        self.compute_radius(scale, rotation);
    }

    fn compute_radius(&mut self, scale: Vec3, rotation: f32) {
        let half = scale * 0.5;
        let (sin_a, cos_a) = rotation.sin_cos();
        let ext_x = half.x * cos_a.abs() + half.z * sin_a.abs();
        let ext_z = half.x * sin_a.abs() + half.z * cos_a.abs();
        let ext_y = half.y;
        // largest extent = radius
        let r = ext_x.max(ext_y).max(ext_z);
        self.radius2 = r * r;
    }

    fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    fn with_detail(mut self, detail: f32) -> Self {
        self.detail = detail;
        self
    }

    fn with_flatness(mut self, flatness: f32) -> Self {
        self.flatness = flatness;
        self
    }

    fn with_streakiness(mut self, streakiness: f32) -> Self {
        self.streakiness = streakiness;
        self
    }

    fn with_anvil(mut self, anvil: f32) -> Self {
        self.anvil = anvil;
        self
    }
}

#[derive(Resource, ExtractResource, Clone)]
struct CloudsBufferData(Vec<Cloud>);

#[derive(Resource)]
struct CloudsBuffer {
    buffer: Buffer,
}
impl FromWorld for CloudsBuffer {
    fn from_world(world: &mut World) -> Self {
        let clouds_data = world.resource::<CloudsBufferData>();
        let render_device = world.resource::<RenderDevice>();
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("clouds_buffer"),
            contents: bytemuck::cast_slice(&clouds_data.0),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        Self { buffer }
    }
}

fn setup(mut commands: Commands) {
    let mut rng = rng();
    let num_clouds = 100;
    let mut clouds = Vec::with_capacity(num_clouds);
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
            .with_density(rng.random_range(0.5..=1.0))
            .with_detail(rng.random_range(0.0..=1.0)),
        );
    }
    clouds.push(Cloud::new(
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(40.0, 10.0, 20.0),
        0.0,
    ));

    let clouds_num = clouds.len() as u32;
    commands.insert_resource(CloudsBufferData(clouds));
    commands
        .spawn((
            Camera3d::default(),
            Camera {
                hdr: true,
                ..default()
            },
            Transform::from_xyz(-1.2, 0.15, 0.0).looking_at(Vec3::Y * 0.1, Vec3::Y),
            Atmosphere::EARTH,
            AtmosphereSettings {
                aerial_view_lut_max_distance: 3.2e5,
                scene_units_to_m: 1.0,
                ..Default::default()
            },
            Exposure::SUNLIGHT,
            Bloom::NATURAL,
            DepthPrepass,
            VolumetricClouds {
                num_clouds: clouds_num,
            },
        ))
        .insert(UnrealCameraBundle::new(
            UnrealCameraController::default(),
            Vec3::new(-15.0, 8.0, 18.0),
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::Y,
        ));
}

fn update(time: Res<Time>, mut clouds_buffer_data: ResMut<CloudsBufferData>) {
    for cloud in &mut clouds_buffer_data.0 {
        cloud.position.x += time.delta_secs() * 10.0;
        cloud.update_transforms(cloud.position.xyz(), cloud.scale.xyz(), cloud.rotation);
    }
}

fn update_buffer(
    mut commands: Commands,
    clouds_data: Res<CloudsBufferData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    clouds_buffer: Option<Res<CloudsBuffer>>,
) {
    if let Some(clouds_buffer) = clouds_buffer {
        // Update existing buffer
        let data = bytemuck::cast_slice(&clouds_data.0);
        render_queue.write_buffer(&clouds_buffer.buffer, 0, data);
    } else {
        // Create new buffer
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("clouds_buffer"),
            contents: bytemuck::cast_slice(&clouds_data.0),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        commands.insert_resource(CloudsBuffer { buffer });
    }
}

// Global data used by the render pipeline
#[derive(Resource)]
struct VolumetricCloudsPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolumetricCloudsLabel;

#[derive(Default)]
struct VolumetricCloudsNode;

impl ViewNode for VolumetricCloudsNode {
    type ViewQuery = (
        Read<ViewTarget>,
        Read<ViewUniformOffset>,
        Read<DynamicUniformIndex<VolumetricClouds>>,
        Read<ViewPrepassTextures>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, cloud_uniforms_index, prepass_textures): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();

        // Fetch the data safely.
        let (
            Some(pipeline),
            Some(cloud_uniforms_binding),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(clouds_buffer),
        ) = (
            world
                .resource::<PipelineCache>()
                .get_render_pipeline(volumetric_clouds_pipeline.pipeline_id),
            world
                .resource::<ComponentUniforms<VolumetricClouds>>()
                .uniforms()
                .binding(),
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
                cloud_uniforms_binding.clone(),
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
        render_pass.set_bind_group(
            0,
            &bind_group,
            &[view_uniform_offset.offset, cloud_uniforms_index.index()],
        );
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
                    uniform_buffer::<VolumetricClouds>(true),
                    storage_buffer_read_only::<Cloud>(false),
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
