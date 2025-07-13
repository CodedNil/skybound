use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    ecs::{query::QueryItem, system::lifetimeless::Read},
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        RenderApp,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::{GlobalsBuffer, GlobalsUniform},
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_3d, texture_depth_2d_multisampled, uniform_buffer,
                uniform_buffer_sized,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};

const FROXEL_HEIGHT: u32 = 96;
const FROXEL_WIDTH: u32 = (FROXEL_HEIGHT * 16 + 9 - 1) / 9;
const FROXEL_DEPTH: u32 = 192;

pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<FroxelTextureHandle>::default(),
            ExtractComponentPlugin::<VolumetricClouds>::default(),
            UniformComponentPlugin::<VolumetricClouds>::default(),
        ))
        .add_systems(Startup, setup_froxels)
        .add_systems(Update, update);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<VolumetricCloudsPipeline>()
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

/// Component that will get passed to the shader
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
pub struct VolumetricClouds {
    froxel_res: UVec3,
    froxel_near: f32,
    froxel_far: f32,
}

/// Update the data including froxels every frame
fn update(
    mut cloud_uniforms: Query<&mut VolumetricClouds>,
    froxel_handle: Res<FroxelTextureHandle>,
    mut images: ResMut<Assets<Image>>,
    camera_query: Query<(&GlobalTransform, &Projection), With<Camera>>,
) {
    // Get camera transforms
    let (camera_transform, projection) = match camera_query.single() {
        Ok((transform, projection)) => (transform, projection),
        Err(e) => {
            warn!("Failed to get single camera: {:?}", e);
            return; // Skip this frame if no unique camera is found
        }
    };
    let view_matrix = camera_transform.compute_matrix();

    // Update froxel texture with a time-based offset
    if let (Some(image), Projection::Perspective(persp)) =
        (images.get_mut(&froxel_handle.0), projection)
    {
        let total_voxels = (FROXEL_WIDTH * FROXEL_HEIGHT * FROXEL_DEPTH) as usize;
        let mut data = vec![0u8; total_voxels * 4];

        let (near, far, fov, aspect) = (persp.near, persp.far, persp.fov, persp.aspect_ratio);
        let tan_half = (fov * 0.5).tan();

        for mut uniform in &mut cloud_uniforms {
            uniform.froxel_res = UVec3::new(FROXEL_WIDTH, FROXEL_HEIGHT, FROXEL_DEPTH);
            uniform.froxel_near = near;
            uniform.froxel_far = far;
        }

        for z in 0..FROXEL_DEPTH {
            let fraction = (z as f32 + 0.5) / FROXEL_DEPTH as f32;
            let z_lin = near * (far / near).powf(fraction);

            for y in 0..FROXEL_HEIGHT {
                let v_ndc = 2.0 * ((y as f32 + 0.5) / FROXEL_HEIGHT as f32) - 1.0;

                for x in 0..FROXEL_WIDTH {
                    let u_ndc = 2.0 * ((x as f32 + 0.5) / FROXEL_WIDTH as f32) - 1.0;

                    // Compute view‐space position
                    let view_pos = Vec4::new(
                        u_ndc * aspect * tan_half * z_lin,
                        v_ndc * tan_half * z_lin,
                        -z_lin,
                        1.0,
                    );

                    // Transform to world space
                    let world_hom = view_matrix * view_pos;
                    let world_pos = world_hom.truncate() / world_hom.w;

                    // Simple sphere distance
                    let distance = (world_pos - Vec3::new(0.0, 10.0, 0.0)).length();
                    let value = (1.0 - (distance / 10.0)).clamp(0.0, 1.0);

                    // Update the texture data with the new value
                    let idx = (((z * FROXEL_HEIGHT + y) * FROXEL_WIDTH + x) as usize) * 4;
                    data[idx] = (value * 255.0) as u8; // R
                    data[idx + 1] = (value * 255.0) as u8; // G
                    data[idx + 2] = (value * 255.0) as u8; // B
                    data[idx + 3] = 255; // A
                }
            }
        }

        image.data = Some(data);
    }
}

// System to generate 3D noise texture
#[derive(Resource, Component, ExtractResource, Clone)]
struct FroxelTextureHandle(Handle<Image>);
fn setup_froxels(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new(
        Extent3d {
            width: FROXEL_WIDTH,
            height: FROXEL_HEIGHT,
            depth_or_array_layers: FROXEL_DEPTH,
        },
        TextureDimension::D3,
        vec![0u8; (FROXEL_WIDTH * FROXEL_HEIGHT * FROXEL_DEPTH * 4) as usize],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        address_mode_w: ImageAddressMode::ClampToEdge,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });

    let handle = images.add(image);
    commands.insert_resource(FroxelTextureHandle(handle));
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

        // Fetch the uniform buffer and binding.
        let (
            Some(pipeline),
            Some(cloud_uniforms_binding),
            Some(view_binding),
            Some(globals_binding),
            Some(depth_view),
            Some(froxel_gpu_image),
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
            world
                .resource::<RenderAssets<GpuImage>>()
                .get(&world.resource::<FroxelTextureHandle>().0),
        )
        else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                cloud_uniforms_binding.clone(),
                view_binding.clone(),
                globals_binding.clone(),
                post_process.source,
                depth_view,
                &froxel_gpu_image.texture_view,
                &froxel_gpu_image.sampler,
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
            &[cloud_uniforms_index.index(), view_uniform_offset.offset],
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
                    uniform_buffer::<VolumetricClouds>(true), // The clouds uniform that will control the effect
                    uniform_buffer::<ViewUniform>(true),      // The view uniform
                    uniform_buffer_sized(false, Some(GlobalsUniform::min_size())), // The globals uniform
                    texture_2d(TextureSampleType::Float { filterable: false }), // The screen texture
                    texture_depth_2d_multisampled(),                            // The depth texture
                    texture_3d(TextureSampleType::Float { filterable: true }),  // Froxel texture
                    sampler(SamplerBindingType::Filtering),                     // Froxel sampler
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
