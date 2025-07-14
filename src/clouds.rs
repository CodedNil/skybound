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

const FROXEL_HEIGHT: usize = 64;
const FROXEL_WIDTH: usize = (FROXEL_HEIGHT * 16 + 9 - 1) / 9;
const FROXEL_DEPTH: usize = 256;

const CLIP_NEAR: f32 = 0.1;
const CLIP_FAR: f32 = 1000.0;

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
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct VolumetricClouds {
    froxel_res: UVec3,
    froxel_near: f32,
    froxel_far: f32,
}
impl Default for VolumetricClouds {
    fn default() -> Self {
        Self {
            froxel_res: UVec3::new(
                FROXEL_WIDTH as u32,
                FROXEL_HEIGHT as u32,
                FROXEL_DEPTH as u32,
            ),
            froxel_near: CLIP_NEAR,
            froxel_far: CLIP_FAR,
        }
    }
}

/// Update the data including froxels every frame
fn update(
    froxel_handle: Res<FroxelTextureHandle>,
    mut images: ResMut<Assets<Image>>,
    camera_query: Query<(&GlobalTransform, &Projection), With<Camera>>,
    clouds: ResMut<CloudsData>,
) {
    // Get camera transforms
    let (camera_transform, projection) = match camera_query.single() {
        Ok((t, Projection::Perspective(p))) => (t, p),
        Ok(_) => return,
        Err(_) => return,
    };
    let view_matrix = camera_transform.compute_matrix();
    let image = match images.get_mut(&froxel_handle.0) {
        Some(image) => image,
        None => return,
    };

    // Update froxel texture with a time-based offset
    let tan_half = (projection.fov * 0.5).tan();
    let u_scale = projection.aspect_ratio * tan_half;
    let v_scale = tan_half;

    // Precompute constants
    let r = CLIP_FAR / CLIP_NEAR;
    let ln_r = r.ln();
    let inv_tan_half = 1.0 / tan_half;
    let inv_aspect = 1.0 / projection.aspect_ratio;

    // Calculate camera position and basis vectors
    let cam_pos = view_matrix.w_axis.truncate() / view_matrix.w_axis.w;
    let x_basis = (view_matrix * Vec4::new(u_scale, 0.0, 0.0, 0.0)).truncate();
    let y_basis = (view_matrix * Vec4::new(0.0, v_scale, 0.0, 0.0)).truncate();
    let z_basis = (view_matrix * Vec4::new(0.0, 0.0, -1.0, 0.0)).truncate();

    // Initialize buffer to accumulate voxel values
    let (wf, hf, df) = (
        FROXEL_WIDTH as f32,
        FROXEL_HEIGHT as f32,
        FROXEL_DEPTH as f32,
    );
    let froxel_max = Vec3::new(
        (FROXEL_WIDTH - 1) as f32,
        (FROXEL_HEIGHT - 1) as f32,
        (FROXEL_DEPTH - 1) as f32,
    );
    let mut v_buffer = vec![0u8; FROXEL_WIDTH * FROXEL_HEIGHT * FROXEL_DEPTH * 4];

    // Process each cloud
    for cloud in &clouds.clouds {
        let bmin = cloud.position - cloud.size / 2.0;
        let bmax = cloud.position + cloud.size / 2.0;

        // Compute eight corners of AABB
        let corners = [
            bmin,
            Vec3::new(bmax.x, bmin.y, bmin.z),
            Vec3::new(bmin.x, bmax.y, bmin.z),
            Vec3::new(bmin.x, bmin.y, bmax.z),
            Vec3::new(bmax.x, bmax.y, bmin.z),
            Vec3::new(bmax.x, bmin.y, bmax.z),
            Vec3::new(bmin.x, bmax.y, bmax.z),
            bmax,
        ];

        // Map each corner to froxel space
        let mut min_froxel = Vec3::MAX;
        let mut max_froxel = Vec3::MIN;

        for corner in corners {
            // Transform to camera space
            let p_world_homogeneous = Vec4::new(corner.x, corner.y, corner.z, 1.0);
            let p_cam_homogeneous = view_matrix.inverse() * p_world_homogeneous;
            let p_cam = p_cam_homogeneous.truncate() / p_cam_homogeneous.w;

            if p_cam.z < 0.0 {
                let d = -p_cam.z;
                if d >= CLIP_NEAR && d <= CLIP_FAR {
                    let inv_d_tan_half = inv_tan_half / d;
                    let u_ndc = p_cam.x * inv_d_tan_half * inv_aspect;
                    let v_ndc = p_cam.y * inv_d_tan_half;

                    let frac = (d / CLIP_NEAR).ln() / ln_r;

                    let froxel_pos = Vec3::new(
                        ((u_ndc + 1.0) * 0.5) * wf - 0.5,
                        ((v_ndc + 1.0) * 0.5) * hf - 0.5,
                        frac * df - 0.5,
                    );

                    min_froxel = min_froxel.min(froxel_pos);
                    max_froxel = max_froxel.max(froxel_pos);
                }
            }
        }

        // Compute froxel ranges
        let start = min_froxel
            .floor()
            .max(Vec3::ZERO)
            .min(froxel_max)
            .as_uvec3();
        let end = max_froxel.ceil().max(Vec3::ZERO).min(froxel_max).as_uvec3();

        // Process only relevant froxels
        if start.x <= end.x && start.y <= end.y && start.z <= end.z {
            for z in start.z..=end.z {
                let z = z as usize;
                let z_lin = clouds.z_pre[z];
                let z_basis_zlin = z_basis * z_lin;

                for y in start.y..=end.y {
                    let y = y as usize;
                    let v_ndc = clouds.v_pre[y];
                    let y_basis_vz = y_basis * (v_ndc * z_lin);
                    let offset = cam_pos + y_basis_vz + z_basis_zlin;

                    for x in start.x..=end.x {
                        let x = x as usize;
                        let u_ndc = clouds.u_pre[x];
                        let world = offset + x_basis * (u_ndc * z_lin);

                        if world.clamp(bmin, bmax) == world {
                            let dist_sq_ellipsoid =
                                ((world - cloud.position) / (cloud.size / 2.0)).length_squared();

                            // Only contribute density if inside the ellipsoid
                            if dist_sq_ellipsoid <= 1.0 {
                                let v = (1.0 - dist_sq_ellipsoid).clamp(0.0, 1.0);
                                let c = (v * 255.0) as u8;
                                let idx = ((z * FROXEL_HEIGHT + y) * FROXEL_WIDTH + x) * 4;
                                v_buffer[idx] = v_buffer[idx].saturating_add(c); // R
                                v_buffer[idx + 1] = 0; // G
                                v_buffer[idx + 2] = 0; // B
                                v_buffer[idx + 3] = 0; // A
                            }
                        }
                    }
                }
            }
        }
    }
    image.data = Some(v_buffer);

    // let data = image.data.as_mut().expect("Image must have data");
    // for z in 0..d {
    //     let z_lin = clouds.z_pre[z];
    //     let z_basis_zlin = z_basis * z_lin;

    //     for y in 0..h {
    //         let v_ndc = clouds.v_pre[y];
    //         let y_basis_vz = y_basis * (v_ndc * z_lin);
    //         let offset = cam_pos + y_basis_vz + z_basis_zlin;

    //         for x in 0..w {
    //             let u_ndc = clouds.u_pre[x];

    //             // World‐space position via basis
    //             let world = offset + x_basis * (u_ndc * z_lin);

    //             // Distance calculation to clouds
    //             let mut v = 0.0;
    //             for cloud in &clouds.clouds {
    //                 let bmin = cloud.position - cloud.size / 2.0;
    //                 let bmax = cloud.position + cloud.size / 2.0;
    //                 if world.clamp(bmin, bmax) == world {
    //                     let dist_sq = world.distance_squared(cloud.position);
    //                     v += (1.0 - dist_sq / 25.0).clamp(0.0, 1.0);
    //                     if v >= 1.0 {
    //                         break;
    //                     }
    //                 };
    //             }

    //             // Write to data buffer
    //             let c = (v * 255.0) as u8;
    //             let idx = ((z * FROXEL_HEIGHT + y) * FROXEL_WIDTH + x) * 4;
    //             data[idx] = c; // R
    //             data[idx + 1] = c; // G
    //             data[idx + 2] = c; // B
    //             data[idx + 3] = 255; // A
    //         }
    //     }
    // }
}

// Froxel setup
#[derive(Resource, Component, ExtractResource, Clone)]
struct FroxelTextureHandle(Handle<Image>);

#[derive(Resource)]
struct CloudsData {
    clouds: Vec<Cloud>,

    u_pre: Vec<f32>,
    v_pre: Vec<f32>,
    z_pre: Vec<f32>,
}

struct Cloud {
    position: Vec3,
    size: Vec3,
}

fn setup_froxels(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new(
        Extent3d {
            width: FROXEL_WIDTH as u32,
            height: FROXEL_HEIGHT as u32,
            depth_or_array_layers: FROXEL_DEPTH as u32,
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

    // Initial clouds
    let clouds = vec![
        Cloud {
            position: Vec3::new(0.0, 10.0, 0.0),
            size: Vec3::new(40.0, 10.0, 20.0),
        },
        Cloud {
            position: Vec3::new(40.0, 20.0, 10.0),
            size: Vec3::new(40.0, 20.0, 10.0),
        },
    ];

    // Froxel scratch
    let mut u_pre = Vec::with_capacity(FROXEL_WIDTH as usize);
    let mut v_pre = Vec::with_capacity(FROXEL_HEIGHT as usize);
    let mut z_pre = Vec::with_capacity(FROXEL_DEPTH as usize);
    for x in 0..FROXEL_WIDTH {
        let u_ndc = 2.0 * (x as f32 + 0.5) / FROXEL_WIDTH as f32 - 1.0;
        u_pre.push(u_ndc);
    }
    for y in 0..FROXEL_HEIGHT {
        let v_ndc = 2.0 * (y as f32 + 0.5) / FROXEL_HEIGHT as f32 - 1.0;
        v_pre.push(v_ndc);
    }
    for z in 0..FROXEL_DEPTH {
        let frac = (z as f32 + 0.5) / FROXEL_DEPTH as f32;
        z_pre.push(CLIP_NEAR * (CLIP_FAR / CLIP_NEAR).powf(frac));
    }

    commands.insert_resource(CloudsData {
        clouds,
        u_pre,
        v_pre,
        z_pre,
    });
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
