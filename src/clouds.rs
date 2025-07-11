use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    ecs::query::QueryItem,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::{
        RenderApp,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{
                sampler, texture_2d, texture_3d, texture_depth_2d_multisampled, uniform_buffer,
            },
            *,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
    },
};
use noiz::{Noise, SampleableFor, prelude::common_noise::Perlin};

pub struct CloudsPlugin;

impl Plugin for CloudsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<NoiseTextureHandle>::default(),
            ExtractComponentPlugin::<VolumetricClouds>::default(),
            UniformComponentPlugin::<VolumetricClouds>::default(),
        ))
        .add_systems(Startup, setup_noise_texture);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ViewNodeRunner<VolumetricCloudsNode>>(
                Core3d,
                VolumetricCloudsLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    VolumetricCloudsLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<VolumetricCloudsPipeline>();
    }
}

// System to generate 3D noise texture
#[derive(Resource, Component, ExtractResource, Clone)]
struct NoiseTextureHandle(Handle<Image>);
fn setup_noise_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = 64;
    let mut data = vec![0u8; size * size * size * 4]; // RGBA
    let noise = Noise::<Perlin>::default();

    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let pos = Vec3::new(x as f32, y as f32, z as f32) / size as f32;
                let sample_value: f32 = noise.sample(pos * 50.0);
                let value = sample_value.mul_add(0.5, 0.5); // Normalize to [0,1]
                let idx = (z * size * size + y * size + x) * 4;
                data[idx] = (value * 255.0) as u8; // R
                data[idx + 1] = (value * 255.0) as u8; // G
                data[idx + 2] = (value * 255.0) as u8; // B
                data[idx + 3] = 255; // A
            }
        }
    }

    let mut image = Image::new(
        Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: size as u32,
        },
        TextureDimension::D3,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );

    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        address_mode_w: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    });

    let handle = images.add(image);
    commands.insert_resource(NoiseTextureHandle(handle));
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
        &'static ViewTarget,
        &'static ViewUniformOffset,
        &'static DynamicUniformIndex<VolumetricClouds>,
        &'static ViewPrepassTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, view_uniform_offset, settings_index, prepass_textures): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let volumetric_clouds_pipeline = world.resource::<VolumetricCloudsPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let noise_texture_handle = world.resource::<NoiseTextureHandle>();
        let view_uniforms = world.resource::<ViewUniforms>();

        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(volumetric_clouds_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<VolumetricClouds>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };
        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            return Ok(());
        };

        let Some(noise_gpu_image) = gpu_images.get(&noise_texture_handle.0) else {
            return Ok(());
        };
        let Some(depth_texture_view) = prepass_textures.depth_view() else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "volumetric_clouds_bind_group",
            &volumetric_clouds_pipeline.layout,
            &BindGroupEntries::sequential((
                settings_binding.clone(),
                view_binding.clone(),
                post_process.source,
                depth_texture_view,
                &noise_gpu_image.texture_view,
                &noise_gpu_image.sampler,
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
            &[settings_index.index(), view_uniform_offset.offset],
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
                    uniform_buffer::<VolumetricClouds>(true), // The settings uniform that will control the effect
                    uniform_buffer::<ViewUniform>(true),      // The view uniform with the view info
                    texture_2d(TextureSampleType::Float { filterable: false }), // The screen texture
                    texture_depth_2d_multisampled(),                            // The depth texture
                    texture_3d(TextureSampleType::Float { filterable: true }),  // Noise texture
                    sampler(SamplerBindingType::Filtering),                     // Noise sampler
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

// Component that will get passed to the shader
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
pub struct VolumetricClouds {
    time: f32,
}

// Update the time every frame
pub fn update_settings(mut settings: Query<&mut VolumetricClouds>, time: Res<Time>) {
    for mut setting in &mut settings {
        setting.time = time.elapsed_secs();
    }
}
