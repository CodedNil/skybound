use crate::{
    game::{camera::DebugView, flight::PlayerModel, world::WorldData},
    graphics::prepare::textures::VolumeTextures,
};
use bevy::{
    core_pipeline::FullscreenShader,
    core_pipeline::prepass::ViewPrepassTextures,
    diagnostic::FrameCount,
    ecs::system::SystemParam,
    prelude::*,
    render::{
        camera::TemporalJitter,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{
            AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntries, BufferUsages, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, FilterMode,
            FragmentState, LoadOp, MultisampleState, Operations, PipelineCache, PrimitiveState,
            RawBufferVec, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
            RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, SamplerDescriptor, ShaderStages, StencilState, StoreOp,
            TextureFormat, TextureSampleType, TextureView,
            binding_types::{
                sampler, texture_2d, texture_3d, texture_depth_2d, uniform_buffer_sized,
            },
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::{ExtractedView, ViewTarget},
    },
};
use skybound::scene::{FrameUniform, ViewUniform};
use std::{mem::size_of, num::NonZeroU64};

#[derive(Resource)]
pub struct SceneBuffer {
    uniform: RawBufferVec<FrameUniform>,
    previous_clip: Mat4,
}
impl Default for SceneBuffer {
    fn default() -> Self {
        Self {
            uniform: RawBufferVec::new(BufferUsages::UNIFORM),
            previous_clip: Mat4::IDENTITY,
        }
    }
}

type ViewQuery = (&'static ExtractedView, Option<&'static TemporalJitter>);

#[derive(SystemParam)]
pub(super) struct PrepareSceneView<'w, 's> {
    render_device: Res<'w, RenderDevice>,
    render_queue: Res<'w, RenderQueue>,
    buffer: ResMut<'w, SceneBuffer>,
    player: Res<'w, PlayerModel>,
    views: Single<'w, 's, ViewQuery, With<Camera3d>>,
    time: Res<'w, Time>,
    frame_count: Res<'w, FrameCount>,
    world: Res<'w, WorldData>,
    debug: Res<'w, DebugView>,
    scale: Res<'w, super::targets::RenderScale>,
    sky: ResMut<'w, super::prepare::sky::SkyLookup>,
    targets: ResMut<'w, super::targets::SceneTargets>,
}

pub(super) fn prepare_scene_view(
    PrepareSceneView {
        render_device,
        render_queue,
        mut buffer,
        player,
        views,
        time,
        frame_count,
        world,
        debug,
        scale,
        mut sky,
        mut targets,
    }: PrepareSceneView,
) {
    buffer.uniform.clear();
    let (extracted_view, temporal_jitter) = *views;
    {
        let viewport = extracted_view.viewport.as_vec4();
        let view_size = (viewport.zw() * scale.0).floor().max(Vec2::ONE);
        targets.resize(&render_device, view_size.as_uvec2());

        let mut clip_from_view = extracted_view.clip_from_view;
        if let Some(jitter) = temporal_jitter {
            jitter.jitter_projection(&mut clip_from_view, viewport.zw());
        }

        let view_from_clip = clip_from_view.inverse();
        let world_from_view = extracted_view.world_from_view.to_matrix();
        let view_from_world = world_from_view.inverse();
        let clip_from_world = clip_from_view * view_from_world;
        let world_from_clip = world_from_view * view_from_clip;
        let world_position = extracted_view.world_from_view.translation();
        let (planet_rotation, latitude, longitude) = world.planet_frame(world_position);

        let world_from_clip_unjittered = world_from_view * extracted_view.clip_from_view.inverse();

        let mut uniform = ViewUniform {
            clip_from_world,
            world_from_clip,
            prev_clip_from_world: buffer.previous_clip,
            world_from_clip_unjittered,
            world_position: world_position.extend(world.camera_offset.z),
            camera_position: vec4(
                latitude,
                longitude,
                world.camera_offset.x,
                world.camera_offset.y,
            ),
            planet_rotation,
            times: vec4(
                time.elapsed_secs_wrapped(),
                frame_count.0 as f32,
                debug.0 as f32,
                time.delta_secs(),
            ),
            ..default()
        };
        uniform.prepare_atmosphere();
        sky.update(&mut uniform, &render_queue);
        buffer.uniform.push(FrameUniform {
            view: uniform,
            player: player.0,
        });

        buffer.previous_clip = extracted_view
            .clip_from_world
            .unwrap_or_else(|| extracted_view.clip_from_view * view_from_world);
    }

    buffer.uniform.write_buffer(&render_device, &render_queue);
}

#[derive(Resource)]
pub struct ScenePipeline {
    layout: BindGroupLayout,
    resolve_layout: BindGroupLayout,
    resolve_pipeline_id: CachedRenderPipelineId,
    pipeline_id: CachedRenderPipelineId,
    linear_sampler: Sampler,
    resolve_sampler: Sampler,
}

fn scene_bind_group(
    world: &World,
    device: &RenderDevice,
    pipeline: &ScenePipeline,
) -> Option<BindGroup> {
    let gpu_images = world.resource::<RenderAssets<GpuImage>>();
    let noise_texture_handle = world.resource::<VolumeTextures>();
    let (Some(view_binding), Some(base_noise), Some(detail_noise), Some(weather_noise)) = (
        world.resource::<SceneBuffer>().uniform.binding(),
        gpu_images.get(&noise_texture_handle.base),
        gpu_images.get(&noise_texture_handle.detail),
        gpu_images.get(&noise_texture_handle.weather),
    ) else {
        return None;
    };

    Some(device.create_bind_group(
        "scene_bind_group",
        &pipeline.layout,
        &BindGroupEntries::sequential((
            view_binding,
            &pipeline.linear_sampler,
            &base_noise.texture_view,
            &detail_noise.texture_view,
            &weather_noise.texture_view,
            &world.resource::<super::prepare::sky::SkyLookup>().view,
        )),
    ))
}

pub fn raymarch_pass(
    world: &World,
    mut render_context: RenderContext,
    view_query: Query<(&ExtractedView, &ViewTarget, &ViewPrepassTextures)>,
) {
    for (view, view_target, prepass_textures) in &view_query {
        let scene_pipeline = world.resource::<ScenePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let device = world.resource::<RenderDevice>();
        let (
            Some(pipeline),
            Some(bind_group),
            Some(depth_view),
            Some(motion_view),
            Some(normal_view),
        ) = (
            pipeline_cache.get_render_pipeline(scene_pipeline.pipeline_id),
            scene_bind_group(world, device, scene_pipeline),
            prepass_textures.depth_only_view(),
            prepass_textures.motion_vectors_view(),
            prepass_textures.normal_view(),
        )
        else {
            continue;
        };

        let targets = world
            .resource::<super::targets::SceneTargets>()
            .0
            .as_ref()
            .expect("prepared scene targets");
        let reduced = targets.size != view.viewport.zw();
        let resolve_pipeline =
            pipeline_cache.get_render_pipeline(scene_pipeline.resolve_pipeline_id);
        if reduced && resolve_pipeline.is_none() {
            continue;
        }
        let native = || PassTargets {
            color: view_target.get_color_attachment(),
            motion: motion_view,
            normal: normal_view,
            depth: depth_view,
            viewport: view.viewport,
            clear: false,
        };
        let scene = if reduced {
            PassTargets {
                color: color_attachment(&targets.color, true),
                motion: &targets.motion,
                normal: &targets.normal,
                depth: &targets.depth,
                viewport: UVec4::new(0, 0, targets.size.x, targets.size.y),
                clear: true,
            }
        } else {
            native()
        };
        draw_scene(
            &mut render_context,
            pipeline,
            &bind_group,
            scene,
            "raymarch",
        );
        if reduced {
            let group = device.create_bind_group(
                "scene_resolve",
                &scene_pipeline.resolve_layout,
                &BindGroupEntries::sequential((
                    &targets.color,
                    &targets.motion,
                    &targets.normal,
                    &targets.depth,
                    &scene_pipeline.resolve_sampler,
                )),
            );
            draw_scene(
                &mut render_context,
                resolve_pipeline.expect("checked resolve pipeline"),
                &group,
                native(),
                "scene_resolve",
            );
        }
    }
}

struct PassTargets<'a> {
    color: RenderPassColorAttachment<'a>,
    motion: &'a TextureView,
    normal: &'a TextureView,
    depth: &'a TextureView,
    viewport: UVec4,
    clear: bool,
}

fn color_attachment(view: &TextureView, clear: bool) -> RenderPassColorAttachment<'_> {
    RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: Operations {
            load: if clear {
                LoadOp::Clear(default())
            } else {
                LoadOp::Load
            },
            store: StoreOp::Store,
        },
    }
}

fn draw_scene(
    context: &mut RenderContext,
    pipeline: &RenderPipeline,
    group: &BindGroup,
    targets: PassTargets<'_>,
    label: &'static str,
) {
    let diagnostics = context.diagnostic_recorder();
    let span = diagnostics
        .as_ref()
        .map(|recorder| recorder.time_span(context.command_encoder(), label));
    let mut pass = context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[
            Some(targets.color),
            Some(color_attachment(targets.motion, targets.clear)),
            Some(color_attachment(targets.normal, targets.clear)),
        ],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: targets.depth,
            depth_ops: Some(Operations {
                load: if targets.clear {
                    LoadOp::Clear(0.0)
                } else {
                    LoadOp::Load
                },
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let vp = targets.viewport;
    pass.set_viewport(vp.x as f32, vp.y as f32, vp.z as f32, vp.w as f32, 0.0, 1.0);
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
    drop(pass);
    if let Some(span) = span {
        span.end(context.command_encoder());
    }
}

impl FromWorld for ScenePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world.resource::<super::SceneShader>().0.clone();
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

        let resolve_sampler = resolve_sampler(render_device);

        let layout_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                uniform_buffer_sized(false, NonZeroU64::new(size_of::<FrameUniform>() as u64)), // 0: View uniforms
                sampler(SamplerBindingType::Filtering), // 1: Linear sampler
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
            ),
        );
        let layout_descriptor =
            BindGroupLayoutDescriptor::new("scene_bind_group_layout", &layout_entries);
        let layout =
            render_device.create_bind_group_layout("scene_bind_group_layout", &layout_entries);

        let pipeline_cache = world.resource::<PipelineCache>();
        let mut descriptor = RenderPipelineDescriptor {
            label: Some("scene_pipeline".into()),
            layout: vec![layout_descriptor],
            immediate_size: 0,
            vertex: fullscreen_shader.to_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                entry_point: Some("main".into()),
                constants: Vec::new(),
                targets: [
                    TextureFormat::Rgba16Float,
                    TextureFormat::Rg16Float,
                    TextureFormat::Rgb10a2Unorm,
                ]
                .map(|format| {
                    Some(ColorTargetState {
                        format,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    })
                })
                .to_vec(),
                shader_defs: Vec::new(),
            }),
            zero_initialize_workgroup_memory: false,
        };
        let pipeline_id = pipeline_cache.queue_render_pipeline(descriptor.clone());
        let resolve_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d(),
                sampler(SamplerBindingType::Filtering),
            ),
        );
        descriptor.label = Some("scene_resolve".into());
        descriptor.layout = vec![BindGroupLayoutDescriptor::new(
            "scene_resolve",
            &resolve_entries,
        )];
        let fragment = descriptor.fragment.as_mut().expect("fragment pipeline");
        fragment.shader = world.resource::<super::ResolveShader>().0.clone();
        fragment.entry_point = Some("fragment".into());
        let resolve_pipeline_id = pipeline_cache.queue_render_pipeline(descriptor);
        let resolve_layout =
            render_device.create_bind_group_layout("scene_resolve", &resolve_entries);

        Self {
            layout,
            resolve_layout,
            resolve_pipeline_id,
            pipeline_id,
            linear_sampler,
            resolve_sampler,
        }
    }
}

fn resolve_sampler(device: &RenderDevice) -> Sampler {
    device.create_sampler(&SamplerDescriptor {
        label: Some("scene_resolve_linear"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..default()
    })
}
