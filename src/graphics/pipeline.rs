use bevy::render::render_resource::MipmapFilterMode;
use std::array::from_fn;
use std::env;
use std::mem::size_of;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

use bevy::core_pipeline::FullscreenShader;
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::diagnostic::FrameCount;
use bevy::ecs::system::SystemParam;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::camera::TemporalJitter;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    sampler, texture_2d, texture_3d, texture_depth_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    AddressMode, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntries, BindGroupLayoutEntry, BindingResource, BufferUsages,
    CachedRenderPipelineId, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, FilterMode, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineCache, PrimitiveState, RawBufferVec, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StencilState, StoreOp, TextureFormat, TextureSampleType, TextureView, TextureViewId,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewTarget};
use skybound::scene::atmosphere::SKY_SIZE;
use skybound::scene::aur::{EJECTION_GRID_HEIGHT, EJECTION_GRID_WIDTH};
use skybound::scene::{FrameUniform, ViewUniform};

use crate::game::camera::DebugView;
use crate::game::flight::PlayerModel;
use crate::game::world::WorldData;
use crate::graphics::prepare::sky::SkyLookup;
use crate::graphics::prepare::textures::VolumeTextures;
use crate::graphics::targets::{SceneTargets, Targets, VolumeScale, texture};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolveUniform {
    world_from_clip: Mat4,
    origin: Vec4,
    size: Vec4,
    native: Vec4,
}
#[derive(Resource)]
pub struct SceneBuffer {
    uniform: RawBufferVec<FrameUniform>,
    resolve: RawBufferVec<ResolveUniform>,
    previous_clip: Mat4,
    previous_position: Vec3,
    previous_origin: DVec3,
    previous_rotation: Quat,
    previous_debug: u32,
    previous_weather: Vec4,
    frame: u32,
    profile: u32,
}
impl Default for SceneBuffer {
    fn default() -> Self {
        Self {
            uniform: RawBufferVec::new(BufferUsages::UNIFORM),
            resolve: RawBufferVec::new(BufferUsages::UNIFORM),
            previous_clip: Mat4::IDENTITY,
            previous_position: Vec3::ZERO,
            previous_origin: DVec3::ZERO,
            previous_rotation: Quat::IDENTITY,
            previous_debug: 0,
            previous_weather: Vec4::ZERO,
            frame: 0,
            profile: env::var("SKYBOUND_PROFILE_VOLUME")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        }
    }
}
type ViewQuery = (&'static ExtractedView, Option<&'static TemporalJitter>);
#[derive(SystemParam)]
pub(super) struct PrepareSceneView<'w, 's> {
    device: Res<'w, RenderDevice>,
    queue: Res<'w, RenderQueue>,
    buffer: ResMut<'w, SceneBuffer>,
    player: Res<'w, PlayerModel>,
    views: Single<'w, 's, ViewQuery, With<Camera3d>>,
    time: Res<'w, Time>,
    frame_count: Res<'w, FrameCount>,
    world: Res<'w, WorldData>,
    debug: Res<'w, DebugView>,
    atmosphere: Res<'w, super::AtmosphereSettings>,
    scale: Res<'w, VolumeScale>,
    sky: ResMut<'w, SkyLookup>,
    targets: ResMut<'w, SceneTargets>,
}
pub(super) fn prepare_scene_view(params: PrepareSceneView) {
    let PrepareSceneView {
        device,
        queue,
        mut buffer,
        player,
        views,
        time,
        frame_count,
        world,
        debug,
        atmosphere,
        scale,
        mut sky,
        mut targets,
    } = params;
    let (view, jitter) = *views;
    let viewport = view.viewport.as_vec4();
    targets.resize(&device, viewport.zw().as_uvec2(), scale.0);
    let targets = targets.0.as_ref().expect("prepared volume targets");
    let mut clip_from_view = view.clip_from_view;
    if let Some(jitter) = jitter {
        jitter.jitter_projection(&mut clip_from_view, viewport.zw());
    }
    let world_from_view = view.world_from_view.to_matrix();
    let world_from_clip = world_from_view * clip_from_view.inverse();
    let position = view.world_from_view.translation();
    let (rotation, latitude, longitude) = world.planet_frame(position);
    let mut uniform = ViewUniform {
        clip_from_world: clip_from_view * world_from_view.inverse(),
        world_from_clip,
        prev_clip_from_world: buffer.previous_clip,
        world_from_clip_unjittered: world_from_view * view.clip_from_view.inverse(),
        world_position: position.extend(0.0),
        planet_center_value: world.local(DVec3::ZERO).extend(0.0),
        camera_position: vec4(
            latitude,
            longitude,
            world.altitude(position) as f32,
            buffer.profile as f32,
        ),
        planet_rotation: rotation,
        weather: atmosphere.0,
        times: vec4(
            time.elapsed_secs_wrapped(),
            frame_count.0 as f32,
            debug.0 as f32,
            time.delta_secs(),
        ),
        ..default()
    };
    uniform.prepare_atmosphere();
    uniform.prepare_flashes();
    sky.update(&mut uniform);
    let valid = targets.valid.load(Ordering::Relaxed)
        && position.distance(buffer.previous_position) < 100.0
        && world.origin == buffer.previous_origin
        && view
            .world_from_view
            .rotation()
            .dot(buffer.previous_rotation)
            .abs()
            > 0.95
        && debug.0 == buffer.previous_debug
        && atmosphere.0 == buffer.previous_weather;
    buffer.uniform.clear();
    buffer.uniform.push(FrameUniform {
        view: uniform,
        players: player.0,
        viewport,
        volume_size: targets.size.as_vec2().extend(0.0).extend(0.0),
    });
    buffer.resolve.clear();
    buffer.resolve.push(ResolveUniform {
        world_from_clip,
        origin: position.extend(0.0),
        size: targets.size.as_vec2().extend(f32::from(valid)).extend(0.0),
        native: viewport.zw().extend(debug.0 as f32).extend(0.0),
    });
    buffer.uniform.write_buffer(&device, &queue);
    buffer.resolve.write_buffer(&device, &queue);
    buffer.previous_clip = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * world_from_view.inverse());
    buffer.previous_position = position;
    buffer.previous_origin = world.origin;
    buffer.previous_rotation = view.world_from_view.rotation();
    buffer.previous_debug = debug.0;
    buffer.previous_weather = atmosphere.0;
    buffer.frame = frame_count.0;
}

struct BindGroups {
    depth: TextureViewId,
    volume: TextureViewId,
    native: [TextureViewId; 2],
    opaque: BindGroup,
    volume_group: BindGroup,
    sky: BindGroup,
    filter: [BindGroup; 2],
    composite: [[BindGroup; 2]; 2],
}
#[derive(Resource)]
pub struct ScenePipeline {
    opaque_layout: BindGroupLayout,
    volume_layout: BindGroupLayout,
    sky_layout: BindGroupLayout,
    filter_layout: BindGroupLayout,
    composite_layout: BindGroupLayout,
    opaque: CachedRenderPipelineId,
    volume: CachedRenderPipelineId,
    thin: CachedRenderPipelineId,
    sky: CachedRenderPipelineId,
    ejection_lookup: CachedRenderPipelineId,
    ejections: TextureView,
    filter: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
    repeat_sampler: Sampler,
    linear_sampler: Sampler,
    groups: Option<BindGroups>,
}
#[derive(SystemParam)]
pub(super) struct PrepareBindings<'w, 's> {
    pipeline: ResMut<'w, ScenePipeline>,
    device: Res<'w, RenderDevice>,
    buffer: Res<'w, SceneBuffer>,
    images: Res<'w, RenderAssets<GpuImage>>,
    textures: Res<'w, VolumeTextures>,
    sky: Res<'w, SkyLookup>,
    targets: Res<'w, SceneTargets>,
    views: Single<'w, 's, (&'static ViewPrepassTextures, &'static ViewTarget), With<Camera3d>>,
}
pub(super) fn prepare_bindings(params: PrepareBindings) {
    let PrepareBindings {
        mut pipeline,
        device,
        buffer,
        images,
        textures,
        sky,
        targets,
        views,
    } = params;
    let (prepass, view) = *views;
    let (
        Some(depth),
        Some(targets),
        Some(frame),
        Some(resolve),
        Some(base),
        Some(detail),
        Some(weather),
        Some(transmittance),
    ) = (
        prepass.depth_only_view(),
        targets.0.as_ref(),
        buffer.uniform.binding(),
        buffer.resolve.binding(),
        images.get(&textures.base),
        images.get(&textures.detail),
        images.get(&textures.weather),
        images.get(&textures.transmittance),
    )
    else {
        return;
    };
    let native = [view.main_texture_view(), view.main_texture_other_view()];
    if pipeline.groups.as_ref().is_some_and(|groups| {
        groups.depth == depth.id()
            && groups.volume == targets.color.id()
            && groups.native.contains(&native[0].id())
            && groups.native.contains(&native[1].id())
    }) {
        return;
    }
    let opaque = device.create_bind_group(
        "opaque",
        &pipeline.opaque_layout,
        &BindGroupEntries::sequential((
            frame.clone(),
            &pipeline.repeat_sampler,
            &base.texture_view,
            &detail.texture_view,
            &weather.texture_view,
            &sky.view,
        )),
    );
    let volume_group = device.create_bind_group(
        "volume",
        &pipeline.volume_layout,
        &BindGroupEntries::sequential((
            frame.clone(),
            &pipeline.repeat_sampler,
            &base.texture_view,
            &detail.texture_view,
            &weather.texture_view,
            &sky.view,
            depth,
            &transmittance.texture_view,
            &pipeline.ejections,
        )),
    );
    let sky_group = device.create_bind_group(
        "sky",
        &pipeline.sky_layout,
        &BindGroupEntries::single(frame),
    );
    let (filter, composite) =
        reconstruction_groups(&pipeline, &device, resolve, targets, depth, native);
    pipeline.groups = Some(BindGroups {
        depth: depth.id(),
        volume: targets.color.id(),
        native: native.map(TextureView::id),
        opaque,
        volume_group,
        sky: sky_group,
        filter,
        composite,
    });
}

fn reconstruction_groups(
    pipeline: &ScenePipeline,
    device: &RenderDevice,
    resolve: BindingResource<'_>,
    targets: &Targets,
    depth: &TextureView,
    native: [&TextureView; 2],
) -> ([BindGroup; 2], [[BindGroup; 2]; 2]) {
    let filter = from_fn(|index| {
        device.create_bind_group(
            "volume_temporal",
            &pipeline.filter_layout,
            &BindGroupEntries::sequential((
                resolve.clone(),
                &targets.color,
                &targets.motion,
                &targets.metadata,
                &targets.history[1 - index],
                &targets.history_metadata[1 - index],
                &pipeline.linear_sampler,
            )),
        )
    });
    let composite = from_fn(|native_index| {
        from_fn(|index| {
            device.create_bind_group(
                "volume_composite",
                &pipeline.composite_layout,
                &BindGroupEntries::sequential((
                    resolve.clone(),
                    native[native_index],
                    &targets.history[index],
                    &targets.history_metadata[index],
                    depth,
                    &targets.thin_color,
                    &targets.thin_transmission,
                    &targets.thin_prefix,
                )),
            )
        })
    });
    (filter, composite)
}

fn attachment(view: &TextureView, clear: bool) -> RenderPassColorAttachment<'_> {
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
fn draw(
    context: &mut RenderContext,
    pipeline: &RenderPipeline,
    group: &BindGroup,
    colors: &[Option<RenderPassColorAttachment<'_>>],
    depth: Option<&TextureView>,
    viewport: UVec4,
    label: &'static str,
) {
    let recorder = context.diagnostic_recorder();
    let span = recorder
        .as_ref()
        .map(|r| r.time_span(context.command_encoder(), label));
    let mut pass = context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(label),
        color_attachments: colors,
        depth_stencil_attachment: depth.map(|view| RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(0.0),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_viewport(
        viewport.x as f32,
        viewport.y as f32,
        viewport.z as f32,
        viewport.w as f32,
        0.0,
        1.0,
    );
    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.push_debug_group(label);
    pass.draw(0..3, 0..1);
    pass.pop_debug_group();
    drop(pass);
    if let Some(span) = span {
        span.end(context.command_encoder());
    }
}

pub fn raymarch_pass(
    world: &World,
    mut context: RenderContext,
    views: Query<(&ExtractedView, &ViewTarget, &ViewPrepassTextures)>,
) {
    let pipeline = world.resource::<ScenePipeline>();
    let cache = world.resource::<PipelineCache>();
    let sky = world.resource::<SkyLookup>();
    let (
        Some(groups),
        Some(targets),
        Some(opaque),
        Some(volume),
        Some(sky_pipeline),
        Some(ejection_pipeline),
    ) = (
        pipeline.groups.as_ref(),
        world.resource::<SceneTargets>().0.as_ref(),
        cache.get_render_pipeline(pipeline.opaque),
        cache.get_render_pipeline(pipeline.volume),
        cache.get_render_pipeline(pipeline.sky),
        cache.get_render_pipeline(pipeline.ejection_lookup),
    )
    else {
        return;
    };
    draw(
        &mut context,
        ejection_pipeline,
        &groups.sky,
        &[Some(attachment(&pipeline.ejections, true))],
        None,
        UVec4::new(0, 0, EJECTION_GRID_WIDTH * 2, EJECTION_GRID_HEIGHT),
        "ejection_lookup",
    );
    for (view, target, prepass) in &views {
        let (Some(depth), Some(motion), Some(normal)) = (
            prepass.depth_only_view(),
            prepass.motion_vectors_view(),
            prepass.normal_view(),
        ) else {
            continue;
        };
        if sky.dirty.swap(false, Ordering::Relaxed) {
            draw(
                &mut context,
                sky_pipeline,
                &groups.sky,
                &[Some(attachment(&sky.view, true))],
                None,
                UVec4::new(0, 0, SKY_SIZE, SKY_SIZE),
                "sky_lookup",
            );
        }
        draw(
            &mut context,
            opaque,
            &groups.opaque,
            &[
                Some(target.get_color_attachment()),
                Some(attachment(motion, true)),
                Some(attachment(normal, true)),
            ],
            Some(depth),
            view.viewport,
            "opaque_creatures",
        );
        draw(
            &mut context,
            volume,
            &groups.volume_group,
            &[
                Some(attachment(&targets.color, true)),
                Some(attachment(&targets.motion, true)),
                Some(attachment(&targets.metadata, true)),
            ],
            Some(&targets.depth),
            UVec4::new(0, 0, targets.size.x, targets.size.y),
            "volumetrics",
        );
    }
}

pub fn composite_pass(
    world: &World,
    mut context: RenderContext,
    views: Query<(&ExtractedView, &ViewTarget)>,
) {
    let pipeline = world.resource::<ScenePipeline>();
    let cache = world.resource::<PipelineCache>();
    let (Some(groups), Some(targets), Some(thin), Some(filter), Some(composite)) = (
        pipeline.groups.as_ref(),
        world.resource::<SceneTargets>().0.as_ref(),
        cache.get_render_pipeline(pipeline.thin),
        cache.get_render_pipeline(pipeline.filter),
        cache.get_render_pipeline(pipeline.composite),
    ) else {
        return;
    };
    if [
        pipeline.opaque,
        pipeline.volume,
        pipeline.sky,
        pipeline.ejection_lookup,
    ]
    .iter()
    .any(|id| cache.get_render_pipeline(*id).is_none())
    {
        targets.valid.store(false, Ordering::Relaxed);
        return;
    }
    let index = world.resource::<SceneBuffer>().frame as usize % 2;
    for (view, target) in &views {
        draw(
            &mut context,
            filter,
            &groups.filter[index],
            &[
                Some(attachment(&targets.history[index], true)),
                Some(attachment(&targets.history_metadata[index], true)),
            ],
            None,
            UVec4::new(0, 0, targets.size.x, targets.size.y),
            "volume_temporal",
        );
        draw(
            &mut context,
            thin,
            &groups.volume_group,
            &[
                Some(attachment(&targets.thin_color, true)),
                Some(attachment(&targets.thin_transmission, true)),
                Some(attachment(&targets.thin_prefix, true)),
            ],
            None,
            UVec4::new(0, 0, targets.native_size.x, targets.native_size.y),
            "thin_surfaces",
        );
        let output = target.post_process_write();
        let native_index = usize::from(output.source.id() == groups.native[1]);
        draw(
            &mut context,
            composite,
            &groups.composite[native_index][index],
            &[Some(attachment(output.destination, true))],
            None,
            view.viewport,
            "volume_composite",
        );
        targets.valid.store(true, Ordering::Relaxed);
    }
}

struct Interfaces {
    opaque: Vec<BindGroupLayoutEntry>,
    volume: Vec<BindGroupLayoutEntry>,
    sky: Vec<BindGroupLayoutEntry>,
    filter: Vec<BindGroupLayoutEntry>,
    composite: Vec<BindGroupLayoutEntry>,
}
impl Interfaces {
    fn new() -> Self {
        let frame_entry =
            || uniform_buffer_sized(false, NonZeroU64::new(size_of::<FrameUniform>() as u64));
        let resolve_entry =
            || uniform_buffer_sized(false, NonZeroU64::new(size_of::<ResolveUniform>() as u64));
        let float = || texture_2d(TextureSampleType::Float { filterable: true });
        let data = || texture_2d(TextureSampleType::Float { filterable: false });
        let opaque = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                frame_entry(),
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                float(),
            ),
        );
        let mut volume = opaque.to_vec();
        let extra = BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((6, texture_depth_2d()), (7, float()), (8, data())),
        );
        volume.extend_from_slice(&extra);
        let sky = BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, frame_entry());
        let filter = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                resolve_entry(),
                float(),
                data(),
                data(),
                float(),
                data(),
                sampler(SamplerBindingType::Filtering),
            ),
        );
        let composite = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                resolve_entry(),
                float(),
                float(),
                data(),
                texture_depth_2d(),
                float(),
                float(),
                float(),
            ),
        );
        Self {
            opaque: opaque.to_vec(),
            volume,
            sky: sky.to_vec(),
            filter: filter.to_vec(),
            composite: composite.to_vec(),
        }
    }
}

struct PipelineCompiler<'a> {
    cache: &'a PipelineCache,
    fullscreen: &'a FullscreenShader,
}
impl PipelineCompiler<'_> {
    fn queue(
        &self,
        name: &'static str,
        shader: Handle<Shader>,
        entry: &'static str,
        entries: &[BindGroupLayoutEntry],
        formats: &[TextureFormat],
        depth: bool,
    ) -> CachedRenderPipelineId {
        self.cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(name.into()),
            layout: vec![BindGroupLayoutDescriptor::new(name, entries)],
            immediate_size: 0,
            vertex: self.fullscreen.to_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: depth.then_some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                entry_point: Some(entry.into()),
                constants: Vec::new(),
                targets: formats
                    .iter()
                    .map(|&format| {
                        Some(ColorTargetState {
                            format,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })
                    })
                    .collect(),
                shader_defs: Vec::new(),
            }),
            zero_initialize_workgroup_memory: false,
        })
    }
}

impl FromWorld for ScenePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let cache = world.resource::<PipelineCache>();
        let fullscreen = world.resource::<FullscreenShader>();
        let scene_shader = world.resource::<super::SceneShader>().0.clone();
        let shaders = world.resource::<super::VolumeShaders>();
        let Interfaces {
            opaque: entries,
            volume: volume_entries,
            sky: sky_entries,
            filter: filter_entries,
            composite: composite_entries,
        } = Interfaces::new();
        let layout =
            |name, entries: &[BindGroupLayoutEntry]| device.create_bind_group_layout(name, entries);
        let compiler = PipelineCompiler { cache, fullscreen };
        let opaque_formats = [
            TextureFormat::Rgba16Float,
            TextureFormat::Rg16Float,
            TextureFormat::Rgb10a2Unorm,
        ];
        let volume_formats = [
            TextureFormat::Rgba16Float,
            TextureFormat::Rg16Float,
            TextureFormat::Rgba32Float,
        ];
        Self {
            opaque_layout: layout("opaque", &entries),
            volume_layout: layout("volume", &volume_entries),
            sky_layout: layout("sky", &sky_entries),
            filter_layout: layout("volume_temporal", &filter_entries),
            composite_layout: layout("volume_composite", &composite_entries),
            opaque: compiler.queue(
                "opaque",
                scene_shader.clone(),
                "opaque",
                &entries,
                &opaque_formats,
                true,
            ),
            volume: compiler.queue(
                "volume",
                scene_shader.clone(),
                "volume",
                &volume_entries,
                &volume_formats,
                true,
            ),
            thin: compiler.queue(
                "transparent",
                scene_shader.clone(),
                "transparent",
                &volume_entries,
                &[TextureFormat::Rgba16Float; 3],
                false,
            ),
            sky: compiler.queue(
                "sky",
                scene_shader.clone(),
                "sky",
                &sky_entries,
                &[TextureFormat::Rgba16Float],
                false,
            ),
            ejection_lookup: compiler.queue(
                "ejection_lookup",
                scene_shader,
                "ejection_lookup",
                &sky_entries,
                &[TextureFormat::Rgba32Float],
                false,
            ),
            ejections: texture(
                device,
                UVec2::new(EJECTION_GRID_WIDTH * 2, EJECTION_GRID_HEIGHT),
                TextureFormat::Rgba32Float,
                "ejection_lookup",
            ),
            filter: compiler.queue(
                "volume_temporal",
                shaders.filter.clone(),
                "reconstruct",
                &filter_entries,
                &[TextureFormat::Rgba16Float, TextureFormat::Rgba32Float],
                false,
            ),
            composite: compiler.queue(
                "volume_composite",
                shaders.composite.clone(),
                "composite",
                &composite_entries,
                &[TextureFormat::Rgba16Float],
                false,
            ),
            repeat_sampler: scene_sampler(device, true),
            linear_sampler: scene_sampler(device, false),
            groups: None,
        }
    }
}

fn scene_sampler(device: &RenderDevice, repeat: bool) -> Sampler {
    let address = if repeat {
        AddressMode::Repeat
    } else {
        AddressMode::ClampToEdge
    };
    device.create_sampler(&SamplerDescriptor {
        address_mode_u: address,
        address_mode_v: address,
        address_mode_w: address,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: if repeat {
            MipmapFilterMode::Linear
        } else {
            MipmapFilterMode::Nearest
        },
        ..default()
    })
}
