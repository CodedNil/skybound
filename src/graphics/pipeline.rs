use crate::{
    game::{
        camera::DebugView,
        flight::PlayerModel,
        weather::{CloudField, prepare_flashes},
        world::WorldData,
    },
    graphics::{
        binding::{Bind, Binder, Pass},
        light::LightVolume,
        targets::{SceneTargets, Targets, VolumeScale, texture},
        textures::{SkyLookup, VolumeTextures},
    },
    math::ToScene,
};
use bevy::{
    core_pipeline::{FullscreenShader, prepass::ViewPrepassTextures},
    diagnostic::FrameCount,
    ecs::system::SystemParam,
    math::DVec3,
    prelude::*,
    render::{
        camera::TemporalJitter,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{
            AddressMode, BindGroup, BufferUsages, FilterMode, LoadOp, MipmapFilterMode, Operations,
            PipelineCache, RawBufferVec, RenderPassColorAttachment,
            RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, Sampler,
            SamplerDescriptor, StoreOp, TextureFormat, TextureView, TextureViewId,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{GpuImage, TextureCache},
        view::{ExtractedView, ViewTarget},
    },
};
use skybound::scene::{
    FrameUniform, ViewUniform,
    atmosphere::SKY_SIZE,
    aur::EJECTION_WINDOW,
    cache as volume_cache,
    climate::{CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CloudParcel},
};
use std::{array::from_fn, env, sync::atomic::Ordering};

#[derive(Resource)]
pub struct SceneBuffer {
    uniform: RawBufferVec<FrameUniform>,
    previous_clip: Mat4,
    previous_position: Vec3,
    previous_origin: DVec3,
    previous_rotation: Quat,
    previous_debug: u32,
    previous_weather: Vec4,
    frame: u32,
    profile: u32,
    parcels: RawBufferVec<CloudParcel>,
    cloud_index: RawBufferVec<u32>,
    cloud_cells: RawBufferVec<Vec4>,
    /// When the uploaded cloud index was built.
    cloud_built: f32,
}

impl Default for SceneBuffer {
    fn default() -> Self {
        Self {
            uniform: RawBufferVec::new(BufferUsages::UNIFORM),
            previous_clip: Mat4::IDENTITY,
            previous_position: Vec3::ZERO,
            previous_origin: DVec3::ZERO,
            previous_rotation: Quat::IDENTITY,
            previous_debug: 0,
            previous_weather: Vec4::ZERO,
            frame: 0,
            parcels: RawBufferVec::new(BufferUsages::STORAGE),
            cloud_index: RawBufferVec::new(BufferUsages::STORAGE),
            cloud_cells: RawBufferVec::new(BufferUsages::STORAGE),
            cloud_built: f32::NAN,
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
    clouds: Res<'w, CloudField>,
    sky: ResMut<'w, SkyLookup>,
    targets: ResMut<'w, SceneTargets>,
    texture_cache: ResMut<'w, TextureCache>,
}

struct BindGroups {
    cloud_built: f32,
    depth: TextureViewId,
    volume: TextureViewId,
    native: [TextureViewId; 2],
    opaque: BindGroup,
    volume_group: BindGroup,
    march_group: [BindGroup; 2],
    cloud_resolve_group: [BindGroup; 2],
    sky: BindGroup,
    columns: BindGroup,
    cloud_map: BindGroup,
    cloud_bounds: BindGroup,
    resolve: [BindGroup; 2],
    composite: [[BindGroup; 2]; 2],
}

/// Every fullscreen scene pass, and the textures the passes share.
#[derive(Resource)]
pub struct ScenePipeline {
    opaque: Pass,
    volume: Pass,
    clouds: Pass,
    cloud_resolve: Pass,
    thin: Pass,
    sky: Pass,
    ejection_lookup: Pass,
    columns: Pass,
    cloud_map: Pass,
    cloud_bounds_pass: Pass,
    resolve: Pass,
    composite: Pass,
    ejections: TextureView,
    column_shape: TextureView,
    cloud_cover: TextureView,
    cloud_layers: TextureView,
    cloud_towers: TextureView,
    cloud_bounds: TextureView,
    light: LightVolume,
    repeat_sampler: Sampler,
    linear_sampler: Sampler,
    groups: Option<BindGroups>,
}

#[derive(SystemParam)]
pub(super) struct PrepareBindings<'w, 's> {
    pipeline: ResMut<'w, ScenePipeline>,
    device: Res<'w, RenderDevice>,
    cache: Res<'w, PipelineCache>,
    fullscreen: Res<'w, FullscreenShader>,
    shader: Res<'w, super::SceneShader>,
    buffer: Res<'w, SceneBuffer>,
    images: Res<'w, RenderAssets<GpuImage>>,
    textures: Res<'w, VolumeTextures>,
    sky: Res<'w, SkyLookup>,
    targets: Res<'w, SceneTargets>,
    views: Single<'w, 's, (&'static ViewPrepassTextures, &'static ViewTarget), With<Camera3d>>,
}

const HALF: TextureFormat = TextureFormat::Rgba16Float;

const FULL: TextureFormat = TextureFormat::Rgba32Float;

impl FromWorld for ScenePipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let map = |format, label| {
            texture(
                device,
                UVec2::new(CLOUD_MAP_WIDTH, CLOUD_MAP_HEIGHT),
                format,
                label,
            )
        };
        Self {
            opaque: Pass::new(
                "opaque",
                &[HALF, TextureFormat::Rg16Float, TextureFormat::Rgb10a2Unorm],
                true,
            ),
            volume: Pass::new("volume", &[HALF, TextureFormat::Rg16Float, FULL], false),
            clouds: Pass::new("clouds", &[HALF, TextureFormat::R32Float], false),
            cloud_resolve: Pass::new("cloud_resolve", &[HALF, TextureFormat::R32Float], false),
            thin: Pass::new("transparent", &[HALF; 3], false),
            sky: Pass::new("sky", &[HALF], false),
            ejection_lookup: Pass::new("ejection_lookup", &[FULL], false),
            columns: Pass::new("column_cache", &[HALF], false),
            cloud_map: Pass::new("cloud_map", &[HALF, HALF, FULL], false),
            cloud_bounds_pass: Pass::new("cloud_bounds", &[HALF], false),
            resolve: Pass::new("volume_resolve", &[HALF, FULL], false),
            composite: Pass::new("volume_composite", &[HALF], false),
            ejections: texture(
                device,
                UVec2::new(EJECTION_WINDOW * 3, EJECTION_WINDOW),
                FULL,
                "ejection_lookup",
            ),
            column_shape: texture(
                device,
                UVec2::new(
                    volume_cache::SIZE * volume_cache::CASCADES,
                    volume_cache::SIZE,
                ),
                HALF,
                "column_shape",
            ),
            cloud_cover: map(HALF, "cloud_cover"),
            cloud_layers: map(HALF, "cloud_layers"),
            cloud_towers: map(FULL, "cloud_towers"),
            cloud_bounds: map(HALF, "cloud_bounds"),
            light: LightVolume::new(device),
            repeat_sampler: scene_sampler(device, true),
            linear_sampler: scene_sampler(device, false),
            groups: None,
        }
    }
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
        clouds,
        mut sky,
        mut targets,
        mut texture_cache,
    } = params;
    prepare_cloud_buffers(&mut buffer, &clouds, &device, &queue);
    let (view, jitter) = *views;
    let viewport = view.viewport.as_vec4();
    targets.resize(
        &device,
        &mut texture_cache,
        viewport.zw().as_uvec2(),
        scale.0,
    );
    let targets = targets.0.as_ref().expect("prepared volume targets");
    let mut clip_from_view = view.clip_from_view;
    if let Some(jitter) = jitter {
        jitter.jitter_projection(&mut clip_from_view, viewport.zw());
    }
    let world_from_view = view.world_from_view.to_matrix();
    let position = view.world_from_view.translation();
    let mut uniform = camera_uniform(
        view,
        clip_from_view,
        &world,
        buffer.previous_clip,
        buffer.profile,
    );
    uniform.weather = atmosphere.0.to_scene();
    uniform.times = vec4(
        time.elapsed_secs_wrapped(),
        frame_count.0 as f32,
        debug.0 as f32,
        time.delta_secs(),
    )
    .to_scene();
    uniform.prepare_atmosphere();
    prepare_flashes(&mut uniform);
    uniform.prepare_cache();
    sky.update(&mut uniform);
    let valid = history_valid(&buffer, targets, view, &world, debug.0, atmosphere.0);
    buffer.uniform.clear();
    buffer.uniform.push(FrameUniform {
        view: uniform,
        players: player.0,
        viewport: viewport.to_scene(),
        history: vec4(
            f32::from(valid),
            (frame_count.0 % 4) as f32,
            targets.trace_size.x as f32,
            targets.trace_size.y as f32,
        )
        .to_scene(),
    });
    buffer.uniform.write_buffer(&device, &queue);
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

fn camera_uniform(
    view: &ExtractedView,
    clip_from_view: Mat4,
    world: &WorldData,
    previous: Mat4,
    profile: u32,
) -> ViewUniform {
    let world_from_view = view.world_from_view.to_matrix();
    let world_from_clip = world_from_view * clip_from_view.inverse();
    let position = view.world_from_view.translation();
    let (rotation, latitude, longitude) = world.planet_frame(position);
    ViewUniform {
        clip_from_world: (clip_from_view * world_from_view.inverse()).to_scene(),
        world_from_clip: world_from_clip.to_scene(),
        prev_clip_from_world: previous.to_scene(),
        direction_from_clip_unjittered: (Mat4::from_quat(view.world_from_view.rotation())
            * view.clip_from_view.inverse())
        .to_scene(),
        direction_from_clip: (Mat4::from_quat(view.world_from_view.rotation())
            * clip_from_view.inverse())
        .to_scene(),
        world_position: position.extend(0.0).to_scene(),
        planet_center_value: world.local(DVec3::ZERO).extend(0.0).to_scene(),
        camera_position: vec4(
            latitude,
            longitude,
            world.altitude(position) as f32,
            profile as f32,
        )
        .to_scene(),
        planet_rotation: rotation.to_scene(),
        ..default()
    }
}

fn prepare_cloud_buffers(
    buffer: &mut SceneBuffer,
    clouds: &CloudField,
    device: &RenderDevice,
    queue: &RenderQueue,
) {
    buffer.parcels.clear();
    buffer.parcels.extend(clouds.parcels.iter().copied());
    buffer.parcels.write_buffer(device, queue);
    if buffer.cloud_built.to_bits() == clouds.built.to_bits() {
        return;
    }
    buffer.cloud_index.clear();
    buffer.cloud_index.extend(clouds.index.iter().copied());
    buffer.cloud_cells.clear();
    buffer
        .cloud_cells
        .extend(clouds.cells.iter().map(|cell| cell.to_array().into()));
    buffer.cloud_index.write_buffer(device, queue);
    buffer.cloud_cells.write_buffer(device, queue);
    buffer.cloud_built = clouds.built;
}

fn history_valid(
    buffer: &SceneBuffer,
    targets: &Targets,
    view: &ExtractedView,
    world: &WorldData,
    debug: u32,
    weather: Vec4,
) -> bool {
    let position = view.world_from_view.translation();
    buffer.profile & 16 == 0
        && targets.valid.load(Ordering::Relaxed)
        && position.distance(buffer.previous_position) < 100.0
        && world.origin == buffer.previous_origin
        && view
            .world_from_view
            .rotation()
            .dot(buffer.previous_rotation)
            .abs()
            > 0.95
        && debug == buffer.previous_debug
        && weather == buffer.previous_weather
}

#[expect(
    clippy::too_many_lines,
    reason = "every scene pass lists its bindings here"
)]
pub(super) fn prepare_bindings(params: PrepareBindings) {
    let PrepareBindings {
        mut pipeline,
        device,
        cache,
        fullscreen,
        shader,
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
        Some(base),
        Some(detail),
        Some(shapes),
        Some(transmittance),
        Some(parcels),
        Some(index),
        Some(cells),
    ) = (
        prepass.depth_only_view(),
        targets.0.as_ref(),
        buffer.uniform.binding(),
        images.get(&textures.base),
        images.get(&textures.detail),
        images.get(&textures.shapes),
        images.get(&textures.transmittance),
        buffer.parcels.binding(),
        buffer.cloud_index.binding(),
        buffer.cloud_cells.binding(),
    )
    else {
        return;
    };
    let native = [view.main_texture_view(), view.main_texture_other_view()];
    if pipeline.groups.as_ref().is_some_and(|groups| {
        groups.cloud_built.to_bits() == buffer.cloud_built.to_bits()
            && groups.depth == depth.id()
            && groups.volume == targets.color.id()
            && groups.native.contains(&native[0].id())
            && groups.native.contains(&native[1].id())
    }) {
        return;
    }
    let binder = Binder {
        device: &device,
        cache: &cache,
        fullscreen: &fullscreen,
        shader: shader.0.clone(),
    };
    let pipeline = &mut *pipeline;
    let (base, detail, shapes, transmittance) = (
        &base.texture_view,
        &detail.texture_view,
        &shapes.texture_view,
        &transmittance.texture_view,
    );
    let common = || {
        [
            Bind::Frame(frame.clone()),
            Bind::Sampler(&pipeline.repeat_sampler),
            Bind::Volume(base),
            Bind::Volume(detail),
            Bind::Float(&sky.view),
        ]
    };
    let opaque = pipeline.opaque.group(&binder, &common());
    // The cloud, thin-surface and volume passes all read the whole scene;
    // the volume pass also reads the clouds rebuilt before it.
    let scene = |slab: Option<usize>| {
        let mut binds = common().into_iter().collect::<Vec<_>>();
        binds.extend([
            Bind::Depth(depth),
            Bind::Float(transmittance),
            Bind::Data(&pipeline.ejections),
            Bind::Float(&pipeline.column_shape),
            Bind::Float(&pipeline.cloud_cover),
            Bind::Volume(&pipeline.light.view),
            Bind::Float(&pipeline.cloud_layers),
            Bind::Storage(cells.clone()),
            Bind::Float(&pipeline.cloud_bounds),
            Bind::Data(&pipeline.cloud_towers),
            Bind::Volume(shapes),
        ]);
        if let Some(index) = slab {
            binds.extend([
                Bind::Data(&targets.cloud_slab[index]),
                Bind::Data(&targets.cloud_front[index]),
            ]);
        }
        binds
    };
    let (layout, volume_group) = binder.group("scene", &scene(None));
    pipeline.clouds.queue(&binder, &layout);
    pipeline.thin.queue(&binder, &layout);
    let march_group = from_fn(|index| pipeline.volume.group(&binder, &scene(Some(index))));
    let cloud_resolve_group = from_fn(|index| {
        pipeline.cloud_resolve.group(
            &binder,
            &[
                Bind::Frame(frame.clone()),
                Bind::Sampler(&pipeline.linear_sampler),
                Bind::Float(&targets.cloud_fresh),
                Bind::Data(&targets.cloud_fresh_front),
                Bind::Float(&targets.cloud_slab[1 - index]),
                Bind::Data(&targets.cloud_front[1 - index]),
            ],
        )
    });
    let (layout, sky_group) = binder.group("sky", &[Bind::Frame(frame.clone())]);
    pipeline.sky.queue(&binder, &layout);
    pipeline.ejection_lookup.queue(&binder, &layout);
    let mut column_binds = common().into_iter().collect::<Vec<_>>();
    column_binds.extend([Bind::Float(transmittance), Bind::Data(&pipeline.ejections)]);
    let columns = pipeline.columns.group(&binder, &column_binds);
    let cloud_map = pipeline.cloud_map.group(
        &binder,
        &[
            Bind::Frame(frame.clone()),
            Bind::Sampler(&pipeline.repeat_sampler),
            Bind::Volume(base),
            Bind::Storage(parcels),
            Bind::Storage(index),
        ],
    );
    let cloud_bounds = pipeline.cloud_bounds_pass.group(
        &binder,
        &[
            Bind::Float(&pipeline.cloud_cover),
            Bind::Float(&pipeline.cloud_layers),
            Bind::Data(&pipeline.cloud_towers),
            Bind::Sampler(&pipeline.repeat_sampler),
            Bind::Volume(shapes),
        ],
    );
    let resolve = from_fn(|index| {
        pipeline.resolve.group(
            &binder,
            &[
                Bind::Frame(frame.clone()),
                Bind::Float(&targets.color),
                Bind::Data(&targets.motion),
                Bind::Data(&targets.metadata),
                Bind::Float(&targets.history[1 - index]),
                Bind::Data(&targets.history_metadata[1 - index]),
                Bind::Sampler(&pipeline.linear_sampler),
            ],
        )
    });
    let composite = from_fn(|native_index| {
        from_fn(|index| {
            pipeline.composite.group(
                &binder,
                &[
                    Bind::Frame(frame.clone()),
                    Bind::Float(native[native_index]),
                    Bind::Float(&targets.history[index]),
                    Bind::Data(&targets.history_metadata[index]),
                    Bind::Depth(depth),
                    Bind::Float(&targets.thin_color),
                    Bind::Float(&targets.thin_transmission),
                    Bind::Float(&targets.thin_prefix),
                ],
            )
        })
    });
    pipeline.light.prepare(
        &binder,
        frame.clone(),
        &pipeline.repeat_sampler,
        [&pipeline.cloud_cover, &pipeline.cloud_layers],
    );
    pipeline.groups = Some(BindGroups {
        cloud_built: buffer.cloud_built,
        depth: depth.id(),
        volume: targets.color.id(),
        native: native.map(TextureView::id),
        opaque,
        volume_group,
        march_group,
        cloud_resolve_group,
        sky: sky_group,
        columns,
        cloud_map,
        cloud_bounds,
        resolve,
        composite,
    });
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
        Some(clouds),
        Some(cloud_resolve),
        Some(sky_pipeline),
    ) = (
        pipeline.groups.as_ref(),
        world.resource::<SceneTargets>().0.as_ref(),
        pipeline.opaque.get(cache),
        pipeline.volume.get(cache),
        pipeline.clouds.get(cache),
        pipeline.cloud_resolve.get(cache),
        pipeline.sky.get(cache),
    )
    else {
        return;
    };
    let index = world.resource::<SceneBuffer>().frame as usize % 2;
    if !draw_lookup_passes(&mut context, pipeline, groups, cache) {
        return;
    }
    if !pipeline.light.draw(&mut context, cache) {
        return;
    }
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
            "opaque",
        );
        draw(
            &mut context,
            clouds,
            &groups.volume_group,
            &[
                Some(attachment(&targets.cloud_fresh, true)),
                Some(attachment(&targets.cloud_fresh_front, true)),
            ],
            None,
            UVec4::new(0, 0, targets.cloud_size.x, targets.cloud_size.y),
            "clouds",
        );
        draw(
            &mut context,
            cloud_resolve,
            &groups.cloud_resolve_group[index],
            &[
                Some(attachment(&targets.cloud_slab[index], true)),
                Some(attachment(&targets.cloud_front[index], true)),
            ],
            None,
            UVec4::new(0, 0, targets.trace_size.x, targets.trace_size.y),
            "cloud_resolve",
        );
        draw(
            &mut context,
            volume,
            &groups.march_group[index],
            &[
                Some(attachment(&targets.color, true)),
                Some(attachment(&targets.motion, true)),
                Some(attachment(&targets.metadata, true)),
            ],
            None,
            UVec4::new(0, 0, targets.cloud_size.x, targets.cloud_size.y),
            "volumetrics",
        );
    }
}

fn draw_lookup_passes(
    context: &mut RenderContext,
    pipeline: &ScenePipeline,
    groups: &BindGroups,
    cache: &PipelineCache,
) -> bool {
    let (
        Some(ejection_pipeline),
        Some(column_pipeline),
        Some(cloud_pipeline),
        Some(bounds_pipeline),
    ) = (
        pipeline.ejection_lookup.get(cache),
        pipeline.columns.get(cache),
        pipeline.cloud_map.get(cache),
        pipeline.cloud_bounds_pass.get(cache),
    )
    else {
        return false;
    };
    draw(
        context,
        cloud_pipeline,
        &groups.cloud_map,
        &[
            Some(attachment(&pipeline.cloud_cover, true)),
            Some(attachment(&pipeline.cloud_layers, true)),
            Some(attachment(&pipeline.cloud_towers, true)),
        ],
        None,
        UVec4::new(0, 0, CLOUD_MAP_WIDTH, CLOUD_MAP_HEIGHT),
        "cloud_map",
    );
    draw(
        context,
        bounds_pipeline,
        &groups.cloud_bounds,
        &[Some(attachment(&pipeline.cloud_bounds, true))],
        None,
        UVec4::new(0, 0, CLOUD_MAP_WIDTH, CLOUD_MAP_HEIGHT),
        "cloud_bounds",
    );
    draw(
        context,
        ejection_pipeline,
        &groups.sky,
        &[Some(attachment(&pipeline.ejections, true))],
        None,
        UVec4::new(0, 0, EJECTION_WINDOW * 3, EJECTION_WINDOW),
        "ejection_lookup",
    );
    let grid = volume_cache::SIZE;
    let width = grid * volume_cache::CASCADES;
    let height = grid;
    draw(
        context,
        column_pipeline,
        &groups.columns,
        &[Some(attachment(&pipeline.column_shape, true))],
        None,
        UVec4::new(0, 0, width, height),
        "column_cache",
    );
    true
}

pub fn composite_pass(
    world: &World,
    mut context: RenderContext,
    views: Query<(&ExtractedView, &ViewTarget)>,
) {
    let pipeline = world.resource::<ScenePipeline>();
    let cache = world.resource::<PipelineCache>();
    let (Some(groups), Some(targets), Some(thin), Some(resolve), Some(composite)) = (
        pipeline.groups.as_ref(),
        world.resource::<SceneTargets>().0.as_ref(),
        pipeline.thin.get(cache),
        pipeline.resolve.get(cache),
        pipeline.composite.get(cache),
    ) else {
        return;
    };
    if [
        &pipeline.opaque,
        &pipeline.volume,
        &pipeline.clouds,
        &pipeline.cloud_resolve,
        &pipeline.sky,
        &pipeline.ejection_lookup,
        &pipeline.columns,
        &pipeline.cloud_map,
        &pipeline.cloud_bounds_pass,
    ]
    .iter()
    .any(|pass| pass.get(cache).is_none())
    {
        targets.valid.store(false, Ordering::Relaxed);
        return;
    }
    let index = world.resource::<SceneBuffer>().frame as usize % 2;
    for (view, target) in &views {
        draw(
            &mut context,
            resolve,
            &groups.resolve[index],
            &[
                Some(attachment(&targets.history[index], true)),
                Some(attachment(&targets.history_metadata[index], true)),
            ],
            None,
            UVec4::new(0, 0, targets.size.x, targets.size.y),
            "volume_resolve",
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
