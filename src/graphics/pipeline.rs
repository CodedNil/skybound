use crate::math::{ToEngine, ToScene};
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
    sampler, storage_buffer_read_only_sized, texture_2d, texture_3d, texture_depth_2d,
    uniform_buffer_sized,
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
use skybound::scene::{
    FrameUniform, ViewUniform, cache as volume_cache,
    life::{CreatureInstance, development},
};

use super::froxel::{FroxelInputs, FroxelVolume};
use crate::game::camera::DebugView;
use crate::game::flight::PlayerModel;
use crate::game::weather::CloudSnapshot;
use crate::game::world::WorldData;
use crate::graphics::prepare::sky::SkyLookup;
use crate::graphics::prepare::textures::VolumeTextures;
use crate::graphics::targets::{ColumnTargets, SceneTargets, Targets, VolumeScale, texture};
use skybound::scene::cloud::Cloud;

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
    opaque_rect: UVec4,
    clouds: RawBufferVec<Cloud>,
    cloud_grid: RawBufferVec<u32>,
    cloud_revision: u64,
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
            clouds: RawBufferVec::new(BufferUsages::STORAGE),
            cloud_grid: RawBufferVec::new(BufferUsages::STORAGE),
            cloud_revision: u64::MAX,
            opaque_rect: UVec4::ZERO,
            profile: env::var("SKYBOUND_PROFILE_VOLUME")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        }
    }
}
type ViewQuery = (&'static ExtractedView, Option<&'static TemporalJitter>);

fn opaque_rectangle(players: &[CreatureInstance; 2], clip: Mat4, viewport: UVec4) -> UVec4 {
    let mut lower = Vec2::ONE;
    let mut upper = Vec2::ZERO;
    for player in players {
        if player.position.w <= 0.0 {
            continue;
        }
        let rotation = Quat::from_array(player.rotation.to_array());
        let scale = development::scale(player).to_engine();
        let mut projected_corners = [Vec4::ZERO; 8];
        let mut behind = 0;
        for (corner, projected) in projected_corners.iter_mut().enumerate() {
            let signs = vec3(
                if corner & 1 == 0 { -1.0 } else { 1.0 },
                if corner & 2 == 0 { -1.0 } else { 1.0 },
                if corner & 4 == 0 { -1.0 } else { 1.0 },
            );
            let point = player.position.truncate().to_engine()
                + rotation * (player.opaque_bounds.truncate().to_engine() * scale * signs);
            *projected = clip * point.extend(1.0);
            behind += usize::from(projected.w <= 0.0);
        }
        if behind == 8 {
            continue;
        }
        if behind > 0 {
            return viewport;
        }
        for projected in projected_corners {
            let uv = projected.truncate().truncate() / projected.w * vec2(0.5, -0.5) + 0.5;
            lower = lower.min(uv);
            upper = upper.max(uv);
        }
    }
    let size = viewport.zw().as_vec2();
    let start = (lower * size - 2.0)
        .floor()
        .clamp(Vec2::ZERO, size)
        .as_uvec2();
    let end = (upper * size + 2.0)
        .ceil()
        .clamp(Vec2::ZERO, size)
        .as_uvec2()
        .max(start);
    UVec4::new(
        viewport.x + start.x,
        viewport.y + start.y,
        end.x - start.x,
        end.y - start.y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::camera::rh::proj::directx::perspective_infinite_reverse;

    #[test]
    fn creature_bounds_behind_the_camera_do_not_expand_visible_work() {
        let viewport = UVec4::new(0, 0, 3840, 2160);
        let clip = perspective_infinite_reverse(1.2, 16.0 / 9.0, 0.1).to_engine();
        let mut creature = CreatureInstance {
            position: vec4(0.0, 0.0, 10.0, 2.0).to_scene(),
            opaque_bounds: Vec4::ONE.to_scene(),
            rotation: Vec4::W.to_scene(),
            ..default()
        };
        let hidden = opaque_rectangle(&[creature, CreatureInstance::default()], clip, viewport);
        assert_eq!(hidden.zw(), UVec2::ZERO);
        creature.position.z = -10.0;
        let visible = opaque_rectangle(&[creature, CreatureInstance::default()], clip, viewport);
        assert!(visible.z > 0 && visible.w > 0 && visible.z < viewport.z);
        creature.position.z = 0.0;
        assert_eq!(
            opaque_rectangle(&[creature, CreatureInstance::default()], clip, viewport),
            viewport
        );
    }
}

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
    clouds: Res<'w, CloudSnapshot>,
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
        clouds,
        mut sky,
        mut targets,
    } = params;
    prepare_cloud_buffers(&mut buffer, &clouds, &device, &queue);
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
    let mut uniform = camera_uniform(
        view,
        clip_from_view,
        &world,
        buffer.previous_clip,
        buffer.profile,
    );
    uniform.weather = atmosphere.0.to_scene();
    uniform.cloud_time = vec4(
        (time.elapsed_secs() - clouds.time).max(0.0),
        clouds.time,
        clouds.clouds.len() as f32,
        0.0,
    )
    .to_scene();
    uniform.times = vec4(
        time.elapsed_secs_wrapped(),
        frame_count.0 as f32,
        debug.0 as f32,
        time.delta_secs(),
    )
    .to_scene();
    prepare_lighting(&mut uniform, &mut sky);
    buffer.opaque_rect = if uniform.altitude() <= 0.0 {
        view.viewport
    } else {
        opaque_rectangle(
            &player.0,
            uniform.clip_from_world.to_engine(),
            view.viewport,
        )
    };
    let valid = history_valid(&buffer, targets, view, &world, debug.0, atmosphere.0);
    buffer.uniform.clear();
    buffer.uniform.push(FrameUniform {
        view: uniform,
        players: player.0,
        viewport: viewport.to_scene(),
    });
    buffer.resolve.clear();
    let profile = buffer.profile;
    buffer.resolve.push(ResolveUniform {
        world_from_clip,
        origin: position.extend(0.0),
        size: targets.size.as_vec2().extend(f32::from(valid)).extend(0.0),
        native: viewport.zw().extend(debug.0 as f32).extend(profile as f32),
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
    clouds: &CloudSnapshot,
    device: &RenderDevice,
    queue: &RenderQueue,
) {
    if buffer.cloud_revision == clouds.revision {
        return;
    }
    buffer.clouds.reserve(clouds.clouds.len(), device);
    buffer.cloud_grid.reserve(clouds.grid.len(), device);
    buffer.clouds.clear();
    buffer.cloud_grid.clear();
    for &cloud in clouds.clouds.iter() {
        buffer.clouds.push(cloud);
    }
    for &index in clouds.grid.iter() {
        buffer.cloud_grid.push(index);
    }
    buffer.clouds.write_buffer(device, queue);
    buffer.cloud_grid.write_buffer(device, queue);
    buffer.cloud_revision = clouds.revision;
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

fn prepare_lighting(uniform: &mut ViewUniform, sky: &mut SkyLookup) {
    uniform.prepare_atmosphere();
    uniform.prepare_flashes();
    uniform.prepare_cache();
    sky.update(uniform);
}

struct BindGroups {
    cloud_revision: u64,
    depth: TextureViewId,
    volume: TextureViewId,
    native: [TextureViewId; 2],
    opaque: BindGroup,
    volume_group: BindGroup,
    sky: BindGroup,
    columns: BindGroup,
    filter: [BindGroup; 2],
    composite: [[BindGroup; 2]; 2],
}
#[derive(Resource)]
pub struct ScenePipeline {
    opaque_layout: BindGroupLayout,
    volume_layout: BindGroupLayout,
    sky_layout: BindGroupLayout,
    column_layout: BindGroupLayout,
    filter_layout: BindGroupLayout,
    composite_layout: BindGroupLayout,
    opaque: CachedRenderPipelineId,
    background: CachedRenderPipelineId,
    volume: CachedRenderPipelineId,
    thin: CachedRenderPipelineId,
    sky: CachedRenderPipelineId,
    ejection_lookup: CachedRenderPipelineId,
    ejections: TextureView,
    columns: CachedRenderPipelineId,
    column_targets: ColumnTargets,
    froxel: FroxelVolume,
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
#[expect(
    clippy::too_many_lines,
    reason = "one descriptor preparation step for the scene passes"
)]
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
        groups.cloud_revision == buffer.cloud_revision
            && groups.depth == depth.id()
            && groups.volume == targets.color.id()
            && groups.native.contains(&native[0].id())
            && groups.native.contains(&native[1].id())
    }) {
        return;
    }
    let opaque = opaque_group(
        &pipeline,
        &device,
        frame.clone(),
        [
            &base.texture_view,
            &detail.texture_view,
            &weather.texture_view,
            &sky.view,
        ],
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
            &pipeline.column_targets.shape,
            &pipeline.column_targets.columns,
            &pipeline.froxel.view,
            buffer.clouds.binding().expect("prepared cloud records"),
            buffer.cloud_grid.binding().expect("prepared cloud index"),
        )),
    );
    let column_group = column_group(
        &pipeline,
        &device,
        frame.clone(),
        [
            &base.texture_view,
            &detail.texture_view,
            &weather.texture_view,
            &sky.view,
            &transmittance.texture_view,
        ],
    );
    let (filter, composite) =
        reconstruction_groups(&pipeline, &device, resolve, targets, depth, native);
    let medium_images = [
        &base.texture_view,
        &detail.texture_view,
        &weather.texture_view,
        &transmittance.texture_view,
        &pipeline.ejections,
        &pipeline.column_targets.shape,
        &pipeline.column_targets.columns,
    ]
    .map(Clone::clone);
    let medium_sampler = pipeline.repeat_sampler.clone();
    pipeline.froxel.prepare(
        &device,
        FroxelInputs {
            frame: buffer.uniform.binding().expect("prepared frame"),
            sampler: &medium_sampler,
            images: medium_images.each_ref(),
            clouds: buffer.clouds.binding().expect("prepared cloud records"),
            grid: buffer.cloud_grid.binding().expect("prepared cloud index"),
        },
    );
    pipeline.groups = Some(BindGroups {
        cloud_revision: buffer.cloud_revision,
        depth: depth.id(),
        volume: targets.color.id(),
        native: native.map(TextureView::id),
        opaque,
        volume_group,
        sky: device.create_bind_group(
            "sky",
            &pipeline.sky_layout,
            &BindGroupEntries::single(frame),
        ),
        columns: column_group,
        filter,
        composite,
    });
}

fn opaque_group(
    pipeline: &ScenePipeline,
    device: &RenderDevice,
    frame: BindingResource<'_>,
    images: [&TextureView; 4],
) -> BindGroup {
    device.create_bind_group(
        "opaque",
        &pipeline.opaque_layout,
        &BindGroupEntries::sequential((
            frame,
            &pipeline.repeat_sampler,
            images[0],
            images[1],
            images[2],
            images[3],
        )),
    )
}

fn column_group(
    pipeline: &ScenePipeline,
    device: &RenderDevice,
    frame: BindingResource<'_>,
    images: [&TextureView; 5],
) -> BindGroup {
    device.create_bind_group(
        "column_cache",
        &pipeline.column_layout,
        &BindGroupEntries::sequential((
            frame,
            &pipeline.repeat_sampler,
            images[0],
            images[1],
            images[2],
            images[3],
            images[4],
            &pipeline.ejections,
        )),
    )
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
    draw_region(
        context, pipeline, group, colors, depth, viewport, label, None,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "render pass parameters plus an optional scissor"
)]
fn draw_region(
    context: &mut RenderContext,
    pipeline: &RenderPipeline,
    group: &BindGroup,
    colors: &[Option<RenderPassColorAttachment<'_>>],
    depth: Option<&TextureView>,
    viewport: UVec4,
    label: &'static str,
    scissor: Option<UVec4>,
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
                load: if scissor.is_some() {
                    LoadOp::Load
                } else {
                    LoadOp::Clear(0.0)
                },
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
    if let Some(rect) = scissor {
        pass.set_scissor_rect(rect.x, rect.y, rect.z, rect.w);
    }
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
        Some(background),
        Some(volume),
        Some(sky_pipeline),
    ) = (
        pipeline.groups.as_ref(),
        world.resource::<SceneTargets>().0.as_ref(),
        cache.get_render_pipeline(pipeline.opaque),
        cache.get_render_pipeline(pipeline.background),
        cache.get_render_pipeline(pipeline.volume),
        cache.get_render_pipeline(pipeline.sky),
    )
    else {
        return;
    };
    if !draw_lookup_passes(&mut context, pipeline, groups, cache) {
        return;
    }
    if !pipeline.froxel.draw(&mut context, cache) {
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
            background,
            &groups.opaque,
            &[
                Some(target.get_color_attachment()),
                Some(attachment(motion, true)),
                Some(attachment(normal, true)),
            ],
            Some(depth),
            view.viewport,
            "background",
        );
        let rect = world.resource::<SceneBuffer>().opaque_rect;
        if rect.z > 0 && rect.w > 0 {
            draw_region(
                &mut context,
                opaque,
                &groups.opaque,
                &[
                    Some(attachment(target.main_texture_view(), false)),
                    Some(attachment(motion, false)),
                    Some(attachment(normal, false)),
                ],
                Some(depth),
                view.viewport,
                "opaque_creatures",
                Some(rect),
            );
        }
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
            UVec4::new(0, 0, targets.trace_size.x, targets.trace_size.y),
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
    let (Some(ejection_pipeline), Some(column_pipeline)) = (
        cache.get_render_pipeline(pipeline.ejection_lookup),
        cache.get_render_pipeline(pipeline.columns),
    ) else {
        return false;
    };
    draw(
        context,
        ejection_pipeline,
        &groups.sky,
        &[Some(attachment(&pipeline.ejections, true))],
        None,
        UVec4::new(0, 0, EJECTION_GRID_WIDTH * 2, EJECTION_GRID_HEIGHT),
        "ejection_lookup",
    );
    let grid = volume_cache::SIZE;
    let width = grid * volume_cache::CASCADES;
    let height = grid * volume_cache::ROWS;
    draw(
        context,
        column_pipeline,
        &groups.columns,
        &[
            Some(attachment(&pipeline.column_targets.shape, true)),
            Some(attachment(&pipeline.column_targets.columns, true)),
        ],
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
        pipeline.background,
        pipeline.volume,
        pipeline.sky,
        pipeline.ejection_lookup,
        pipeline.columns,
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
    columns: Vec<BindGroupLayoutEntry>,
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
            (
                (6, texture_depth_2d()),
                (7, float()),
                (8, data()),
                (9, float()),
                (10, float()),
                (
                    11,
                    texture_3d(TextureSampleType::Float { filterable: true }),
                ),
                (12, storage_buffer_read_only_sized(false, None)),
                (13, storage_buffer_read_only_sized(false, None)),
            ),
        );
        volume.extend_from_slice(&extra);
        let sky = BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, frame_entry());
        let mut columns = opaque.to_vec();
        columns.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((6, float()), (7, data())),
        ));
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
            columns,
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
            columns: column_entries,
            filter: filter_entries,
            composite: composite_entries,
        } = Interfaces::new();
        let layout =
            |name, entries: &[BindGroupLayoutEntry]| device.create_bind_group_layout(name, entries);
        let compiler = PipelineCompiler { cache, fullscreen };
        let scene_pipeline =
            |entry, entries: &[BindGroupLayoutEntry], formats: &[TextureFormat], depth| {
                compiler.queue(entry, scene_shader.clone(), entry, entries, formats, depth)
            };
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
            column_layout: layout("column_cache", &column_entries),
            filter_layout: layout("volume_temporal", &filter_entries),
            composite_layout: layout("volume_composite", &composite_entries),
            opaque: scene_pipeline("opaque", &entries, &opaque_formats, true),
            background: scene_pipeline("background", &entries, &opaque_formats, true),
            volume: scene_pipeline("volume", &volume_entries, &volume_formats, true),
            thin: scene_pipeline(
                "transparent",
                &volume_entries,
                &[TextureFormat::Rgba16Float; 3],
                false,
            ),
            sky: scene_pipeline("sky", &sky_entries, &[TextureFormat::Rgba16Float], false),
            ejection_lookup: scene_pipeline(
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
            columns: scene_pipeline(
                "column_cache",
                &column_entries,
                &[TextureFormat::Rgba16Float; 2],
                false,
            ),
            column_targets: ColumnTargets::new(device),
            froxel: FroxelVolume::new(world),
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
