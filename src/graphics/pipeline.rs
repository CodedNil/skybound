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
    DepthStencilState, FilterMode, FragmentState, LoadOp, MipmapFilterMode, MultisampleState,
    Operations, PipelineCache, PrimitiveState, RawBufferVec, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
    StencilState, StoreOp, TextureFormat, TextureSampleType, TextureView, TextureViewId,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ExtractedView, ViewTarget};
use skybound::scene::atmosphere::SKY_SIZE;
use skybound::scene::aur::{EJECTION_GRID_HEIGHT, EJECTION_GRID_WIDTH};
use skybound::scene::climate::{CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CloudParcel};
use skybound::scene::{FrameUniform, ViewUniform, cache as volume_cache};

use super::light::LightVolume;
use crate::game::camera::DebugView;
use crate::game::flight::PlayerModel;
use crate::game::weather::{CloudField, prepare_flashes};
use crate::game::world::WorldData;
use crate::graphics::targets::{ColumnTargets, SceneTargets, Targets, VolumeScale, texture};
use crate::graphics::textures::{SkyLookup, VolumeTextures};
use crate::math::ToScene;

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
            resolve: RawBufferVec::new(BufferUsages::UNIFORM),
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
    filter: [BindGroup; 2],
    composite: [[BindGroup; 2]; 2],
}

#[derive(Resource)]
pub struct ScenePipeline {
    opaque_layout: BindGroupLayout,
    volume_layout: BindGroupLayout,
    march_layout: BindGroupLayout,
    cloud_resolve_layout: BindGroupLayout,
    sky_layout: BindGroupLayout,
    column_layout: BindGroupLayout,
    cloud_map_layout: BindGroupLayout,
    filter_layout: BindGroupLayout,
    composite_layout: BindGroupLayout,
    opaque: CachedRenderPipelineId,
    volume: CachedRenderPipelineId,
    clouds: CachedRenderPipelineId,
    cloud_resolve: CachedRenderPipelineId,
    thin: CachedRenderPipelineId,
    sky: CachedRenderPipelineId,
    ejection_lookup: CachedRenderPipelineId,
    ejections: TextureView,
    columns: CachedRenderPipelineId,
    column_targets: ColumnTargets,
    cloud_map: CachedRenderPipelineId,
    cloud_cover: TextureView,
    cloud_layers: TextureView,
    cloud_towers: TextureView,
    cloud_bounds: TextureView,
    cloud_bounds_pipeline: CachedRenderPipelineId,
    cloud_bounds_layout: BindGroupLayout,
    light: LightVolume,
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

struct Interfaces {
    opaque: Vec<BindGroupLayoutEntry>,
    volume: Vec<BindGroupLayoutEntry>,
    march: Vec<BindGroupLayoutEntry>,
    cloud_resolve: Vec<BindGroupLayoutEntry>,
    sky: Vec<BindGroupLayoutEntry>,
    columns: Vec<BindGroupLayoutEntry>,
    cloud_map: Vec<BindGroupLayoutEntry>,
    cloud_bounds: Vec<BindGroupLayoutEntry>,
    filter: Vec<BindGroupLayoutEntry>,
    composite: Vec<BindGroupLayoutEntry>,
}

impl Interfaces {
    #[expect(clippy::too_many_lines, reason = "one layout per scene pass")]
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
                float(),
            ),
        );
        let mut volume = opaque.to_vec();
        let extra = BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (5, texture_depth_2d()),
                (6, float()),
                (7, data()),
                (8, float()),
                (9, float()),
                (
                    10,
                    texture_3d(TextureSampleType::Float { filterable: true }),
                ),
                (11, float()),
                (12, storage_buffer_read_only_sized(false, None)),
                (13, float()),
                (14, data()),
            ),
        );
        volume.extend_from_slice(&extra);
        // The volume pass also reads the clouds marched before it.
        let mut march = volume.clone();
        march.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((15, data()), (16, data())),
        ));
        let cloud_resolve = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                frame_entry(),
                sampler(SamplerBindingType::Filtering),
                float(),
                data(),
                float(),
                data(),
            ),
        );
        let sky = BindGroupLayoutEntries::single(ShaderStages::FRAGMENT, frame_entry());
        let mut columns = opaque.to_vec();
        columns.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((5, float()), (6, data())),
        ));
        let cloud_map = BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                frame_entry(),
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
            ),
        );
        let cloud_bounds =
            BindGroupLayoutEntries::sequential(ShaderStages::FRAGMENT, (float(), float(), data()));
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
            march,
            cloud_resolve: cloud_resolve.to_vec(),
            sky: sky.to_vec(),
            columns,
            cloud_map: cloud_map.to_vec(),
            cloud_bounds: cloud_bounds.to_vec(),
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
    #[expect(clippy::too_many_lines, reason = "every scene pipeline and its layout")]
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let cache = world.resource::<PipelineCache>();
        let fullscreen = world.resource::<FullscreenShader>();
        let scene_shader = world.resource::<super::SceneShader>().0.clone();
        let shaders = world.resource::<super::VolumeShaders>();
        let Interfaces {
            opaque: entries,
            volume: volume_entries,
            march: march_entries,
            cloud_resolve: cloud_resolve_entries,
            sky: sky_entries,
            columns: column_entries,
            cloud_map: cloud_map_entries,
            cloud_bounds: cloud_bounds_entries,
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
            march_layout: layout("volume_march", &march_entries),
            cloud_resolve_layout: layout("cloud_resolve", &cloud_resolve_entries),
            sky_layout: layout("sky", &sky_entries),
            column_layout: layout("column_cache", &column_entries),
            cloud_map_layout: layout("cloud_map", &cloud_map_entries),
            filter_layout: layout("volume_temporal", &filter_entries),
            composite_layout: layout("volume_composite", &composite_entries),
            opaque: scene_pipeline("opaque", &entries, &opaque_formats, true),
            volume: scene_pipeline("volume", &march_entries, &volume_formats, false),
            cloud_resolve: scene_pipeline(
                "cloud_resolve",
                &cloud_resolve_entries,
                &[TextureFormat::Rgba16Float, TextureFormat::R32Float],
                false,
            ),
            clouds: scene_pipeline(
                "clouds",
                &volume_entries,
                &[TextureFormat::Rgba16Float, TextureFormat::R32Float],
                false,
            ),
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
                UVec2::new(EJECTION_GRID_WIDTH * 3, EJECTION_GRID_HEIGHT),
                TextureFormat::Rgba32Float,
                "ejection_lookup",
            ),
            columns: scene_pipeline(
                "column_cache",
                &column_entries,
                &[TextureFormat::Rgba16Float],
                false,
            ),
            column_targets: ColumnTargets::new(device),
            cloud_map: scene_pipeline(
                "cloud_map",
                &cloud_map_entries,
                &[
                    TextureFormat::Rgba16Float,
                    TextureFormat::Rgba16Float,
                    TextureFormat::Rgba32Float,
                ],
                false,
            ),
            cloud_cover: cloud_texture(device, "cloud_cover"),
            cloud_layers: cloud_texture(device, "cloud_layers"),
            cloud_towers: texture(
                device,
                UVec2::new(CLOUD_MAP_WIDTH, CLOUD_MAP_HEIGHT),
                TextureFormat::Rgba32Float,
                "cloud_towers",
            ),
            cloud_bounds: cloud_texture(device, "cloud_bounds"),
            cloud_bounds_pipeline: scene_pipeline(
                "cloud_bounds",
                &cloud_bounds_entries,
                &[TextureFormat::Rgba16Float],
                false,
            ),
            cloud_bounds_layout: layout("cloud_bounds", &cloud_bounds_entries),
            light: LightVolume::new(world),
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
    uniform.times = vec4(
        time.elapsed_secs_wrapped(),
        frame_count.0 as f32,
        debug.0 as f32,
        time.delta_secs(),
    )
    .to_scene();
    prepare_lighting(&mut uniform, &mut sky);
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
    buffer.resolve.clear();
    let profile = buffer.profile;
    buffer.resolve.push(ResolveUniform {
        world_from_clip,
        origin: position.extend(0.0),
        size: targets
            .size
            .as_vec2()
            .extend(f32::from(valid))
            .extend((frame_count.0 % 4) as f32),
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

fn prepare_lighting(uniform: &mut ViewUniform, sky: &mut SkyLookup) {
    uniform.prepare_atmosphere();
    prepare_flashes(uniform);
    uniform.prepare_cache();
    sky.update(uniform);
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
        Some(transmittance),
    ) = (
        prepass.depth_only_view(),
        targets.0.as_ref(),
        buffer.uniform.binding(),
        buffer.resolve.binding(),
        images.get(&textures.base),
        images.get(&textures.detail),
        images.get(&textures.transmittance),
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
    let opaque = opaque_group(
        &pipeline,
        &device,
        frame.clone(),
        [&base.texture_view, &detail.texture_view, &sky.view],
    );
    // The volume pass reads everything the cloud pass does, and the clouds
    // that pass rebuilt.
    let volume_group = |layout: &BindGroupLayout, slab: Option<usize>| {
        let shared = (
            frame.clone(),
            &pipeline.repeat_sampler,
            &base.texture_view,
            &detail.texture_view,
            &sky.view,
            depth,
            &transmittance.texture_view,
            &pipeline.ejections,
            &pipeline.column_targets.shape,
            &pipeline.cloud_cover,
            &pipeline.light.view,
            &pipeline.cloud_layers,
            buffer.cloud_cells.binding().expect("prepared cloud cells"),
            &pipeline.cloud_bounds,
            &pipeline.cloud_towers,
        );
        match slab {
            None => {
                device.create_bind_group("volume", layout, &BindGroupEntries::sequential(shared))
            }
            Some(index) => {
                let mut entries = BindGroupEntries::sequential(shared).to_vec();
                entries.extend(
                    BindGroupEntries::with_indices((
                        (15, &targets.cloud_slab[index]),
                        (16, &targets.cloud_front[index]),
                    ))
                    .to_vec(),
                );
                device.create_bind_group("volume_march", layout, &entries)
            }
        }
    };
    let march_group = from_fn(|index| volume_group(&pipeline.march_layout, Some(index)));
    let volume_group = volume_group(&pipeline.volume_layout, None);
    let cloud_resolve_group = from_fn(|index: usize| {
        device.create_bind_group(
            "cloud_resolve",
            &pipeline.cloud_resolve_layout,
            &BindGroupEntries::sequential((
                frame.clone(),
                &pipeline.linear_sampler,
                &targets.cloud_fresh,
                &targets.cloud_fresh_front,
                &targets.cloud_slab[1 - index],
                &targets.cloud_front[1 - index],
            )),
        )
    });

    let column_group = column_group(
        &pipeline,
        &device,
        frame.clone(),
        [
            &base.texture_view,
            &detail.texture_view,
            &sky.view,
            &transmittance.texture_view,
        ],
    );
    let (filter, composite) =
        reconstruction_groups(&pipeline, &device, resolve, targets, depth, native);
    let light_sampler = pipeline.repeat_sampler.clone();
    let maps = [pipeline.cloud_cover.clone(), pipeline.cloud_layers.clone()];
    pipeline.light.prepare(
        &device,
        buffer.uniform.binding().expect("prepared frame"),
        &light_sampler,
        maps.each_ref(),
    );
    let cloud_map = device.create_bind_group(
        "cloud_map",
        &pipeline.cloud_map_layout,
        &BindGroupEntries::sequential((
            frame.clone(),
            &pipeline.repeat_sampler,
            &base.texture_view,
            buffer.parcels.binding().expect("prepared cloud parcels"),
            buffer.cloud_index.binding().expect("prepared cloud index"),
        )),
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
        sky: device.create_bind_group(
            "sky",
            &pipeline.sky_layout,
            &BindGroupEntries::single(frame),
        ),
        columns: column_group,
        cloud_map,
        cloud_bounds: device.create_bind_group(
            "cloud_bounds",
            &pipeline.cloud_bounds_layout,
            &BindGroupEntries::sequential((
                &pipeline.cloud_cover,
                &pipeline.cloud_layers,
                &pipeline.cloud_towers,
            )),
        ),
        filter,
        composite,
    });
}

fn opaque_group(
    pipeline: &ScenePipeline,
    device: &RenderDevice,
    frame: BindingResource<'_>,
    images: [&TextureView; 3],
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
        )),
    )
}

fn column_group(
    pipeline: &ScenePipeline,
    device: &RenderDevice,
    frame: BindingResource<'_>,
    images: [&TextureView; 4],
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
        cache.get_render_pipeline(pipeline.opaque),
        cache.get_render_pipeline(pipeline.volume),
        cache.get_render_pipeline(pipeline.clouds),
        cache.get_render_pipeline(pipeline.cloud_resolve),
        cache.get_render_pipeline(pipeline.sky),
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
        cache.get_render_pipeline(pipeline.ejection_lookup),
        cache.get_render_pipeline(pipeline.columns),
        cache.get_render_pipeline(pipeline.cloud_map),
        cache.get_render_pipeline(pipeline.cloud_bounds_pipeline),
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
        UVec4::new(0, 0, EJECTION_GRID_WIDTH * 3, EJECTION_GRID_HEIGHT),
        "ejection_lookup",
    );
    let grid = volume_cache::SIZE;
    let width = grid * volume_cache::CASCADES;
    let height = grid;
    draw(
        context,
        column_pipeline,
        &groups.columns,
        &[Some(attachment(&pipeline.column_targets.shape, true))],
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
        pipeline.volume,
        pipeline.clouds,
        pipeline.cloud_resolve,
        pipeline.sky,
        pipeline.ejection_lookup,
        pipeline.columns,
        pipeline.cloud_map,
        pipeline.cloud_bounds_pipeline,
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

fn cloud_texture(device: &RenderDevice, label: &'static str) -> TextureView {
    texture(
        device,
        UVec2::new(CLOUD_MAP_WIDTH, CLOUD_MAP_HEIGHT),
        TextureFormat::Rgba16Float,
        label,
    )
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
