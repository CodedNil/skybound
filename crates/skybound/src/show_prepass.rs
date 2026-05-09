use bevy::{
    asset::embedded_asset,
    core_pipeline::{Core3d, Core3dSystems, FullscreenShader, prepass::ViewPrepassTextures},
    material::descriptor::BindGroupLayoutDescriptor,
    platform::collections::HashMap,
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayoutEntry, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FragmentState, IntoBinding, Operations, PipelineCache,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines,
            TextureFormat, TextureSampleType,
            binding_types::{
                texture_2d, texture_2d_multisampled, texture_depth_2d,
                texture_depth_2d_multisampled, uniform_buffer,
            },
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget},
    },
};

/// A Bevy plugin to visualize depth, normal and motion vector prepasses.
#[derive(Debug, Default)]
pub struct ShowPrepassPlugin;

impl Plugin for ShowPrepassPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "show_prepass.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<ShowPrepass>::default(),
            ExtractComponentPlugin::<ShowPrepassDepthPower>::default(),
            UniformComponentPlugin::<ShowPrepassUniform>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<ShowPrepassPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_uniforms
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareResources),
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    prepare_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                show_prepass_render_system.in_set(Core3dSystems::PostProcess),
            );
    }
}

/// Add this component to a camera to visualize a prepass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Component, ExtractComponent)]
pub enum ShowPrepass {
    /// Visualize the depth prepass.
    Depth,
    /// Visualize the normal prepass.
    Normals,
    /// Visualize the motion vector prepass.
    MotionVectors,
}

/// Optional component to scale the depth visualization.
///
/// For example, a value of `0.75` will visualize depth as `depth = depth^0.75`.
#[derive(Debug, Clone, Copy, PartialEq, Component, ExtractComponent)]
pub struct ShowPrepassDepthPower(pub f32);

#[derive(Component, Clone, ShaderType)]
struct ShowPrepassUniform {
    depth_power: f32,
    delta_time: f32,
}

#[allow(clippy::type_complexity)]
fn show_prepass_render_system(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    views: Query<
        (
            &ViewTarget,
            &ExtractedCamera,
            &CachedShowPrepassPipeline,
            &ShowPrepassBindGroup,
            &DynamicUniformIndex<ShowPrepassUniform>,
        ),
        With<ShowPrepass>,
    >,
) {
    for (view_target, camera, pipeline_id, bind_group, uniform_index) in &views {
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            continue;
        };

        let post_process = view_target.post_process_write();

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("show_prepass_render_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[uniform_index.index()]);
        render_pass.draw(0..3, 0..1);
    }
}

fn prepare_uniforms(
    mut commands: Commands,
    views: Query<(Entity, Option<&ShowPrepassDepthPower>), With<ShowPrepass>>,
    time: Res<Time>,
) {
    for (view_entity, depth_power) in &views {
        commands.entity(view_entity).insert(ShowPrepassUniform {
            depth_power: depth_power.map_or(1.0, |d| d.0),
            delta_time: time.delta_secs(),
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShowPrepassPipelineKey {
    show_prepass: ShowPrepass,
    hdr: bool,
    multisampled: bool,
}

impl ShowPrepassPipelineKey {
    const fn layout_key(self) -> ShowPrepassPipelineLayoutKey {
        ShowPrepassPipelineLayoutKey {
            show_prepass: self.show_prepass,
            multisampled: self.multisampled,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ShowPrepassPipelineLayoutKey {
    show_prepass: ShowPrepass,
    multisampled: bool,
}

impl ShowPrepassPipelineLayoutKey {
    fn keys() -> impl Iterator<Item = Self> {
        [
            ShowPrepass::Depth,
            ShowPrepass::Normals,
            ShowPrepass::MotionVectors,
        ]
        .into_iter()
        .flat_map(|show_prepass| {
            [true, false].into_iter().map(move |multisampled| Self {
                show_prepass,
                multisampled,
            })
        })
    }
}

#[derive(Resource)]
struct ShowPrepassPipeline {
    shader: Handle<Shader>,
    fullscreen_shader: FullscreenShader,
    layouts: HashMap<ShowPrepassPipelineLayoutKey, Vec<BindGroupLayoutEntry>>,
}

impl SpecializedRenderPipeline for ShowPrepassPipeline {
    type Key = ShowPrepassPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let entries = self.layouts.get(&key.layout_key()).unwrap();

        RenderPipelineDescriptor {
            label: Some("show prepass pipeline".into()),
            layout: vec![BindGroupLayoutDescriptor {
                label: "show_prepass_bind_group_layout".into(),
                entries: entries.clone(),
            }],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: {
                    let mut defs = match key.show_prepass {
                        ShowPrepass::Depth => vec!["SHOW_DEPTH".into()],
                        ShowPrepass::Normals => vec!["SHOW_NORMALS".into()],
                        ShowPrepass::MotionVectors => vec!["SHOW_MOTION_VECTORS".into()],
                    };
                    if key.multisampled {
                        defs.push("MULTISAMPLED".into());
                    }
                    defs
                },
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: if key.hdr {
                        TextureFormat::Rgba16Float
                    } else {
                        TextureFormat::Rgba8UnormSrgb
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            zero_initialize_workgroup_memory: false,
            immediate_size: 0,
        }
    }
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    let layouts = ShowPrepassPipelineLayoutKey::keys()
        .map(|key| {
            let uniform = uniform_buffer::<ShowPrepassUniform>(true);
            let texture = match key.show_prepass {
                ShowPrepass::Depth => {
                    if key.multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    }
                }
                _ => {
                    if key.multisampled {
                        texture_2d_multisampled(TextureSampleType::Float { filterable: false })
                    } else {
                        texture_2d(TextureSampleType::Float { filterable: false })
                    }
                }
            };

            let entries = vec![
                uniform.build(0, ShaderStages::FRAGMENT),
                texture.build(1, ShaderStages::FRAGMENT),
            ];

            (key, entries)
        })
        .collect();

    commands.insert_resource(ShowPrepassPipeline {
        shader: asset_server.load("embedded://skybound/show_prepass.wgsl"),
        fullscreen_shader: fullscreen_shader.clone(),
        layouts,
    });
}

#[derive(Component)]
struct CachedShowPrepassPipeline(CachedRenderPipelineId);

fn prepare_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ShowPrepassPipeline>>,
    pipeline: Res<ShowPrepassPipeline>,
    views: Query<(Entity, &ExtractedView, Option<&ShowPrepass>, Option<&Msaa>)>,
) {
    for (view_entity, view, show_prepass, msaa) in &views {
        if let Some(show_prepass) = show_prepass {
            let key = ShowPrepassPipelineKey {
                show_prepass: *show_prepass,
                hdr: view.target_format == TextureFormat::Rgba16Float,
                multisampled: msaa.is_some_and(|msaa| msaa.samples() > 1),
            };
            let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, key);
            commands
                .entity(view_entity)
                .insert(CachedShowPrepassPipeline(pipeline_id));
        } else {
            commands
                .entity(view_entity)
                .remove::<CachedShowPrepassPipeline>();
        }
    }
}

#[derive(Component)]
struct ShowPrepassBindGroup(BindGroup);

fn prepare_bind_groups(
    mut commands: Commands,
    views: Query<(
        Entity,
        &ShowPrepass,
        Option<&ViewPrepassTextures>,
        Option<&Msaa>,
    )>,
    uniforms: Res<ComponentUniforms<ShowPrepassUniform>>,
    render_device: Res<RenderDevice>,
    pipeline: Res<ShowPrepassPipeline>,
) {
    for (view_entity, show_prepass, view_prepass_textures, msaa) in &views {
        let key = ShowPrepassPipelineLayoutKey {
            show_prepass: *show_prepass,
            multisampled: msaa.is_some_and(|msaa| msaa.samples() > 1),
        };
        let entries = pipeline.layouts.get(&key).unwrap();
        let layout = render_device.create_bind_group_layout("show_prepass_layout", entries);

        let Some(uniform) = uniforms.uniforms().binding() else {
            continue;
        };
        let uniform_entry = BindGroupEntry {
            binding: 0,
            resource: uniform,
        };

        let resource = match show_prepass {
            ShowPrepass::Depth => view_prepass_textures
                .and_then(|t| t.depth_view())
                .map(IntoBinding::into_binding),
            ShowPrepass::Normals => view_prepass_textures
                .and_then(|t| t.normal_view())
                .map(IntoBinding::into_binding),
            ShowPrepass::MotionVectors => view_prepass_textures
                .and_then(|t| t.motion_vectors_view())
                .map(IntoBinding::into_binding),
        };

        if let Some(resource) = resource {
            let bind_group = render_device.create_bind_group(
                "show_prepass_bind_group",
                &layout,
                &[
                    uniform_entry,
                    BindGroupEntry {
                        binding: 1,
                        resource,
                    },
                ],
            );
            commands
                .entity(view_entity)
                .insert(ShowPrepassBindGroup(bind_group));
        } else {
            commands
                .entity(view_entity)
                .remove::<ShowPrepassBindGroup>();
        }
    }
}
