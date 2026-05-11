use super::physics::ExtractedShipData;
use crate::render::raymarch::{CloudsViewUniformOffset, ViewUniforms};
use bevy::{
    camera::MainPassResolutionOverride,
    prelude::*,
    render::{
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, Operations,
            PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderStages, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, UniformBuffer,
            binding_types::uniform_buffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};
use skybound_shared::{ShipUniform, ViewUniform};

#[derive(Resource, Default)]
pub struct ShipRenderTargets {
    surface_tex: Option<bevy::render::render_resource::Texture>,
    gbuf_tex: Option<bevy::render::render_resource::Texture>,
    pub surface_view: Option<TextureView>,
    pub gbuf_view: Option<TextureView>,
    pub size: UVec2,
}

impl ShipRenderTargets {
    fn ensure_size(&mut self, device: &RenderDevice, size: UVec2) {
        if self.size == size {
            return;
        }
        let mk = |label: &str| {
            let tex = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = tex.create_view(&TextureViewDescriptor::default());
            (tex, view)
        };
        let (st, sv) = mk("ship_surface");
        let (gt, gv) = mk("ship_gbuf");
        self.surface_tex = Some(st);
        self.surface_view = Some(sv);
        self.gbuf_tex = Some(gt);
        self.gbuf_view = Some(gv);
        self.size = size;
    }
}

#[derive(Resource, Default)]
pub struct ShipUniforms {
    pub buffer: UniformBuffer<ShipUniform>,
}

#[derive(Resource)]
pub struct ShipPipeline {
    pub layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
}

pub fn init_ship_resources(world: &mut World) {
    let layout = {
        let device = world.resource::<RenderDevice>();
        device.create_bind_group_layout(
            "ship_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<ShipUniform>(false),
                ),
            ),
        )
    };

    let asset_server = world.resource::<AssetServer>();
    let shader = asset_server.load("shaders/raymarch.spv");

    let fullscreen_shader = world
        .resource::<bevy::core_pipeline::FullscreenShader>()
        .clone();

    let pipeline_id = {
        let layout_desc = bevy::render::render_resource::BindGroupLayoutDescriptor::new(
            "ship_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<ShipUniform>(false),
                ),
            ),
        );
        let pipeline_cache = world.resource::<PipelineCache>();
        pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("ship_pipeline".into()),
            layout: vec![layout_desc],
            immediate_size: 0,
            vertex: fullscreen_shader.to_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader,
                entry_point: Some("ship_main".into()),
                targets: vec![
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
                shader_defs: Vec::new(),
            }),
            zero_initialize_workgroup_memory: false,
        })
    };

    world.insert_resource(ShipPipeline {
        layout,
        pipeline_id,
    });
    world.insert_resource(ShipUniforms::default());
    world.insert_resource(ShipRenderTargets::default());
}

pub fn prepare_ship_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut ship_uniforms: ResMut<ShipUniforms>,
    data: Res<ExtractedShipData>,
) {
    ship_uniforms.buffer.set(data.uniform);
    ship_uniforms
        .buffer
        .write_buffer(&render_device, &render_queue);
}

pub fn prepare_ship_render_targets(
    render_device: Res<RenderDevice>,
    mut ship_targets: ResMut<ShipRenderTargets>,
    views: Query<(&ExtractedView, Option<&MainPassResolutionOverride>), With<Camera>>,
) {
    let Some((view, res_override)) = views.iter().next() else {
        return;
    };
    let vp = view.viewport.as_vec4();
    let size = res_override.map_or_else(
        || UVec2::new(vp.z as u32, vp.w as u32),
        |r| UVec2::new(r.x, r.y),
    );
    ship_targets.ensure_size(&render_device, size);
}

pub fn ship_pass(
    world: &World,
    mut render_context: RenderContext,
    view_query: Query<(
        &ExtractedView,
        &CloudsViewUniformOffset,
        Option<&MainPassResolutionOverride>,
    )>,
) {
    let ship_pipeline = world.resource::<ShipPipeline>();
    let pipeline_cache = world.resource::<PipelineCache>();
    let ship_targets = world.resource::<ShipRenderTargets>();
    let ship_uniforms = world.resource::<ShipUniforms>();

    let (
        Some(pipeline),
        Some(view_binding),
        Some(ship_binding),
        Some(surface_view),
        Some(gbuf_view),
    ) = (
        pipeline_cache.get_render_pipeline(ship_pipeline.pipeline_id),
        world.resource::<ViewUniforms>().uniforms.binding(),
        ship_uniforms.buffer.binding(),
        ship_targets.surface_view.as_ref(),
        ship_targets.gbuf_view.as_ref(),
    )
    else {
        return;
    };

    for (view, view_uniform_offset, resolution_override) in &view_query {
        let device = render_context.render_device();
        let bind_group = device.create_bind_group(
            "ship_bind_group",
            &ship_pipeline.layout,
            &BindGroupEntries::sequential((view_binding.clone(), ship_binding.clone())),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ship_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: Operations {
                        load: bevy::render::render_resource::LoadOp::Load,
                        store: bevy::render::render_resource::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(RenderPassColorAttachment {
                    view: gbuf_view,
                    resolve_target: None,
                    ops: Operations {
                        load: bevy::render::render_resource::LoadOp::Load,
                        store: bevy::render::render_resource::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let vp = view.viewport;
        let (x, y, w, h) =
            resolution_override.map_or((vp.x, vp.y, vp.z, vp.w), |r| (0u32, 0u32, r.x, r.y));
        render_pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);
    }
}
