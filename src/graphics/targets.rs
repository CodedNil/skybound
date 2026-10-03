use bevy::render::{RenderApp, extract_resource::ExtractResource};
use bevy::{
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureView, TextureViewDescriptor,
        },
        renderer::RenderDevice,
    },
};
use std::env;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct RenderScale(pub f32);
impl Default for RenderScale {
    fn default() -> Self {
        Self(
            env::var("SKYBOUND_RENDER_SCALE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| {
                    if env::args().any(|arg| arg == "--benchmark") {
                        1.0
                    } else {
                        0.75
                    }
                })
                .clamp(0.25, 1.0),
        )
    }
}

#[derive(Resource, Default)]
pub struct SceneTargets(pub Option<Targets>);
pub struct Targets {
    pub size: UVec2,
    pub color: TextureView,
    pub motion: TextureView,
    pub normal: TextureView,
    pub depth: TextureView,
}
impl SceneTargets {
    pub fn resize(&mut self, device: &RenderDevice, size: UVec2) {
        if self.0.as_ref().is_some_and(|targets| targets.size == size) {
            return;
        }
        let texture = |format| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some("scene_low_resolution"),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        self.0 = Some(Targets {
            size,
            color: texture(TextureFormat::Rgba16Float),
            motion: texture(TextureFormat::Rg16Float),
            normal: texture(TextureFormat::Rgb10a2Unorm),
            depth: texture(TextureFormat::Depth32Float),
        });
    }
}
