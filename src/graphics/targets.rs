use std::array::from_fn;
use std::env;
use std::sync::atomic::AtomicBool;

use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};
use bevy::render::renderer::RenderDevice;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct VolumeScale(pub f32);
impl Default for VolumeScale {
    fn default() -> Self {
        Self(
            env::var("SKYBOUND_VOLUME_SCALE")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(0.75)
                .clamp(0.25, 1.0),
        )
    }
}

pub fn texture(
    device: &RenderDevice,
    size: UVec2,
    format: TextureFormat,
    label: &'static str,
) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some(label),
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
}
#[derive(Resource, Default)]
pub struct SceneTargets(pub Option<Targets>);
pub struct Targets {
    pub size: UVec2,
    pub native_size: UVec2,
    pub color: TextureView,
    pub motion: TextureView,
    pub metadata: TextureView,
    pub depth: TextureView,
    pub history: [TextureView; 2],
    pub history_metadata: [TextureView; 2],
    pub thin_color: TextureView,
    pub thin_transmission: TextureView,
    pub thin_prefix: TextureView,
    pub valid: AtomicBool,
}
impl SceneTargets {
    pub fn resize(&mut self, device: &RenderDevice, native: UVec2, scale: f32) {
        let size = (native.as_vec2() * scale).floor().max(Vec2::ONE).as_uvec2();
        if self
            .0
            .as_ref()
            .is_some_and(|t| t.size == size && t.native_size == native)
        {
            return;
        }
        let low = |format, label| texture(device, size, format, label);
        let native_texture = |label| texture(device, native, TextureFormat::Rgba16Float, label);
        self.0 = Some(Targets {
            size,
            native_size: native,
            color: low(TextureFormat::Rgba16Float, "volume_radiance"),
            motion: low(TextureFormat::Rg16Float, "volume_motion"),
            metadata: low(TextureFormat::Rgba32Float, "volume_depth"),
            depth: low(TextureFormat::Depth32Float, "volume_depth_attachment"),
            history: from_fn(|_| low(TextureFormat::Rgba16Float, "volume_history")),
            history_metadata: from_fn(|_| low(TextureFormat::Rgba32Float, "volume_history_depth")),
            thin_color: native_texture("thin_radiance"),
            thin_transmission: native_texture("thin_transmission"),
            thin_prefix: native_texture("thin_foreground_volume"),
            valid: AtomicBool::new(false),
        });
    }
}
