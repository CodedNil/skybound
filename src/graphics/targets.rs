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
use skybound::scene::cache;

pub struct ColumnTargets {
    pub shape: TextureView,
}

impl ColumnTargets {
    pub fn new(device: &RenderDevice) -> Self {
        let size = UVec2::new(cache::SIZE * cache::CASCADES, cache::SIZE);
        Self {
            shape: texture(device, size, TextureFormat::Rgba16Float, "column_shape"),
        }
    }
}

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
                .unwrap_or(1.0)
                .clamp(0.25, 1.0),
        )
    }
}

#[derive(Resource, Default)]
pub struct SceneTargets(pub Option<Targets>);

pub struct Targets {
    pub size: UVec2,
    pub trace_size: UVec2,
    pub native_size: UVec2,
    pub color: TextureView,
    pub motion: TextureView,
    pub metadata: TextureView,
    pub history: [TextureView; 2],
    pub history_metadata: [TextureView; 2],
    /// The clouds' light and transmittance, and the depth of the first.
    /// Traced this frame at a quarter of the pixels, then rebuilt in full
    /// with history into one of the alternating pairs.
    pub cloud_fresh: TextureView,
    pub cloud_fresh_front: TextureView,
    pub cloud_slab: [TextureView; 2],
    pub cloud_front: [TextureView; 2],
    pub cloud_size: UVec2,
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
        info!(
            "Render targets: native={}x{}, volumetrics={}x{}",
            native.x, native.y, size.x, size.y
        );
        let trace_size = size;
        let cloud_size = (trace_size + 1) / 2;
        let low = |format, label| texture(device, size, format, label);
        let traced = |format, label| texture(device, trace_size, format, label);
        let native_texture = |label| texture(device, native, TextureFormat::Rgba16Float, label);
        self.0 = Some(Targets {
            size,
            trace_size,
            native_size: native,
            color: texture(
                device,
                cloud_size,
                TextureFormat::Rgba16Float,
                "volume_radiance",
            ),
            motion: texture(
                device,
                cloud_size,
                TextureFormat::Rg16Float,
                "volume_motion",
            ),
            metadata: texture(
                device,
                cloud_size,
                TextureFormat::Rgba32Float,
                "volume_depth",
            ),
            history: from_fn(|_| low(TextureFormat::Rgba16Float, "volume_history")),
            history_metadata: from_fn(|_| low(TextureFormat::Rgba32Float, "volume_history_depth")),
            cloud_fresh: texture(
                device,
                cloud_size,
                TextureFormat::Rgba16Float,
                "cloud_fresh",
            ),
            cloud_fresh_front: texture(
                device,
                cloud_size,
                TextureFormat::R32Float,
                "cloud_fresh_front",
            ),
            cloud_slab: from_fn(|_| traced(TextureFormat::Rgba16Float, "cloud_slab")),
            cloud_front: from_fn(|_| traced(TextureFormat::R32Float, "cloud_front")),
            cloud_size,
            thin_color: native_texture("thin_radiance"),
            thin_transmission: native_texture("thin_transmission"),
            thin_prefix: native_texture("thin_foreground_volume"),
            valid: AtomicBool::new(false),
        });
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
