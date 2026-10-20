use bevy::{
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureView, TextureViewDescriptor,
        },
        renderer::RenderDevice,
        texture::TextureCache,
    },
};
use std::{
    env,
    sync::atomic::{AtomicBool, Ordering},
};

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
    /// Fetch this frame's targets from the texture cache. Each has its own
    /// label, so it is the same texture every frame while the size holds;
    /// history stays valid only for as long as that does.
    pub fn resize(
        &mut self,
        device: &RenderDevice,
        cache: &mut TextureCache,
        native: UVec2,
        scale: f32,
    ) {
        let size = (native.as_vec2() * scale).floor().max(Vec2::ONE).as_uvec2();
        let kept = self
            .0
            .as_ref()
            .is_some_and(|t| t.size == size && t.native_size == native);
        if !kept {
            info!(
                "Render targets: native={}x{}, volumetrics={}x{}",
                native.x, native.y, size.x, size.y
            );
        }
        let valid = kept
            && self
                .0
                .as_ref()
                .is_some_and(|t| t.valid.load(Ordering::Relaxed));
        let cloud_size = (size + 1) / 2;
        let mut get = |size: UVec2, format, label| {
            cache
                .get(
                    device,
                    TextureDescriptor {
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
                    },
                )
                .default_view
        };
        let (half, full) = (TextureFormat::Rgba16Float, TextureFormat::Rgba32Float);
        let front = TextureFormat::R32Float;
        self.0 = Some(Targets {
            size,
            trace_size: size,
            native_size: native,
            color: get(cloud_size, half, "volume_radiance"),
            motion: get(cloud_size, TextureFormat::Rg16Float, "volume_motion"),
            metadata: get(cloud_size, full, "volume_depth"),
            history: [
                get(size, half, "volume_history_0"),
                get(size, half, "volume_history_1"),
            ],
            history_metadata: [
                get(size, full, "volume_history_depth_0"),
                get(size, full, "volume_history_depth_1"),
            ],
            cloud_fresh: get(cloud_size, half, "cloud_fresh"),
            cloud_fresh_front: get(cloud_size, front, "cloud_fresh_front"),
            cloud_slab: [
                get(size, half, "cloud_slab_0"),
                get(size, half, "cloud_slab_1"),
            ],
            cloud_front: [
                get(size, front, "cloud_front_0"),
                get(size, front, "cloud_front_1"),
            ],
            cloud_size,
            thin_color: get(native, half, "thin_radiance"),
            thin_transmission: get(native, half, "thin_transmission"),
            thin_prefix: get(native, half, "thin_foreground_volume"),
            valid: AtomicBool::new(valid),
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
