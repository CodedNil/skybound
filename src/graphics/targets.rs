use bevy::{
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureView, TextureViewDescriptor,
        },
        renderer::RenderDevice,
        texture::TextureCache,
    },
};

#[derive(Resource, Default)]
pub struct SceneTargets(pub Option<Targets>);

pub struct Targets {
    /// The resolution the scene renders at, below the window's under an upscaler.
    pub size: UVec2,
    /// The aur's fog, arcs and pole beams, with the clouds slotted in, and their motion: traced at
    /// every pixel, every frame.
    pub color: TextureView,
    pub motion: TextureView,
    /// The clouds' light and transmittance, and the depth of the first.
    pub cloud_slab: TextureView,
    pub cloud_front: TextureView,
    pub thin_color: TextureView,
    pub thin_transmission: TextureView,
    pub thin_prefix: TextureView,
    /// The solids' motion, before the composite lays the volumes' over it into the view's motion
    /// vectors. Drawn with the view's own targets, so at its full size.
    pub scene_motion: TextureView,
}

impl SceneTargets {
    /// Fetch this frame's targets from the texture cache, each by its own label. `size` is the
    /// resolution the scene renders at, below the view's `display` one under an upscaler.
    pub fn resize(
        &mut self,
        device: &RenderDevice,
        cache: &mut TextureCache,
        size: UVec2,
        display: UVec2,
    ) {
        if self.0.as_ref().is_none_or(|t| t.size != size) {
            info!("Render targets: {}x{}", size.x, size.y);
        }
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
        let half = TextureFormat::Rgba16Float;
        let front = TextureFormat::Rgba32Float;
        self.0 = Some(Targets {
            size,
            color: get(size, half, "volume_radiance"),
            motion: get(size, TextureFormat::Rg16Float, "volume_motion"),
            cloud_slab: get(size, half, "cloud_slab"),
            cloud_front: get(size, front, "cloud_front"),
            thin_color: get(size, half, "thin_radiance"),
            thin_transmission: get(size, half, "thin_transmission"),
            thin_prefix: get(size, half, "thin_foreground_volume"),
            scene_motion: get(display, TextureFormat::Rg16Float, "scene_motion"),
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
