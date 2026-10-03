use bevy::{
    prelude::*,
    render::{
        render_resource::{
            Extent3d, TexelCopyBufferLayout, Texture, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};
use half::f16;
use rayon::prelude::*;
use skybound::scene::{
    PLANET_RADIUS, ViewUniform,
    atmosphere::{SKY_SIZE, render_sky, sky_direction},
};

#[derive(Resource)]
pub struct SkyLookup {
    texture: Texture,
    pub view: TextureView,
    last: Vec2,
}

impl FromWorld for SkyLookup {
    fn from_world(world: &mut World) -> Self {
        let texture = world
            .resource::<RenderDevice>()
            .create_texture(&TextureDescriptor {
                label: Some("sky lookup"),
                size: Extent3d {
                    width: SKY_SIZE,
                    height: SKY_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            last: Vec2::splat(f32::INFINITY),
        }
    }
}

impl SkyLookup {
    pub fn update(&mut self, view: &mut ViewUniform, queue: &RenderQueue) {
        let altitude = view.ro_relative().z - PLANET_RADIUS;
        let sun = (view.sun_position.xyz() - view.world_position.xyz()).normalize();
        let horizon = -(1.0 - (PLANET_RADIUS / (PLANET_RADIUS + altitude.max(0.0))).powi(2)).sqrt();
        view.sky_horizon = Vec4::splat(horizon);
        if (altitude - self.last.x).abs() < 50.0 && (sun.z - self.last.y).abs() < 0.0005 {
            return;
        }
        self.last = vec2(altitude, sun.z);
        let ro = view.ro_relative();
        let sun = vec3((1.0 - sun.z * sun.z).max(0.0).sqrt(), 0.0, sun.z);
        let data: Vec<[u16; 4]> = (0..SKY_SIZE * SKY_SIZE)
            .into_par_iter()
            .map(|i| {
                let uv = vec2((i % SKY_SIZE) as f32, (i / SKY_SIZE) as f32) / (SKY_SIZE - 1) as f32;
                let color = render_sky(sky_direction(uv, horizon), ro, sun);
                [
                    f16::from_f32(color.x).to_bits(),
                    f16::from_f32(color.y).to_bits(),
                    f16::from_f32(color.z).to_bits(),
                    0,
                ]
            })
            .collect();
        queue.write_texture(
            self.texture.as_image_copy(),
            bytemuck::cast_slice(&data),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SKY_SIZE * 8),
                rows_per_image: Some(SKY_SIZE),
            },
            self.texture.size(),
        );
    }
}
