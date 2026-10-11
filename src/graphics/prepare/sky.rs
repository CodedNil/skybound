use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureView};
use bevy::render::renderer::RenderDevice;
use skybound::scene::atmosphere::SKY_SIZE;
use skybound::scene::{PLANET_RADIUS, ViewUniform};

use crate::graphics::targets::texture;

#[derive(Resource)]
pub struct SkyLookup {
    pub view: TextureView,
    pub dirty: AtomicBool,
    last: Vec3,
}
impl FromWorld for SkyLookup {
    fn from_world(world: &mut World) -> Self {
        Self {
            view: texture(
                world.resource::<RenderDevice>(),
                UVec2::splat(SKY_SIZE),
                TextureFormat::Rgba16Float,
                "sky_lookup",
            ),
            dirty: AtomicBool::new(true),
            last: Vec3::splat(f32::INFINITY),
        }
    }
}
impl SkyLookup {
    pub fn update(&mut self, view: &mut ViewUniform) {
        let altitude = view.altitude();
        let direction = view.pole_direction();
        let activity = 0.10 + 0.90 * (0.5 + 0.5 * (view.time() * TAU / 2400.0).cos());
        let horizon = -(1.0 - (PLANET_RADIUS / (PLANET_RADIUS + altitude.max(0.0))).powi(2)).sqrt();
        view.sky_horizon = vec4(horizon, activity, 0.0, 0.0);
        view.pole_color *= activity;
        view.ambient_color *= activity;
        let current = vec3(
            altitude,
            direction.dot(view.up(view.world_position.truncate())),
            activity,
        );
        let dirty = self.dirty.load(Ordering::Relaxed)
            || (current.x - self.last.x).abs() >= 0.5
            || (current.y - self.last.y).abs() >= 0.000_01
            || (current.z - self.last.z).abs() >= 0.0001;
        self.dirty.store(dirty, Ordering::Relaxed);
        if dirty {
            self.last = current;
        }
    }
}
