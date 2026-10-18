use bevy::prelude::*;

pub mod camera;
pub mod ecosystem;
pub mod flight;
mod hud;
pub mod weather;
pub mod world;

pub struct GamePlugin;
impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            world::WorldPlugin,
            weather::WeatherPlugin,
            flight::PlayerPlugin,
            camera::CameraPlugin,
            hud::HudPlugin,
            ecosystem::EcosystemPlugin,
        ));
    }
}
