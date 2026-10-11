pub mod camera;
pub mod flight;
mod hud;
pub mod world;

use bevy::prelude::*;

pub struct GamePlugin;
impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            world::WorldPlugin,
            flight::PlayerPlugin,
            camera::CameraPlugin,
            hud::HudPlugin,
        ));
    }
}
