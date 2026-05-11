use bevy::prelude::*;

use crate::show_prepass::{ShowPrepass, ShowPrepassDepthPower};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, choose_show_prepass_mode);
    }
}

fn choose_show_prepass_mode(
    mut commands: Commands,
    camera: Single<Entity, With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if keyboard.just_pressed(KeyCode::Digit1) {
        commands
            .entity(*camera)
            .remove::<ShowPrepass>()
            .remove::<ShowPrepassDepthPower>();
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        commands
            .entity(*camera)
            .insert(ShowPrepass::Depth)
            .insert(ShowPrepassDepthPower(0.5));
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        commands.entity(*camera).insert(ShowPrepass::Normals);
    } else if keyboard.just_pressed(KeyCode::Digit4) {
        commands.entity(*camera).insert(ShowPrepass::MotionVectors);
    }
}
