use crate::world::WorldCoordinates;
use bevy::{
    diagnostic::{Diagnostic, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use std::time::Duration;

const UPDATE_INTERVAL: Duration = Duration::from_millis(100);

pub struct DebugTextPlugin;

impl Plugin for DebugTextPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::new(3))
            .add_systems(Startup, spawn_text)
            .add_systems(Update, update)
            .init_resource::<FpsCounter>();
    }
}

#[derive(Resource)]
struct FpsCounter {
    timer: Timer,
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self {
            timer: Timer::new(UPDATE_INTERVAL, TimerMode::Repeating),
        }
    }
}

#[derive(Component)]
struct FpsCounterText;

fn update(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut fps_state: ResMut<FpsCounter>,
    world_coords: Res<WorldCoordinates>,
    camera_query: Query<&Transform, With<Camera>>,
    mut display: Single<&mut Text, With<FpsCounterText>>,
) {
    let fps_res = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(Diagnostic::average);

    let should_update = fps_state.timer.tick(time.delta()).just_finished();
    if !should_update {
        return;
    }

    let fps = fps_res.unwrap_or(0.0);

    let (lat_deg, lon_deg, alt) = camera_query
        .single()
        .map(|camera_transform| {
            (
                world_coords
                    .latitude(camera_transform.translation)
                    .to_degrees(),
                world_coords
                    .longitude(camera_transform.translation)
                    .to_degrees(),
                camera_transform.translation.y,
            )
        })
        .unwrap_or((0.0, 0.0, 0.0));

    display.0 = format!(
        "FPS: {fps:.0}\n\
         Lat: {lat_deg:+.2}\n\
         Lon: {lon_deg:+.2}\n\
         Alt: {alt:+.2}m",
    );
}

fn spawn_text(mut commands: Commands) {
    commands.spawn((
        Text::default(),
        TextFont::default(),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
        FpsCounterText,
    ));
}
