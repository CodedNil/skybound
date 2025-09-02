use crate::world::WorldData;
use bevy::{
    diagnostic::{Diagnostic, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use std::time::Duration;

const UPDATE_INTERVAL: Duration = Duration::from_millis(100);

/// Plugin that displays FPS and camera world coordinates on-screen.
pub struct DebugTextPlugin;

impl Plugin for DebugTextPlugin {
    /// Register debug text systems and diagnostics plugin.
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
    /// Create the default fps update timer resource.
    fn default() -> Self {
        Self {
            timer: Timer::new(UPDATE_INTERVAL, TimerMode::Repeating),
        }
    }
}

#[derive(Component)]
struct FpsCounterText;

/// Periodically updates the on-screen debug text (FPS and camera coords).
fn update(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    mut fps_state: ResMut<FpsCounter>,
    world_coords: Res<WorldData>,
    camera_query: Query<&Transform, With<Camera>>,
    mut display: Single<&mut Text, With<FpsCounterText>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(Diagnostic::average)
        .unwrap_or(0.0);

    if !fps_state.timer.tick(time.delta()).just_finished() {
        return;
    }

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
                camera_transform.translation.z,
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

/// Spawns the UI text node used for displaying debug information.
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
