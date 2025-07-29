use crate::world::WorldCoordinates;
use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
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
    coords_query: Query<&WorldCoordinates, With<Camera>>,
    mut display: Single<&mut Text, With<FpsCounterText>>,
) {
    let fps_res = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.average());

    let should_update = fps_state.timer.tick(time.delta()).just_finished();
    if !should_update {
        return;
    }

    let fps = fps_res.unwrap_or(0.0);
    let (lat, lon, alt) = if let Ok(wc) = coords_query.single() {
        (wc.latitude, wc.longitude, wc.altitude)
    } else {
        (0.0, 0.0, 0.0)
    };

    display.0 = format!(
        "FPS: {:.0}\nLatitude: {:+.2}\nLongitude: {:+.2}\nAltitude: {:+.2}m",
        fps, lat, lon, alt
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
