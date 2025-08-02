use crate::world::{CameraCoordinates, WorldCoordinates};
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
    world_coords: Res<WorldCoordinates>,
    camera_query: Query<(&Transform, &CameraCoordinates), With<Camera>>,
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

    let (lat_deg, lon_deg, lat_m, lon_m, alt) = camera_query
        .single()
        .map(|(transform, camera_coords)| {
            let planet_rotation = camera_coords.planet_rotation(&world_coords, transform);
            let latitude_rad = camera_coords.latitude(planet_rotation, transform);
            let longitude_rad = camera_coords.longitude(planet_rotation, transform);
            (
                latitude_rad.to_degrees(),
                longitude_rad.to_degrees(),
                camera_coords.latitude_meters(latitude_rad),
                camera_coords.longitude_meters(latitude_rad, longitude_rad),
                camera_coords.altitude(transform),
            )
        })
        .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0));

    display.0 = format!(
        "FPS:  {fps:.0}\n\
         Lat:  {lat_deg:+.2}° ({lat_m:+.0} m)\n\
         Lon:  {lon_deg:+.2}° ({lon_m:+.0} m)\n\
         Alt:  {alt:+.2} m",
        fps = fps,
        lat_deg = lat_deg,
        lat_m = lat_m,
        lon_deg = lon_deg,
        lon_m = lon_m,
        alt = alt,
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
