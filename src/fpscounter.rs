use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use std::time::Duration;

const FONT_SIZE: f32 = 32.;
const FONT_COLOR: Color = Color::WHITE;
const UPDATE_INTERVAL: Duration = Duration::from_millis(100);

const STRING_FORMAT: &str = "FPS: ";
const STRING_INITIAL: &str = "FPS: ...";
const STRING_MISSING: &str = "FPS: ???";

pub struct FpsCounterPlugin;

impl Plugin for FpsCounterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, spawn_text)
            .add_systems(Update, update)
            .init_resource::<FpsCounter>();
    }
}

#[derive(Resource)]
struct FpsCounter {
    timer: Timer,
    update_now: bool,
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self {
            timer: Timer::new(UPDATE_INTERVAL, TimerMode::Repeating),
            update_now: true,
        }
    }
}

#[derive(Component)]
struct FpsCounterText;

fn update(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    state_resources: Option<ResMut<FpsCounter>>,
    query: Query<Entity, With<FpsCounterText>>,
    mut writer: TextUiWriter,
) {
    let Some(mut state) = state_resources else {
        return;
    };
    if !(state.update_now || state.timer.tick(time.delta()).just_finished()) {
        return;
    }
    if state.timer.paused() {
        for entity in query {
            writer.text(entity, 0).clear();
        }
    } else {
        let fps_dialog: Option<f64> = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(bevy::diagnostic::Diagnostic::average);

        for entity in query {
            if let Some(fps) = fps_dialog {
                *writer.text(entity, 0) = format!("{STRING_FORMAT}{fps:.0}");
            } else {
                *writer.text(entity, 0) = STRING_MISSING.to_string();
            }
        }
    }
}

fn spawn_text(mut commands: Commands) {
    commands
        .spawn((
            Text::new(STRING_INITIAL),
            TextFont {
                font_size: FONT_SIZE,
                ..Default::default()
            },
            TextColor(FONT_COLOR),
        ))
        .insert(FpsCounterText);
}
