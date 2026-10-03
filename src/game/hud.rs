use crate::{
    game::camera::CameraController,
    game::flight::{Flight, Player},
    game::world::WorldData,
};
use bevy::{
    diagnostic::{Diagnostic, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    time::common_conditions::on_timer,
};
use std::time::Duration;

pub struct HudPlugin;
impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::new(3))
            .add_systems(Startup, spawn)
            .add_systems(Update, update.run_if(on_timer(Duration::from_millis(100))));
    }
}

#[derive(Component)]
struct Hud;

fn update(
    diagnostics: Res<DiagnosticsStore>,
    world: Res<WorldData>,
    camera: Single<(&Transform, Has<CameraController>), With<Camera3d>>,
    flight: Single<&Flight, With<Player>>,
    mut text: Single<&mut Text, With<Hud>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(Diagnostic::average)
        .unwrap_or(0.0);
    let altitude = camera.0.translation.z + world.camera_offset.z;
    let state = if camera.1 {
        "FREE CAMERA · P to return"
    } else if flight.tuck > 0.5 {
        "DIVING"
    } else if flight.flap_cooldown > 0.0 {
        "FLAPPING"
    } else {
        "GLIDING"
    };
    text.0 = format!(
        "NAETU   {:.0} m/s   {state}\n{altitude:.0}m altitude   {fps:.0} FPS\n\nW / S  Dive / climb    A / D  Turn    Space  Boost\nScroll  Camera distance    P  Free camera\n1–4  Scene / depth / normals / motion",
        flight.speed
    );
}

fn spawn(mut commands: Commands) {
    commands.spawn((
        Hud,
        Text::default(),
        TextFont {
            font_size: FontSize::Px(17.0),
            ..default()
        },
        TextColor(Color::srgb(0.8, 0.9, 1.0)),
        BackgroundColor(Color::srgba(0.025, 0.045, 0.08, 0.78)),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(22.0),
            left: Val::Px(24.0),
            max_width: Val::Percent(90.0),
            padding: UiRect::all(Val::Px(12.0)),
            ..default()
        },
    ));
}
