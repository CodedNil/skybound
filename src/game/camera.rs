use bevy::render::{RenderApp, extract_resource::ExtractResource};
use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    input::{
        InputSystems,
        mouse::{AccumulatedMouseMotion, MouseWheel},
    },
    prelude::*,
    window::WindowResized,
};

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct DebugView(pub u32);

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugView>()
            .add_systems(PreUpdate, camera_controller.after(InputSystems))
            .add_systems(Update, choose_show_prepass_mode)
            .add_systems(Update, toggle_freecam)
            .add_systems(Update, reset_history_on_resize);
    }
}

#[derive(Component)]
pub struct CameraController {
    pub speed: f32,
    pub sensitivity: f32,
}

fn camera_controller(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut CameraController), With<Camera>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mut mouse_wheel_events: MessageReader<MouseWheel>,
) {
    for (mut transform, controller) in &mut query {
        let movement = movement(&keyboard_input, &transform);
        transform.translation += movement * controller.speed * time.delta_secs();

        if mouse_button_input.pressed(MouseButton::Right) {
            let delta = mouse_motion.delta;
            if delta != Vec2::ZERO {
                transform.rotate_z(-delta.x * controller.sensitivity);
                let pitch_delta = -delta.y * controller.sensitivity;
                let current_pitch_sin = transform.forward().z;
                if (pitch_delta > 0.0 && current_pitch_sin < 0.99)
                    || (pitch_delta < 0.0 && current_pitch_sin > -0.99)
                {
                    transform.rotate_local_x(pitch_delta);
                }
                let final_forward = *transform.forward();
                transform.look_to(final_forward, Vec3::Z);
            }
        }
    }
    for event in mouse_wheel_events.read() {
        for (_, mut controller) in &mut query {
            controller.speed = (controller.speed * (1.0 + event.y * 0.5)).clamp(0.1, 5000.0);
        }
    }
}

fn choose_show_prepass_mode(
    mut mode: ResMut<DebugView>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut taa: Single<&mut TemporalAntiAliasing>,
) {
    if let Some(index) = [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
    ]
    .iter()
    .position(|&key| keyboard.just_pressed(key))
    {
        mode.0 = index as u32;
        taa.reset = true;
    }
}

fn toggle_freecam(
    mut commands: Commands,
    camera: Single<(Entity, Has<CameraController>), With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }

    let (entity, has_controller) = *camera;
    if has_controller {
        commands.entity(entity).remove::<CameraController>();
    } else {
        commands.entity(entity).insert(CameraController {
            speed: 40.0,
            sensitivity: 0.005,
        });
    }
}

pub fn movement(keys: &ButtonInput<KeyCode>, transform: &Transform) -> Vec3 {
    let axis =
        |positive, negative| f32::from(keys.pressed(positive)) - f32::from(keys.pressed(negative));
    let direction = *transform.forward() * axis(KeyCode::KeyW, KeyCode::KeyS)
        + *transform.right() * axis(KeyCode::KeyD, KeyCode::KeyA)
        + *transform.up() * axis(KeyCode::KeyE, KeyCode::KeyQ);
    direction.normalize_or_zero()
        * if keys.pressed(KeyCode::ShiftLeft) {
            10.0
        } else {
            1.0
        }
}

fn reset_history_on_resize(
    mut events: MessageReader<WindowResized>,
    mut taa: Single<&mut TemporalAntiAliasing>,
) {
    if events.read().next().is_some() {
        taa.reset = true;
    }
}
