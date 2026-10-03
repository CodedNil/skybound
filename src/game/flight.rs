use crate::game::{camera::CameraController, world::WorldAnchor};
use bevy::{
    camera::{
        primitives::{Frustum, Sphere},
        visibility::VisibilitySystems,
    },
    input::{InputSystems, mouse::MouseWheel},
    prelude::*,
    render::{RenderApp, extract_resource::ExtractResource},
    transform::TransformSystems,
};
use std::f32::consts::{FRAC_PI_2, TAU};

use skybound::scene::life::{
    CreatureInstance, Kind, tetralata::viviparia::ferrilaminata::naeturidae::naeturae::naetu,
};

pub const SPAWN: Vec3 = vec3(0.0, 4.0, 2400.0);

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct PlayerModel(pub CreatureInstance);

const FLAP_INTERVAL: f32 = 0.7;
const CRUISE_SPEED: f32 = 48.0;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Flight {
    pub speed: f32,
    pub flap_cooldown: f32,
    pub tuck: f32,
    yaw: f32,
    pitch: f32,
    bank: f32,
    animation_time: f32,
    camera_distance: f32,
}

impl Default for Flight {
    fn default() -> Self {
        Self {
            speed: CRUISE_SPEED,
            flap_cooldown: 0.0,
            tuck: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            bank: 0.0,
            animation_time: 0.0,
            camera_distance: 52.0,
        }
    }
}

impl Flight {
    fn steer(&mut self, intent: Vec2, boost: bool, dt: f32) -> Vec3 {
        let response = 1.0 - (-8.0 * dt).exp();
        let pitch_response = 1.0 - (-(if intent.y.abs() < 0.001 { 4.0 } else { 8.0 }) * dt).exp();
        self.pitch += (intent.y * 0.95 - self.pitch) * pitch_response;
        let bank = intent.x * 0.75;
        let turning = bank * dt + (self.bank - bank) * response / 8.0;
        self.bank += (bank - self.bank) * response;
        self.yaw = (self.yaw + turning * 1.8).rem_euclid(TAU);
        let tuck = ((-self.pitch - 0.25) / 0.6).clamp(0.0, 1.0);
        self.tuck += (tuck - self.tuck) * (1.0 - (-6.0 * dt).exp());
        let target_speed = CRUISE_SPEED - self.pitch.sin() * 32.0 + if boost { 35.0 } else { 0.0 };
        self.speed += (target_speed.max(32.0) - self.speed) * (1.0 - (-1.5 * dt).exp());
        self.flap_cooldown = (self.flap_cooldown - dt).max(0.0);
        if self.flap_cooldown <= 0.0
            && self.tuck < 0.5
            && (boost || self.pitch > 0.12 || self.speed < 42.0)
        {
            self.flap_cooldown = FLAP_INTERVAL;
        }
        self.animation_time += dt;
        let heading = Quat::from_rotation_z(self.yaw) * Quat::from_rotation_x(self.pitch);
        heading * Vec3::Y * self.speed
    }

    fn advance(&mut self, intent: Vec2, boost: bool, dt: f32) -> Vec3 {
        let steps = (dt * 120.0).ceil() as u32;
        let step = dt / steps.max(1) as f32;
        let mut displacement = Vec3::ZERO;
        for _ in 0..steps {
            displacement += self.steer(intent, boost, step) * step;
        }
        displacement
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_rotation_z(self.yaw)
            * Quat::from_rotation_x(self.pitch + FRAC_PI_2)
            * Quat::from_rotation_z(self.bank)
    }
}

pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerModel>()
            .add_systems(Startup, spawn_player)
            .add_systems(PreUpdate, fly.after(InputSystems))
            .add_systems(
                PostUpdate,
                follow_camera.before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                extract_player
                    .after(TransformSystems::Propagate)
                    .after(VisibilitySystems::UpdateFrusta),
            );
    }
}

fn spawn_player(mut commands: Commands) {
    let flight = Flight::default();
    commands.spawn((
        Player,
        WorldAnchor,
        Transform::from_translation(SPAWN).with_rotation(flight.rotation()),
        flight,
    ));
}

fn extract_player(
    mut model: ResMut<PlayerModel>,
    player: Single<(&Transform, &Flight), With<Player>>,
    camera: Single<&Frustum, With<Camera3d>>,
    mut previous: Local<Option<Transform>>,
) {
    let (transform, flight) = *player;
    let old = previous.replace(*transform).unwrap_or(*transform);
    let radius = naetu::TAXON.radius;
    if !camera.intersects_sphere(
        &Sphere {
            center: transform.translation.into(),
            radius,
        },
        false,
    ) {
        model.0 = CreatureInstance::default();
        return;
    }
    let stroke = if flight.flap_cooldown > 0.0 {
        ((1.0 - flight.flap_cooldown / FLAP_INTERVAL) * TAU).sin() * 0.65
    } else {
        (flight.animation_time * 1.8).sin() * 0.06
    };
    model.0 = CreatureInstance::new(
        transform.translation.extend(radius),
        transform.rotation.to_array().into(),
        Vec4::new(stroke, flight.tuck, flight.animation_time, flight.speed),
        Kind::Naetu as u32,
    );
    model.0.previous_position = old.translation.extend(radius);
    model.0.previous_rotation = old.rotation.to_array().into();
}

fn fly(
    time: Res<Time>,
    mut naetu: Single<(&mut Transform, &mut Flight), With<Player>>,
    camera: Single<Has<CameraController>, With<Camera3d>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
) {
    let (transform, flight) = &mut *naetu;
    let scroll: f32 = wheel.read().map(|event| event.y).sum();
    if *camera {
        return;
    }
    flight.camera_distance = (flight.camera_distance - scroll * 4.0).clamp(32.0, 95.0);
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }
    let axis = |a, b| f32::from(keys.pressed(a)) - f32::from(keys.pressed(b));
    let intent = vec2(
        axis(KeyCode::KeyA, KeyCode::KeyD),
        axis(KeyCode::KeyS, KeyCode::KeyW),
    );
    transform.translation += flight.advance(intent, keys.pressed(KeyCode::Space), dt);
    transform.rotation = flight.rotation();
}

type FollowingCamera = (With<Camera3d>, Without<Player>, Without<CameraController>);

fn follow_camera(
    time: Res<Time>,
    naetu: Single<(&Transform, &Flight), With<Player>>,
    mut camera: Single<&mut Transform, FollowingCamera>,
) {
    let (body, flight) = *naetu;
    let forward = *body.forward();
    // A stable horizon lets the creature bank without rolling the whole screen.
    let target = body.translation - forward * (flight.camera_distance + flight.speed * 0.06)
        + Vec3::Z * 13.0;
    let lag = 1.0 - (-8.0 * time.delta_secs()).exp();
    camera.translation = camera.translation.lerp(target, lag);
    let aim = Transform::from_translation(camera.translation)
        .looking_at(body.translation + forward * 12.0 + Vec3::Z * 2.0, Vec3::Z);
    camera.rotation = camera.rotation.slerp(aim.rotation, lag);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate(intent: Vec2, boost: bool, hz: u32) -> Flight {
        let mut flight = Flight::default();
        for _ in 0..hz * 5 {
            flight.steer(intent, boost, 1.0 / hz as f32);
        }
        flight
    }

    #[test]
    fn instinctive_flight_and_smooth_release() {
        let dive = simulate(Vec2::NEG_Y, false, 120);
        assert!(
            dive.speed > CRUISE_SPEED && dive.tuck > 0.9,
            "dive should gain speed and fold wings"
        );
        let mut climb = simulate(Vec2::Y, false, 120);
        assert!(
            climb.flap_cooldown > 0.0 && climb.speed > 28.0,
            "climb should flap automatically without stalling"
        );
        let pitch = climb.pitch;
        climb.steer(Vec2::ZERO, false, 1.0 / 120.0);
        assert!(climb.pitch > pitch * 0.95, "release must not snap to level");
        for _ in 0..600 {
            climb.steer(Vec2::ZERO, false, 1.0 / 120.0);
        }
        assert!(climb.pitch.abs() < 0.01, "release should gently level out");
        assert!(
            simulate(Vec2::ZERO, true, 120).speed > CRUISE_SPEED + 20.0,
            "optional boost should add speed"
        );
    }

    #[test]
    fn frame_rate_independence_and_smooth_thrust() {
        let a = simulate(vec2(0.5, -0.5), false, 30);
        let b = simulate(vec2(0.5, -0.5), false, 144);
        assert!(
            (a.speed - b.speed).abs() < 0.3 && (a.yaw - b.yaw).abs() < 0.03,
            "flight depends on frame rate"
        );
        let mut flight = Flight::default();
        let speed = flight.speed;
        flight.steer(Vec2::ZERO, true, 1.0 / 120.0);
        assert!(
            (flight.speed - speed).abs() < 0.5,
            "boost must accelerate smoothly"
        );
        for _ in 0..120 {
            flight.steer(Vec2::ZERO, true, 1.0 / 120.0);
        }
        let speed = flight.speed;
        flight.steer(Vec2::ZERO, false, 1.0 / 120.0);
        assert!(
            (flight.speed - speed).abs() < 0.5,
            "boost release must decelerate smoothly"
        );
    }

    #[test]
    fn slow_frames_preserve_flight_distance_and_steering() {
        let simulate = |hz: u32| {
            let mut flight = Flight::default();
            let mut position = Vec3::ZERO;
            for _ in 0..hz * 5 {
                position += flight.advance(vec2(0.5, -0.5), true, 1.0 / hz as f32);
            }
            (flight, position)
        };
        let (reference, position) = simulate(144);
        for hz in [8, 30, 60] {
            let (flight, actual) = simulate(hz);
            assert!(
                actual.distance(position) < 0.5,
                "flight distance depends on frame rate"
            );
            assert!(
                (flight.yaw - reference.yaw).abs() < 0.001,
                "steering depends on frame rate"
            );
        }
    }

    #[test]
    fn steering_reverses_promptly_and_gliding_is_level() {
        let mut flight = simulate(Vec2::X, false, 120);
        for _ in 0..30 {
            flight.steer(Vec2::NEG_X, false, 1.0 / 120.0);
        }
        assert!(flight.bank < -0.5, "turn reversal takes too long");
        let velocity = flight.steer(Vec2::ZERO, false, 1.0 / 120.0);
        assert!(velocity.z.abs() < 0.001, "neutral flight must stay level");
        assert!(
            flight.rotation().is_finite(),
            "flight rotation must stay finite"
        );
    }
}
