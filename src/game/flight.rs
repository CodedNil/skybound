use bevy::ecs::system::SystemParam;
use bevy::input::InputSystems;
use bevy::input::mouse::{AccumulatedMouseMotion, MouseWheel};
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::transform::TransformSystems;
use skybound::scene::aur;
use skybound::scene::life::animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::{
    nae, naetu,
};
use skybound::scene::life::behaviour::{Motion, Senses};
use skybound::scene::life::surface::{Ray, estimate_normal};
use skybound::scene::life::{CreatureInstance, Kind, development};

use crate::game::camera::CameraController;
use crate::game::ecosystem::{Individual, Physiology};
use crate::game::world::{WorldAnchor, WorldData};
use crate::math::{ToEngine, ToScene};

pub const SPAWN: Vec3 = vec3(0.0, 4.0, 2400.0);

const ROSETTE: Vec3 = vec3(0.0, 5.8, 2.0);

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct PlayerModel(pub [CreatureInstance; 2]);

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Mount;

#[derive(Component, Default)]
pub struct Velocity(pub Vec3);

#[derive(Clone, Copy)]
pub struct PerchAnchor {
    point: DVec3,
    normal: Vec3,
    on_mount: bool,
}

#[derive(Clone, Copy, Default)]
pub enum ControlMode {
    Flying,
    Perched(PerchAnchor),
    #[default]
    Mounted,
}

#[derive(Resource)]
pub struct Flight {
    pub mode: ControlMode,
    pub speed: f32,
    orbit: Vec2,
    steering: Vec2,
    zoom: f32,
}

impl Default for Flight {
    fn default() -> Self {
        Self {
            mode: ControlMode::Mounted,
            speed: 0.0,
            orbit: Vec2::ZERO,
            steering: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerModel>()
            .init_resource::<Flight>()
            .add_systems(Startup, spawn_player)
            .add_systems(
                PreUpdate,
                (fly, advance_companion).chain().after(InputSystems),
            )
            .add_systems(
                PostUpdate,
                follow_camera.before(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                extract_player.after(TransformSystems::Propagate),
            );
    }
}

type NaeQuery = (With<Player>, Without<Mount>, Without<Camera3d>);

type NaetuQuery = (With<Mount>, Without<Player>, Without<Camera3d>);

type BodyView<'a> = (
    &'a Transform,
    &'a Velocity,
    Option<&'a Physiology>,
    Option<&'a Individual>,
);

type BodyMotion<'a> = (
    &'a mut Transform,
    &'a mut Velocity,
    Option<&'a Physiology>,
    Option<&'a Individual>,
);

type CompanionMotion<'a> = (
    &'a mut Transform,
    &'a mut Velocity,
    Option<&'a super::ecosystem::Relations>,
    Option<&'a Physiology>,
    Option<&'a Individual>,
);

type FollowCameraQuery = (
    With<Camera3d>,
    Without<Player>,
    Without<Mount>,
    Without<CameraController>,
);

type CameraQuery = (With<Camera3d>, Without<Player>, Without<Mount>);

#[derive(SystemParam)]
struct FlightInput<'w, 's> {
    time: Res<'w, Time>,
    keys: Res<'w, ButtonInput<KeyCode>>,
    mouse: Res<'w, ButtonInput<MouseButton>>,
    motion: Res<'w, AccumulatedMouseMotion>,
    wheel: MessageReader<'w, 's, MouseWheel>,
    world: Res<'w, WorldData>,
}

struct Steering<'a> {
    mount_velocity: Vec3,
    forward: Vec3,
    horizontal: Vec3,
    dt: f32,
    time: f32,
    physiology: Option<&'a Physiology>,
    individual: Option<&'a Individual>,
}

fn spawn_player(mut commands: Commands) {
    let body = Transform::from_translation(SPAWN).looking_to(Vec3::Y, Vec3::Z);
    commands.spawn((Mount, Velocity::default(), WorldAnchor, body));
    commands.spawn((
        Player,
        Velocity::default(),
        WorldAnchor,
        Transform::from_translation(body.transform_point(ROSETTE)).with_rotation(body.rotation),
    ));
}

fn instance(
    transform: &Transform,
    kind: Kind,
    time: f32,
    speed: f32,
    world: &WorldData,
    physiology: Option<&Physiology>,
    individual: Option<&Individual>,
) -> CreatureInstance {
    let radius = if kind == Kind::Nae {
        nae::RENDER_RADIUS
    } else {
        naetu::RENDER_RADIUS
    };
    let dive = (-transform.forward().dot(world.up(transform.translation))).clamp(0.0, 1.0);
    let stroke = (time * 4.0).sin() * (0.06 + (speed / 100.0).min(1.0) * 0.28) * (1.0 - dive);
    let fold = dive * 1.1;
    CreatureInstance::specimen(
        transform.translation.extend(radius).to_scene(),
        transform.rotation.to_array().into(),
        Vec4::new(stroke, fold, time, speed).to_scene(),
        kind as u32,
        physiology.map_or_else(
            || kind.biology().maturity_days * 1.25,
            |body| body.0.age_days as f32,
        ),
        individual.map_or(kind as u32 + 1, |identity| identity.0),
    )
}

fn extract_player(
    world: Res<WorldData>,
    mut model: ResMut<PlayerModel>,
    player: Single<BodyView, With<Player>>,
    mount: Single<BodyView, With<Mount>>,
    time: Res<Time>,
    flight: Res<Flight>,
) {
    let poses = [
        instance(
            mount.0,
            Kind::Naetu,
            time.elapsed_secs(),
            mount.1.0.length(),
            &world,
            mount.2,
            mount.3,
        ),
        instance(
            player.0,
            Kind::Nae,
            time.elapsed_secs(),
            player.1.0.length(),
            &world,
            player.2,
            player.3,
        ),
    ];
    for (index, mut pose) in poses.into_iter().enumerate() {
        if index == 1 && !matches!(flight.mode, ControlMode::Flying) {
            for (knot, point) in pose.tail.iter_mut().zip([
                vec3(0.0, -0.95, 0.12),
                vec3(0.0, -1.8, 0.9),
                vec3(0.0, -2.6, 0.8),
                vec3(0.0, -2.7, 0.5),
                vec3(0.0, -2.5, 0.2),
                vec3(0.0, -2.3, 0.3),
            ]) {
                *knot = point.extend(0.0).to_scene();
            }
        }
        let old = model.0[index];
        if old.position.w > 0.0
            && old.position.truncate().distance(pose.position.truncate()) < 100.0
        {
            pose.previous_position = old.position;
            pose.previous_rotation = old.rotation;
            pose.previous_animation = old.animation;
            pose.previous_proportions = old.proportions;
            pose.previous_tail = old.tail;
        }
        model.0[index] = pose;
    }
}

fn advance(velocity: &mut Vec3, target: Vec3, response: f32, dt: f32) -> Vec3 {
    let decay = (-response * dt).exp();
    let displacement = target * dt + (*velocity - target) * ((1.0 - decay) / response);
    *velocity = target + (*velocity - target) * decay;
    displacement
}

fn pilot(
    body: &mut Transform,
    velocity: &mut Vec3,
    steering: Vec2,
    cruise: f32,
    dt: f32,
    up: Vec3,
) {
    let tangent = Quat::from_rotation_arc(Vec3::Z, up);
    let forward = tangent.conjugate() * *body.forward();
    let yaw = forward.y.atan2(forward.x) - steering.x * 1.1 * dt;
    let pitch = (forward.z.asin() + steering.y * 0.9 * dt).clamp(-1.25, 1.25);
    let heading = tangent
        * vec3(
            yaw.cos() * pitch.cos(),
            yaw.sin() * pitch.cos(),
            pitch.sin(),
        );
    let level = Transform::default().looking_to(heading, up);
    let bank = -body
        .up()
        .dot(*level.right())
        .atan2(body.up().dot(*level.up()));
    let bank = bank.lerp(-steering.x * 0.65, 1.0 - (-6.0 * dt).exp());
    body.rotation = level.rotation * Quat::from_rotation_z(bank);
    let speed = cruise * (1.0 + (-heading.dot(up)).max(0.0) * 0.7);
    body.translation += advance(velocity, heading * speed, 3.5, dt);
}

fn contact(
    position: Vec3,
    mount: &Transform,
    c: &CreatureInstance,
    world: &WorldData,
) -> Option<PerchAnchor> {
    let mut nearest = 1.2f32;
    let mut found = None;
    for direction in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
        let ray = Ray {
            origin: position.to_scene(),
            direction: direction.to_scene(),
            light: Vec3::Z.to_scene(),
            max_distance: nearest,
            footprint: 0.001,
            ambient: vec3(0.30, 0.36, 0.45).to_scene(),
            radiance: Vec3::ONE.to_scene(),
        };
        let result = naetu::render(ray, c);
        let mut hit = result.opaque;
        for layer in result.layers.iter().take(result.count as usize) {
            if layer.part != 2 && layer.distance() < hit.distance() {
                hit = *layer;
            }
        }
        if hit.is_hit() && hit.distance() < nearest {
            nearest = hit.distance();
            let point = position + direction * nearest + hit.normal.to_engine() * 0.62;
            found = Some(PerchAnchor {
                point: (mount.rotation.inverse() * (point - mount.translation)).as_dvec3(),
                normal: mount.rotation.inverse() * hit.normal.to_engine(),
                on_mount: true,
            });
        }
    }
    let terrain_point = world.surface_position(position);
    let distance = aur::spire_field(terrain_point.to_scene());
    if distance.abs() < nearest {
        let normal = estimate_normal(position.to_scene(), |p| {
            aur::spire_field(world.surface_position(p.to_engine()).to_scene())
        })
        .to_engine();
        found = Some(PerchAnchor {
            point: world.position(position + normal * (0.62 - distance)),
            normal: world.basis.as_quat() * normal,
            on_mount: false,
        });
    }
    found
}

fn fly(
    input: FlightInput,
    player: Single<(&mut Transform, &mut Velocity), NaeQuery>,
    mount: Single<BodyMotion, NaetuQuery>,
    camera: Single<(&Transform, Has<CameraController>), CameraQuery>,
    mut flight: ResMut<Flight>,
) {
    let (mut player, mut player_velocity) = player.into_inner();
    let (mut mount, mut mount_velocity, physiology, individual) = mount.into_inner();
    let FlightInput {
        time,
        keys,
        mouse,
        motion,
        mut wheel,
        world,
    } = input;
    let scroll: f32 = wheel.read().map(|e| e.y).sum();
    flight.zoom = (flight.zoom * (-scroll * 0.12).exp()).clamp(0.45, 2.5);
    if camera.1 {
        return;
    }
    if mouse.pressed(MouseButton::Right) {
        flight.orbit += motion.delta * vec2(-0.004, 0.004);
        flight.orbit.y = flight.orbit.y.clamp(-0.5, 1.1);
    }
    let dt = time.delta_secs().min(0.1);
    if dt <= 0.0 {
        return;
    }
    let axis = |a, b| f32::from(keys.pressed(a)) - f32::from(keys.pressed(b));
    let up = world.up(player.translation);
    let forward = (*camera.0.forward() - up * camera.0.forward().dot(up)).normalize_or_zero();
    let right = forward.cross(up);
    let horizontal = (forward * axis(KeyCode::KeyW, KeyCode::KeyS)
        + right * axis(KeyCode::KeyD, KeyCode::KeyA))
    .normalize_or_zero();
    let pitch = axis(KeyCode::KeyS, KeyCode::KeyW) + axis(KeyCode::Space, KeyCode::ControlLeft);
    let steering = vec2(axis(KeyCode::KeyD, KeyCode::KeyA), pitch.clamp(-1.0, 1.0));
    flight.steering = flight.steering.lerp(steering, 1.0 - (-8.0 * dt).exp());
    let steering = flight.steering;
    let boost = 1.0 + f32::from(keys.pressed(KeyCode::ShiftLeft)) * 0.65;
    let c = instance(
        &mount,
        Kind::Naetu,
        time.elapsed_secs(),
        mount_velocity.0.length(),
        &world,
        physiology,
        individual,
    );
    let rosette = ROSETTE * development::scale(&c).to_engine();
    if keys.just_pressed(KeyCode::Space) && matches!(flight.mode, ControlMode::Perched(_)) {
        flight.mode = ControlMode::Flying;
        player_velocity.0 = mount_velocity.0 + up * 3.0;
    }
    if keys.just_pressed(KeyCode::KeyE) {
        if matches!(flight.mode, ControlMode::Mounted | ControlMode::Perched(_)) {
            flight.mode = ControlMode::Flying;
        } else if matches!(flight.mode, ControlMode::Flying)
            && let Some(anchor) = contact(player.translation, &mount, &c, &world)
        {
            flight.mode = if anchor.on_mount && anchor.point.as_vec3().distance(rosette) < 1.5 {
                ControlMode::Mounted
            } else {
                ControlMode::Perched(anchor)
            };
        }
    }
    match flight.mode {
        ControlMode::Mounted | ControlMode::Flying => {
            let mounted = matches!(flight.mode, ControlMode::Mounted);
            let (body, velocity, cruise) = if mounted {
                (&mut *mount, &mut mount_velocity.0, 95.0)
            } else {
                (&mut *player, &mut player_velocity.0, 22.0)
            };
            let up = world.up(body.translation);
            pilot(body, velocity, steering, cruise * boost, dt, up);
            if mounted {
                player.translation = mount.transform_point(rosette);
                player.rotation = mount.rotation;
                player_velocity.0 = mount_velocity.0;
            }
        }
        ControlMode::Perched(anchor) => crawl(
            &mut player,
            &mount,
            &world,
            &mut flight,
            &mut player_velocity.0,
            anchor,
            Steering {
                forward,
                horizontal,
                dt,
                time: time.elapsed_secs(),
                mount_velocity: mount_velocity.0,
                physiology,
                individual,
            },
        ),
    }
}

fn advance_companion(
    time: Res<Time>,
    world: Res<WorldData>,
    player: Single<(&mut Transform, &mut Velocity), NaeQuery>,
    mount: Single<CompanionMotion, NaetuQuery>,
    camera: Single<Has<CameraController>, With<Camera3d>>,
    mut flight: ResMut<Flight>,
) {
    let dt = time.delta_secs().min(0.1);
    if dt <= 0.0 || *camera {
        return;
    }
    let (mut player, mut velocity) = player.into_inner();
    let (mut mount, mut mount_velocity, ecology, physiology, individual) = mount.into_inner();
    if !matches!(flight.mode, ControlMode::Mounted) {
        let on_back = matches!(flight.mode, ControlMode::Perched(a) if a.on_mount);
        let mut motion = Motion {
            position: mount.translation.to_scene(),
            velocity: mount_velocity.0.to_scene(),
            rotation: mount.rotation.to_scene(),
        };
        motion.advance(
            Kind::Naetu,
            &Senses {
                up: world.up(mount.translation).to_scene(),
                companion: Some(
                    if on_back {
                        Vec3::ZERO
                    } else {
                        player.translation - mount.translation
                    }
                    .to_scene(),
                ),
                ..ecology.map_or_else(Senses::default, |relations| relations.senses)
            },
            dt,
        );
        mount.translation = motion.position.to_engine();
        mount.rotation = motion.rotation.to_engine();
        mount_velocity.0 = motion.velocity.to_engine();
        if let ControlMode::Perched(anchor) = flight.mode
            && anchor.on_mount
        {
            player.translation = mount.transform_point(anchor.point.as_vec3());
            velocity.0 = mount_velocity.0;
        }
    }
    if matches!(flight.mode, ControlMode::Flying) {
        let body = instance(
            &mount,
            Kind::Naetu,
            time.elapsed_secs(),
            mount_velocity.0.length(),
            &world,
            physiology,
            individual,
        );
        separate_from_mount(
            &mut player,
            &mount,
            &body,
            &mut velocity.0,
            mount_velocity.0,
        );
    }
    flight.speed = velocity.0.length();
}

fn separate_from_mount(
    player: &mut Transform,
    mount: &Transform,
    body: &CreatureInstance,
    velocity: &mut Vec3,
    mount_velocity: Vec3,
) {
    for _ in 0..4 {
        let scale = development::scale(body).to_engine();
        let point = mount.rotation.inverse() * (player.translation - mount.translation) / scale;
        if point
            .abs()
            .cmpgt(body.opaque_bounds.truncate().to_engine() + 1.0)
            .any()
        {
            return;
        }
        let surface = |p| naetu::surface(development::body_point(p, body), body).distance;
        let distance = surface(point.to_scene()) * scale.min_element();
        if distance >= 0.35 {
            return;
        }
        let normal = (mount.rotation
            * (estimate_normal(point.to_scene(), surface).to_engine() / scale))
            .normalize();
        player.translation += normal * (0.35 - distance);
        let inward = (*velocity - mount_velocity).dot(normal).min(0.0);
        *velocity -= normal * inward;
    }
}

fn crawl(
    player: &mut Transform,
    mount: &Transform,
    world: &WorldData,
    flight: &mut Flight,
    velocity: &mut Vec3,
    mut anchor: PerchAnchor,
    steering: Steering,
) {
    let normal = if anchor.on_mount {
        mount.rotation * anchor.normal
    } else {
        world.rotate_to_local(anchor.normal)
    };
    let tangent = steering.horizontal - normal * steering.horizontal.dot(normal);
    let movement = tangent.normalize_or_zero() * (steering.dt * 2.4);
    let point = if anchor.on_mount {
        mount.transform_point(anchor.point.as_vec3())
    } else {
        world.local(anchor.point)
    };
    let c = instance(
        mount,
        Kind::Naetu,
        steering.time,
        steering.mount_velocity.length(),
        world,
        steering.physiology,
        steering.individual,
    );
    if let Some(projected) = contact(point + movement, mount, &c, world) {
        anchor = projected;
    }
    player.translation = if anchor.on_mount {
        mount.transform_point(anchor.point.as_vec3())
    } else {
        world.local(anchor.point)
    };
    let normal = if anchor.on_mount {
        mount.rotation * anchor.normal
    } else {
        world.rotate_to_local(anchor.normal)
    };
    let facing = (steering.forward - normal * steering.forward.dot(normal)).normalize_or_zero();
    if facing.length_squared() > 0.1 {
        player.look_to(facing, normal);
    }
    flight.mode = if anchor.on_mount && anchor.point.as_vec3().distance(ROSETTE) < 1.3 {
        ControlMode::Mounted
    } else {
        ControlMode::Perched(anchor)
    };
    *velocity = if anchor.on_mount {
        steering.mount_velocity
    } else {
        Vec3::ZERO
    };
}

pub fn follow_camera(
    time: Res<Time>,
    world: Res<WorldData>,
    flight: Res<Flight>,
    player: Single<&Transform, NaeQuery>,
    mount: Single<&Transform, NaetuQuery>,
    mut camera: Single<&mut Transform, FollowCameraQuery>,
) {
    let mounted = matches!(flight.mode, ControlMode::Mounted);
    let body = if mounted { *mount } else { *player };
    let up = world.up(body.translation);
    let focus = body.translation + if mounted { up * 5.0 } else { up * 0.2 };
    let heading = (*body.forward() - up * body.forward().dot(up)).normalize_or_zero();
    let back = Quat::from_axis_angle(up, flight.orbit.x) * -heading;
    let distance = (if mounted { 85.0 } else { 5.0 }) * flight.zoom;
    let pitch = 0.18 + flight.orbit.y;
    let target = focus + back * distance * pitch.cos() + up * distance * pitch.sin();
    let lag = 1.0 - (-7.0 * time.delta_secs()).exp();
    camera.translation = camera.translation.lerp(target, lag);
    let aim = Transform::from_translation(camera.translation)
        .looking_at(focus, world.up(camera.translation));
    camera.rotation = camera.rotation.slerp(aim.rotation, lag);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    #[test]
    fn companion_motion_keeps_perched_entities_and_velocities_together() {
        let mut world = World::new();
        let mut time = Time::<()>::default();
        time.advance_by(Duration::from_secs_f32(0.1));
        world.insert_resource(time);
        world.insert_resource(WorldData::default());
        world.insert_resource(Flight {
            mode: ControlMode::Perched(PerchAnchor {
                point: ROSETTE.as_dvec3(),
                normal: Vec3::Y,
                on_mount: true,
            }),
            ..Flight::default()
        });
        world.spawn(Camera3d::default());
        let body = Transform::from_translation(SPAWN).looking_to(Vec3::Y, Vec3::Z);
        let mount = world.spawn((Mount, body, Velocity(Vec3::X * 10.0))).id();
        let player = world
            .spawn((
                Player,
                Transform::from_translation(body.transform_point(ROSETTE)),
                Velocity::default(),
            ))
            .id();
        world
            .run_system_once(advance_companion)
            .expect("companion ECS system must run without conflicting queries");
        let moved = world.get::<Transform>(mount).expect("mount transform");
        let rider = world.get::<Transform>(player).expect("player transform");
        let velocity = world.get::<Velocity>(player).expect("player velocity").0;
        assert!(moved.translation.x > SPAWN.x, "companion did not move");
        assert!(
            rider.translation.distance(moved.transform_point(ROSETTE)) < 1e-5,
            "perched player detached from moving companion"
        );
        assert_eq!(
            velocity,
            world.get::<Velocity>(mount).expect("mount velocity").0,
            "perched entities must share velocity"
        );
        assert!(
            (world.resource::<Flight>().speed - velocity.length()).abs() < 1e-5,
            "HUD speed must reflect the entity velocity"
        );
    }

    #[test]
    fn acceleration_integrates_independently_of_frame_rate() {
        let simulate = |hz: u32| {
            let mut velocity = Vec3::ZERO;
            let mut position = Vec3::ZERO;
            for _ in 0..hz * 5 {
                position += advance(&mut velocity, Vec3::Y * 48.0, 2.2, 1.0 / hz as f32);
            }
            (position, velocity)
        };
        let a = simulate(30);
        let b = simulate(144);
        assert!(
            a.0.distance(b.0) < 0.002 && a.1.distance(b.1) < 0.002,
            "flight depends on frame rate"
        );
    }

    #[test]
    fn banking_turns_forward_flight_and_diving_tucks_wings() {
        let simulate = |hz: u32, steering: Vec2| {
            let mut body = Transform::default().looking_to(Vec3::Y, Vec3::Z);
            let mut velocity = Vec3::ZERO;
            for _ in 0..hz {
                pilot(
                    &mut body,
                    &mut velocity,
                    steering,
                    95.0,
                    1.0 / hz as f32,
                    Vec3::Z,
                );
            }
            (body, velocity)
        };
        let (slow, _) = simulate(30, Vec2::X);
        let (fast, velocity) = simulate(144, Vec2::X);
        assert!(
            slow.forward().distance(*fast.forward()) < 0.001,
            "steering depends on frame rate"
        );
        assert!(
            fast.forward().x > 0.7 && velocity.x > 40.0,
            "banking must turn forward flight"
        );
        let (dive, velocity) = simulate(60, -Vec2::Y);
        assert!(
            dive.translation.z < -20.0 && velocity.length() > 110.0,
            "diving must descend and gain speed"
        );
        assert!(
            instance(
                &dive,
                Kind::Naetu,
                0.0,
                velocity.length(),
                &WorldData::default(),
                None,
                None,
            )
            .animation
            .y > 0.7,
            "diving must tuck wings"
        );
    }

    #[test]
    fn contact_uses_body_and_ignores_aur_dome() {
        let mount = Transform::from_translation(SPAWN).looking_to(Vec3::Y, Vec3::Z);
        let body = instance(
            &mount,
            Kind::Naetu,
            0.0,
            0.0,
            &WorldData::default(),
            None,
            None,
        );
        let world = WorldData::default();
        let anchor = contact(mount.transform_point(ROSETTE), &mount, &body, &world)
            .expect("rosette must be reachable from the back");
        assert!(
            anchor.on_mount && anchor.point.as_vec3().distance(ROSETTE) < 1.5,
            "perching must resolve to the physical back"
        );
        assert!(
            contact(
                mount.transform_point(vec3(0.0, 15.1, 3.0)),
                &mount,
                &body,
                &world
            )
            .is_none(),
            "the aur dome must not act as a physical perch"
        );
    }

    #[test]
    fn flying_body_collision_removes_inward_velocity() {
        let mount = Transform::from_translation(SPAWN);
        let body = instance(
            &mount,
            Kind::Naetu,
            0.0,
            0.0,
            &WorldData::default(),
            None,
            None,
        );
        let mut player = Transform::from_translation(SPAWN + Vec3::X * 3.0);
        let mut velocity = -Vec3::X * 10.0;
        separate_from_mount(&mut player, &mount, &body, &mut velocity, Vec3::ZERO);
        assert!(
            naetu::surface((player.translation - SPAWN).to_scene(), &body).distance > 0.30,
            "Næ remains embedded in Nætu"
        );
        assert!(velocity.x >= -0.01, "collision must remove inward velocity");
    }
}
