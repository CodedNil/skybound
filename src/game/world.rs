use crate::math::{ToEngine, ToScene};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use bevy::math::{DQuat, DVec3};
use bevy::transform::TransformSystems;

use bevy::anti_alias::contrast_adaptive_sharpening::ContrastAdaptiveSharpening;
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::{
    DepthPrepass, MotionVectorPrepass, NoBackgroundMotionVectors, NormalPrepass,
};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::view::DebandDither;
use skybound::scene::{PLANET_RADIUS, planet::surface_coordinates};

use crate::game::ecosystem::{EcologicalMotion, Relations};
use crate::game::flight::{SPAWN, Velocity, follow_camera};

/// Every world-space entity moves together when the floating origin changes.
#[derive(Component)]
pub struct WorldAnchor;

const CAMERA_RESET_THRESHOLD: f32 = 2048.0;

#[derive(Component)]
pub struct PlanetPosition {
    pub position: DVec3,
    local: Vec3,
}

impl PlanetPosition {
    pub fn translate(&mut self, displacement: Vec3, transform: &mut Transform, world: &WorldData) {
        self.position += world.basis * displacement.as_dvec3();
        self.local = world.local(self.position);
        transform.translation = self.local;
    }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WorldData {
    pub origin: DVec3,
    pub basis: DQuat,
}
impl Default for WorldData {
    fn default() -> Self {
        Self::at(FRAC_PI_4, -FRAC_PI_2)
    }
}
impl WorldData {
    pub fn at(latitude: f64, longitude: f64) -> Self {
        let up = DVec3::new(
            latitude.cos() * longitude.cos(),
            latitude.cos() * longitude.sin(),
            latitude.sin(),
        );
        Self {
            origin: up * f64::from(PLANET_RADIUS),
            basis: DQuat::from_rotation_arc(DVec3::Z, up),
        }
    }
    pub fn position(&self, local: Vec3) -> DVec3 {
        self.origin + self.basis * local.as_dvec3()
    }
    pub fn local(&self, position: DVec3) -> Vec3 {
        (self.basis.conjugate() * (position - self.origin)).as_vec3()
    }
    pub fn altitude(&self, local: Vec3) -> f64 {
        self.position(local).length() - f64::from(PLANET_RADIUS)
    }
    pub fn up(&self, local: Vec3) -> Vec3 {
        (self.basis.conjugate() * self.position(local).normalize()).as_vec3()
    }
    pub fn down(&self, local: Vec3) -> Vec3 {
        -self.up(local)
    }
    pub fn surface_position(&self, local: Vec3) -> Vec3 {
        surface_coordinates(self.position(local).as_vec3().to_scene()).to_engine()
    }
    pub fn planet_frame(&self, local: Vec3) -> (Quat, f32, f32) {
        let point = self.position(local);
        (
            self.basis.conjugate().as_quat(),
            (point.z / point.length()).asin() as f32,
            point.y.atan2(point.x) as f32,
        )
    }
    pub fn rotate_to_local(&self, direction: Vec3) -> Vec3 {
        self.basis.conjugate().as_quat() * direction
    }
}

pub struct WorldPlugin;
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorldSystems {
    Rebase,
}
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldData>()
            .add_systems(Startup, setup)
            .add_systems(
                PostUpdate,
                (sync_positions, update)
                    .chain()
                    .in_set(WorldSystems::Rebase)
                    .before(TransformSystems::Propagate)
                    .after(follow_camera),
            );
    }
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        WorldAnchor,
        Camera::default(),
        Projection::default(),
        NormalPrepass,
        DepthPrepass,
        MotionVectorPrepass,
        NoBackgroundMotionVectors,
        Msaa::Off,
        Hdr,
        DebandDither::Enabled,
        Bloom::NATURAL,
        TemporalAntiAliasing::default(),
        ContrastAdaptiveSharpening {
            sharpening_strength: 0.3,
            ..default()
        },
        Transform::from_translation(SPAWN + vec3(0.0, -56.0, 13.0)).looking_at(SPAWN, Vec3::Z),
    ));
}

fn sync_positions(
    mut commands: Commands,
    world: Res<WorldData>,
    mut anchors: Query<(Entity, &Transform, Option<&mut PlanetPosition>), With<WorldAnchor>>,
) {
    for (entity, transform, coordinate) in &mut anchors {
        if let Some(mut coordinate) = coordinate {
            let displacement = (transform.translation - coordinate.local).as_dvec3();
            coordinate.position += world.basis * displacement;
            coordinate.local = transform.translation;
        } else {
            commands.entity(entity).insert(PlanetPosition {
                position: world.position(transform.translation),
                local: transform.translation,
            });
        }
    }
}

type RebaseAnchor<'a> = (
    &'a mut Transform,
    &'a mut PlanetPosition,
    Option<&'a mut Velocity>,
    Option<&'a mut EcologicalMotion>,
    Option<&'a mut Relations>,
);
fn update(
    mut world: ResMut<WorldData>,
    camera: Single<Entity, With<Camera3d>>,
    mut anchors: Query<RebaseAnchor, With<WorldAnchor>>,
    mut taa: Single<&mut TemporalAntiAliasing>,
) {
    let (transform, coordinate, _, _, _) = anchors.get(*camera).expect("camera world anchor");
    if transform.translation.abs().max_element() < CAMERA_RESET_THRESHOLD {
        return;
    }
    let origin = coordinate.position;
    let basis = DQuat::from_rotation_arc(DVec3::Z, origin.normalize());
    let rotation = (basis.conjugate() * world.basis).as_quat();
    world.origin = origin;
    world.basis = basis;
    taa.reset = true;
    for (mut transform, mut coordinate, velocity, ecological, relations) in &mut anchors {
        transform.translation = world.local(coordinate.position);
        coordinate.local = transform.translation;
        transform.rotation = rotation * transform.rotation;
        if let Some(mut velocity) = velocity {
            velocity.0 = rotation * velocity.0;
        }
        if let Some(mut velocity) = ecological {
            velocity.0 = rotation * velocity.0;
        }
        if let Some(mut relations) = relations {
            relations.rotate(rotation);
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::ecs::system::RunSystemOnce;

    use super::*;

    #[test]
    fn down_points_to_planet_at_every_latitude_and_longitude() {
        for latitude in [-FRAC_PI_2, -FRAC_PI_4, 0.0, FRAC_PI_4, FRAC_PI_2] {
            for longitude in [-3.0, -1.0, 0.0, 2.0] {
                let world = WorldData::at(latitude, longitude);
                let local = vec3(40_000.0, -20_000.0, 300_000.0);
                let global_down = world.basis * world.down(local).as_dvec3();
                assert!(global_down.dot(-world.position(local).normalize()) > 0.999_999);
                let lower = local + world.down(local) * 10.0;
                assert!((world.altitude(local) - world.altitude(lower) - 10.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn rebasing_preserves_planet_positions_heading_and_velocity() {
        let frame = WorldData::default();
        let mut ecs = World::new();
        ecs.insert_resource(frame.clone());
        let camera = ecs
            .spawn((
                Camera3d::default(),
                WorldAnchor,
                TemporalAntiAliasing::default(),
                Transform::from_translation(vec3(40_000.0, 20_000.0, 90_000.0)),
            ))
            .id();
        let transform =
            Transform::from_translation(vec3(12.0, 23.0, 3000.0)).looking_to(Vec3::Y, Vec3::Z);
        let heading = frame.basis * transform.forward().as_dvec3();
        let velocity = vec3(12.0, 30.0, -4.0);
        let creature = ecs.spawn((WorldAnchor, transform, Velocity(velocity))).id();
        ecs.run_system_once(sync_positions).unwrap();
        let before = ecs.get::<PlanetPosition>(creature).unwrap().position;
        ecs.run_system_once(update).unwrap();
        let after = ecs.resource::<WorldData>();
        let transform = ecs.get::<Transform>(creature).unwrap();
        assert_eq!(
            ecs.get::<PlanetPosition>(creature).unwrap().position,
            before
        );
        assert!((after.position(transform.translation) - before).length() < 0.02);
        assert!((after.basis * transform.forward().as_dvec3() - heading).length() < 1e-6);
        assert!(
            (after.basis * ecs.get::<Velocity>(creature).unwrap().0.as_dvec3()
                - frame.basis * velocity.as_dvec3())
            .length()
                < 1e-5
        );
        assert!(ecs.get::<Transform>(camera).unwrap().translation.length() < 0.001);
        assert!(ecs.get::<TemporalAntiAliasing>(camera).unwrap().reset);
    }
}
