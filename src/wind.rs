use avian3d::prelude::{ExternalForce, RigidBody};
use bevy::prelude::*;

// Define the wind force at specific heights as constants.
const WIND_FORCE_AT_SEA_LEVEL: f32 = 6.0; // The max force at or below y=0 (sea level)
const WIND_FORCE_AT_100_Y: f32 = 3.0; // The force at y=100 (e.g., half of sea level force)
const DECAY_RATIO: f32 = WIND_FORCE_AT_100_Y / WIND_FORCE_AT_SEA_LEVEL;

/// Calculates an upward wind force based on the given Y-position.
pub fn get_wind_force(position: Vec3) -> Vec3 {
    if position.y <= 0.0 {
        return Vec3::Y * WIND_FORCE_AT_SEA_LEVEL;
    }

    let wind_force_magnitude = WIND_FORCE_AT_SEA_LEVEL * DECAY_RATIO.powf(position.y / 100.0);

    Vec3::Y * wind_force_magnitude
}

/// Bevy system that applies an upward wind force to dynamic rigid bodies.
pub fn apply_wind_force(mut query: Query<(&Transform, &mut ExternalForce, &RigidBody)>) {
    for (transform, mut external_force, rigid_body) in &mut query {
        if matches!(rigid_body, RigidBody::Dynamic) {
            let wind_force = get_wind_force(transform.translation);
            *external_force = ExternalForce::new(wind_force);
        }
    }
}
