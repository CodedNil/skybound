use crate::scene::PLANET_RADIUS;
use core::f32::consts::FRAC_PI_2;
use spirv_std::glam::{Mat3, Quat, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// The pull of the planet at the aur's surface, in metres per second squared.
///
/// It is about a twelfth of Earth's, from a world of rocky density only 750 km across, and it is
/// why so much of its life lives in the sky: the gentlest updraft holds up a body that would fall
/// like a stone on a larger world.
pub const GRAVITY: f32 = 0.8;

pub const REACTION_HEIGHT: f32 = 100_000.0;

pub const REACTION_RADIUS: f32 = PLANET_RADIUS + REACTION_HEIGHT;

pub const REACTION_SPREAD: f32 = 110_000.0;

/// The direction towards its pole light from a point in the planet.
pub fn light_towards(point: Vec3) -> Vec3 {
    (light_position(point) - point).normalize()
}

pub fn surface_coordinates(point: Vec3) -> Vec3 {
    let radius = point.length().max(1.0);
    vec3(
        point.y.atan2(point.x) * PLANET_RADIUS,
        (FRAC_PI_2 - (point.z / radius).clamp(-1.0, 1.0).asin()) * PLANET_RADIUS,
        radius - PLANET_RADIUS,
    )
}

pub fn surface_point(coordinates: Vec3) -> Vec3 {
    let longitude = coordinates.x / PLANET_RADIUS;
    let colatitude = coordinates.y / PLANET_RADIUS;
    vec3(
        colatitude.sin() * longitude.cos(),
        colatitude.sin() * longitude.sin(),
        colatitude.cos(),
    ) * (PLANET_RADIUS + coordinates.z)
}

pub fn surface_frame(point: Vec3) -> Mat3 {
    let up = point / point.length().max(1e-8);
    let east = vec3(-up.y, up.x, 0.0);
    let east = if east.length_squared() < 1e-12 {
        Vec3::X
    } else {
        east / east.length().max(1e-8)
    };
    Mat3::from_cols(east, up.cross(east), up)
}

pub fn light_position(point: Vec3) -> Vec3 {
    Vec3::Z
        * (if point.z < 0.0 {
            -REACTION_RADIUS
        } else {
            REACTION_RADIUS
        })
}

pub fn reaction_coordinates(relative: Vec3, axis: Vec3) -> Vec2 {
    let radius = relative.length().max(1.0);
    let along = relative.dot(axis);
    let radial = (relative - axis * along).length();
    let angle = radial.atan2(along.abs());
    vec2(angle * REACTION_RADIUS, radius - REACTION_RADIUS)
}

pub fn get_pole_position(planet_center: Vec3, planet_rotation: Quat, ro_relative: Vec3) -> Vec3 {
    let north_axis = planet_rotation.mul_vec3(Vec3::Z).normalize();
    let up_vector = ro_relative.normalize();
    let pole_axis = if north_axis.dot(up_vector) > 0.0 {
        north_axis
    } else {
        -north_axis
    };
    let pole_altitude = PLANET_RADIUS + REACTION_HEIGHT;
    planet_center + pole_axis * pole_altitude
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_reactions_follow_the_same_spherical_boundary() {
        for sign in [-1.0, 1.0] {
            let center = Vec3::Z * (REACTION_RADIUS * sign);
            assert!(reaction_coordinates(center, Vec3::Z).length() < 0.01);
            let point = Vec3::new(0.1f32.sin(), 0.0, 0.1f32.cos()) * (REACTION_RADIUS * sign);
            let coordinates = reaction_coordinates(point, Vec3::Z);
            assert!(coordinates.y.abs() < 0.2);
            assert!((coordinates.x - REACTION_RADIUS * 0.1).abs() < 2.0);
        }
    }
}
