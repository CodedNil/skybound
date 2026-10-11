use spirv_std::glam::{Vec2, Vec3, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::PLANET_RADIUS;

pub const REACTION_HEIGHT: f32 = 200_000.0;
pub const REACTION_RADIUS: f32 = PLANET_RADIUS + REACTION_HEIGHT;
pub const REACTION_SPREAD: f32 = 220_000.0;

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
