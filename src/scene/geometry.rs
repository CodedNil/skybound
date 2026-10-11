use spirv_std::glam::{Quat, Vec2, Vec3, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::PLANET_RADIUS;

pub const MAGNETOSPHERE_HEIGHT: f32 = 400_000.0;

pub fn intersect_sphere(ro: Vec3, rd: Vec3, radius: f32) -> Vec2 {
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(ro);
    let c = ro.dot(ro) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return vec2(1.0, -1.0);
    }
    let sqrt_disc = disc.sqrt();
    vec2(-b - sqrt_disc, -b + sqrt_disc) / (2.0 * a)
}

pub fn get_pole_position(planet_center: Vec3, planet_rotation: Quat, ro_relative: Vec3) -> Vec3 {
    let north_axis = planet_rotation.mul_vec3(Vec3::Z).normalize();
    let up_vector = ro_relative.normalize();
    let pole_axis = if north_axis.dot(up_vector) > 0.0 {
        north_axis
    } else {
        -north_axis
    };
    let pole_altitude = PLANET_RADIUS + MAGNETOSPHERE_HEIGHT;
    planet_center + pole_axis * pole_altitude
}
