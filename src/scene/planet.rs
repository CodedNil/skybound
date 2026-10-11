use core::f32::consts::FRAC_PI_2;
use spirv_std::glam::{Mat3, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::PLANET_RADIUS;

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
