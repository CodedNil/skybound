use crate::scene::PLANET_RADIUS;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const CELL: f32 = 2200.0;

pub fn spike_distance(p: Vec3, offset: Vec2) -> f32 {
    let grid = (p.truncate() + offset) / CELL;
    let cell = grid.floor();
    let local = (grid - cell - 0.5) * CELL;
    let seed = (cell.dot(vec2(127.1, 311.7)).sin() * 43_758.547).fract_gl();
    let seed2 = (cell.dot(vec2(269.5, 183.3)).sin() * 43_758.547).fract_gl();
    let center = vec2(seed - 0.5, seed2 - 0.5) * (CELL * 0.4);
    let height = 900.0 + seed * 4500.0;
    let radius = 180.0 + seed2 * 260.0;
    let z = p.z + 6000.0;
    let slope = radius / height;
    let side = ((local - center).length() + slope * z - radius) / (1.0 + slope * slope).sqrt();
    let cone = side.max(-z).max(z - height);
    // The gap bound prevents a step across the cell from skipping a neighbouring cone.
    let gap = CELL * 0.3 - 440.0;
    let edge = CELL * 0.5 - local.abs().max_element();
    cone.min(edge + gap).min(z)
}

pub fn spike_interval(altitude: f32, direction: Vec3, max_distance: f32) -> Vec2 {
    let a = direction.truncate().length_squared() / (2.0 * PLANET_RADIUS);
    let b = direction.z;
    let c = altitude + 600.0;
    if a < 1e-12 {
        if b < 0.0 {
            return vec2((-c / b).max(0.0), max_distance);
        }
        if c <= 0.0 {
            return vec2(
                0.0,
                if b > 0.0 {
                    (-c / b).min(max_distance)
                } else {
                    max_distance
                },
            );
        }
        return vec2(1.0, -1.0);
    }
    let discriminant = b * b - 4.0 * a * c;
    if discriminant <= 0.0 || (c > 0.0 && b >= 0.0) {
        return vec2(1.0, -1.0);
    }
    let root = discriminant.sqrt();
    let q = -0.5 * (b + if b < 0.0 { -root } else { root });
    let r1 = q / a;
    let r2 = c / q;
    vec2(r1.min(r2).max(0.0), r1.max(r2).min(max_distance))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_handle_vertical_horizontal_and_inside_rays() {
        let down = spike_interval(2400.0, -Vec3::Z, 1e6);
        assert!((down.x - 3000.0).abs() < 0.01, "incorrect entry altitude");
        assert!(
            spike_interval(2400.0, Vec3::Z, 1e6).x > spike_interval(2400.0, Vec3::Z, 1e6).y,
            "upward ray should miss"
        );
        assert!(
            spike_interval(2400.0, Vec3::X, 1e6).x > spike_interval(2400.0, Vec3::X, 1e6).y,
            "horizontal ray should miss"
        );
        let inside = spike_interval(-1000.0, Vec3::Z, 1e6);
        assert!(
            inside.x.abs() < 0.01 && (inside.y - 400.0).abs() < 0.01,
            "inside ray needs an exit"
        );
        for elevation in -10..=10 {
            let direction = Vec3::new(1.0, 0.0, elevation as f32 / 10.0).normalize();
            assert!(
                spike_interval(-1000.0, direction, 1e6).is_finite(),
                "non-finite interval"
            );
        }
    }
    #[test]
    fn cones_are_signed_and_cell_edges_cannot_be_false_hits() {
        let center = Vec3::new(660.0, 660.0, -5500.0);
        assert!(
            spike_distance(center, Vec2::ZERO) < 0.0,
            "cone interior must be signed"
        );
        assert!(
            spike_distance(Vec3::new(0.0, 0.0, -5500.0), Vec2::ZERO) > 0.0,
            "cell boundary must stay empty"
        );
        assert!(
            spike_distance(Vec3::new(660.0, 660.0, 0.0), Vec2::ZERO) > 0.0,
            "above every cone must stay empty"
        );
        let p = Vec3::new(100.0, 200.0, -5000.0);
        let shifted = p + Vec3::new(2200.0, -4400.0, 0.0);
        assert!(
            (spike_distance(p, Vec2::ZERO) - spike_distance(shifted, Vec2::new(-2200.0, 4400.0)))
                .abs()
                < 0.01,
            "rebasing changed terrain"
        );
    }
}
