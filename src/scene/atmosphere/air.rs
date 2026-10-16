use spirv_std::glam::Vec3;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::{PLANET_RADIUS, geometry::intersect_sphere};

const SCALE_HEIGHT: f32 = 11_000.0;
const EXTINCTION: f32 = 0.000_004;

pub fn density(altitude: f32) -> f32 {
    EXTINCTION * (-altitude.max(0.0) / SCALE_HEIGHT).exp()
}

// Integrate an exponential height profile along a linear approximation to altitude.
// Unlike a midpoint sample this also resolves thin air near the end of a very long ray.
fn panel(origin: Vec3, direction: Vec3, start: f32, end: f32) -> f32 {
    let a = ((origin + direction * start).length() - PLANET_RADIUS).max(0.0);
    let b = ((origin + direction * end).length() - PLANET_RADIUS).max(0.0);
    let delta = (a - b).abs() / SCALE_HEIGHT;
    let mean = if delta < 0.01 {
        1.0 - delta * 0.5 + delta * delta / 6.0
    } else {
        (1.0 - (-delta).exp()) / delta
    };
    density(a.min(b)) * mean * (end - start)
}

pub fn optical_depth(origin: Vec3, direction: Vec3, length: f32) -> f32 {
    if length < 2000.0 {
        let altitude = (origin + direction * (length * 0.5)).length() - PLANET_RADIUS;
        return density(altitude) * length.max(0.0);
    }
    // Above twelve scale heights the remaining density is negligible. Clip first so
    // orbital rays spend the same fixed work on the air as rays close to the planet.
    let shell = intersect_sphere(origin, direction, PLANET_RADIUS + SCALE_HEIGHT * 12.0);
    let start = shell.x.max(0.0);
    let end = shell.y.min(length);
    if end <= start {
        return 0.0;
    }
    let closest = (-origin.dot(direction)).clamp(start, end);
    let mut depth = 0.0;
    for side in 0..2 {
        let (a, b) = if side == 0 {
            (start, closest)
        } else {
            (closest, end)
        };
        if b <= a {
            continue;
        }
        let step = (b - a) * 0.125;
        for index in 0..8 {
            let from = a + index as f32 * step;
            depth += panel(origin, direction, from, from + step);
        }
    }
    depth
}

pub fn transmittance(origin: Vec3, direction: Vec3, length: f32) -> f32 {
    (-optical_depth(origin, direction, length)).exp()
}

pub fn cached_transmittance(
    origin: Vec3,
    direction: Vec3,
    length: f32,
    lookup: impl Fn(Vec3, Vec3) -> f32,
) -> f32 {
    if length < 2000.0 {
        return transmittance(origin, direction, length);
    }
    let shell = intersect_sphere(origin, direction, PLANET_RADIUS + super::ATMOSPHERE_HEIGHT);
    let start = shell.x.max(0.0);
    let end = shell.y.min(length);
    if end <= start {
        return 1.0;
    }
    let a = origin + direction * start;
    let b = origin + direction * end;
    // Use the outward orientation to avoid subtracting large, nearly equal
    // columns through the planet when the visible segment is descending.
    let depth = if (a + b).dot(direction) >= 0.0 {
        lookup(a, direction) - lookup(b, direction)
    } else {
        lookup(b, -direction) - lookup(a, -direction)
    };
    (-depth.max(0.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_air_matches_the_exponential_integral() {
        for height in [0.0, 5000.0, 40_000.0] {
            for length in [10.0, 1000.0, 100_000.0, 1_000_000.0] {
                let origin = Vec3::Z * (PLANET_RADIUS + height);
                let expected =
                    (-density(height) * SCALE_HEIGHT * (1.0 - (-length / SCALE_HEIGHT).exp()))
                        .exp();
                assert!((transmittance(origin, Vec3::Z, length) - expected).abs() < 0.00001);
            }
        }
    }

    #[test]
    fn grazing_air_is_reciprocal_and_close_to_numerical_integration() {
        let origin = Vec3::Z * (PLANET_RADIUS + 2000.0) - Vec3::X * 600_000.0;
        let length = 1_200_000.0;
        let value = transmittance(origin, Vec3::X, length);
        assert!((value - transmittance(origin + Vec3::X * length, -Vec3::X, length)).abs() < 1e-5);
        let mut depth = 0.0;
        for index in 0..4096 {
            let point = origin + Vec3::X * ((index as f32 + 0.5) * length / 4096.0);
            depth += density(point.length() - PLANET_RADIUS) * length / 4096.0;
        }
        assert!((value - (-depth).exp()).abs() < 0.015);
        assert!(
            (transmittance(Vec3::Z * (PLANET_RADIUS + 200_000.0), Vec3::X, length) - 1.0).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn cached_air_columns_match_direct_integration_on_visible_segments() {
        use crate::scene::atmosphere::{TRANSMITTANCE_SIZE, air_texel, transmittance_coordinates};
        use spirv_std::glam::{Vec2, vec2};
        let size = TRANSMITTANCE_SIZE as usize;
        let table: Vec<_> = (0..size * size)
            .map(|index| {
                air_texel(vec2((index % size) as f32, (index / size) as f32) / (size - 1) as f32)
            })
            .collect();
        let lookup = |point, direction| {
            let uv = transmittance_coordinates(point, direction) * size as f32 - 0.5;
            let uv = uv.clamp(Vec2::ZERO, Vec2::splat((size - 1) as f32));
            let column = uv.x.floor() as usize;
            let row = uv.y.floor() as usize;
            let horizontal = uv.x.fract();
            let vertical = uv.y.fract();
            let fetch = |x: usize, y: usize| table[y.min(size - 1) * size + x.min(size - 1)];
            let top = fetch(column, row) * (1.0 - horizontal) + fetch(column + 1, row) * horizontal;
            let bottom = fetch(column, row + 1) * (1.0 - horizontal)
                + fetch(column + 1, row + 1) * horizontal;
            top * (1.0 - vertical) + bottom * vertical
        };
        for height in [0.0, 2500.0, 20_000.0, 100_000.0, 350_000.0] {
            let origin = Vec3::Z * (PLANET_RADIUS + height);
            for cosine in [-1.0f32, -0.3, -0.05, 0.0, 0.1, 0.7, 1.0] {
                let direction = Vec3::new((1.0 - cosine * cosine).sqrt(), 0.0, cosine);
                for length in [500.0, 5000.0, 100_000.0, 1_000_000.0] {
                    let closest = (-origin.dot(direction)).clamp(0.0, length);
                    if (origin + direction * closest).length() < PLANET_RADIUS {
                        continue;
                    }
                    let expected = transmittance(origin, direction, length);
                    let cached = cached_transmittance(origin, direction, length, lookup);
                    assert!(
                        (cached - expected).abs() < 0.02,
                        "height={height} cosine={cosine} length={length}: {cached} vs {expected}"
                    );
                }
            }
        }
    }
}
