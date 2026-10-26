use core::f32::consts::PI;
use spirv_std::glam::{IVec3, Vec2, Vec3, Vec4, ivec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub fn square(value: f32) -> f32 {
    value * value
}

pub fn cube(value: f32) -> f32 {
    value * value * value
}

pub fn fourth(value: f32) -> f32 {
    square(square(value))
}

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

/// Where a ray crosses a shell between two altitudes over a sphere of `radius` at the origin: in
/// and out through the top, then through the bottom, if it reaches that.
pub fn ray_shell_intersect(
    ro: Vec3,
    rd: Vec3,
    radius: f32,
    bottom_altitude: f32,
    top_altitude: f32,
) -> Vec4 {
    let top_interval = intersect_sphere(ro, rd, radius + top_altitude);
    if top_interval.x > top_interval.y {
        return vec4(1.0, 0.0, 1.0, 0.0);
    }
    let bottom_interval = intersect_sphere(ro, rd, radius + bottom_altitude);
    if bottom_interval.x > bottom_interval.y {
        return vec4(top_interval.x, top_interval.y, 1.0, 0.0);
    }
    vec4(
        top_interval.x,
        bottom_interval.x,
        bottom_interval.y,
        top_interval.y,
    )
}

/// Close an optically opaque ray with its last source value. Dropping a different residual at the
/// cutoff draws contours between marching steps.
pub fn finish_opaque(color: Vec4, source: Vec3) -> Vec4 {
    if color.w < 0.005 {
        (color.truncate() + source * color.w).extend(0.0)
    } else {
        color
    }
}

pub fn exponential_column(a: f32, b: f32, length: f32, scale_height: f32) -> f32 {
    let a = a.max(0.0);
    let b = b.max(0.0);
    let delta = (a - b).abs() / scale_height;
    let mean = if delta < 0.01 {
        1.0 - delta * 0.5 + delta * delta / 6.0
    } else {
        (1.0 - (-delta).exp()) / delta
    };
    (-a.min(b).max(0.0) / scale_height).exp() * mean * length.max(0.0)
}

pub fn simpson(a: f32, midpoint: f32, b: f32, length: f32) -> f32 {
    (a + midpoint * 4.0 + b).max(0.0) * (length / 6.0)
}

/// Exact mean of a clamped cubic transition over a linear density segment. Filtering the transition
/// itself removes marching contours without dither.
pub fn smoothstep_mean(a: f32, b: f32) -> f32 {
    let low = a.min(b);
    let high = a.max(b);
    if high <= 0.0 {
        return 0.0;
    }
    if low >= 1.0 {
        return 1.0;
    }
    if high - low < 0.001 {
        let x = a.midpoint(b).clamp(0.0, 1.0);
        return x * x * (3.0 - 2.0 * x);
    }
    let primitive = |x: f32| {
        let t = x.clamp(0.0, 1.0);
        t * t * t * (1.0 - t * 0.5) + (x - 1.0).max(0.0)
    };
    (primitive(high) - primitive(low)) / (high - low)
}

pub fn cellular(position: Vec3, period: i32, seed: u32) -> f32 {
    let p = position * period as f32;
    let cell = p.floor().as_ivec3();
    let mut nearest = 3.0f32;
    for z in -1..=1 {
        for y in -1..=1 {
            for x in -1..=1 {
                let c = cell + ivec3(x, y, z);
                let feature = c.as_vec3()
                    + vec3(
                        lattice(c, period, seed),
                        lattice(c, period, seed.wrapping_add(31)),
                        lattice(c, period, seed.wrapping_add(67)),
                    );
                nearest = nearest.min(p.distance_squared(feature));
            }
        }
    }
    nearest.sqrt().min(1.0)
}

pub fn rotated(p: Vec3) -> Vec3 {
    vec3(
        p.dot(vec3(0.36, 0.48, -0.8)),
        p.dot(vec3(-0.8, 0.6, 0.0)),
        p.dot(vec3(0.48, 0.64, 0.6)),
    ) * 0.754_877
        + vec3(0.37, 0.19, 0.63)
}

pub fn lattice(cell: IVec3, period: i32, seed: u32) -> f32 {
    let mut value = (cell.x.rem_euclid(period) as u32).wrapping_mul(73_856_093)
        ^ (cell.y.rem_euclid(period) as u32).wrapping_mul(19_349_663)
        ^ (cell.z.rem_euclid(period) as u32).wrapping_mul(83_492_791)
        ^ seed;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value >> 8) as f32 / 16_777_215.0
}

fn periodic_noise(position: Vec3, period: i32, seed: u32) -> f32 {
    let position = position * period as f32;
    let cell = position.floor().as_ivec3();
    let fraction = position - position.floor();
    let weight = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0);
    let mut value = 0.0;
    for corner in 0..8 {
        let offset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let blend = Vec3::select(offset.cmpeq(IVec3::ZERO), Vec3::ONE - weight, weight);
        let hash = (lattice(cell + offset, period, seed) * 255.0) as u32 & 15;
        let delta = fraction - offset.as_vec3();
        let u = if hash < 8 { delta.x } else { delta.y };
        let v = if hash < 4 {
            delta.y
        } else if hash == 12 || hash == 14 {
            delta.x
        } else {
            delta.z
        };
        let gradient = (if hash & 1 == 0 { u } else { -u }) + (if hash & 2 == 0 { v } else { -v });
        value += gradient * blend.x * blend.y * blend.z;
    }
    value * 0.5 + 0.5
}

pub fn fbm(position: Vec3, period: i32, seed: u32) -> f32 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    for octave in 0..4 {
        value += periodic_noise(position, period << octave, seed.wrapping_add(octave * 1013))
            * amplitude;
        amplitude *= 0.5;
    }
    ((value / 1.875 - 0.5) * 2.0 + 0.5).clamp(0.0, 1.0)
}

/// Pack a non-negative value as half-precision bits, for compute passes that write textures
/// through buffers.
pub fn half(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    let bits = value.min(65_504.0).to_bits();
    let exponent = (bits >> 23) & 255;
    if exponent < 103 {
        return 0;
    }
    if exponent < 113 {
        let shift = 126 - exponent;
        return (((bits & 0x7f_ffff) | 0x80_0000) + (1 << (shift - 1))) >> shift;
    }
    let rounded = bits + 0x1000;
    (((rounded >> 23) - 112) << 10) | ((rounded >> 13) & 0x3ff)
}

pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    (1.0 - g2) / (4.0 * PI * denom.max(1e-4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_encoding_matches_ieee_bits() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(32.0), 0x5000);
    }

    #[test]
    fn opaque_emission_has_no_partition_dependent_cutoff_tail() {
        let source = Vec3::new(3.0, 0.8, 1.5);
        for parts in [1, 16, 257] {
            let mut color = Vec4::W;
            let transmission = (-8.0 / parts as f32).exp();
            for _ in 0..parts {
                let weight = color.w * (1.0 - transmission);
                color = finish_opaque(
                    (color.truncate() + source * weight).extend(color.w * transmission),
                    source,
                );
            }
            assert!(color.truncate().distance(source) < 0.00001);
            assert!(color.w.abs() < f32::EPSILON);
        }
        let thin = Vec4::new(0.2, 0.3, 0.4, 0.25);
        assert_eq!(finish_opaque(thin, source), thin);
    }

    #[test]
    fn integrated_transition_is_invariant_to_segment_partition_and_direction() {
        for (a, b) in [(-2.0, 3.0), (-0.2, 0.8), (0.3, 0.7), (1.2, 2.0)] {
            let whole = smoothstep_mean(a, b);
            assert!((whole - smoothstep_mean(b, a)).abs() < 1e-6);
            let parts = (0..64)
                .map(|i| {
                    let x = a + (b - a) * i as f32 / 64.0;
                    let y = a + (b - a) * (i + 1) as f32 / 64.0;
                    smoothstep_mean(x, y) / 64.0
                })
                .sum::<f32>();
            assert!((whole - parts).abs() < 1e-5);
            assert!((0.0..=1.0).contains(&whole));
        }
        assert!((smoothstep_mean(-1.0, 2.0) - 0.5).abs() < 1e-6);
    }
    #[test]
    fn exponential_veil_is_exact_for_horizontal_and_vertical_segments() {
        for height in [0.0f32, 20.0, 300.0, 600.0] {
            let horizontal = exponential_column(height, height, 1000.0, 120.0);
            assert!((horizontal - (-height / 120.0).exp() * 1000.0).abs() < 0.0001);
            let vertical = exponential_column(height, height + 600.0, 600.0, 120.0);
            let expected = 120.0 * ((-height / 120.0).exp() - (-(height + 600.0) / 120.0).exp());
            assert!((vertical - expected).abs() < 0.0001);
        }
    }
    #[test]
    fn smooth_cloud_segments_integrate_without_dither() {
        // Simpson quadrature exactly integrates a cubic density profile.
        let density = |x: f32| x * x * (3.0 - 2.0 * x);
        assert!((simpson(density(0.0), density(0.5), density(1.0), 1.0) - 0.5).abs() < 0.000_001);
        let split = simpson(density(0.0), density(0.25), density(0.5), 0.5)
            + simpson(density(0.5), density(0.75), density(1.0), 0.5);
        assert!((split - 0.5).abs() < 0.000_001);
    }
}
