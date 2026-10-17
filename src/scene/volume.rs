//! Stable integration of smooth volume segments; no stochastic point estimates.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use spirv_std::glam::{Vec3, Vec4};

/// Close an optically opaque ray with its last source value. Dropping a
/// different residual at the cutoff draws contours between marching steps.
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

/// Exact mean of a clamped cubic transition over a linear density segment.
/// Filtering the transition itself removes marching contours without dither.
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

#[cfg(test)]
mod tests {
    use super::*;
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
