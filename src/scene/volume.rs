//! Stable integration of smooth volume segments; no stochastic point estimates.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

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

#[cfg(test)]
mod tests {
    use super::*;
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
