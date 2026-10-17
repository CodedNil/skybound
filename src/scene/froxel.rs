//! A single view volume; logarithmic depth and native-resolution integration.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const WIDTH: u32 = 256;
pub const HEIGHT: u32 = 144;
pub const DEPTH: u32 = 96;
pub const COUNT: usize = (WIDTH * HEIGHT * DEPTH) as usize;
const NEAR_SCALE: f32 = 64.0;
const LOG_RANGE: f32 = 13.931_66; // log2(1 + 1,000,000 / 64)

pub fn distance(slice: f32) -> f32 {
    NEAR_SCALE * ((slice / DEPTH as f32 * LOG_RANGE).exp2() - 1.0)
}

pub fn depth(distance: f32) -> f32 {
    (1.0 + distance.max(0.0) / NEAR_SCALE).log2() / LOG_RANGE
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depth_mapping_and_half_encoding_are_consistent() {
        for slice in 0..=DEPTH {
            let mapped = depth(distance(slice as f32)) * DEPTH as f32;
            assert!((mapped - slice as f32).abs() < 0.0001);
        }
        assert_eq!(half(0.0), 0);
        assert_eq!(half(0.5), 0x3800);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(32.0), 0x5000);
    }
}
