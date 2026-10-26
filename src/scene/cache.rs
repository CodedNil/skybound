use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    aur::{spire_coordinates, spire_point},
    planet::surface_frame,
};
use spirv_std::glam::{Vec2, Vec3, Vec4Swizzles, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const SIZE: u32 = 512;
/// Each level doubles the spacing. A factor of four left the texel up to eight times the pixel
/// footprint just past every level boundary.
pub const CASCADES: u32 = 10;
const FINEST: f32 = 4.0;

/// The cache origin moves in steps of this many metres.
const SNAP: f32 = 256.0;

pub const fn spacing(cascade: u32) -> f32 {
    (4u32 << cascade) as f32
}

impl ViewUniform {
    pub fn prepare_cache(&mut self) {
        // Fixed to a planet grid, so a moving camera does not resample the waves at shifting texel
        // positions and set distant fog shimmering.
        let camera = spire_coordinates(self.camera());
        let snapped = (camera.truncate() / SNAP).round() * SNAP;
        let up = spire_point(snapped.extend(0.0)).normalize();
        let frame = surface_frame(up);
        self.cache_origin =
            (up * PLANET_RADIUS).extend(f32::from(self.camera_position.w as u32 & 8 == 0));
        self.cache_east = frame.x_axis.extend(0.0);
        self.cache_north = frame.y_axis.extend(0.0);
    }
}

pub fn coordinates(point: Vec3, view: &ViewUniform) -> Option<Vec2> {
    let up = view.cache_origin.xyz() / PLANET_RADIUS;
    let denominator = point.dot(up);
    if denominator <= 0.0 {
        return None;
    }
    Some(
        vec2(
            point.dot(view.cache_east.xyz()),
            point.dot(view.cache_north.xyz()),
        ) * (PLANET_RADIUS / denominator),
    )
}

pub fn point(coordinates: Vec2, view: &ViewUniform) -> Vec3 {
    (view.cache_origin.xyz()
        + view.cache_east.xyz() * coordinates.x
        + view.cache_north.xyz() * coordinates.y)
        .normalize()
        * PLANET_RADIUS
}

pub fn cascade(coordinates: Vec2) -> u32 {
    let extent = coordinates.abs().max_element();
    let mut level = 0;
    while level < CASCADES && extent > spacing(level) * (SIZE as f32 * 0.46 - 2.0) {
        level += 1;
    }
    level
}

/// Screen footprints must also choose the filter scale. Otherwise an orbital camera sees an
/// unfiltered high-detail disc directly below itself.
pub fn filter_level(footprint: f32) -> f32 {
    (footprint.max(FINEST) / FINEST).log2()
}

/// Levels finer than the nearest visible ocean footprint are never sampled.
pub fn first_level(altitude: f32, angle: f32) -> u32 {
    filter_level((altitude - 500.0).max(0.0) * angle * 0.7) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::{Vec4, vec3};

    #[test]
    fn pixel_filter_scale_is_continuous_and_independent_of_camera_centre() {
        assert!(filter_level(2.0).abs() < f32::EPSILON);
        for level in 0..CASCADES {
            assert!((filter_level(spacing(level)) - level as f32).abs() < 0.0001);
        }
        assert!((filter_level(64.0) - 4.0).abs() < 0.0001);
        assert_eq!(first_level(0.0, 0.002), 0);
        assert_eq!(first_level(10_000.0, 0.002), 1);
    }

    fn view(up: Vec3) -> ViewUniform {
        let mut view = ViewUniform {
            world_position: (up * (PLANET_RADIUS + 7500.0)).extend(0.0),
            ..Default::default()
        };
        view.prepare_cache();
        view
    }

    #[test]
    fn projection_round_trips_at_poles_and_across_longitude_seam() {
        for up in [
            Vec3::Z,
            -Vec3::Z,
            Vec3::X,
            vec3(-1.0, -0.00001, 0.3).normalize(),
        ] {
            let view = view(up);
            for c in [Vec2::ZERO, vec2(625.0, -875.0), vec2(87_500.0, -37_500.0)] {
                let projected = coordinates(point(c, &view), &view).unwrap();
                assert!(
                    // f32 holds points in the planet to a tenth of a metre.
                    projected.distance(c) < 0.3,
                    "cache projection drift: {c} -> {projected}"
                );
            }
            assert!(coordinates(-up * PLANET_RADIUS, &view).is_none());
        }
    }

    #[test]
    fn cascade_coverage_and_reference_switch() {
        assert_eq!(cascade(Vec2::ZERO), 0);
        assert_eq!(cascade(Vec2::splat(1_000_000.0)), CASCADES);
        assert!(spacing(CASCADES - 1) * SIZE as f32 * 0.46 > 375_000.0);
        let mut view = view(Vec3::X);
        view.camera_position = Vec4::new(0.0, 0.0, 0.0, 8.0);
        view.prepare_cache();
        assert!(view.cache_origin.w.abs() < f32::EPSILON);
    }
}
