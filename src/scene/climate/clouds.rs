use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::{PLANET_RADIUS, geometry::intersect_sphere, noise::lattice};

pub const LAYERS: usize = 6;
pub const STORM: usize = LAYERS;
pub const ALL_KINDS: u32 = (1 << (LAYERS + 1)) - 1;
/// One coherent formation per altitude regime, plus independently bounded storms.
pub const ACTIVE_KINDS: u32 = (1 << 0) | (1 << 2) | (1 << 4) | (1 << STORM);
pub const HIGH_KINDS: u32 = (1 << 4) | (1 << 5);

pub const fn cloud_type(kind: usize) -> usize {
    kind / 2
}

pub fn kinds_at_altitude(altitude: f32) -> u32 {
    let mut kinds = 0;
    for kind in 0..LAYERS {
        let bounds = layer_bounds(kind);
        kinds |= u32::from(altitude > bounds.x && altitude < bounds.y) << kind;
    }
    kinds
}

#[derive(Default)]
pub struct LayerTraversal {
    pub end: f32,
    pub kinds: u32,
}

impl LayerTraversal {
    pub fn advance(&mut self, origin: Vec3, direction: Vec3, distance: f32, active: u32) {
        if distance < self.end {
            return;
        }
        self.end = f32::MAX;
        self.kinds = 0;
        let mut pending = active & ((1 << LAYERS) - 1);
        while pending != 0 {
            let kind = pending.trailing_zeros() as usize;
            pending &= pending - 1;
            let bounds = layer_bounds(kind);
            let top = intersect_sphere(origin, direction, PLANET_RADIUS + bounds.y + 128.0);
            if top.x >= top.y {
                continue;
            }
            let bottom = intersect_sphere(origin, direction, PLANET_RADIUS + bounds.x - 128.0);
            for half in 0..2 {
                let interval = if bottom.x >= bottom.y {
                    if half == 1 {
                        continue;
                    }
                    top
                } else if half == 0 {
                    vec2(top.x, bottom.x)
                } else {
                    vec2(bottom.y, top.y)
                };
                if interval.x >= interval.y || interval.y <= distance {
                    continue;
                }
                if interval.x <= distance {
                    self.kinds |= 1 << kind;
                    self.end = self.end.min(interval.y);
                } else {
                    self.end = self.end.min(interval.x);
                }
            }
        }
    }
}

pub const fn region_size(kind: usize) -> f32 {
    match cloud_type(kind) {
        0 => 36_000.0,
        1 => 60_000.0,
        2 => 126_000.0,
        _ => 160_000.0,
    }
}

pub const fn cloud_scale(kind: usize) -> f32 {
    match cloud_type(kind) {
        0 => 0.000_12,
        1 => 0.000_069,
        _ => 0.000_036,
    }
}

pub const fn layer_bounds(kind: usize) -> Vec2 {
    match kind {
        0 => vec2(6000.0, 15_000.0),
        1 => vec2(10_000.0, 19_000.0),
        2 => vec2(16_500.0, 25_500.0),
        3 => vec2(23_500.0, 33_000.0),
        4 => vec2(33_000.0, 73_500.0),
        5 => vec2(46_000.0, 73_500.0),
        _ => vec2(super::CLOUD_BOTTOM, super::CLOUD_TOP),
    }
}

// Smooth the interpolation coordinates, not the sampled value. This retains one
// hardware-filtered lookup while removing the planar height slopes of lattice cells.
pub fn region_coordinates(point: Vec3, kind: usize) -> Vec3 {
    let p = point / region_size(kind) + vec3(11.31, 7.17, 3.73) * kind as f32;
    let cell = p.floor();
    let f = p - cell;
    cell + f * f * (Vec3::splat(3.0) - f * 2.0)
}

pub const fn base_variation(kind: usize) -> f32 {
    match cloud_type(kind) {
        0 => 700.0,
        1 => 1200.0,
        _ => 28_000.0,
    }
}

pub fn column_point(point: Vec3) -> Vec3 {
    point.normalize() * PLANET_RADIUS
}

pub fn region_value(seed: f32, seconds: f32) -> f32 {
    seed * 2.4 - 0.65 + (seconds * 0.000_2 + seed * 6.0).sin() * 0.10
}

pub fn column_bounds(base: f32, kind: usize) -> Vec2 {
    let layer = layer_bounds(kind);
    let variation = base_variation(kind);
    let fine_variation = variation * 0.5;
    let bottom = layer.x + base * (variation - fine_variation);
    vec2(
        bottom - 100.0,
        (bottom + fine_variation + 350.0 + column_height(1.0, kind)).min(layer.y),
    )
}

pub fn column_clearance(altitude: f32, base: f32, kind: usize) -> f32 {
    let bounds = column_bounds(base, kind);
    let variation = base_variation(kind);
    // Surface projection is non-expansive above the planet; include the regional height gradient.
    let slope = 1.0 + variation * 2.598_077 / region_size(kind);
    ((bounds.x - altitude).max(altitude - bounds.y) / slope - 16.0).max(0.0)
}

pub fn column_height(seed: f32, kind: usize) -> f32 {
    match cloud_type(kind) {
        0 => 1800.0 + seed * 4700.0,
        1 => 700.0 + seed * 2200.0,
        _ => 350.0 + seed * 1100.0,
    }
}

pub struct CloudFlow {
    pub previous: Vec3,
    pub current: Vec3,
    pub blend: f32,
}

pub fn flow(point: Vec3, seconds: f32, kind: usize) -> CloudFlow {
    let (lifetime, rise, side, sink) = match cloud_type(kind) {
        0 => (900.0, 6.0, 8.0, 4.0),
        1 => (1200.0, 3.0, 24.0, 8.0),
        _ => (1800.0, 1.0, 45.0, 12.0),
    };
    let clock = seconds + kind as f32 * 317.0;
    let epoch = (clock / lifetime).floor();
    let age = clock - epoch * lifetime;
    let up = point.normalize();
    let latitude = up.z;
    let pole = (latitude * latitude).powi(4);
    let velocity =
        (Vec3::Z - up * latitude) * (latitude * side) + up * (rise - (rise + sink) * pole);
    let offset = |index: f32| {
        (vec3(0.618_034, 0.414_213_6, 0.732_050_8) * (index + kind as f32 * 47.0)).fract_gl() * 12.0
    };
    let scale = cloud_scale(kind);
    CloudFlow {
        previous: (point - velocity * (age + lifetime)) * scale + offset(epoch - 1.0),
        current: (point - velocity * age) * scale + offset(epoch),
        blend: (age / lifetime).smoothstep(0.8, 1.0),
    }
}

pub fn layer_density(altitude: f32, coverage: f32, kind: usize, noise: Vec4, filter: f32) -> f32 {
    let bounds = layer_bounds(kind);
    if coverage <= 0.0 || altitude <= bounds.x || altitude >= bounds.y {
        return 0.0;
    }
    let threshold = if cloud_type(kind) == 0 { 0.72 } else { 0.82 } - coverage * 0.30;
    let interior = ((noise.y - threshold + filter) / (1.0 - threshold + filter)).saturate();
    let bottom = bounds.x + noise.z * base_variation(kind) + (1.0 - interior) * 350.0;
    let height = column_height(noise.w, kind);
    let dome = if cloud_type(kind) == 0 {
        interior.sqrt()
    } else {
        interior.sqrt().sqrt()
    };
    let top = (bottom + height * dome).min(bounds.y);
    let base_softness = if cloud_type(kind) == 0 { 180.0 } else { 250.0 };
    let top_softness = if cloud_type(kind) == 0 { 650.0 } else { 350.0 };
    let envelope = altitude.smoothstep(bottom - 100.0, bottom + base_softness)
        * (1.0 - altitude.smoothstep(top - top_softness, top));
    let density = if cloud_type(kind) == 2 { 0.055 } else { 0.22 };
    envelope
        * interior.smoothstep(0.0, 0.25)
        * (density + noise.w * density * 0.5)
        * coverage.smoothstep(0.0, 0.2)
}

#[derive(Clone, Copy, Debug)]
pub struct CloudVolume {
    pub center: Vec3,
    pub up: Vec3,
    pub radius: f32,
    pub height: f32,
}

impl CloudVolume {
    pub fn coordinates(self, point: Vec3) -> Vec3 {
        let delta = point - self.center;
        let vertical = delta.dot(self.up);
        (delta - self.up * vertical) / self.radius + self.up * (vertical / self.height)
    }

    pub fn ray_interval(self, origin: Vec3, direction: Vec3) -> Vec2 {
        let local_origin = self.coordinates(origin);
        let vertical = direction.dot(self.up);
        let local_direction =
            (direction - self.up * vertical) / self.radius + self.up * (vertical / self.height);
        let height = local_origin.dot(self.up);
        let along = local_direction.dot(self.up);
        let radial = local_origin - self.up * height;
        let across = local_direction - self.up * along;
        let tower = intersect_sphere(
            radial / 0.55 + self.up * height,
            across / 0.55 + self.up * along,
            1.0,
        );
        let anvil = intersect_sphere(
            radial / 0.72 + self.up * ((height - 0.4) / 0.55),
            across / 0.72 + self.up * (along / 0.55),
            1.0,
        );
        let tower = if tower.x < tower.y && tower.y > 0.0 {
            tower
        } else {
            vec2(f32::MAX, -f32::MAX)
        };
        let anvil = if anvil.x < anvil.y && anvil.y > 0.0 {
            anvil
        } else {
            vec2(f32::MAX, -f32::MAX)
        };
        vec2(tower.x.min(anvil.x), tower.y.max(anvil.y))
    }
}

pub fn storm_at(point: Vec3, seconds: f32) -> Option<CloudVolume> {
    let size = region_size(STORM);
    let cell = (point / size).floor().as_ivec3();
    let salt = 3 * 1013;
    let seed = vec3(
        lattice(cell, 65_521, 0x81 + salt),
        lattice(cell, 65_521, 0x97 + salt),
        lattice(cell, 65_521, 0xb3 + salt),
    );
    let drift = vec3(
        (seconds * 0.000_02 + seed.x * 6.0).sin(),
        (seconds * 0.000_017 + seed.y * 6.0).sin(),
        (seconds * 0.000_013 + seed.z * 6.0).sin(),
    ) * 0.04;
    let center = (cell.as_vec3() + Vec3::splat(0.3) + seed * 0.4 + drift) * size;
    let altitude = center.length() - PLANET_RADIUS;
    if altitude < 28_000.0 || altitude > 46_000.0 {
        return None;
    }
    Some(CloudVolume {
        center,
        up: center.normalize(),
        radius: size * (0.16 + seed.y * 0.08),
        height: (size * 0.2)
            .min(altitude - super::CLOUD_BOTTOM - 500.0)
            .min(super::CLOUD_TOP - altitude - 500.0),
    })
}

pub fn ray_event(point: Vec3, direction: Vec3, seconds: f32, kind: usize, seed: f32) -> Vec2 {
    let bounds = layer_bounds(kind);
    let altitude = point.length() - PLANET_RADIUS;
    if altitude < bounds.x - 128.0 {
        let interval = intersect_sphere(point, direction, PLANET_RADIUS + bounds.x - 128.0);
        return Vec2::splat((interval.y - 16.0).max(0.0));
    }
    if altitude > bounds.y + 128.0 {
        let interval = intersect_sphere(point, direction, PLANET_RADIUS + bounds.y + 128.0);
        return Vec2::splat(if interval.y > 0.0 && interval.x < interval.y {
            (interval.x - 16.0).max(0.0)
        } else {
            f32::MAX
        });
    }
    if kind != STORM {
        let value = region_value(seed, seconds);
        // Smoothed trilinear seed gradients are bounded by 1.5*sqrt(3)/cell;
        // the coverage remapping has slope at most 3.
        let clearance = -value * region_size(kind) / 7.794_229 - region_size(kind) * 0.01;
        return if clearance > 16.0 {
            Vec2::splat(clearance)
        } else {
            vec2(0.0, region_size(kind) * 0.125)
        };
    }
    storm_ray_event(point, direction, storm_at(point + direction * 8.0, seconds))
}

pub fn storm_ray_event(point: Vec3, direction: Vec3, storm: Option<CloudVolume>) -> Vec2 {
    // Forward cells avoid zero-length exits at negative-direction crossings.
    let cell_point = point + direction * 8.0;
    let size = region_size(STORM);
    let cell = (cell_point / size).floor();
    let edge = (cell + Vec3::select(direction.cmpgt(Vec3::ZERO), Vec3::ONE, Vec3::ZERO)) * size;
    let exit = Vec3::select(
        direction.abs().cmpgt(Vec3::splat(1e-10)),
        (edge - point) / direction,
        Vec3::splat(f32::MAX),
    )
    .min_element()
    .max(0.0);
    if let Some(cloud) = storm {
        let interval = cloud.ray_interval(point, direction);
        if interval.y > 0.0 && interval.x < interval.y {
            return vec2(
                interval.x.max(0.0).min(exit),
                (if interval.x > 0.0 {
                    interval.x
                } else {
                    interval.y
                })
                .min(exit),
            );
        }
    }
    Vec2::splat(exit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::climate::weather_at;

    #[test]
    fn cached_shell_traversal_keeps_overlapping_layers_and_both_sides_of_the_planet() {
        let origin = Vec3::Z * (PLANET_RADIUS + 100_000.0);
        for direction in [-Vec3::Z, Vec3::X, vec3(1.0, 0.0, -0.1).normalize()] {
            let mut traversal = LayerTraversal::default();
            for index in 0..4000 {
                let distance = index as f32 * 1000.0;
                traversal.advance(origin, direction, distance, ALL_KINDS);
                let altitude = (origin + direction * distance).length() - PLANET_RADIUS;
                let required = kinds_at_altitude(altitude);
                assert_eq!(traversal.kinds & required, required);
                assert!(traversal.end > distance);
                for kind in 0..LAYERS {
                    if traversal.kinds & (1 << kind) != 0 {
                        let bounds = layer_bounds(kind);
                        assert!(altitude >= bounds.x - 130.0 && altitude <= bounds.y + 130.0);
                    }
                }
            }
        }
        assert!(kinds_at_altitude(12_000.0).count_ones() > 1);
    }

    #[test]
    fn cloud_columns_have_rounded_tops_and_type_specific_depth() {
        let up = vec3(1.0, 2.0, 3.0).normalize();
        let a = column_point(up * (PLANET_RADIUS + 7000.0));
        let b = column_point(up * (PLANET_RADIUS + 70_000.0));
        assert!(a.distance(b) < 0.2);
        let core = Vec4::new(0.8, 0.95, 0.0, 1.0);
        let edge = Vec4::new(0.8, 0.65, 0.0, 1.0);
        assert!(layer_density(11_000.0, 1.0, 0, core, 0.0) > 0.0);
        assert!(layer_density(11_000.0, 1.0, 0, edge, 0.0).abs() < 1e-6);
        assert!(layer_density(18_000.0, 1.0, 2, core, 0.0) > 0.0);
        assert!(layer_density(24_000.0, 1.0, 2, core, 0.0).abs() < 1e-6);
        assert!(layer_density(34_000.0, 1.0, 4, core, 0.0) > 0.0);
        assert!(layer_density(40_000.0, 1.0, 4, core, 0.0).abs() < 1e-6);
    }

    #[test]
    fn cloud_formations_remain_continuous_when_their_flow_epoch_changes() {
        use crate::scene::noise::fbm;
        let point = vec3(1.0, 0.3, 0.7).normalize() * (PLANET_RADIUS + 25_000.0);
        for (kind, lifetime) in [
            (0, 900.0),
            (1, 900.0),
            (2, 1200.0),
            (3, 1200.0),
            (4, 1800.0),
            (5, 1800.0),
        ] {
            for epoch in [1.0, 2.0, 20.0] {
                let boundary = epoch * lifetime - kind as f32 * 317.0;
                let sample = |seconds| {
                    let flow = flow(point, seconds, kind);
                    fbm(flow.previous, 4, 17).lerp(fbm(flow.current, 4, 17), flow.blend)
                };
                assert!((sample(boundary - 0.01) - sample(boundary + 0.01)).abs() < 0.002);
            }
        }
    }

    #[test]
    fn height_clearance_cannot_cross_a_cloud_column() {
        let mut skips = 0;
        for kind in 0..LAYERS {
            for index in 0..64 {
                let up = vec3(1.0, index as f32 * 0.13 - 4.0, 0.7).normalize();
                let layer = layer_bounds(kind);
                let altitude = layer.x + (layer.y - layer.x) * (index as f32 / 64.0);
                let point = up * (PLANET_RADIUS + altitude);
                let base = weather_at(region_coordinates(column_point(point), kind)).w;
                let clearance = column_clearance(altitude, base, kind);
                if clearance <= 0.0 {
                    continue;
                }
                skips += 1;
                for direction in [up, -up, Vec3::X, Vec3::Y, Vec3::Z] {
                    for step in 0..=16 {
                        let sample = point + direction * clearance * (step as f32 / 16.0);
                        let base = weather_at(region_coordinates(column_point(sample), kind)).w;
                        let bounds = column_bounds(base, kind);
                        let height = sample.length() - PLANET_RADIUS;
                        assert!(height < bounds.x || height > bounds.y);
                    }
                }
            }
        }
        assert!(skips > 32);
    }

    #[test]
    fn regional_clearance_cannot_cross_into_cloudy_weather() {
        let mut skips = 0;
        for kind in 0..LAYERS {
            let size = region_size(kind);
            for x in -12..=12 {
                for y in -12..=12 {
                    let point = vec3(
                        x as f32 * size * 0.23,
                        y as f32 * size * 0.31,
                        PLANET_RADIUS,
                    )
                    .normalize()
                        * (PLANET_RADIUS + layer_bounds(kind).x.midpoint(layer_bounds(kind).y));
                    for seconds in [0.0, 20_000.0] {
                        let seed = weather_at(region_coordinates(column_point(point), kind))
                            [cloud_type(kind)];
                        for direction in
                            [Vec3::X, -Vec3::Y, Vec3::Z, vec3(1.0, 1.0, 1.0).normalize()]
                        {
                            let query = ray_event(point, direction, seconds, kind, seed);
                            if query.x <= 0.0 {
                                continue;
                            }
                            skips += 1;
                            for step in 0..=8 {
                                let sample = point + direction * query.x * (step as f32 / 8.0);
                                assert!(
                                    region_value(
                                        weather_at(region_coordinates(column_point(sample), kind))
                                            [cloud_type(kind)],
                                        seconds
                                    ) <= 0.0001,
                                    "weather clearance skipped a cloudy region"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(skips > 100, "exercise empty-region traversal");
    }

    #[test]
    fn layers_have_varied_bases_and_vanish_outside_their_bounds() {
        for kind in 0..LAYERS {
            let bounds = layer_bounds(kind);
            let low = Vec4::new(0.8, 0.8, 0.0, 1.0);
            let high = Vec4::new(0.8, 0.8, 1.0, 1.0);
            let offset = match cloud_type(kind) {
                0 => 400.0,
                1 => 600.0,
                _ => 700.0,
            };
            assert!(layer_density(bounds.x + offset, 1.0, kind, low, 0.0) > 0.0);
            assert!(layer_density(bounds.x + offset, 1.0, kind, high, 0.0).abs() < 1e-6);
            for noise in [low, high, Vec4::ZERO, Vec4::ONE] {
                for altitude in [bounds.x - 1000.0, bounds.y + 1000.0] {
                    assert!(layer_density(altitude, 1.0, kind, noise, 0.0).abs() < 1e-6);
                }
                assert!(layer_density(bounds.x + 8000.0, 0.0, kind, noise, 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn weather_and_storm_traversal_remain_conservative() {
        let mut storms = 0;
        for kind in 0..=STORM {
            let size = region_size(kind);
            for x in -7..=7 {
                for y in -7..=7 {
                    for z in -5..=5 {
                        let point = Vec3::Z * (PLANET_RADIUS + 40_000.0)
                            + vec3(x as f32, y as f32, z as f32) * size;
                        for seconds in [0.0, 40_000.0, 180_000.0] {
                            if kind != STORM {
                                let altitude = point.length() - PLANET_RADIUS;
                                let bounds = layer_bounds(kind);
                                let seed =
                                    weather_at(region_coordinates(column_point(point), kind))
                                        [cloud_type(kind)];
                                if altitude > bounds.x
                                    && altitude < bounds.y
                                    && region_value(seed, seconds) > 0.0
                                {
                                    assert!(
                                        ray_event(point, Vec3::X, seconds, kind, seed).x.abs()
                                            < 0.1
                                    );
                                }
                                continue;
                            }
                            let Some(cloud) = storm_at(point, seconds) else {
                                continue;
                            };
                            storms += 1;
                            let cell_min = (point / size).floor() * size;
                            let extent = Vec3::splat(cloud.radius.max(cloud.height));
                            assert!((cloud.center - extent).cmpgt(cell_min).all());
                            assert!((cloud.center + extent).cmplt(cell_min + size).all());
                            for direction in [cloud.up, Vec3::X, Vec3::Y, Vec3::Z, -Vec3::Z] {
                                let origin = cloud.center - direction * extent.x * 2.0;
                                let interval = cloud.ray_interval(origin, direction);
                                assert!(
                                    ray_event(origin, direction, seconds, kind, 0.0).x
                                        <= interval.x + 1.0
                                );
                                let inside = ray_event(cloud.center, direction, seconds, kind, 0.0);
                                assert!(inside.x.abs() < 0.1);
                                assert!(
                                    inside.y > 0.0
                                        && inside.y
                                            <= cloud.ray_interval(cloud.center, direction).y + 1.0
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(storms > 20, "storm geometry must be exercised");
    }
}
