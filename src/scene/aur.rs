use core::f32::consts::{PI, TAU};
#[cfg(not(target_arch = "spirv"))]
pub mod lightning;
use spirv_std::glam::{
    FloatExt, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const CURRENT: Vec3 = vec3(35.0, -20.0, 0.0);

pub fn curl(mut pos: Vec2, time: f32) -> Vec2 {
    let mut direction = vec2(0.6, 0.8);
    let mut frequency = 1.6;
    for octave in 0..4 {
        let perpendicular = vec2(-direction.y, direction.x);
        let phase = frequency * pos.dot(perpendicular)
            + time * (0.18 + octave as f32 * 0.035)
            + octave as f32;
        pos += direction * (0.6 * phase.sin() / frequency);
        direction = vec2(
            direction.x * -0.737 - direction.y * 0.676,
            direction.x * 0.676 + direction.y * -0.737,
        );
        frequency *= 1.4;
    }
    pos
}

pub fn ambient(pos: Vec3, time: f32) -> Vec3 {
    let fade = (1.0 - pos.z.max(0.0) / 6000.0).clamp(0.0, 1.0);
    let flow = curl((pos - CURRENT * time).xy() * 0.001, time);
    let pulse = 0.8 + 0.2 * (time * 1.2 + flow.x * 7.0).sin();
    vec3(0.035, 0.008, 0.065) * fade * pulse
}

pub const EJECTION_CELL: f32 = 40_000.0;
pub const EJECTION_WIDTH: f32 = 1050.0;
pub const EJECTION_REACH: f32 = 36_000.0;
pub const EJECTION_HEIGHT: f32 = 12_150.0;
pub const EJECTION_GRID_HALF: i32 = (PI * super::PLANET_RADIUS / EJECTION_CELL) as i32 + 2;
pub const EJECTION_GRID_WIDTH: u32 = EJECTION_GRID_HALF as u32 * 2;
pub const EJECTION_GRID_HEIGHT: u32 = EJECTION_GRID_HALF as u32 + 1;

fn eruption_hash(cell: Vec2) -> Vec3 {
    let mut hash = (cell.xyx() * vec3(0.1031, 0.1030, 0.0973)).fract_gl();
    hash += hash.dot(hash.yzx() + 33.33);
    ((hash.xxy() + hash.yzz()) * hash.zyx()).fract_gl()
}

pub fn ejection(cell: Vec2, time: f32) -> [Vec4; 2] {
    let seed = eruption_hash(cell + vec2(19.7, 63.1));
    let period = 55.0 + seed.x * 45.0;
    let cycle = time / period + seed.y;
    let phase = (cycle.fract_gl() * period / (5.0 + seed.z * 3.0)).min(1.0);
    if phase >= 1.0 {
        return [Vec4::ZERO; 2];
    }
    let random = eruption_hash(cell + vec2(37.1, 91.7) * (cycle.floor() + 1.0));
    let center = (cell + random.xy()) * EJECTION_CELL;
    let angle = random.z * TAU;
    let radius = 10_800.0 + seed.x * 5400.0;
    [
        center.extend(phase).extend(1.0),
        Vec4::new(
            angle.cos(),
            angle.sin(),
            radius,
            (5850.0 + seed.z * 6300.0).min(radius - 2700.0),
        ),
    ]
}

pub fn ejection_fill(progress: f32, phase: f32) -> f32 {
    let head = phase / 0.35;
    let tail = (phase - 0.60) / 0.40;
    progress.smoothstep(tail - 0.025, tail + 0.025)
        * (1.0 - progress.smoothstep(head - 0.025, head + 0.025))
}

fn ejection_coordinates(position: Vec3, event: [Vec4; 2]) -> Vec3 {
    let delta = position.xy() - event[0].truncate().truncate();
    let axis = event[1].truncate().truncate();
    let along = delta.dot(axis);
    let across = delta.dot(vec2(-axis.y, axis.x));
    let vertical = position.z - (event[1].w - event[1].z);
    vec3(along, vertical, across)
}

pub fn ejection_distance(position: Vec3, event: [Vec4; 2]) -> Vec2 {
    let local = ejection_coordinates(position, event);
    let radial = local.xy().length() - event[1].z;
    let distance = vec2(radial, local.z).length();
    let progress = local.y.atan2(local.x) / PI;
    vec2(distance, progress.clamp(0.0, 1.0))
}

pub fn ejection_field(position: Vec3, time: f32) -> Vec3 {
    ejection_field_with(position, 511, f32::MAX, |cell| ejection(cell, time))
}

pub(crate) fn ejection_field_with(
    position: Vec3,
    mut active: u32,
    limit: f32,
    event: impl Fn(Vec2) -> [Vec4; 2],
) -> Vec3 {
    if active == 0 {
        return vec3(1.0e10, 0.0, 0.0);
    }
    let cell = (position.xy() / EJECTION_CELL).floor();
    let mut nearest = Vec4::new(1.0e20, 0.0, 0.0, 1.0);
    while active != 0 {
        let index = active.trailing_zeros();
        active &= active - 1;
        let neighbour = cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
        let outside =
            ((position.xy() / EJECTION_CELL - neighbour - 0.5).abs() - 0.5).max(Vec2::ZERO);
        if outside.max_element() * EJECTION_CELL > EJECTION_REACH {
            continue;
        }
        let event = event(neighbour);
        if event[0].w < 0.01 {
            continue;
        }
        if limit < 1.0e9 {
            let delta = position - event[0].xy().extend(event[1].w - event[1].z);
            let extent = (event[1].xy().abs() * event[1].z).extend(event[1].z) + Vec3::splat(limit);
            if delta.abs().cmpgt(extent).any() {
                continue;
            }
        }
        let local = ejection_coordinates(position, event);
        let radial = local.xy().length() - event[1].z;
        let distance_squared = radial * radial + local.z * local.z;
        if distance_squared < nearest.x {
            nearest = Vec4::new(distance_squared, local.x, local.y, event[0].z);
        }
    }
    let progress = (nearest.z.atan2(nearest.y) / PI).clamp(0.0, 1.0);
    vec3(
        nearest.x.sqrt(),
        progress,
        ejection_fill(progress, nearest.w),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::ivec2;
    #[test]
    fn eruptions_fill_then_drain_from_the_same_end() {
        assert!(ejection_fill(0.25, 0.15) > 0.99);
        assert!(ejection_fill(0.85, 0.15) < 0.01);
        assert!(ejection_fill(0.25, 0.5) > 0.99 && ejection_fill(0.85, 0.5) > 0.99);
        assert!(ejection_fill(0.25, 0.85) < 0.01);
        assert!(ejection_fill(0.85, 0.85) > 0.99);
        assert!(ejection_fill(0.85, 1.0) < 0.01);
    }

    #[test]
    fn eruption_distance_is_continuous_and_tracks_position_along_curve() {
        let event = [Vec4::ZERO, Vec4::new(1.0, 0.0, 2000.0, 1000.0)];
        let top = ejection_distance(vec3(0.0, 0.0, 1000.0), event);
        assert!(top.x < 0.001 && (top.y - 0.5).abs() < 0.001);
        let side = ejection_distance(vec3(2000.0, 0.0, -1000.0), event);
        assert!(side.x < 0.001 && side.y < 0.001);
        assert!((ejection_distance(vec3(0.0, 100.0, 1000.0), event).x - 100.0).abs() < 0.001);
    }
    #[test]
    fn neighbouring_cells_preserve_every_visible_eruption_across_boundaries() {
        for x in -8..8 {
            for y in -8..8 {
                for time in [2.0, 10.0, 20.0] {
                    let point = vec3(
                        x as f32 * EJECTION_CELL * 0.37,
                        y as f32 * EJECTION_CELL * 0.53,
                        650.0,
                    );
                    let cell = (point.xy() / EJECTION_CELL).floor();
                    let mut distance: f32 = 1.0e10;
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            let event = ejection(cell + vec2(dx as f32, dy as f32), time);
                            if event[0].w >= 0.01 {
                                distance = distance.min(ejection_distance(point, event).x);
                            }
                        }
                    }
                    if distance < EJECTION_WIDTH * 3.0 {
                        assert!((ejection_field(point, time).x - distance).abs() < 0.01);
                    }
                }
            }
        }
    }

    #[test]
    fn cached_arc_candidates_cover_the_planet_and_preserve_the_full_field() {
        let radius = super::super::PLANET_RADIUS;
        for longitude in [-PI, -PI * 0.5, 0.0, PI * 0.5, PI] {
            for colatitude in [0.0, 0.01, PI * 0.5, PI - 0.01, PI] {
                for time in [0.0, 10.0, 120.0, 50_000.0] {
                    let point = vec3(longitude * radius, colatitude * radius, 650.0);
                    let cell = (point.xy() / EJECTION_CELL).floor();
                    let mut active = 0;
                    for index in 0..9 {
                        let neighbour =
                            cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
                        let pixel = neighbour.as_ivec2() + ivec2(EJECTION_GRID_HALF, 1);
                        assert!(pixel.x >= 0 && pixel.x < EJECTION_GRID_WIDTH.cast_signed());
                        assert!(pixel.y >= 0 && pixel.y < EJECTION_GRID_HEIGHT.cast_signed());
                        active |= u32::from(ejection(neighbour, time)[0].w > 0.0) << index;
                    }
                    assert_eq!(
                        ejection_field(point, time),
                        ejection_field_with(point, active, f32::MAX, |cell| ejection(cell, time))
                    );
                    let limit = EJECTION_WIDTH + 4000.0;
                    let bounded =
                        ejection_field_with(point, active, limit, |cell| ejection(cell, time));
                    let expected = ejection_field(point, time);
                    assert!((bounded.x.min(limit) - expected.x.min(limit)).abs() < 0.01);
                }
            }
        }
    }
}
