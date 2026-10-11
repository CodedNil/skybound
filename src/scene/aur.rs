use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, vec2, vec3};
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

pub const EJECTION_CELL: f32 = 8000.0;

pub fn ejection(cell: Vec2, time: f32) -> [Vec4; 2] {
    let mut hash = (cell.xyx() * vec3(0.1031, 0.1030, 0.0973)).fract_gl();
    hash += hash.dot(hash.yzx() + 33.33);
    let seed = ((hash.x + hash.y) * hash.z).fract_gl();
    let period = 12.0 + seed * 9.0;
    let phase = ((time / period + seed).fract_gl() / 0.35).min(1.0);
    let envelope = phase.smoothstep(0.0, 0.12) * (1.0 - phase.smoothstep(0.78, 1.0));
    let center =
        (cell + vec2(0.3 + seed * 0.4, 0.3 + (seed * 17.37).fract_gl() * 0.4)) * EJECTION_CELL;
    let angle = seed * TAU;
    [
        center.extend(phase).extend(envelope),
        Vec4::new(
            angle.cos(),
            angle.sin(),
            1200.0 + seed * 600.0,
            (650.0 + seed * 700.0) * (0.45 + 0.55 * (phase * PI).sin()),
        ),
    ]
}

pub fn ejection_distance(position: Vec3, event: [Vec4; 2]) -> Vec2 {
    let delta = position.xy() - event[0].truncate().truncate();
    let axis = event[1].truncate().truncate();
    let along = delta.dot(axis);
    let across = delta.dot(vec2(-axis.y, axis.x));
    let vertical = position.z - (event[1].w - event[1].z);
    let radial = vec2(along, vertical).length() - event[1].z;
    let distance = vec2(radial, across).length();
    let progress = vertical.atan2(along) / PI;
    vec2(distance, progress.clamp(0.0, 1.0))
}

pub fn ejection_field(position: Vec3, time: f32) -> Vec3 {
    let cell = (position.xy() / EJECTION_CELL - Vec2::splat(0.5)).floor();
    let mut nearest = vec3(1.0e10, 0.0, 0.0);
    for index in 0..4 {
        let event = ejection(cell + vec2((index / 2) as f32, (index % 2) as f32), time);
        if event[0].w < 0.01 {
            continue;
        }
        let curve = ejection_distance(position, event);
        if curve.x < nearest.x {
            nearest = curve.extend(event[0].w);
        }
    }
    nearest
}

pub fn ejection_clearance(position: Vec3, time: f32) -> f32 {
    ((ejection_field(position, time).x - 625.0) * 0.5).clamp(0.0, 2000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
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
                    let point = vec3(x as f32 * 1300.0, y as f32 * 1700.0, 650.0);
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
                    if distance < 1875.0 {
                        assert!((ejection_field(point, time).x - distance).abs() < 0.01);
                    }
                }
            }
        }
    }
}
