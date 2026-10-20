use crate::scene::life::CreatureInstance;
use core::array::from_fn;
use spirv_std::glam::{Vec3, vec3};

const STEP: f32 = 1.0 / 240.0;

pub struct Cloth {
    points: [Vec3; 36],
    previous: [Vec3; 36],
    rest: [Vec3; 36],
    constraints: Vec<(usize, usize, f32, f32)>,
    time: f32,
    remainder: f32,
}

impl Cloth {
    pub fn new(anchor: Vec3, length: f32, width: f32) -> Self {
        let rest = from_fn(|i| {
            let t = (i / 3) as f32 / 11.0;
            let phase = anchor.x * 0.45;
            let curl = t * 3.8;
            let twist = t * 4.5 + (t * 7.0 + phase).sin() * t * 0.6;
            let across = vec3(twist.cos(), twist.sin(), 0.0);
            anchor
                + across * ((i % 3) as f32 - 1.0) * width * (1.0 - t.powi(2) * 0.98)
                + vec3(
                    ((curl + phase).sin() - phase.sin()) * length * 0.18,
                    -length * t * 0.42 + curl.sin() * length * 0.07,
                    length * t * 0.78 + curl.sin() * length * 0.12,
                )
        });
        let mut constraints = Vec::new();
        for row in 0..12 {
            for col in 0usize..3 {
                let a = row * 3 + col;
                for (dr, dc, compliance) in [
                    (0, 1, 1e-7),
                    (1, 0, 1e-7),
                    (1, 1, 2e-6),
                    (1, -1, 2e-6),
                    (2, 0, 8e-4),
                ] {
                    let r = row + dr;
                    if let Some(column) = col.checked_add_signed(dc)
                        && r < 12
                        && column < 3
                    {
                        let b = r * 3 + column;
                        constraints.push((a, b, rest[a].distance(rest[b]), compliance));
                    }
                }
            }
        }
        Self {
            points: rest,
            previous: rest,
            rest,
            constraints,
            time: 0.0,
            remainder: 0.0,
        }
    }

    pub fn advance(&mut self, seconds: f32) {
        self.remainder += seconds.max(0.0);
        let steps = (self.remainder / STEP) as usize;
        self.remainder -= steps as f32 * STEP;
        for _ in 0..steps {
            self.time += STEP;
            for row in 1..12 {
                let normal = (self.points[row * 3 + 2] - self.points[row * 3])
                    .cross(self.points[row * 3 + 1] - self.points[(row - 1) * 3 + 1])
                    .normalize_or_zero();
                for col in 0..3 {
                    let i = row * 3 + col;
                    let velocity = (self.points[i] - self.previous[i]) * 0.997;
                    self.previous[i] = self.points[i];
                    let point = self.points[i];
                    let gust = self.time * 5.5 - point.z * 0.9 + point.x * 1.8;
                    let wind = vec3(4.0 * gust.sin(), 0.6 + 6.0 * (gust * 1.3).cos(), 7.0)
                        - velocity / STEP;
                    let pressure = normal * wind.dot(normal) * wind.dot(normal).abs() * 0.12;
                    self.points[i] +=
                        velocity + (vec3(0.0, -3.0, 0.0) + pressure + wind * 0.08) * STEP * STEP;
                }
            }
            for &(a, b, length, compliance) in &self.constraints {
                let delta = self.points[b] - self.points[a];
                let distance = delta.length().max(1e-6);
                let wa = if a < 3 { 0.0 } else { 1.0 };
                let wb = if b < 3 { 0.0 } else { 1.0 };
                let correction = delta / distance
                    * ((distance - length) / (wa + wb + compliance / (STEP * STEP)));
                self.points[a] += correction * wa;
                self.points[b] -= correction * wb;
            }
            self.points[..3].copy_from_slice(&self.rest[..3]);
        }
    }

    pub fn write(&self, c: &mut CreatureInstance, ribbon: usize) {
        c.previous_cloth[ribbon] = c.cloth[ribbon];
        for row in 0..12 {
            for col in 0..3 {
                c.cloth[ribbon][row][col] = self.points[row * 3 + col].extend(0.0);
                c.opaque_bounds = c
                    .opaque_bounds
                    .truncate()
                    .max(self.points[row * 3 + col].abs() + Vec3::splat(0.1))
                    .extend(0.0);
            }
        }
    }
}

pub fn prepare(c: &mut CreatureInstance, ribbon: usize, anchor: Vec3, length: f32, width: f32) {
    let mut cloth = Cloth::new(anchor, length, width);
    cloth.advance(2.0 + c.animation.z.max(0.0));
    cloth.write(c, ribbon);
    c.previous_cloth[ribbon] = c.cloth[ribbon];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::FrameUniform;
    use core::mem::size_of;

    #[test]
    fn fixed_steps_preserve_attachments_and_material_lengths() {
        let mut cloth = Cloth::new(Vec3::ZERO, 12.0, 1.5);
        cloth.advance(4.0);
        assert_eq!(&cloth.points[..3], &cloth.rest[..3]);
        for &(a, b, length, compliance) in &cloth.constraints {
            assert!(cloth.points[a].is_finite());
            if compliance < 1e-6 {
                assert!((cloth.points[a].distance(cloth.points[b]) - length).abs() < length * 0.1);
            }
        }
        assert!(cloth.points[34].distance(cloth.rest[34]) > 0.1);
        let mut partitioned = Cloth::new(Vec3::ZERO, 12.0, 1.5);
        for _ in 0..8 {
            partitioned.advance(0.5);
        }
        assert_eq!(
            cloth.points, partitioned.points,
            "cloth depends on frame timing"
        );
        assert!(
            size_of::<FrameUniform>() <= 16384,
            "scene exceeds portable uniform limit"
        );
    }
}
