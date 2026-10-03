use crate::scene::{geometry::intersect_sphere, life::CreatureInstance};
use spirv_std::glam::{Quat, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub light: Vec3,
    pub max_distance: f32,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub color_depth: Vec4,
    pub normal: Vec3,
    pub previous_position: Vec3,
}

impl Hit {
    pub const MISS: Self = Self {
        color_depth: Vec4::new(0.0, 0.0, 0.0, 1e6),
        normal: Vec3::ZERO,
        previous_position: Vec3::ZERO,
    };
    pub fn is_hit(self) -> bool {
        self.color_depth.w < Self::MISS.color_depth.w
    }
    pub fn distance(self) -> f32 {
        self.color_depth.w
    }
    pub fn color(self) -> Vec3 {
        self.color_depth.truncate()
    }
}

pub struct Surface {
    pub distance: f32,
    pub material: u32,
}
impl Surface {
    pub const fn new(distance: f32, material: u32) -> Self {
        Self { distance, material }
    }
    pub fn join(&mut self, distance: f32, material: u32) {
        if distance < self.distance {
            *self = Self::new(distance, material);
        }
    }
}

pub fn estimate_normal<F: FnMut(Vec3) -> f32>(point: Vec3, mut sdf: F) -> Vec3 {
    let offset_a = vec3(1.0, -1.0, -1.0);
    let offset_b = vec3(-1.0, -1.0, 1.0);
    let offset_c = vec3(-1.0, 1.0, -1.0);
    let offset_d = Vec3::ONE;
    let normal = offset_a * sdf(point + offset_a * 0.1)
        + offset_b * sdf(point + offset_b * 0.1)
        + offset_c * sdf(point + offset_c * 0.1)
        + offset_d * sdf(point + offset_d * 0.1);
    if normal.length_squared() > 1e-10 {
        normal.normalize()
    } else {
        Vec3::Z
    }
}

pub fn trace<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    pigment: Vec4,
    membrane: Vec3,
) -> Hit {
    let interval = intersect_sphere(
        ray.origin - c.position.truncate(),
        ray.direction,
        c.position.w,
    );
    let end = interval.y.min(ray.max_distance);
    let mut t = interval.x.max(0.0);
    if t >= end {
        return Hit::MISS;
    }
    let rotation = Quat::from_xyzw(c.rotation.x, c.rotation.y, c.rotation.z, c.rotation.w);
    let origin = rotation.conjugate() * (ray.origin - c.position.truncate());
    let direction = rotation.conjugate() * ray.direction;
    for _ in 0..192 {
        if t >= end {
            break;
        }
        let p = origin + direction * t;
        let sample = surface(p, c);
        if sample.distance < 0.045 {
            let normal = rotation * estimate_normal(p, |q| surface(q, c).distance);
            let diffuse = normal.dot(ray.light).max(0.0);
            let rim = (1.0 - normal.dot(-ray.direction).abs()).powi(3);
            let color = (match sample.material {
                0 => pigment.truncate(),
                3 => vec3(0.015, 0.04, 0.09),
                _ => membrane,
            }) * (0.35 + diffuse * 0.65 + pigment.w)
                + vec3(0.2, 0.45, 0.7) * rim * 0.4;
            return Hit {
                color_depth: color.extend(t),
                normal,
                previous_position: Quat::from_xyzw(
                    c.previous_rotation.x,
                    c.previous_rotation.y,
                    c.previous_rotation.z,
                    c.previous_rotation.w,
                ) * p
                    + c.previous_position.truncate(),
            };
        }
        t += (sample.distance * 0.85).max(0.015);
    }
    Hit::MISS
}
