use core::f32::consts::PI;
use core::mem::swap;

use spirv_std::glam::{FloatExt, Quat, Vec2, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Kind, development};

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub light: Vec3,
    pub ambient: Vec3,
    pub radiance: Vec3,
    pub max_distance: f32,
    pub footprint: f32,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub color_depth: Vec4,
    pub normal: Vec3,
    pub previous_position: Vec3,
    pub transmittance: Vec3,
    pub part: u32,
}
impl Hit {
    pub const MISS: Self = Self {
        color_depth: Vec4::new(0.0, 0.0, 0.0, 1e6),
        normal: Vec3::ZERO,
        previous_position: Vec3::ZERO,
        transmittance: Vec3::ONE,
        part: 0,
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

#[derive(Clone, Copy)]
pub struct TraceResult {
    pub opaque: Hit,
    pub layers: [Hit; 16],
    pub count: u32,
}
impl TraceResult {
    pub const MISS: Self = Self {
        opaque: Hit::MISS,
        layers: [Hit::MISS; 16],
        count: 0,
    };
    pub const fn opaque(hit: Hit) -> Self {
        Self {
            opaque: hit,
            ..Self::MISS
        }
    }
    pub fn front(self) -> Hit {
        if self.count > 0 && self.layers[0].distance() < self.opaque.distance() {
            self.layers[0]
        } else {
            self.opaque
        }
    }
    pub fn is_hit(self) -> bool {
        self.front().is_hit()
    }
    pub fn distance(self) -> f32 {
        self.front().distance()
    }
    pub fn composite(self, background: Vec3) -> Vec3 {
        let mut color = if self.opaque.is_hit() {
            self.opaque.color()
        } else {
            background
        };
        for i in (0..self.layers.len()).rev() {
            if i < self.count as usize && self.layers[i].distance() < self.opaque.distance() {
                color = self.layers[i].color() + color * self.layers[i].transmittance;
            }
        }
        color
    }
    pub fn add(&mut self, hit: Hit) {
        if !hit.is_hit() {
            return;
        }
        let mut incoming = hit;
        for i in 0..self.layers.len() {
            if incoming.distance() < self.layers[i].distance() {
                swap(&mut self.layers[i], &mut incoming);
            }
        }
        self.count = (self.count + 1).min(self.layers.len() as u32);
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

#[derive(Clone, Copy)]
pub struct Material {
    pub color: Vec3,
    pub roughness: f32,
    pub metal: f32,
    pub emission: f32,
    pub transmission: Vec3,
    pub ior: f32,
    pub translucency: f32,
}
impl Material {
    pub fn transmittance(self, cosine: f32) -> Vec3 {
        let dielectric = ((self.ior - 1.0) / (self.ior + 1.0)).powi(2);
        let fresnel = dielectric + (1.0 - dielectric) * (1.0 - cosine.clamp(0.0, 1.0)).powi(5);
        self.transmission * (1.0 - fresnel)
    }
    pub const fn skin(color: Vec3, roughness: f32) -> Self {
        Self {
            color,
            roughness,
            metal: 0.0,
            emission: 0.0,
            transmission: Vec3::ZERO,
            ior: 1.5,
            translucency: 0.0,
        }
    }
    pub fn shade(self, normal: Vec3, ray: Ray) -> Vec3 {
        let view = -ray.direction;
        let light = normalize(ray.light);
        let halfway = normalize(light + view);
        let nl = normal.dot(light).max(0.0);
        let backlight = (-normal.dot(light)).max(0.0);
        let nv = normal.dot(view).abs().max(0.0001);
        let nh = normal.dot(halfway).max(0.0);
        let vh = view.dot(halfway).max(0.0);
        let a2 = self.roughness.clamp(0.045, 1.0).powi(4);
        let distribution = a2 / (PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2)).max(1e-12);
        let visibility = 0.5
            / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt())
                .max(1e-6);
        let dielectric = ((self.ior - 1.0) / (self.ior + 1.0)).powi(2);
        let f0 = Vec3::splat(dielectric).lerp(self.color, self.metal);
        let fresnel = f0 + (Vec3::ONE - f0) * (1.0 - vh).powi(5);
        let specular = fresnel * distribution * visibility * nl;
        let fill = ray.ambient * (0.85 + normal.y * 0.15);
        let diffuse = self.color
            * (Vec3::ONE - fresnel)
            * (Vec3::ONE - self.transmission)
            * (1.0 - self.metal)
            * (fill + ray.radiance * (nl.lerp(backlight, self.translucency) / PI));
        let reflected = normal * (2.0 * normal.dot(view)) - view;
        let environment = ray.ambient * (0.85 + reflected.y * 0.15);
        let grazing = Vec3::splat(1.0 - self.roughness).max(f0);
        let environment_fresnel = f0 + (grazing - f0) * (1.0 - nv).powi(5);
        diffuse
            + specular * ray.radiance
            + environment * environment_fresnel
            + self.color * self.emission
    }
}

pub fn coat(p: Vec3, id: u32, pigment: Vec4, accent: Vec3) -> Material {
    let stripes = (p.z * 2.2 + (p.x * 1.5).sin() * 1.4).sin();
    let flecks = ((p.x * 31.0).sin() * (p.z * 27.0).sin() * (p.y * 19.0).sin()).abs();
    let dorsal = (p.y * 0.30 + 0.5).saturate();
    let mut m = Material::skin(
        pigment
            .truncate()
            .lerp(accent, stripes.smoothstep(0.30, 0.85) * dorsal * 0.55),
        0.43,
    );
    m.color *= 0.92 + flecks * 0.08;
    m.emission = pigment.w;
    match id {
        1 => {
            m.color = accent;
            m.roughness = 0.62;
        }
        2 => {
            m.color = accent.lerp(vec3(0.95, 0.29, 0.31), 0.32);
            m.transmission = Vec3::splat(0.35);
        }
        3 => {
            m.color = vec3(0.012, 0.022, 0.036);
            m.roughness = 0.14;
            m.emission = 0.0;
        }
        4 => {
            m.color = accent.lerp(pigment.truncate(), stripes * 0.25 + 0.25);
            m.roughness = 0.32;
        }
        5 => {
            m.color = accent * (0.6 + flecks * 0.4);
            m.metal = 0.55;
            m.roughness = 0.3;
        }
        6 => {
            m.color = vec3(0.60, 0.73, 0.78);
            m.metal = 0.85;
            m.roughness = 0.24;
        }
        7 => {
            m.color = vec3(0.91, 0.87, 0.73) * (0.9 + flecks * 0.1);
            m.roughness = 0.8;
        }
        8 => {
            m.color = vec3(0.65, 0.25, 0.32);
            m.roughness = 0.23;
        }
        9 => {
            m.color = accent;
            m.emission = 2.5;
            m.roughness = 0.22;
        }
        10 => {
            m.color = vec3(0.17, 0.34, 0.21).lerp(vec3(0.43, 0.59, 0.27), flecks);
            m.roughness = 0.85;
        }
        _ => {}
    }
    m
}

fn normalize(vector: Vec3) -> Vec3 {
    vector / vector.length_squared().max(1e-20).sqrt()
}

pub fn estimate_normal<F: FnMut(Vec3) -> f32>(point: Vec3, sdf: F) -> Vec3 {
    normal_at(point, 0.03, sdf)
}
fn normal_at<F: FnMut(Vec3) -> f32>(point: Vec3, epsilon: f32, mut sdf: F) -> Vec3 {
    let a = vec3(1.0, -1.0, -1.0);
    let b = vec3(-1.0, -1.0, 1.0);
    let c = vec3(-1.0, 1.0, -1.0);
    let d = Vec3::ONE;
    normalize(
        a * sdf(point + a * epsilon)
            + b * sdf(point + b * epsilon)
            + c * sdf(point + c * epsilon)
            + d * sdf(point + d * epsilon),
    )
}

pub fn local_ray(ray: Ray, c: &CreatureInstance) -> Ray {
    let rotation = Quat::from_array(c.rotation.to_array()).conjugate();
    let scale = development::scale(c);
    Ray {
        origin: (rotation * (ray.origin - c.position.truncate())) / scale,
        direction: (rotation * ray.direction) / scale,
        light: rotation * ray.light,
        ..ray
    }
}

pub fn make_hit(
    ray: Ray,
    c: &CreatureInstance,
    t: f32,
    normal: Vec3,
    previous: Vec3,
    material: Material,
) -> Hit {
    let rotation = Quat::from_array(c.rotation.to_array());
    let scale = development::scale(c);
    let world_normal = (rotation * (normal / scale)).normalize();
    let material = development::appearance(material, previous, c);
    let cosine = world_normal.dot(ray.direction).abs().clamp(0.0, 1.0);
    Hit {
        color_depth: material.shade(world_normal, ray).extend(t),
        normal: world_normal,
        previous_position: Quat::from_array(c.previous_rotation.to_array())
            * (previous
                * if c.previous_proportions.x > 0.0 {
                    c.previous_proportions.truncate()
                } else {
                    scale
                })
            + c.previous_position.truncate(),
        transmittance: material.transmittance(cosine),
        part: 0,
    }
}

pub fn trace<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    pigment: Vec4,
    accent: Vec3,
) -> Hit {
    let mut hit = trace_material(ray, c, surface, |point, id| {
        coat(point / (c.position.w * 0.15).max(0.7), id, pigment, accent)
    });
    hit.transmittance = Vec3::ZERO;
    hit
}

pub fn trace_material<F: Fn(Vec3, &CreatureInstance) -> Surface, M: Fn(Vec3, u32) -> Material>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    material: M,
) -> Hit {
    let local = local_ray(ray, c);
    let scale = development::scale(c).min_element();
    let surface = |point, c: &CreatureInstance| {
        let mut hit = surface(development::body_point(point, c), c);
        hit.distance *= 1.0 - (1.0 - c.life.y) * 0.2;
        hit
    };
    let bounds = c.opaque_bounds.truncate();
    let nonzero = |value: f32| if value.abs() < 1e-8 { 1e-8 } else { value };
    let safe = vec3(
        nonzero(local.direction.x),
        nonzero(local.direction.y),
        nonzero(local.direction.z),
    );
    let inv = Vec3::ONE / safe;
    let near = (-bounds - local.origin) * inv;
    let far = (bounds - local.origin) * inv;
    let mut distance = near.min(far).max_element().max(0.0);
    let end = near.max(far).min_element().min(ray.max_distance);
    for _ in 0..128 {
        if distance >= end {
            break;
        }
        let point = local.origin + local.direction * distance;
        let sample = surface(point, c);
        let epsilon = (ray.footprint * distance * 0.35 / scale)
            .clamp(c.position.w * 0.000_01, c.position.w * 0.000_35);
        if sample.distance < epsilon {
            let normal = normal_at(point, epsilon.max(c.position.w * 0.000_05), |q| {
                surface(q, c).distance
            });
            let mut previous = point;
            let tail_length = c.tail[5].z - c.tail[0].z;
            if point.z > c.tail[0].z && tail_length > 0.01 {
                let fraction = ((point.z - c.tail[0].z) / tail_length).clamp(0.0, 1.0) * 5.0;
                let index = (fraction as usize).min(4);
                let delta = c.previous_tail[index]
                    .lerp(c.previous_tail[index + 1], fraction - index as f32)
                    - c.tail[index].lerp(c.tail[index + 1], fraction - index as f32);
                let scale = if c.kind.x == Kind::Nae as u32 {
                    vec3(0.2, 0.25, 1.0)
                } else if c.kind.x == Kind::Naetu as u32 {
                    vec3(3.0, 3.0, 1.0)
                } else {
                    Vec3::ONE
                };
                previous += delta.truncate() * scale;
            }
            let scale = (c.position.w * 0.04).max(0.025);
            let mut occlusion = 0.0;
            for index in 1..=4 {
                let reach = scale * index as f32 * 0.25;
                occlusion += (reach - surface(point + normal * reach, c).distance).max(0.0) / scale
                    * (0.5f32).powi(index);
            }
            let shaded_ray = Ray {
                ambient: ray.ambient * (1.0 - occlusion * 4.0).clamp(0.25, 1.0),
                ..ray
            };
            return make_hit(
                shaded_ray,
                c,
                distance,
                normal,
                previous,
                material(point, sample.material),
            );
        }
        distance += (sample.distance * scale * 0.80).max(epsilon * scale * 0.5);
    }
    Hit::MISS
}

pub fn tonemap(color: Vec3) -> Vec3 {
    let a = color * (color * 2.51 + Vec3::splat(0.03));
    let b = color * (color * 2.43 + Vec3::splat(0.59)) + Vec3::splat(0.14);
    (a / b).clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2)
}

#[derive(Clone, Copy)]
pub struct Sheet {
    pub center: Vec3,
    pub previous_center: Vec3,
    pub radii: Vec2,
    pub bend: Vec2,
    pub side: f32,
}

impl Sheet {
    fn coordinates(self, point: Vec3) -> Vec2 {
        let span = point.x / self.radii.x;
        let fraction = span.mul_add(self.side * 0.5, 0.5);
        Vec2::new(
            span,
            (point.z - (fraction * fraction - 0.25) * self.radii.x * 0.55)
                / (self.radii.y * (1.0 - fraction * 0.55)),
        )
    }
}

pub fn sheet(ray: Ray, c: &CreatureInstance, shape: Sheet, pigment: Vec4) -> Hit {
    let Sheet {
        center,
        previous_center,
        radii,
        bend,
        ..
    } = shape;
    let tint = pigment.truncate();
    let opacity = pigment.w;
    let local = local_ray(ray, c);
    let origin = local.origin - center;
    let curvature = bend.x / (radii.x * radii.x);
    let quadratic = curvature * local.direction.x * local.direction.x;
    let linear = 2.0 * curvature * origin.x * local.direction.x - local.direction.y;
    let constant = curvature * origin.x * origin.x - origin.y;
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return Hit::MISS;
    }
    let roots = if quadratic.abs() < 1e-8 {
        Vec2::splat(-constant / linear)
    } else {
        let root = discriminant.sqrt();
        Vec2::new(
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        )
    };
    let mut distance = ray.max_distance;
    for index in 0..2 {
        let candidate = if index == 0 { roots.x } else { roots.y };
        if !(candidate > 0.0 && candidate < distance) {
            continue;
        }
        let point = origin + local.direction * candidate;
        let uv = shape.coordinates(point);
        let scallop = (uv.x * if opacity > 0.9 { 42.0 } else { 17.0 }).sin()
            * if opacity > 0.9 { 0.025 } else { 0.035 };
        if uv.length_squared() + scallop * uv.x.abs() <= 1.0 {
            distance = candidate;
        }
    }
    if distance >= ray.max_distance {
        return Hit::MISS;
    }
    let point = origin + local.direction * distance;
    let uv = shape.coordinates(point);
    let scallop = (uv.x * if opacity > 0.9 { 42.0 } else { 17.0 }).sin()
        * if opacity > 0.9 { 0.025 } else { 0.035 };
    let vein = ((uv.y.atan2(uv.x) * 9.0 + uv.length() * 3.0).sin().abs())
        .min((uv.length() * 27.0).sin().abs());
    let vein_width = (ray.footprint * distance / radii.min_element()).max(0.012);
    let rib = 1.0 - vein.smoothstep(vein_width, vein_width * 2.5);
    let panel = (uv.x * 4.0 + uv.y * 3.0).sin() * 0.5 + 0.5;
    let colour = if opacity > 0.9 {
        tint.lerp(
            vec3(0.80, 0.85, 0.86),
            (uv.y * 0.4 + panel * 0.25).clamp(0.0, 0.35),
        ) * (0.90 + (uv.x * 55.0 + uv.y * 4.0).sin().abs() * 0.10)
    } else {
        tint.lerp(vec3(0.94, 0.59, 0.17), panel * 0.65)
    };
    let coverage = ((1.0 - uv.length_squared() - scallop * uv.x.abs())
        / (ray.footprint * distance / radii.min_element()).max(0.001))
    .saturate();
    let alpha = (opacity + rib * 0.65).clamp(0.0, 0.98);
    let mut material = Material::skin(
        colour,
        if opacity > 0.9 {
            0.65
        } else {
            0.25 + rib * 0.35
        },
    );
    material.transmission = (Vec3::ONE.lerp(colour.sqrt(), alpha * 0.5)) * (1.0 - alpha);
    let normal = vec3(-2.0 * curvature * point.x, 1.0, 0.0).normalize();
    let previous = previous_center
        + vec3(
            point.x,
            bend.y * point.x * point.x / (radii.x * radii.x),
            point.z,
        );
    let mut hit = make_hit(
        ray,
        c,
        distance,
        if normal.dot(local.direction) > 0.0 {
            -normal
        } else {
            normal
        },
        previous,
        material,
    );
    hit.color_depth = (hit.color() * coverage).extend(distance);
    hit.transmittance = Vec3::ONE.lerp(hit.transmittance, coverage);
    hit.part = 1;
    hit
}

pub fn paired_sheets(
    result: &mut TraceResult,
    ray: Ray,
    c: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    pigment: Vec4,
) {
    visit_paired_sheets(ray, c, root, dimensions, pigment, |hit| result.add(hit));
}

pub fn visit_paired_sheets<F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    pigment: Vec4,
    mut emit: F,
) {
    let development = if c.proportions.x > 0.0 {
        0.25 + c.proportions.w * 0.75
    } else {
        1.0
    };
    let chord = dimensions.y * development.sqrt();
    let reach = dimensions.x * development * (1.0 - c.animation.y * 0.65);
    let old_reach = dimensions.x * development * (1.0 - c.previous_animation.y * 0.65);
    for i in 0..2 {
        let side = if i == 0 { -1.0 } else { 1.0 };
        let center = root
            + vec3(
                side * reach * 0.5,
                c.animation.x * reach * 0.25,
                reach * 0.12,
            );
        let bend = reach * (0.08 + c.animation.x * 0.2);
        let old_bend = old_reach * (0.08 + c.previous_animation.x * 0.2);
        emit(sheet(
            ray,
            c,
            Sheet {
                center,
                previous_center: root
                    + vec3(
                        side * old_reach * 0.5,
                        c.previous_animation.x * old_reach * 0.25,
                        old_reach * 0.12,
                    ),
                radii: Vec2::new(reach * 0.52, chord),
                bend: Vec2::new(bend, old_bend),
                side,
            },
            pigment,
        ));
    }
}

pub fn shell(
    result: &mut TraceResult,
    ray: Ray,
    c: &CreatureInstance,
    center: Vec3,
    radii: Vec3,
    pigment: Vec4,
    bottom: f32,
) {
    visit_shell(ray, c, center, radii, pigment, bottom, |hit| {
        result.add(hit);
    });
}

pub fn visit_shell<F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    center: Vec3,
    radii: Vec3,
    pigment: Vec4,
    bottom: f32,
    emit: F,
) {
    let tint = pigment.truncate();
    let opacity = pigment.w;
    visit_shell_material(
        ray,
        c,
        Shell {
            center,
            radii,
            bottom,
            scallop: 0.0,
        },
        |_, cosine| {
            let mut material = Material::skin(tint, 0.18);
            material.ior = 1.36;
            material.transmission =
                (-(Vec3::ONE - tint) * (opacity / cosine.max(0.08))).exp() * (1.0 - opacity);
            if c.kind.x == Kind::Naetu as u32 {
                material.emission = 0.0125 + 0.4875 * (1.0 - cosine).powi(5);
            }
            material
        },
        emit,
    );
}

pub struct Shell {
    pub center: Vec3,
    pub radii: Vec3,
    pub bottom: f32,
    pub scallop: f32,
}

pub fn visit_shell_material<M: Fn(Vec3, f32) -> Material, F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    shape: Shell,
    material: M,
    mut emit: F,
) {
    let Shell {
        center,
        radii,
        bottom,
        scallop,
    } = shape;
    let squared_radii = radii * radii;
    let local = local_ray(ray, c);
    let origin = (local.origin - center) / radii;
    let direction = local.direction / radii;
    let quadratic = direction.length_squared();
    let linear = origin.dot(direction);
    let discriminant = linear * linear - quadratic * (origin.length_squared() - 1.0);
    if discriminant < 0.0 {
        return;
    }
    let root = discriminant.sqrt();
    for i in 0..2 {
        let distance = (-linear + if i == 0 { -root } else { root }) / quadratic;
        if distance <= 0.0 || distance >= ray.max_distance {
            continue;
        }
        let point = local.origin + local.direction * distance;
        if point.y < bottom + (point.x.atan2(point.z) * 14.0).cos() * scallop {
            continue;
        }
        let normal = ((point - center) / squared_radii).normalize();
        let cosine = normal.dot(local.direction).abs();
        let mut hit = make_hit(
            ray,
            c,
            distance,
            if normal.dot(local.direction) > 0.0 {
                -normal
            } else {
                normal
            },
            point,
            material(point, cosine),
        );
        hit.part = 2;
        emit(hit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction,
            light: Vec3::Y,
            ambient: Vec3::ONE,
            radiance: Vec3::ONE,
            max_distance: 100.0,
            footprint: 0.001,
        }
    }

    #[test]
    fn scattering_conserves_energy_under_uniform_illumination() {
        for roughness in [0.3, 0.6, 1.0] {
            for (metal, translucency) in [(0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (0.0, 1.0)] {
                let mut material = Material::skin(Vec3::ONE, roughness);
                material.metal = metal;
                material.translucency = translucency;
                let mut energy = Vec3::ZERO;
                for latitude in 0..128 {
                    let cosine = (latitude as f32 + 0.5) / 64.0 - 1.0;
                    let sine = (1.0 - cosine * cosine).sqrt();
                    for longitude in 0..128 {
                        let angle = (longitude as f32 + 0.5) * (2.0 * PI / 128.0);
                        let illumination = Ray {
                            direction: -Vec3::Y,
                            light: vec3(sine * angle.cos(), cosine, sine * angle.sin()),
                            ambient: Vec3::ZERO,
                            ..ray(Vec3::ZERO, -Vec3::Y)
                        };
                        energy += material.shade(Vec3::Y, illumination) * (4.0 * PI / 16384.0);
                    }
                }
                assert!(
                    energy.max_element() <= 1.01,
                    "BRDF creates energy: {energy:?}"
                );
            }
        }
    }

    #[test]
    fn curved_sheet_keeps_second_root_when_first_is_outside_wing() {
        let instance = CreatureInstance {
            rotation: Vec4::W,
            previous_rotation: Vec4::W,
            ..CreatureInstance::default()
        };
        let hit = sheet(
            ray(vec3(-2.0, 0.25, 3.0), vec3(1.0, 0.0, -1.0).normalize()),
            &instance,
            Sheet {
                center: Vec3::ZERO,
                previous_center: Vec3::ZERO,
                radii: Vec2::ONE,
                bend: Vec2::ONE,
                side: 0.0,
            },
            Vec3::ONE.extend(0.3),
        );
        assert!(hit.is_hit(), "the second valid wing root was discarded");
        assert!(
            (hit.distance() - 2.5 * 2.0f32.sqrt()).abs() < 0.001,
            "wrong wing root"
        );
        assert!(
            (hit.normal.length() - 1.0).abs() < 0.001,
            "invalid wing normal"
        );
        assert!(
            hit.transmittance.min_element() > 0.0,
            "membrane must transmit light"
        );
    }

    #[test]
    fn transparency_is_sorted_and_clipped_by_opaque_body() {
        let layer = |distance, color: Vec3| Hit {
            color_depth: color.extend(distance),
            transmittance: Vec3::splat(0.5),
            ..Hit::MISS
        };
        let mut result = TraceResult::opaque(layer(5.0, Vec3::Z));
        result.add(layer(7.0, Vec3::splat(100.0)));
        result.add(layer(3.0, Vec3::Y));
        result.add(layer(1.0, Vec3::X));
        assert!(
            result.front().distance() < 2.0,
            "layers must sort front to back"
        );
        assert!(
            result.composite(Vec3::ZERO).distance(vec3(1.0, 0.5, 0.25)) < 0.001,
            "transparent layers behind the opaque body must be excluded"
        );
    }
}
