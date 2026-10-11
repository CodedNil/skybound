pub use super::BIOLOGY;
use spirv_std::glam::{FloatExt, Quat, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, smooth_min};
use crate::scene::surface::{Hit, Material, Ray, Surface, TraceResult, local_ray, trace_material};

pub mod aerogigantiform;
#[cfg(not(target_arch = "spirv"))]
pub mod cloth;
pub mod nubidriftonidae;

fn cubic(before: Vec3, start: Vec3, end: Vec3, after: Vec3, t: f32) -> (Vec3, Vec3) {
    let linear = (end - before) * 0.5;
    let quadratic = (before * 2.0 - start * 5.0 + end * 4.0 - after) * 0.5;
    let cubic = (-before + start * 3.0 - end * 3.0 + after) * 0.5;
    (
        start + t * (linear + t * (quadratic + t * cubic)),
        linear + t * (quadratic * 2.0 + t * cubic * 3.0),
    )
}

fn ribbon_patch(points: &[[Vec4; 3]; 12], row: usize, t: f32, v: f32) -> (Vec3, Vec3, Vec3) {
    let mut across = [Vec3::ZERO; 3];
    let mut tangents = [Vec3::ZERO; 3];
    for (col, point) in across.iter_mut().enumerate() {
        (*point, tangents[col]) = cubic(
            points[row.saturating_sub(1)][col].truncate(),
            points[row][col].truncate(),
            points[row + 1][col].truncate(),
            points[(row + 2).min(11)][col].truncate(),
            t,
        );
    }
    let point = across[1]
        + (across[2] - across[0]) * (v * 0.5)
        + (across[0] + across[2] - across[1] * 2.0) * (v * v * 0.5);
    let along = tangents[1]
        + (tangents[2] - tangents[0]) * (v * 0.5)
        + (tangents[0] + tangents[2] - tangents[1] * 2.0) * (v * v * 0.5);
    let across = (across[2] - across[0]) * 0.5 + (across[0] + across[2] - across[1] * 2.0) * v;
    (point, along, across)
}

fn ribbon_point(points: &[[Vec4; 3]; 12], row: usize, t: f32, v: f32) -> Vec3 {
    ribbon_patch(points, row, t, v).0
}

fn ribbon_coordinates(p: Vec3, points: &[[Vec4; 3]; 12]) -> (usize, f32, f32) {
    let mut best = f32::MAX;
    let mut row = 0;
    let mut t = 0.0;
    for i in 0..11 {
        let start = points[i][1].truncate();
        let edge = points[i + 1][1].truncate() - start;
        let fraction = ((p - start).dot(edge) / edge.length_squared().max(1e-6)).clamp(0.0, 1.0);
        let distance = (p - start - edge * fraction).length_squared();
        if distance < best {
            best = distance;
            row = i;
            t = fraction;
        }
    }
    let mut closest = (row, t, 0.0);
    best = f32::MAX;
    for candidate in row.saturating_sub(1)..=(row + 1).min(10) {
        let center = ribbon_point(points, candidate, 0.5, 0.0);
        let edge =
            ribbon_point(points, candidate, 0.5, 1.0) - ribbon_point(points, candidate, 0.5, -1.0);
        let mut along = if candidate == row { t } else { 0.5 };
        let mut across =
            ((p - center).dot(edge) * 2.0 / edge.length_squared().max(1e-6)).clamp(-1.0, 1.0);
        for _ in 0..5 {
            let (point, du, dv) = ribbon_patch(points, candidate, along, across);
            let delta = p - point;
            let uu = du.length_squared();
            let vv = dv.length_squared();
            let uv = du.dot(dv);
            let determinant = (uu * vv - uv.powi(2)).max(1e-8);
            along = (along
                + ((delta.dot(du) * vv - delta.dot(dv) * uv) / determinant).clamp(-0.3, 0.3))
            .clamp(0.0, 1.0);
            across = (across
                + ((delta.dot(dv) * uu - delta.dot(du) * uv) / determinant).clamp(-0.5, 0.5))
            .clamp(-1.0, 1.0);
        }
        let distance = p.distance_squared(ribbon_point(points, candidate, along, across));
        if distance < best {
            best = distance;
            closest = (candidate, along, across);
        }
    }
    closest
}

pub fn ribbon_distance(p: Vec3, c: &CreatureInstance, ribbon: usize) -> f32 {
    let points = &c.cloth[ribbon];
    let (row, t, v) = ribbon_coordinates(p, points);
    (p - ribbon_point(points, row, t, v)).length() * 0.4 - 0.012
}

#[derive(Clone, Copy)]
pub struct GasChamber {
    pub center: Vec3,
    pub radii: Vec3,
}

impl GasChamber {
    pub fn distance(self, p: Vec3) -> f32 {
        let mut local = p - self.center;
        let q = local / self.radii;
        local += vec3(
            (q.y * 3.0 + q.z).sin(),
            (q.z * 2.7 + q.x).sin(),
            (q.x * 2.5 + q.y).sin(),
        ) * self.radii
            * 0.025;
        ellipsoid(local, self.radii) * 0.8
    }
}

pub fn skin<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    p: Vec3,
    c: &CreatureInstance,
    organs: F,
    chambers: &[GasChamber],
    scale: f32,
) -> Surface {
    let mut distance = organs(p, c).distance - scale * 0.035;
    for chamber in chambers {
        distance = smooth_min(distance, chamber.distance(p) - scale * 0.018, scale * 0.16);
    }
    Surface::new(distance, 0)
}

fn closed_hits<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
) -> [Hit; 2] {
    let front = trace_material(ray, c, &surface, |_, _| Material::skin(Vec3::ZERO, 1.0));
    if !front.is_hit() {
        return [Hit::MISS; 2];
    }
    let end = (c.position.truncate() - ray.origin).dot(ray.direction) + c.position.w * 1.1;
    let reverse = Ray {
        origin: ray.origin + ray.direction * end,
        direction: -ray.direction,
        max_distance: end,
        ..ray
    };
    let mut back = trace_material(reverse, c, surface, |_, _| Material::skin(Vec3::ZERO, 1.0));
    if back.is_hit() {
        back.color_depth.w = end - back.distance();
    }
    [front, back]
}

pub fn drift_body<
    F: Fn(Vec3, &CreatureInstance) -> Surface,
    G: Fn(Vec3, &CreatureInstance) -> Surface,
>(
    ray: Ray,
    c: &CreatureInstance,
    envelope: F,
    organs: G,
    tint: Vec3,
    scale: f32,
    chambers: &[GasChamber],
) -> TraceResult {
    let [front, back] = closed_hits(ray, c, &envelope);
    if !front.is_hit() {
        return TraceResult::MISS;
    }
    let tissue = |p, c: &CreatureInstance| {
        let mut distance = organs(p, c)
            .distance
            .max(envelope(p, c).distance + scale * 0.025);
        for chamber in chambers {
            distance = distance.max(-chamber.distance(p));
        }
        Surface::new(distance, 0)
    };
    let mut result = TraceResult::MISS;
    let local = local_ray(ray, c);
    let [tissue_front, tissue_back] = closed_hits(ray, c, tissue);
    let depth = (tissue_back.distance() - tissue_front.distance()).max(0.0) / scale;
    for mut hit in [tissue_front, tissue_back] {
        if !hit.is_hit() {
            continue;
        }
        let point = local.origin + local.direction * hit.distance();
        let patches = mottling(point / scale * 9.0).smoothstep(-0.35, 0.5);
        let mut material = gel(point, scale * 0.45, tint * (0.55 + patches * 0.45));
        material.roughness = 0.3;
        material.transmission = (-(Vec3::ONE - tint * 0.6) * (0.8 + depth * 2.0)).exp();
        if hit.normal.dot(ray.direction) > 0.0 {
            hit.normal = -hit.normal;
        }
        hit.transmittance = material.transmittance(hit.normal.dot(ray.direction).abs());
        hit.color_depth = material.shade(hit.normal, ray).extend(hit.distance());
        result.add(hit);
    }
    let thickness = if back.is_hit() {
        (back.distance() - front.distance()).max(0.0) / scale
    } else {
        0.0
    };
    for (index, mut hit) in [front, back].into_iter().enumerate() {
        if !hit.is_hit() || hit.distance() <= 0.0 || hit.distance() >= ray.max_distance {
            continue;
        }
        let point = local.origin + local.direction * hit.distance();
        let q = point / scale;
        let food = mottling(q * 11.0).smoothstep(0.05, 0.65) * (1.0 - q.y.smoothstep(-0.25, 0.3));
        let ivory = vec3(0.88, 0.86, 0.61);
        let mut material = gel(point, scale, ivory.lerp(tint, 0.55 + food * 0.35));
        material.roughness = 0.19 + food * 0.22;
        let absorb = (Vec3::ONE - tint) * (0.12 + food * 0.3 + thickness * 0.08);
        material.transmission = (-absorb).exp() * (0.92 - food * 0.18);
        if hit.normal.dot(ray.direction) > 0.0 {
            hit.normal = -hit.normal;
        }
        let cosine = hit.normal.dot(ray.direction).abs();
        hit.transmittance = material.transmittance(cosine);
        hit.color_depth = material.shade(hit.normal, ray).extend(hit.distance());
        hit.part = 2;
        if index == 0 || back.distance() > front.distance() + scale * 0.001 {
            result.add(hit);
        }
    }
    for chamber in chambers {
        for mut hit in closed_hits(ray, c, |p, _| Surface::new(chamber.distance(p), 0)) {
            if hit.is_hit()
                && hit.distance() > 0.0
                && hit.distance() < result.opaque.distance().min(ray.max_distance)
            {
                let point = local.origin + local.direction * hit.distance();
                if hit.normal.dot(ray.direction) > 0.0 {
                    hit.normal = -hit.normal;
                }
                let cosine = hit.normal.dot(ray.direction).abs();
                let mut material = gel(point, scale * 0.8, tint.lerp(vec3(0.95, 0.88, 0.53), 0.65));
                material.roughness = 0.23;
                material.transmission =
                    (-(Vec3::ONE - tint) * (0.045 / cosine.max(0.1))).exp() * 0.97;
                hit.color_depth = material.shade(hit.normal, ray).extend(hit.distance());
                hit.transmittance = material.transmittance(cosine);
                hit.part = 2;
                result.add(hit);
            }
        }
    }
    result
}

pub fn mottling(p: Vec3) -> f32 {
    (p.x + (p.z * 0.73).sin()).sin()
        * (p.y * 1.17 + p.x.sin()).sin()
        * (p.z * 0.91 + (p.y * 0.8).cos()).sin()
}

pub fn tissue_distance(p: Vec3, scale: f32) -> f32 {
    let q = p / scale;
    scale * (mottling(q * 6.0) * 0.022 + mottling(q * 21.0) * 0.004)
}

pub fn gel(p: Vec3, scale: f32, tint: Vec3) -> Material {
    let mut q = p / scale * 9.0;
    q += vec3(
        (q.y * 1.7 + q.z).sin(),
        (q.z * 1.5 + q.x).sin(),
        (q.x * 1.8 + q.y).sin(),
    ) * 0.3;
    let cell = q.floor();
    let mut nearest = f32::MAX;
    let mut second = f32::MAX;
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let neighbor = cell + vec3(x as f32, y as f32, z as f32);
                let seed = vec3(
                    neighbor.dot(vec3(127.1, 311.7, 74.7)),
                    neighbor.dot(vec3(269.5, 183.3, 246.1)),
                    neighbor.dot(vec3(113.5, 271.9, 124.6)),
                );
                let offset = (seed.sin() * 43758.547).fract_gl();
                let distance = (neighbor + offset - q).length_squared();
                second = second.min(distance.max(nearest));
                nearest = nearest.min(distance);
            }
        }
    }
    let vein = 1.0 - (second.sqrt() - nearest.sqrt()).smoothstep(0.008, 0.055);
    let algae = mottling(q * 2.5).smoothstep(-0.1, 0.6);
    let pores = (mottling(q * 10.0)).smoothstep(0.45, 0.8);
    let mut material = Material::skin(
        tint * (0.75 + algae * 0.25) * (1.0 - pores * 0.3),
        0.2 + algae * 0.12,
    );
    material.color = material.color.lerp(vec3(0.95, 0.83, 0.43), vein * 0.65);
    material.ior = 1.36;
    material.translucency = 0.5;
    material
}

pub fn ribbons(result: &mut TraceResult, ray: Ray, c: &CreatureInstance, count: usize, tint: Vec3) {
    for i in 0..count {
        let mut offset = 0.0;
        for _ in 0..2 {
            let segment = Ray {
                origin: ray.origin + ray.direction * offset,
                max_distance: ray.max_distance.min(result.opaque.distance()) - offset,
                ..ray
            };
            let mut hit = trace_material(
                segment,
                c,
                |p, c| Surface::new(ribbon_distance(p, c, i), 0),
                |_, _| Material::skin(tint, 0.3),
            );
            if !hit.is_hit() {
                break;
            }
            hit.color_depth.w += offset;
            let cosine = hit.normal.dot(-ray.direction).abs();
            let local = local_ray(ray, c);
            let point = local.origin + local.direction * hit.distance();
            let (row, t, v) = ribbon_coordinates(point, &c.cloth[i]);
            let margin = v.abs().smoothstep(0.78, 0.98);
            let midrib = 1.0 - v.abs().smoothstep(0.02, 0.07);
            let veins = ((row as f32 + t) * 4.0 + v.abs() * 7.0)
                .sin()
                .abs()
                .smoothstep(0.95, 0.995);
            let mut material = gel(
                point,
                3.0,
                tint.lerp(
                    vec3(0.92, 0.83, 0.35),
                    (margin + midrib + veins * 0.4).min(1.0) * 0.55,
                ),
            );
            material.transmission =
                (-vec3(0.35, 0.10, 0.6) * ((0.09 + margin * 0.1) / cosine.max(0.15))).exp() * 0.93;
            hit.transmittance = material.transmittance(cosine);
            hit.color_depth = material.shade(hit.normal, ray).extend(hit.distance());
            hit.previous_position = Quat::from_array(c.previous_rotation.to_array())
                * ribbon_point(&c.previous_cloth[i], row, t, v)
                + c.previous_position.truncate();
            hit.part = 1;
            offset = hit.distance() + 0.07 / cosine.max(0.05);
            result.add(hit);
        }
    }
}

pub const TAXON: Taxon = Taxon {
    name: "Caeliformes",
    english_name: "Sky Shapes",
    description: "Sky Shapes are buoyant sponge-like animals whose bodies range from rounded balloons and ribbon-bearing leaves to enormous arched canopies. \
        Translucent skins enclose multiple gas chambers and a lobed mass of gelatinous organs. \
        Their wet surfaces collect algae and plant debris as they drift with the updraft.",
    rank: Rank::Class,
    parent: Some(&super::TAXON),
    radius: 0.0,
};

#[cfg(test)]
mod tests {
    use aerogigantiform::aerogigantidae::aerogigans;

    use super::*;
    use crate::scene::life::Kind;

    #[test]
    fn twisted_cloth_has_no_gaps_between_patches() {
        let c = CreatureInstance::new(Vec4::W * 25.0, Vec4::W, Vec4::ZERO, Kind::Levita as u32);
        for ribbon in 0..3 {
            for row in 0..11 {
                for t in [0.001, 0.25, 0.5, 0.75, 0.999] {
                    for v in [-0.9, -0.5, 0.0, 0.5, 0.9] {
                        let point = ribbon_point(&c.cloth[ribbon], row, t, v);
                        assert!(
                            ribbon_distance(point, &c, ribbon) < 0.001,
                            "hole in cloth patch {row} at {t}, {v}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn giant_skin_is_closed_from_above_below_and_both_sides() {
        let c = CreatureInstance::new(
            Vec3::ZERO.extend(46.0),
            Vec4::W,
            Vec4::ZERO,
            Kind::Aerogigans as u32,
        );
        for direction in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let ray = Ray {
                origin: direction * 80.0,
                direction: -direction,
                light: Vec3::Y,
                ambient: Vec3::ONE,
                radiance: Vec3::ONE,
                footprint: 0.0001,
                max_distance: 160.0,
            };
            let [front, back] = closed_hits(ray, &c, aerogigans::envelope);
            assert!(
                front.is_hit() && back.is_hit(),
                "open skin along {direction:?}"
            );
            assert!(
                back.distance() > front.distance() + 1.0,
                "missing enclosed volume"
            );
            let midpoint = ray.origin + ray.direction * front.distance().midpoint(back.distance());
            assert!(
                aerogigans::envelope(midpoint, &c).distance < 0.0,
                "skin has no interior"
            );
        }
    }
}
