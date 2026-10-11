pub use super::{BIOLOGY, render_wings};
use crate::scene::life::CreatureInstance;
use crate::scene::life::{Rank, Taxon, ellipsoid, tapered_tail};
use crate::scene::surface::{Hit, Material, Ray, Surface, local_ray, make_hit};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub mod lucidus;
pub mod robustus;

pub const TAXON: Taxon = Taxon {
    name: "Naeturus",
    english_name: "Nae Kin",
    description: "Aur-adapted relatives in this genus share quadratic close bond between small Næ and enormous Nætu. \
        Feathered tails and paired wings connect very different body sizes, while specialised aur organs enable unusually intimate use of the energy below. \
        Cooperative travel, grooming and protection are central to the association.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
};

pub fn feather(point: Vec3, root: Vec3, length: f32, width: f32, sweep: f32) -> f32 {
    let stable = point - root;
    let stable = vec3(
        stable.x - stable.z * sweep,
        stable.y,
        stable.z - length * 0.5,
    );
    ellipsoid(stable, vec3(width, width * 0.13, length * 0.55)) * 0.65
}

pub fn feathered_tail(
    point: Vec3,
    creature: &CreatureInstance,
    scale: Vec3,
    thickness: f32,
    length: f32,
    width: f32,
) -> Surface {
    let mut hit = Surface::new(1e6, 4);
    for segment in 0..5 {
        let start = creature.tail[segment].truncate() * scale;
        let end = creature.tail[segment + 1].truncate() * scale;
        let fraction = segment as f32 / 5.0;
        hit.join(
            tapered_tail(
                point,
                start,
                end,
                thickness * (1.06 - fraction),
                thickness * (0.86 - fraction),
            ),
            4,
        );
        for layer in 1..=3 {
            let anchor = start.lerp(end, layer as f32 / 3.0);
            for wing_index in 0..2 {
                let handedness = if wing_index == 0 { -1.0 } else { 1.0 };
                hit.join(
                    feather(
                        point,
                        anchor,
                        length * (1.0 - fraction * 0.3),
                        width * (1.0 - fraction * 0.4),
                        handedness * (0.8 - fraction * 0.35),
                    ),
                    1,
                );
            }
        }
    }
    hit
}

pub fn wings<F: FnMut(Hit)>(
    ray: Ray,
    creature: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    tint: Vec3,
    mut emit: F,
) {
    let local = local_ray(ray, creature);
    let reach = dimensions.x * (1.0 - creature.animation.y.clamp(0.0, 1.2) * 0.6);
    let old_reach = dimensions.x * (1.0 - creature.previous_animation.y.clamp(0.0, 1.2) * 0.6);
    let lift = creature.animation.x * 0.45 - creature.animation.y * 0.12;
    let bend = 0.12 + creature.animation.x * 0.2;
    for wing_index in 0..2 {
        let handedness = if wing_index == 0 { -1.0 } else { 1.0 };
        let origin = local.origin - vec3(handedness * root.x, root.y, root.z);
        let curvature = bend / reach;
        let roots = wing_roots(
            Vec2::new(origin.x * handedness, origin.y),
            Vec2::new(local.direction.x * handedness, local.direction.y),
            lift,
            curvature,
        );
        let mut nearest = ray.max_distance;
        for root_index in 0..2 {
            let distance = if root_index == 0 { roots.x } else { roots.y };
            if distance <= 0.0 || distance >= nearest {
                continue;
            }
            let point = origin + local.direction * distance;
            let fraction = point.x * handedness / reach;
            if !(0.0..1.0).contains(&fraction) {
                continue;
            }
            let chord =
                dimensions.y * (0.55 + (fraction * PI).sin().sqrt()) * (1.0 - fraction).sqrt();
            let chord_position = point.z - reach * fraction * fraction * 0.35;
            let fringe = (fraction * 19.0).fract_gl();
            let trailing = chord * (0.7 - fraction * 0.15 * (fringe * PI).sin());
            if chord_position >= -chord * 0.45 && chord_position <= trailing {
                nearest = distance;
            }
        }
        if nearest >= ray.max_distance {
            continue;
        }
        let point = origin + local.direction * nearest;
        let fraction = point.x * handedness / reach;
        let chord_position = point.z - reach * fraction * fraction * 0.35;
        let chord = dimensions.y * (0.55 + (fraction * PI).sin().sqrt()) * (1.0 - fraction).sqrt();
        let feather_coordinate = fraction * 24.0 + chord_position / dimensions.y * 0.6;
        let ridge = ((feather_coordinate.fract_gl() - 0.5).abs() * 2.0).powi(3);
        let tips = (chord_position / chord).smoothstep(0.05, 0.65);
        let covert = (-chord_position / chord).smoothstep(0.05, 0.4);
        let variation = (feather_coordinate.floor() * 2.4).sin() * 0.06;
        let detail = 1.0 - (ray.footprint * nearest / dimensions.y).smoothstep(0.001, 0.02);
        let filaments =
            (chord_position / dimensions.y * 180.0 + fraction * 60.0).sin() * 0.025 * detail;
        let colour = tint.lerp(vec3(0.045, 0.095, 0.15), tips * 0.65).lerp(
            vec3(0.72, 0.53, 0.25),
            covert * fraction.smoothstep(0.25, 0.95) * 0.6,
        );
        let mut material =
            Material::skin(colour * (1.0 + variation + filaments - ridge * 0.16), 0.65);
        material.translucency = 0.18;
        let mut normal = vec3(
            -handedness * (lift + 2.0 * curvature * point.x * handedness),
            1.0,
            0.0,
        )
        .normalize();
        if normal.dot(local.direction) > 0.0 {
            normal = -normal;
        }
        let old_lift = creature.previous_animation.x * 0.45 - creature.previous_animation.y * 0.12;
        let old_bend = 0.12 + creature.previous_animation.x * 0.2;
        let previous = vec3(
            handedness * (root.x + fraction * old_reach),
            root.y + old_reach * (old_lift * fraction + old_bend * fraction * fraction),
            root.z + old_reach * fraction * fraction * 0.35 + chord_position,
        );
        let mut hit = make_hit(ray, creature, nearest, normal, previous, material);
        hit.part = 1;
        emit(hit);
    }
}

fn wing_roots(origin: Vec2, direction: Vec2, lift: f32, curvature: f32) -> Vec2 {
    let quadratic = curvature * direction.x * direction.x;
    let linear = (lift + 2.0 * curvature * origin.x) * direction.x - direction.y;
    let constant = lift * origin.x + curvature * origin.x * origin.x - origin.y;
    let disc = linear * linear - 4.0 * quadratic * constant;
    if disc < 0.0 {
        return Vec2::splat(-1.0);
    }
    if quadratic.abs() < 1e-8 {
        if linear.abs() < 1e-8 {
            return Vec2::splat(-1.0);
        }
        Vec2::splat(-constant / linear)
    } else {
        let stable = -0.5
            * (linear
                + if linear < 0.0 {
                    -disc.sqrt()
                } else {
                    disc.sqrt()
                });
        Vec2::new(
            stable / quadratic,
            if stable.abs() > 1e-10 {
                constant / stable
            } else {
                -linear / (2.0 * quadratic)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::Vec4;

    #[test]
    fn feathered_wings_remain_attached_and_reproject_when_folded() {
        let root = vec3(4.6, 0.7, -5.0);
        let dimensions = Vec2::new(34.0, 7.0);
        for stroke in [-0.3, 0.0, 0.3] {
            for fold in [0.0, 1.0] {
                let animation = Vec4::new(stroke, fold, 0.0, 0.0);
                let creature = CreatureInstance {
                    rotation: Vec4::W,
                    previous_rotation: Vec4::W,
                    animation,
                    previous_animation: animation,
                    ..CreatureInstance::default()
                };
                let reach = dimensions.x * (1.0 - fold * 0.6);
                for handedness in [-1.0, 1.0] {
                    for fraction in [0.001, 0.5, 0.9] {
                        let point = vec3(
                            handedness * (root.x + reach * fraction),
                            root.y
                                + reach
                                    * ((stroke * 0.45 - fold * 0.12) * fraction
                                        + (0.12 + stroke * 0.2) * fraction * fraction),
                            root.z + reach * fraction * fraction * 0.35,
                        );
                        let ray = Ray {
                            origin: point + Vec3::Y * 30.0,
                            direction: -Vec3::Y,
                            light: Vec3::Y,
                            ambient: Vec3::ONE,
                            radiance: Vec3::ONE,
                            max_distance: 100.0,
                            footprint: 0.001,
                        };
                        let mut hit = Hit::MISS;
                        wings(ray, &creature, root, dimensions, Vec3::ONE, |layer| {
                            hit = layer;
                        });
                        assert!(
                            hit.is_hit(),
                            "missing wing at stroke={stroke}, fold={fold}, span={fraction}"
                        );
                        assert!(
                            (hit.distance() - 30.0).abs() < 0.001,
                            "wing detached from its profile"
                        );
                        assert!(
                            hit.previous_position.distance(point) < 0.001,
                            "stationary feather has incorrect motion"
                        );
                        assert!(
                            (hit.normal.length() - 1.0).abs() < 0.001,
                            "invalid wing normal"
                        );
                    }
                }
            }
        }
    }
}
