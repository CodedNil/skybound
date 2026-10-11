use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use super::biology::{Biology, Digestion, Exchange, Lift, Transport};
#[cfg(not(target_arch = "spirv"))]
use super::{CreatureInstance, Kind};
use super::{ellipsoid, smooth_min};
use crate::scene::surface::Surface;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::surface::{Ray, TraceResult};

pub fn organs(point: Vec3, biology: Biology) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let mut surface = Surface::new(1e6, 0);
    if biology.organs.digestion != Digestion::None {
        let mut crop = 1e6f32;
        for index in 0..3 {
            let segment = index as f32;
            crop = smooth_min(
                crop,
                ellipsoid(
                    point - vec3(0.0, -0.2 + segment * 0.07, -0.48 + segment * 0.35),
                    vec3(0.32, 0.26, 0.3),
                ),
                0.13,
            );
        }
        surface.join(crop, 1);
    }
    let exchange = match biology.organs.exchange {
        Exchange::Surface => ellipsoid(point, vec3(0.72, 0.66, 0.78)).abs() - 0.025,
        Exchange::Pores | Exchange::FlowChannels => {
            let folds = (point.z * 45.0).sin() * 0.025;
            ellipsoid(mirrored - vec3(0.48, 0.1, -0.2), vec3(0.16, 0.28, 0.5)) + folds
        }
    };
    surface.join(exchange, 2);
    if biology.organs.transport != Transport::Diffusion {
        let radial = point.x.hypot(point.y);
        let pulse = (point.z * 12.0).cos() * 0.018;
        surface.join((radial - 0.42).hypot(point.z * 0.55) - 0.07 - pulse, 3);
    }
    surface.join(
        ellipsoid(mirrored - vec3(0.33, -0.32, 0.3), vec3(0.14, 0.15, 0.25)),
        4,
    );
    if matches!(biology.organs.lift, Lift::Bladders | Lift::Membrane) {
        for index in 0..3 {
            surface.join(
                ellipsoid(
                    mirrored - vec3(0.32, 0.35, (index as f32 - 1.0) * 0.48),
                    vec3(0.25, 0.3, 0.3),
                )
                .abs()
                    - 0.025,
                5,
            );
        }
    }
    if biology.organs.aur_capacity > 0.0 {
        surface.join(
            ellipsoid(mirrored - vec3(0.42, 0.36, 0.23), vec3(0.16, 0.16, 0.25)),
            6,
        );
    }
    surface
}

#[cfg(not(target_arch = "spirv"))]
pub fn render(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    use crate::scene::surface::{Material, trace_material};
    let biology = super::KINDS[creature.kind.x as usize].biology();
    let (center, scale) = match super::KINDS[creature.kind.x as usize] {
        Kind::Nae => (vec3(0.0, 0.03, 0.0), vec3(0.22, 0.33, 0.16)),
        Kind::Naetu => (vec3(0.0, 0.0, -5.0), vec3(4.0, 4.0, 12.0)),
        _ => (
            Vec3::ZERO,
            creature.opaque_bounds.truncate() * vec3(0.55, 0.55, 0.35),
        ),
    };
    let mut result = super::render(ray, creature);
    result.opaque = trace_material(
        ray,
        creature,
        |point, c| {
            let body = super::body_surface(point, c);
            let mut cut = Surface::new(body.distance.max(-point.x), 0);
            let internal = organs((point - center) / scale, biology);
            cut.join(
                (internal.distance * scale.min_element()).max(body.distance),
                internal.material,
            );
            cut
        },
        |point, id| {
            let color = match id {
                1 => vec3(0.75, 0.34, 0.14),
                2 => vec3(0.25, 0.7, 0.62),
                3 => vec3(0.32, 0.22, 0.48),
                4 => vec3(0.22, 0.46, 0.72),
                5 => vec3(0.78, 0.72, 0.36),
                6 => vec3(0.38, 0.85, 0.95),
                _ => vec3(0.55, 0.62, 0.58),
            };
            let texture = 0.94 + 0.06 * (point.z / scale.z * 80.0).sin();
            let mut material = Material::skin(color * texture, 0.35);
            material.translucency = 0.2;
            if id == 6 {
                material.emission = 1.5;
            }
            material
        },
    );
    result
}
