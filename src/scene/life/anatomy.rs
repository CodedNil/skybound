//! A cutaway of a creature, its near half removed to show its organs where its species
//! carries them.

use crate::scene::life::{
    self, CreatureInstance, KINDS, ellipsoid,
    organs::{Function, Viscera},
    surface::{Material, Ray, Surface, TraceResult, trace_material},
};
use spirv_std::glam::{Vec3, vec3};

/// The organs as one surface, each site its own material by its index.
pub fn organs(point: Vec3, viscera: &Viscera) -> Surface {
    let mut surface = Surface::new(1e6, 0);
    for (index, site) in viscera.sites().iter().enumerate() {
        let local = if site.paired {
            vec3(point.x.abs(), point.y, point.z)
        } else {
            point
        };
        surface.join(ellipsoid(local - site.centre, site.radii), index as u32 + 1);
    }
    surface
}

pub fn render(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    let viscera = KINDS[creature.kind.x as usize].organs();
    let mut result = life::render(ray, creature);
    result.opaque = trace_material(
        ray,
        creature,
        |point, c| {
            let body = life::body_surface(point, c);
            let mut cut = Surface::new(body.distance.max(-point.x), 0);
            let internal = organs(point, &viscera);
            cut.join(internal.distance.max(body.distance), internal.material);
            cut
        },
        |_, id| {
            let tint = if id == 0 {
                vec3(0.55, 0.62, 0.58)
            } else {
                viscera.sites()[id as usize - 1].organ.colour
            };
            let mut material = Material::skin(tint, 0.35);
            material.translucency = 0.2;
            if id > 0 && viscera.sites()[id as usize - 1].organ.function == Function::AurStore {
                material.emission = 1.5;
            }
            material
        },
    );
    result
}
