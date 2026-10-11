use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, capsule, ellipsoid, eyes, legs, prepare_pose,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Sputorexidae",
    english_name: "Spitfire Beetles",
    description: "Spitfire Beetles are small, armoured predators with specialised mouths that launch sticky acidic saliva. \
        The projected secretion can immobilise prey or corrode a threat before the animal approaches. \
        Compact wings and sturdy limbs let them work around vegetation and Weaver structures. \
        Despite their size, their ability to attack from a distance makes them dangerous to disturb.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 24.7,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 15.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 3.0, 4.0, 3);
    c.opaque_bounds = vec3(5.0, 5.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, _c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.7, 1.4, 2.8)), 5);
    hit.join(ellipsoid(p - vec3(0.0, 0.15, -2.2), vec3(1.2, 1.0, 1.2)), 5);
    hit.join(legs(p, vec3(1.7, 1.4, 2.8), 3), 0);
    hit.join(
        capsule(p, vec3(0.0, -0.3, -2.8), vec3(0.0, -0.2, -4.0), 0.3),
        8,
    );
    hit.join(eyes(p, 2.8, 0.65), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.18, 0.44, 0.29, 0.0),
        vec3(0.28, 0.58, 0.62),
        [vec4(6.0, 2.2, 0.0, 0.95), vec4(4.0, 1.8, 3.0, 0.32)],
    )
}
