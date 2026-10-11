use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    speed: 24.0,
    adult_mass: 0.01,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, eyes, feathers, prepare_pose};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Ptilohirudo",
    english_name: "Feathered Leeches",
    description: "Feathered Leeches are small gliders whose feathered wings support quick approaches and retreats through cloud layers. \
        Their compact bodies and wide-spectrum vision suit dim aur light and poor visibility. \
        They attach briefly to feed, then release and glide back into cover. \
        Clouds provide concealment between these hit-and-run visits.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 30.4,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 5.0, 0);
    c.opaque_bounds = vec3(4.0, 2.0, 10.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.7, 0.55, 2.8)), 0);
    hit.join(feathers(p, c, 2.0), 1);
    hit.join(eyes(p, 2.8, 0.5), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.25, 0.32, 0.42, 0.0),
        vec3(0.69, 0.73, 0.80),
        [vec4(9.0, 3.0, 0.0, 0.98)],
    )
}
