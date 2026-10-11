use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, eyes, feathers, prepare_pose, smooth_min, tail, tapered_tail,
    torso,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Umbraevolidae",
    english_name: "Shadowings",
    description: "Shadowings are large, intelligent scavengers covered in muted, light-absorbing dark plumage. \
        Fine sensitivity to wind and thermal currents helps them search vast territories for carcasses left by apex predators. \
        Sharp, echoing cries warn kin and other animals, coordinate safe feeding and discourage threats. \
        Broad wings support long journeys and patient circling over potential feeding sites.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 49.8,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 32.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 8.0, 14.0, 0);
    c.opaque_bounds = vec3(7.0, 6.0, 21.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(3.4, 3.7, 7.2), 1.0), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.45);
    hit.join(feathers(p, c, 4.6), 1);
    hit.join(
        tapered_tail(p, vec3(0.0, 0.5, -9.0), vec3(0.0, -0.4, -12.0), 0.8, 0.1),
        6,
    );
    hit.join(eyes(p, 7.2, 1.0), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.06, 0.08, 0.14, 0.0),
        vec3(0.16, 0.23, 0.29),
        [vec4(20.0, 5.0, 0.0, 0.98), vec4(13.0, 3.0, 9.0, 0.98)],
    )
}
