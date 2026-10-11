use crate::scene::life::biology::{Biology, Feeding, Social};

pub const BIOLOGY: Biology = Biology {
    feeding: Feeding::Filter,
    speed: 10.0,
    social: Social::Swarm,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, ellipsoid, prepare_pose, ribbons, smooth_min, tail,
};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Lucentiae",
    english_name: "Shimmerays",
    description: "Shimmerays are long, softly bioluminescent drifters with broad wings and trailing structures. \
        They filter microscopic nutrients and discarded auric particles from the air, and their glow also attracts Sparkies that they consume. \
        During the day they rise into higher currents; at night they descend toward the air just above the Veilands. \
        Their elongated luminous bodies make these vertical migrations visible across dim skies.",
    rank: Rank::Family,
    parent: Some(&super::TAXON),
    radius: 40.4,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 17.0, 2);
    c.opaque_bounds = vec3(5.0, 5.0, 24.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.0, 0.75, 9.0)), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.3);
    for i in 0..8 {
        let z = i as f32 * 2.0 - 7.0;
        hit.join(ellipsoid(p - vec3(0.0, 0.68, z), vec3(0.65, 0.16, 0.3)), 9);
    }
    hit.join(ribbons(p, c, 2), 4);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.16, 0.26, 0.41, 0.0),
        vec3(0.22, 0.82, 0.89),
        [vec4(12.0, 5.0, 0.0, 0.25), vec4(9.0, 3.0, 9.0, 0.25)],
    )
}
