use super::blade;
pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth, prepare_pose, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1800.0,
    lifespan_days: 30000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Ferrispina falcata",
    "Shredders",
    [
        "Shredders are long, flexible predators with two wing pairs and hardened blades running along the spine.",
        "They bank past prey in curved cutting passes, keeping the vulnerable underside away from a struggling target.",
        "Young animals hunt in loose coordinated groups that split feeding herds and exploit the openings created by a companion.",
        "Mouth tentacles retain a captured piece of food while the body recovers into rising air.",
        "Their hunting follows herd routes and open gaps between plant communities, where a long body has room to turn.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 14.0, 0);
    c.opaque_bounds = vec3(5.0, 6.0, 29.0).extend(0.0);
    rig_tentacles(c, mouth(8.0, 0.8), 1.0, BIOLOGY.oral_pairs);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(1.7, 1.5, 8.0), 0.8), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.35);
    // Born as soft nubs, so they cannot wound the brood chamber; the
    // blades harden and lengthen as the animal matures.
    let grown = c.life.y * c.life.y;
    for i in 0..9 {
        let z = i as f32 * 1.6 - 6.0;
        hit.join(
            blade(
                p,
                vec3(0.0, 1.2, z),
                vec3(0.0, 1.2, z).lerp(vec3(0.0, 4.5, z + 2.0), grown),
                0.5 * (0.4 + 0.6 * grown),
            ),
            6,
        );
    }
    hit.join(eyes(p, 8.0, 0.8), 3);
    hit.join(
        oral_tentacles(p, c, mouth(8.0, 0.8), 1.0, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.19, 0.22, 0.31, 0.0),
    vec3(0.66, 0.19, 0.14),
    [vec4(15.0, 2.8, -2.0, 0.98), vec4(10.0, 2.0, 3.2, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
