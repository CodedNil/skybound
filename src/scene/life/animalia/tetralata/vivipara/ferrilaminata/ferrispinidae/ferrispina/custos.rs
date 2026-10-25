use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Social},
    eyes, gills, mouth, prepare_pose, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail, torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 2200.0,
    lifespan_days: 42000.0,
    social: Social::Guardian,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

taxon!(
    Species,
    "Ferrispina custos",
    "Iron Spined Guardians",
    [
        "Iron Spined Guardians are heavy-bodied Shredders that remain near communal nursery currents.",
        "Conspicuous spinal blades and repeated banking patrols discourage predators from approaching slowly developing young.",
        "Their attacks retain the sweeping motion of roaming relatives, but begin from a defended area rather than a prolonged pursuit.",
        "Vocal exchanges and the motion of nearby adults help maintain a loose moving boundary around the nursery.",
        "The nursery travels with a suitable current, so protection depends on recognising companions rather than defending a fixed place.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 6.0, 14.0, 0);
    c.opaque_bounds = vec3(6.0, 7.0, 27.0).extend(0.0);
    rig_tentacles(c, mouth(7.0, 1.1), 1.1, BIOLOGY.oral_pairs);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(2.5, 2.3, 7.0), 1.1), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.5);
    // Born as soft nubs, so they cannot wound the brood chamber; the
    // blades harden and lengthen as the animal matures.
    let grown = c.life.y * c.life.y;
    for i in 0..7 {
        let z = i as f32 * 2.0 - 5.5;
        hit.join(
            super::blade(
                p,
                vec3(0.0, 1.8, z),
                vec3(0.0, 1.8, z).lerp(vec3(0.0, 5.6, z + 2.0), grown),
                0.7 * (0.4 + 0.6 * grown),
            ),
            6,
        );
    }
    hit.join(gills(p, 7.0, 2), 9);
    // A guardian meets attackers head on: its cowl is roofed by an iron
    // shield, thickest at the rim it strikes with.
    let shield = rounded(
        p - vec3(0.0, 0.75, -8.3),
        vec3(2.55, 0.32, 2.3) * (0.6 + 0.4 * grown),
    );
    hit.join(shield, 6);
    hit.join(eyes(p, 7.0, 1.1), 3);
    hit.join(
        oral_tentacles(p, c, mouth(7.0, 1.1), 1.1, BIOLOGY.oral_pairs),
        8,
    );
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.26, 0.45, 0.43, 0.0),
    vec3(0.86, 0.68, 0.28),
    [vec4(17.0, 3.0, -2.0, 0.98), vec4(11.0, 2.2, 2.8, 0.98)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
