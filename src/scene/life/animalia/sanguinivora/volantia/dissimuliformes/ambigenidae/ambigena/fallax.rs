use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 90.0,
    lifespan_days: 2400.0,
    speed: 8.0,
    adult_mass: 0.04,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 22.6;

taxon!(
    Species,
    "Ambigena fallax",
    "Double Natured",
    [
        "Double Natured animals resemble Sweepers, with swollen bodies and mobile mouth fringes used during close grooming visits.",
        "They remove debris and apply antiseptic secretions while taking small amounts of host fluid through the inherited feeding sucker.",
        "Genuine cleaning reduces the irritation of feeding and gives the host a reason to tolerate their presence.",
        "Frequent short visits are safer than remaining attached where a host can repeatedly inspect the same patch of skin.",
        "Their deceptive resemblance arose in a winged parasite lineage distinct from the four-winged cleaners they imitate.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(5.0, 5.0, 4.8).extend(0.0);
    rig_tentacles(c, mouth(2.3, 0.55), 0.8, BIOLOGY.oral_pairs);
}

/// A stout flyer that passes for a harmless grazer: a deep, round-backed
/// body and a short, soft face, until the mouth opens.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = torso(p, vec3(1.45, 1.35, 2.3), 0.55);
    // A soft hump over the shoulders, like the grazers it copies.
    let hump = rounded(p - vec3(0.0, 0.9, -0.6), vec3(1.0, 0.75, 1.3));
    let mut hit = Surface::new(smooth_min(body, hump, 0.5), 0);
    hit.join(
        oral_tentacles(p, c, mouth(2.3, 0.55), 0.8, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 2.3, 0.55), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.67, 0.72, 0.46, 0.0),
    vec3(0.57, 0.38, 0.29),
    [vec4(6.0, 2.3, -0.5, 0.35), Vec4::ZERO],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
