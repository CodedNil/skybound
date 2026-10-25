use crate::scene::life::{
    CreatureInstance, Look, annulated,
    biology::Biology,
    eyespots, prepare_pose, ribbons, rig_ribbons, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tail,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 60.0,
    lifespan_days: 900.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 40.4;

taxon!(
    Species,
    "Lucerna undulata",
    "Shimmerays",
    [
        "Shimmerays spread broad wings and luminous trailing structures through currents carrying algae and other organic particles.",
        "Their glow also draws Sparkies close enough for the mouth appendages to sweep them into the feeding stream.",
        "During the bright aur phase they rise, then descend toward the upper edge of the Veilands as the cycle dims.",
        "Food comes from living particles and organic residues carried above the ocean, while concentrated aur remains dangerous to their tissues.",
        "Seen from afar, their coordinated vertical movement forms long shifting ribbons of light over the lower Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 17.0, 2);
    c.opaque_bounds = vec3(5.0, 5.0, 24.0).extend(0.0);
    // Sticky streamers under the forebody comb particles from the air it
    // ripples through.
    rig_ribbons(c, 2, vec3(0.0, -0.55, -3.5), 1.2, 14.0);
}

/// A long, ribbon-bodied drifter rippling as it goes: a flattened, finely
/// ringed body with a row of glowing organs along its back and trailing
/// ribbons beneath.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let ripple = (p.z * 0.7 - c.animation.z * 2.0).sin() * 0.25;
    let q = p - vec3(0.0, ripple, 0.0);
    let body = annulated(
        q,
        vec3(0.0, 0.0, -8.6),
        vec3(0.0, 0.0, 8.8),
        vec3(0.65, 1.05, 0.55),
        20.0,
        0.05,
        0.62,
    );
    let mut hit = Surface::new(smooth_min(body, tail(p, c), 0.3), 0);
    for i in 0..8 {
        let z = i as f32 * 2.0 - 7.0;
        hit.join(rounded(q - vec3(0.0, 0.6, z), vec3(0.42, 0.13, 0.26)), 9);
    }
    hit.join(eyespots(q, vec3(0.32, 0.25, -8.6), 0.4, 2, 0.11), 3);
    hit.join(ribbons(p, c, 2), 4);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.16, 0.26, 0.41, 0.0),
    vec3(0.22, 0.82, 0.89),
    [vec4(12.0, 5.0, -3.0, 0.25), vec4(9.0, 3.0, 3.0, 0.25)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
