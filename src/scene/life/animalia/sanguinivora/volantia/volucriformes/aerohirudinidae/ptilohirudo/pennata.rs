use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    genome::Gene,
    rounded, smooth_min, sucker,
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::sanguinivora::volantia::WING_MUSCLE,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.2,
    maturity_days: 21.0,
    lifespan_days: 480.0,
    adult_mass: 0.01,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 30.4;

/// How broad its disc, how finely feathered its fringe, and how deeply it ripples: a broad disc
/// glides further on the updraft, and a fine, deeply rippling fringe flies quieter onto a host.
pub const DISC: Gene = Gene(38);
pub const FEATHERING: Gene = Gene(39);
pub const RIPPLE: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Ptilohirudo pennata",
    "Feathered Leeches",
    [
        "Feathered Leeches glide along cloud margins on finely fringed wings surrounding a compact body.",
        "Their vision detects weak aur-lit contrasts, helping them find hosts hidden by mist.",
        "A quick attachment supplies a fluid meal before they release and return to cover.",
        "They favour broken cloud edges where passing animals are visible without requiring a long flight through dry air.",
        "They favour humid cloud pockets on the warmer margins of the equatorial Highlands, avoiding both frozen Shard Peaks and the strongest poleward winds.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(3.6, 1.6, 3.6).extend(0.0);
}

/// A glider of the cloud margins whose one wing pair has grown right round it.
///
/// A single finely fringed disc encircles a compact body, rippling as it steers, so it slips along
/// the mist edge in any direction. Two large eyes flank the sucker, built for the weak aur-lit
/// contrasts by which it finds hosts hidden in cloud.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = rounded(p, vec3(0.6, 0.5, 0.95));
    let radial = vec2(p.x, p.z).length();
    let around = p.z.atan2(p.x);
    let ripple =
        (around * 3.0 + c.animation.z * 2.5).sin() * (0.07 + c.gene(RIPPLE) * 0.1) * radial / 3.2;
    let droop = radial * radial * 0.03;
    let fringe = 2.8
        + c.gene(DISC) * 0.5
        + (around * 64.0).sin() * (0.05 + c.gene(FEATHERING) * 0.15)
        + (around * 9.0).sin() * 0.1;
    let thickness = 0.07 * (1.0 - radial / 3.4).max(0.15);
    let disc = ((p.y + droop - ripple).abs() - thickness)
        .max(radial - fringe)
        .max(0.45 - radial)
        * 0.6;
    let mut hit = Surface::new(smooth_min(body, disc, 0.25), 0);
    if disc < body {
        hit.material = 1;
    }
    hit.blend(
        sucker(
            p,
            vec3(0.0, -0.05, -0.95),
            vec3(0.0, -0.2, -1.0).normalize(),
            0.3,
        ),
        8,
        0.06,
    );
    hit.join(
        rounded(
            vec3(p.x.abs(), p.y, p.z) - vec3(0.32, 0.22, -0.72),
            vec3(0.2, 0.2, 0.18),
        ),
        3,
    );
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.01, -1.00), vec3(0.05, 0.05, 0.05)),
    site(PUMP, vec3(0.00, -0.04, -0.55), vec3(0.14, 0.08, 0.26)),
    site(CROP, vec3(0.00, 0.01, 0.35), vec3(0.36, 0.20, 0.40)),
    pair(WING_MUSCLE, vec3(0.32, 0.09, -0.19), vec3(0.14, 0.08, 0.17)),
    site(BROOD, vec3(0.00, -0.04, 0.93), vec3(0.06, 0.06, 0.06)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.25, 0.32, 0.42, 0.0),
    vec3(0.69, 0.73, 0.80),
    // Its disc is its wing: there are no separate membranes.
    [Vec4::ZERO; 2],
)
.skinned(vec4(5.0, 0.8, 1.4, -0.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::Physiology;

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 24.0,
        ..super::super::PHYSIOLOGY
    };
}
