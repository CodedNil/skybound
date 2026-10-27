#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::sanguinivora::volantia::WING_MUSCLE,
    organs::{Function, Organ, Viscera, pair, site},
};
use crate::scene::{
    life::{
        CreatureInstance, Look, annulated,
        biology::Biology,
        capsule, eyespots,
        genome::Gene,
        rounded, smooth_min, sucker,
        surface::{Ray, Surface, TraceResult},
    },
    math::Power,
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.1,
    maturity_days: 12.0,
    lifespan_days: 240.0,
    adult_mass: 0.002,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 16.0;

/// How large the lamp it carries, how heavily hooped, and how long the teeth round its mouth: a
/// large, bare lamp draws prey from further off, and long teeth hold what comes.
pub const LAMP_SIZE: Gene = Gene(38);
pub const HOOPS: Gene = Gene(39);
pub const TEETH: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Candela lucens",
    "Glowgnomes",
    [
        "Glowgnomes are tiny luminous sap feeders that inhabit sheltered burrows in Whisperwood, their soft ringed fronts trailing a swollen, translucent lamp sac hooped with ribs and laced with the sap veins that feed its light.",
        "They retain the single wing pair of their parasitic ancestors, but coordinated wing strokes now help draw the host toward favourable currents.",
        "Pheromone signals carried through six vertical sap veins recruit groups on different sides of the trunk.",
        "Their light draws visitors through nearby flowering growth in the dim equatorial skies, while sap and wood cavities provide food and shelter.",
        "Limited feeding and shared movement have turned an ancestral parasitic visit into a partnership on which the drifting tree depends.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(2.5, 2.0, 2.6).extend(0.0);
}

/// A burrow-dweller that is mostly lamp.
///
/// Its front is a short, soft, ringed grub with a small sucker and a fringe for sipping sap, and
/// only faint eyespots, for it lives in the dark of the wood; its whole rear has swollen into a
/// translucent sac lit from within, hooped by ribs and laced with the sap veins that feed the
/// light. Stubby wings are enough to carry it between burrows and to beat in time with its
/// neighbours, drawing the tree towards better air.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let grub = annulated(
        p,
        vec3(0.0, 0.0, -1.7),
        vec3(0.0, 0.05, 0.1),
        vec3(0.32, 0.5, 0.42),
        5.0,
        0.1,
        0.92,
    );
    let lamp_scale = 0.85 + c.gene(LAMP_SIZE) * 0.3;
    let lamp = rounded(p - vec3(0.0, 0.1, 1.0), vec3(0.78, 0.72, 1.0) * lamp_scale);
    let mut hit = Surface::new(smooth_min(grub, lamp + 0.03, 0.2), 0);
    hit.join(lamp, 9);
    // Ribs hooping the lamp, darker than the light between them.
    let q = p - vec3(0.0, 0.1, 1.0);
    for rib in 0..4 {
        let z = rib as f32 * 0.45 - 0.68;
        let across = (1.0 - (z / 1.0).power(2)).max(0.05).sqrt();
        let hoop = vec2(
            vec2(q.x / (0.78 * lamp_scale), q.y / (0.72 * lamp_scale)).length()
                * across
                * 0.75
                * lamp_scale
                - across * 0.75 * lamp_scale,
            q.z - z * lamp_scale,
        )
        .length()
            - (0.025 + c.gene(HOOPS) * 0.035);
        hit.join(hoop, 0);
    }
    let facing = vec3(0.0, -0.3, -1.0).normalize();
    hit.blend(sucker(p, vec3(0.0, -0.05, -1.75), facing, 0.28), 8, 0.05);
    let mouth = p - vec3(0.0, -0.05, -1.8);
    let around = mouth.y.atan2(mouth.x);
    let sector = TAU / 8.0;
    let turn = (around / sector).round() * sector;
    let (sine, cosine) = turn.sin_cos();
    let sway = (c.animation.z * 1.5 + turn).sin() * 0.04;
    hit.join(
        capsule(
            mouth,
            vec3(cosine * 0.25, sine * 0.25, 0.0),
            vec3(cosine, sine, 0.0) * (0.32 + c.gene(TEETH) * 0.16 + sway) + vec3(0.0, 0.0, -0.22),
            0.025,
        ),
        8,
    );
    hit.pit(eyespots(p, vec3(0.12, 0.3, -1.55), 0.12, 2, 0.025), 0.025);
    hit
}

/// The lamp gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const LAMP: Organ = Organ::new("lamp gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.09, -1.34), vec3(0.10, 0.09, 0.18)),
    site(PUMP, vec3(0.00, -0.02, -0.63), vec3(0.12, 0.11, 0.22)),
    site(CROP, vec3(0.00, 0.05, 0.46), vec3(0.31, 0.28, 0.31)),
    pair(WING_MUSCLE, vec3(0.23, 0.18, -0.16), vec3(0.09, 0.09, 0.09)),
    site(BROOD, vec3(0.00, -0.02, 1.34), vec3(0.13, 0.12, 0.39)),
    site(LAMP, vec3(0.00, 0.52, 0.96), vec3(0.20, 0.20, 0.20)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.55, 0.32, 0.15, 0.0),
    vec3(1.0, 0.72, 0.20),
    [vec4(2.0, 1.1, -0.6, 0.35), Vec4::ZERO],
)
.hinged([vec2(0.42, 0.3), Vec2::ZERO])
.styled([vec4(0.3, 1.0, 0.0, 0.0), vec4(0.3, 1.0, 0.0, 0.0)])
.framed([0.4, 0.4])
.skinned(vec4(5.0, 0.8, 1.6, -0.2))
.flying(vec4(4.00, 0.40, 0.20, 0.60))
.marked(vec4(5.00, 4.00, 0.70, 0.40), vec3(1.00, 0.85, 0.40));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{
        Kind,
        animalia::physiology::{Light, Physiology},
        organs::Function,
    };
    pub const PHYSIOLOGY: Physiology = Physiology {
        crops: Function::Sap,
        light: Light::Aurlit,
        eats: |_, food| food == Kind::Susurrus,
        parasite: false,
        attaches: true,
        partner: |_, tree| tree == Kind::Susurrus,
        speed: 1.0,
        ..super::super::PHYSIOLOGY
    };
}
