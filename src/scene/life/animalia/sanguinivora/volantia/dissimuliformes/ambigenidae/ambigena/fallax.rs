use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, eyespots, rounded, smooth_min, sucker,
    surface::{Ray, Surface, TraceResult},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

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
    c.opaque_bounds = vec3(3.0, 3.0, 4.0).extend(0.0);
}

/// A stout flyer that passes for a harmless grazer: a deep, round-backed
/// body and a short, soft face, until the mouth opens.
/// The sucker, leading.
const CUP: Vec3 = vec3(0.0, -0.1, -2.0);

/// A cleaner's impostor, and a blood-eater underneath.
///
/// It shares nothing with the Sweepers it copies but their outline: a soft,
/// swollen sac of a body, a soft crest of papillae along the back where theirs
/// bristles, and a broad flared sucker ringed by a fringe of mobile filaments,
/// which from a little way off reads as a Sweeper's brush face. Up close the
/// fringe is what it is: delicate mouthparts that groom the host and dab
/// antiseptic on the wound while the sucker takes its small, concealed meal. A
/// short arc of eyespots crowns the cup. Like all blood-eaters it breathes
/// through its glistening skin, and flies on a single pair of wings that grew
/// from folds of the body wall.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut body = rounded(p - vec3(0.0, 0.0, 0.4), vec3(1.45, 1.25, 2.0));
    for i in 0..7 {
        let z = i as f32 * 0.5 - 1.2;
        let rise = 1.2 - (z - 0.4).powi(2) * 0.09;
        body = smooth_min(
            body,
            rounded(p - vec3(0.0, rise, z), vec3(0.16, 0.32, 0.16)),
            0.14,
        );
    }
    let facing = vec3(0.0, -0.2, -1.0).normalize();
    let mut hit = Surface::new(body, 0);
    hit.join(sucker(p, CUP, facing, 0.85), 8);
    // The fringe: a filament to each sector of the cup's rim, swaying.
    let q = p - CUP;
    let around = q.y.atan2(q.x);
    let sector = TAU / 14.0;
    let turn = (around / sector).round() * sector;
    let (sine, cosine) = turn.sin_cos();
    let sway = (c.animation.z * 2.0 + turn * 3.0).sin() * 0.12;
    let base = vec3(cosine * 0.8, sine * 0.8, 0.1);
    let tip = vec3(cosine * (1.25 + sway), sine * (1.25 + sway), -0.55);
    hit.join(capsule(q, base, tip, 0.045), 8);
    hit.join(eyespots(p, vec3(0.3, 0.72, -1.75), 0.22, 3, 0.05), 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.67, 0.72, 0.46, 0.0),
    vec3(0.57, 0.38, 0.29),
    [vec4(6.0, 2.3, 0.2, 0.35), Vec4::ZERO],
)
.hinged([vec2(1.35, 0.4), Vec2::ZERO])
.styled([vec4(0.7, 0.8, 0.0, 0.6), vec4(0.7, 0.8, 0.0, 0.6)])
.framed([0.3, 0.3])
.skinned(vec4(5.0, 1.0, 1.2, -0.25))
.flying(vec4(1.20, 0.35, 0.30, 0.90))
.marked(vec4(2.00, 4.00, 0.80, 0.42), vec3(0.25, 0.20, 0.12));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
