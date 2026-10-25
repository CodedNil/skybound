use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Brood, Social},
    capsule, ellipsoid, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    brood: Brood::Carried,
    maturity_days: 75.0,
    lifespan_days: 1500.0,
    speed: 10.0,
    social: Social::Solitary,
    adult_mass: 0.12,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 24.7;

taxon!(
    Species,
    "Sputator acidus",
    "Spitfire Beetles",
    [
        "Spitfire Beetles are compact, armoured predators that project the adhesive secretion inherited from silk-making ancestors as a sticky acidic droplet, fired from a spout under the head.",
        "Two great acid glands glow amber through windows in the overlapping plates of the flanks, and a pair of eyes set to look straight ahead, unusual in a world where most eyes look down, judges the range of each shot.",
        "Thick mouth tentacles grip vegetation, and the narrow, stiff forewings brace against the recoil while the broad hindwings hold the animal steady.",
        "A short attack across a feeding passage avoids the energy cost of chasing prey into open air, and careless discharge can destroy a useful Weaver anchor.",
        "Mothers carry their eggs glued in a band round the body, guarded by the same acid, until they hatch.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.6, 3.2, 4.6).extend(0.0);
    rig_tentacles(c, MOUTH, 0.75, BIOLOGY.oral_pairs);
}

/// Where the gripping tentacles hang, behind the spout.
const MOUTH: Vec3 = vec3(0.0, -0.55, -1.5);

/// A sniper in armour.
///
/// A short, wedge-shaped body lies under overlapping plates, each riding over
/// the next, broad at the shoulders where the two acid glands lie and showing
/// through windows in the plates. The spout juts forward under the blunt head,
/// and above it a pair of eyes looks straight ahead along the line of fire,
/// with a second pair looking down.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut body = 1e6f32;
    for plate in 0..4 {
        let f = plate as f32;
        let z = -1.1 + f * 0.75;
        let size = vec3(1.05 - f * 0.17, 0.62 - f * 0.08, 0.55);
        body = smooth_min(body, rounded(p - vec3(0.0, 0.05 - f * 0.04, z), size), 0.08);
    }
    let head = rounded(p - vec3(0.0, -0.05, -1.75), vec3(0.62, 0.45, 0.48));
    let mut hit = Surface::new(smooth_min(body, head, 0.2), 0);
    // The acid glands, swelling through the plates at the shoulders.
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        ellipsoid(side - vec3(0.72, -0.05, -0.55), vec3(0.42, 0.32, 0.55)),
        2,
    );
    // The spout, a hollow horn under the head.
    let spout = tapered_tail(
        p,
        vec3(0.0, -0.32, -1.9),
        vec3(0.0, -0.42, -2.75),
        0.2,
        0.09,
    );
    let bore = capsule(p, vec3(0.0, -0.37, -2.2), vec3(0.0, -0.43, -3.0), 0.05);
    hit.join(spout.max(-bore), 7);
    hit.join((side - vec3(0.3, 0.12, -2.12)).length() - 0.15, 3);
    hit.join((side - vec3(0.42, -0.32, -1.85)).length() - 0.11, 3);
    hit.join(oral_tentacles(p, c, MOUTH, 0.75, BIOLOGY.oral_pairs), 8);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.18, 0.44, 0.29, 0.0),
    vec3(0.95, 0.55, 0.12),
    [vec4(3.4, 0.9, -0.7, 0.6), vec4(4.6, 2.0, 0.6, 0.35)],
)
.hinged([vec2(0.85, 0.35), vec2(0.7, 0.3)])
.styled([vec4(0.9, 0.2, 0.0, 0.0), vec4(0.4, 0.8, 3.0, 0.0)])
.framed([1.6, 0.6])
.edged(5)
.skinned(vec4(2.0, 1.2, 1.2, 0.10))
.flying(vec4(2.20, 0.30, 0.20, 0.80))
.marked(vec4(4.00, 4.00, 0.80, 0.42), vec3(0.10, 0.16, 0.10))
.brooding(vec4(-0.2, 0.4, 0.95, 0.62));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
