use crate::scene::life::Tetralate;

pub const BIOLOGY: Biology = Biology {
    size: 0.1,
    maturity_days: 75.0,
    lifespan_days: 1500.0,
    adult_mass: 0.12,
    // A clutch of four to nine glued round the mother and guarded with acid.
    litter: Litter::new(4, 1, [0.1, 0.2, 0.3, 0.22, 0.12, 0.06]),
    gestation_days: 8.0,
    altitude: (1500.0, 6000.0),
};

pub const RENDER_RADIUS: f32 = 24.7;

/// How large its acid glands swell, how long a spout it aims with, and how heavy and sharply
/// stepped its plates.
///
/// Big glands fire more shots between refills, a long spout fires truer, and thick plates turn more
/// strikes at the cost of weight.
pub const GLANDS: Gene = Gene(38);
pub const SPOUT: Gene = Gene(39);
pub const PLATES: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

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

pub fn animate(c: &mut Tetralate) {
    c.opaque_bounds = vec3(4.6, 3.2, 4.6).extend(0.0);
    rig_tentacles(c, MOUTH, 0.75, ORAL_PAIRS);
}

/// Where the gripping tentacles hang, behind the spout.
const MOUTH: Vec3 = vec3(0.0, -0.55, -1.5);

/// A sniper in armour.
///
/// A short, wedge-shaped body lies under overlapping plates, each riding over the next, broad at
/// the shoulders where the two acid glands lie and showing through windows in the plates. The spout
/// juts forward under the blunt head, and above it a pair of eyes looks straight ahead along the
/// line of fire, with a second pair looking down.
pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
    let mut body = 1e6f32;
    // Heavy plates stand proud of one another at sharp steps; light ones run smoothly together.
    let heavy = c.gene(PLATES);
    for plate in 0..4 {
        let f = plate as f32;
        let z = -1.1 + f * 0.75;
        let size = vec3(1.05 - f * 0.17, 0.62 - f * 0.08 + heavy * 0.06, 0.55);
        body = smooth_min(
            body,
            rounded(p - vec3(0.0, 0.05 - f * 0.04, z), size),
            0.14 - heavy * 0.11,
        );
    }
    let head = rounded(p - vec3(0.0, -0.05, -1.75), vec3(0.62, 0.45, 0.48));
    // The breathing louvres every tetralate has, along the flanks behind the acid glands.
    let body = louvres(
        smooth_min(body, head, 0.2),
        p - vec3(0.0, 0.05, 1.5),
        vec3(1.0, 0.62, 2.0),
    );
    let mut hit = Surface::new(body, 0);
    // The acid glands, swelling through the plates at the shoulders.
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.blend(
        ellipsoid(
            side - vec3(0.72, -0.05, -0.55),
            vec3(0.42, 0.32, 0.55) * (0.85 + c.gene(GLANDS) * 0.35),
        ),
        2,
        0.15,
    );
    // The spout, a hollow horn under the head.
    let tip = -2.5 - c.gene(SPOUT) * 0.6;
    let spout = tapered_tail(p, vec3(0.0, -0.32, -1.9), vec3(0.0, -0.42, tip), 0.2, 0.09);
    let bore = capsule(
        p,
        vec3(0.0, -0.37, -2.2),
        vec3(0.0, -0.43, tip - 0.25),
        0.05,
    );
    hit.join(spout.max(-bore), 7);
    hit.socket((side - vec3(0.3, 0.12, -2.12)).length() - 0.15, 0.15);
    hit.socket((side - vec3(0.42, -0.32, -1.85)).length() - 0.11, 0.11);
    // A pair of short, toothed mandibles either side of the spout, which part sideways as it
    // feeds to shear the softened meat its acid has opened, and close over the spout's root in
    // flight.
    let open = gape(c);
    let hinge = vec3(0.32, -0.42, -1.95);
    let tip = vec3(0.1 + open * 0.42, -0.5, -2.55 + open * 0.12);
    let bend = hinge.lerp(tip, 0.5) + vec3(0.16 + open * 0.08, -0.04, 0.0);
    let mandible = curved_tube(side, hinge, bend, tip, 0.09);
    let notch = ((side.z - hinge.z) / 0.12).fract_gl() - 0.5;
    hit.join(mandible + notch.abs() * 0.02, 7);
    hit.blend(oral_tentacles(p, c, MOUTH, 0.75, ORAL_PAIRS), 8, 0.08);
    hit
}

/// The acid gland, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const ACID: Organ = Organ::new("acid gland", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[pair(ACID, vec3(0.72, -0.06, -0.48), vec3(0.22, 0.22, 0.22))];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.18, 0.44, 0.29, 0.0),
    vec3(0.95, 0.55, 0.12),
    [vec4(3.4, 0.9, -0.7, 0.6), vec4(4.6, 2.0, 0.6, 0.35)],
)
.hinged([vec2(0.85, 0.35), vec2(0.7, 0.3)])
.stalked([vec2(0.28, 0.45), vec2(0.28, 0.45)])
.styled([vec4(0.9, 0.2, 0.0, 0.0), vec4(0.4, 0.8, 3.0, 0.0)])
.framed([1.6, 0.6])
.edged(5)
.skinned(vec4(2.0, 1.2, 1.2, 0.10))
.flying(vec4(2.20, 0.30, 0.20, 0.80))
.marked(vec4(4.00, 4.00, 0.80, 0.42), vec3(0.10, 0.16, 0.10))
.brooding(vec4(-0.2, 0.4, 0.95, 0.62));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use super::*;
    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent},
            physiology::{Brood, Physiology},
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        brood: Brood::Carried,
        speed: 10.0,
        ..super::super::PHYSIOLOGY
    };
    /// A Spitfire with acid to spare turns on a hunter that comes close rather than flee, and a
    /// mother with her eggs glued round her all the more so: it holds off at its range and spits.
    pub const ETHOLOGY: Ethology = Ethology {
        urges: |s, urges| {
            (super::super::ETHOLOGY.urges)(s, urges);
            if s.seen
                .threat
                .is_some_and(|threat| threat.offset.length() < 40.0)
            {
                let brooding = s.drives.term.max(s.drives.borne);
                urges.add(
                    STAND,
                    s.threat * (0.5 + s.temper(AGGRESSION) * 0.6 + brooding * 0.6),
                );
            }
        },
        ..super::super::ETHOLOGY
    };

    /// Holding at spitting range from a hunter, and firing.
    const STAND: Intent = Intent {
        name: "stand its ground",
        act: |_, s| {
            Goal::new(
                Activity::Defend,
                s.seen
                    .threat
                    .map(|threat| threat.offset - threat.offset.normalize_or_zero() * 10.0),
                0.6,
            )
            .hovering()
        },
    };
}
