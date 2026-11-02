use crate::scene::life::Tetralate;

pub const BIOLOGY: Biology = Biology {
    size: 0.083,
    maturity_days: 30.0,
    lifespan_days: 750.0,
    adult_mass: 0.08,
    // One string of eggs at a laying, now and then two or three.
    litter: Litter::new(1, 1, [0.6, 0.3, 0.1, 0.0, 0.0, 0.0]),
    gestation_days: 4.0,
    altitude: (1500.0, 10000.0),
};

pub const RENDER_RADIUS: f32 = 21.6;

/// How bold its signal stripe, how far its feelers reach, and how long and close-set the bristles
/// of its sole.
///
/// A bolder stripe is known to hosts from further off, longer feelers find parasites sooner, and a
/// denser sole sweeps cleaner but tires sooner.
pub const STRIPE: Gene = Gene(38);
pub const FEELERS: Gene = Gene(39);
pub const BRISTLES: Gene = Gene(40);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Species,
    "Purgator mutualis",
    "Sweepers",
    [
        "Sweepers are small cleaners built to work along the skin of far larger animals: a domed back over a flat sole lined with stiff bristles, which they sweep across the host as they work along it.",
        "Their wings rise from high on the back, so the strokes never strike the surface beneath, and are short, broad and slotted for the fine hovering that close work demands.",
        "Two long, sensitive feelers search ahead for parasites and debris, which the mouth tentacles lift away without any biting grip, while a row of eyes round the sole's rim watches the skin below.",
        "A bold stripe down the back is a signal every host learns: food and the sheltered wake reward the cleaner, and parasite removal rewards the animal being groomed.",
        "That familiarity creates an opportunity exploited by unrelated Double Natured mimics.",
    ],
);

pub fn animate(c: &mut Tetralate) {
    c.opaque_bounds = vec3(4.0, 3.0, 5.2).extend(0.0);
    rig_tentacles(c, MOUTH, 0.7, ORAL_PAIRS);
}

/// Where the mouth tentacles hang, under the front of the sole.
const MOUTH: Vec3 = vec3(0.0, -0.45, -1.7);

/// A cleaner shaped by the surfaces it works on.
///
/// A low dome of a back sits over a flat sole, so it can press close along a host's skin, the
/// sole lined with rows of stiff bristles that sweep it. Its wings rise from high on the dome,
/// clear of the host. Two long feelers search ahead, and a row of eyes round the rim of the sole
/// watches the skin beneath. Down the back runs a raised stripe in a signal colour, which hosts
/// learn to tolerate.
pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
    let dome = rounded(p - vec3(0.0, 0.25, 0.0), vec3(1.5, 0.95, 2.1));
    // The breathing louvres every tetralate has, low on the dome's flanks.
    let body = louvres(
        dome.max(-(p.y + 0.35)),
        p - vec3(0.0, 0.25, 0.4),
        vec3(1.5, 0.95, 2.1),
    );
    let mut hit = Surface::new(body, 0);
    // Rows of bristles beneath the sole.
    let spacing = 0.34 - c.gene(BRISTLES) * 0.12;
    let length = 0.52 + c.gene(BRISTLES) * 0.2;
    let cell = (vec2(p.x, p.z) / spacing).round() * spacing;
    let inside = (cell.x / 1.3).power(2) + (cell.y / 1.9).power(2) < 1.0;
    if inside && p.y < -0.2 {
        let bristle = capsule(
            p,
            vec3(cell.x, -0.35, cell.y),
            vec3(cell.x, -0.1 - length, cell.y + 0.08),
            0.035,
        );
        hit.join(bristle, 7);
    }
    hit.blend(
        capsule(
            p,
            vec3(0.0, 1.05, -1.4),
            vec3(0.0, 1.05, 1.5),
            0.11 + c.gene(STRIPE) * 0.11,
        ),
        4,
        0.08,
    );
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.blend(
        curved_tube(
            side,
            vec3(0.3, -0.15, -1.9),
            vec3(0.75, 0.15, -3.0),
            vec3(0.55, -0.35, -3.6 - c.gene(FEELERS) * 0.8),
            0.06,
        ),
        8,
        0.05,
    );
    hit.socket(eyespots(p, vec3(1.25, -0.3, -1.1), 0.6, 4, 0.1), 0.1);
    hit.blend(oral_tentacles(p, c, MOUTH, 0.7, ORAL_PAIRS), 8, 0.07);
    hit
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.68, 0.74, 0.51, 0.0),
    vec3(0.16, 0.26, 0.58),
    [vec4(4.8, 2.1, -0.7, 0.4), vec4(3.6, 1.8, 0.9, 0.4)],
)
.hinged([vec2(0.65, 0.85), vec2(0.6, 0.8)])
.stalked([vec2(0.30, 0.45), vec2(0.30, 0.45)])
.styled([vec4(0.3, 0.8, 4.0, 0.0), vec4(0.3, 0.9, 3.0, 0.0)])
.framed([0.6, 0.6])
.skinned(vec4(1.0, 1.2, 1.4, 0.10))
.flying(vec4(3.50, 0.30, 0.30, 0.70))
.marked(vec4(3.00, 4.00, 0.80, 0.45), vec3(0.90, 0.95, 0.90));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use super::*;
    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent},
            physiology::{Brood, Light, Physiology, Take},
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Sunlit,
        brood: Brood::Strings,
        // Parasites, and the moss and lumaweed grown over a big host's hide.
        eats: |_, food| {
            food.animal()
                .is_some_and(|animal| animal.parasite || food.overgrown() > 0.1)
        },
        grooms: true,
        takes: Take::Living,
        gleans: false,
        flocks: false,
        speed: 8.0,
        ..super::super::PHYSIOLOGY
    };
    /// Between cleanings, a fed Sweeper rides in the sheltered air downwind of a client, as the
    /// stripe promises it may, ready when the next one wants it.
    pub const ETHOLOGY: Ethology = Ethology {
        urges: |s, urges| {
            (super::super::ETHOLOGY.urges)(s, urges);
            if s.seen
                .mark(CLIENT)
                .is_some_and(|client| client.length() < 60.0)
            {
                urges.add(WAKE, (1.0 - s.drives.hunger) * 0.4);
            }
        },
        ..super::super::ETHOLOGY
    };

    /// In a client's lee.
    const WAKE: Intent = Intent {
        name: "ride the wake",
        act: |_, s| {
            let downwind = s.seen.wind.normalize_or_zero();
            Goal::new(
                Activity::Drift,
                s.seen.mark(CLIENT).map(|client| client + downwind * 12.0),
                0.6,
            )
            .hovering()
        },
    };
}
