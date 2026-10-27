#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::tetralata::CISTERN,
    animalia::tetralata::FLIGHT_MUSCLE,
    organs::{Viscera, pair, site},
};
use crate::scene::{
    life::{
        CreatureInstance, Look,
        animalia::tetralata::{ORAL_PAIRS, curved_tube, oral_tentacles, rig_tentacles},
        biology::Biology,
        capsule, eyespots,
        genome::Gene,
        rounded,
        surface::{Ray, Surface, TraceResult},
    },
    math::Power,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.083,
    maturity_days: 30.0,
    lifespan_days: 750.0,
    adult_mass: 0.08,
    ..super::BIOLOGY
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
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

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

pub fn animate(c: &mut CreatureInstance) {
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
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let dome = rounded(p - vec3(0.0, 0.25, 0.0), vec3(1.5, 0.95, 2.1));
    let body = dome.max(-(p.y + 0.35));
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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.35, -1.49), vec3(0.27, 0.12, 0.39)),
    site(PUMP, vec3(0.00, 0.13, -0.77), vec3(0.29, 0.13, 0.39)),
    site(CROP, vec3(0.00, 0.13, 0.15), vec3(0.39, 0.26, 0.39)),
    pair(BREATHER, vec3(0.74, 0.27, -0.92), vec3(0.31, 0.14, 0.39)),
    site(FLOAT, vec3(0.00, 0.72, 0.05), vec3(0.32, 0.16, 0.32)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.53, 0.48, -0.31),
        vec3(0.29, 0.15, 0.42),
    ),
    site(CISTERN, vec3(0.00, 0.08, 0.97), vec3(0.29, 0.13, 0.34)),
    site(BROOD, vec3(0.00, 0.24, 1.49), vec3(0.29, 0.13, 0.42)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.68, 0.74, 0.51, 0.0),
    vec3(0.16, 0.26, 0.58),
    [vec4(4.8, 2.1, -0.7, 0.4), vec4(3.6, 1.8, 0.9, 0.4)],
)
.hinged([vec2(0.65, 0.85), vec2(0.6, 0.8)])
.styled([vec4(0.3, 0.8, 4.0, 0.0), vec4(0.3, 0.9, 3.0, 0.0)])
.framed([0.6, 0.6])
.skinned(vec4(1.0, 1.2, 1.4, 0.10))
.flying(vec4(3.50, 0.30, 0.30, 0.70))
.marked(vec4(3.00, 4.00, 0.80, 0.45), vec3(0.90, 0.95, 0.90));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Brood, Light, Physiology, Take};

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
}
