use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{curved_tube, oral_tentacles, rig_tentacles},
    biology::{Biology, Brood, Feeding, Social},
    capsule, eyespots, rounded,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    brood: Brood::Strings,
    maturity_days: 30.0,
    lifespan_days: 750.0,
    feeding: Feeding::Cleaner,
    social: Social::Mutualist,
    speed: 8.0,
    adult_mass: 0.08,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 21.6;

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
    rig_tentacles(c, MOUTH, 0.7, BIOLOGY.oral_pairs);
}

/// Where the mouth tentacles hang, under the front of the sole.
const MOUTH: Vec3 = vec3(0.0, -0.45, -1.7);

/// A cleaner shaped by the surfaces it works on.
///
/// A low dome of a back sits over a flat sole, so it can press close along a
/// host's skin, the sole lined with rows of stiff bristles that sweep it. Its
/// wings rise from high on the dome, clear of the host. Two long feelers search
/// ahead, and a row of eyes round the rim of the sole watches the skin beneath.
/// Down the back runs a raised stripe in a signal colour, which hosts learn to
/// tolerate.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let dome = rounded(p - vec3(0.0, 0.25, 0.0), vec3(1.5, 0.95, 2.1));
    let body = dome.max(-(p.y + 0.35));
    let mut hit = Surface::new(body, 0);
    // Rows of bristles beneath the sole.
    let spacing = 0.28;
    let cell = (vec2(p.x, p.z) / spacing).round() * spacing;
    let inside = (cell.x / 1.3).powi(2) + (cell.y / 1.9).powi(2) < 1.0;
    if inside && p.y < -0.2 {
        let bristle = capsule(
            p,
            vec3(cell.x, -0.35, cell.y),
            vec3(cell.x, -0.62, cell.y + 0.08),
            0.035,
        );
        hit.join(bristle, 7);
    }
    hit.join(
        capsule(p, vec3(0.0, 1.05, -1.4), vec3(0.0, 1.05, 1.5), 0.16),
        4,
    );
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        curved_tube(
            side,
            vec3(0.3, -0.15, -1.9),
            vec3(0.75, 0.15, -3.0),
            vec3(0.55, -0.35, -3.9 - c.identity.y * 0.4),
            0.06,
        ),
        8,
    );
    hit.join(eyespots(p, vec3(1.25, -0.3, -1.1), 0.6, 4, 0.1), 3);
    hit.join(oral_tentacles(p, c, MOUTH, 0.7, BIOLOGY.oral_pairs), 8);
    hit
}

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
