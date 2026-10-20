use crate::scene::life::{
    CreatureInstance,
    animalia::tetralata::oral_tentacles,
    biology::{Biology, Feeding, Social},
    ellipsoid, eyes,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
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
        "Sweepers carry small, swollen bodies on four wings that hold them close to the surfaces of larger animals.",
        "Sensitive mouth appendages lift debris and parasites without requiring a biting grip on the host.",
        "They work through short approaches, retreating when the host moves sharply and returning to tolerated areas.",
        "Food and the sheltered wake reward the cleaner, while parasite removal rewards the animal being groomed.",
        "Regular visits also make hosts familiar with their shape and movements, creating an opportunity exploited by unrelated Double Natured mimics.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 5.0, 5.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(2.3, 2.0, 2.7)), 0);
    hit.join(ellipsoid(p - vec3(0.0, 0.4, -2.5), vec3(1.0, 0.9, 0.9)), 0);
    hit.join(
        oral_tentacles(p, c, vec3(0.0, -0.3, -3.10), 1.15, BIOLOGY.oral_pairs),
        8,
    );
    hit.join(eyes(p, 2.7, 0.5), 3);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.68, 0.74, 0.51, 0.0),
        vec3(0.35, 0.59, 0.43),
        [vec4(6.0, 2.5, 0.0, 0.4), vec4(4.0, 1.8, 3.0, 0.4)],
    )
}
