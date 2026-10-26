#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::Viscera;
use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    surface::{Material, Ray, Shell, Surface, TraceResult, trace_material, visit_shell_material},
};
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 0.25,
    lifespan_days: 2.0,
    adult_mass: 0.001,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 0.00035;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;

taxon!(
    Species,
    "Auracoccus communis",
    "Algae",
    [
        "Algae drift as minute rounded cells through air carrying diluted aur and moisture.",
        "Individual cells collect energy and divide rapidly, allowing a suitable current to fill with food for larger organisms.",
        "Small size keeps exchange distances short and lets the updraft carry them without a specialised lifting organ.",
        "Sticky animal folds and plant membranes intercept them, concentrating microscopic growth into meals.",
        "Only brief, resilient blooms reach the Veiland boundary, where the same energy source becomes too intense for sustained ordinary growth.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(0.0002, 0.0002, 0.0002).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let p = p * 10000.0;
    let local = p + vec3((p.y * 7.0).sin(), (p.z * 8.0).sin(), (p.x * 6.0).sin()) * 0.018;
    let mut shape = Surface::new(ellipsoid(local, vec3(0.85, 0.72, 0.85)), 1);
    shape.join(
        ellipsoid(local - vec3(0.75, 0.3, 0.2), vec3(0.6, 0.55, 0.6)),
        1,
    );
    shape.join(ellipsoid(p, Vec3::splat(0.35)), 2);
    shape.distance *= 0.0001;
    shape
}

/// It is a drifting cloud of single cells, with no organs to wound.
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::NONE;

/// Its cells, for the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.16, 0.56, 0.28), vec3(0.08, 0.4, 0.12));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::opaque(trace_material(
        ray,
        c,
        |p, _| Surface::new(ellipsoid(p, Vec3::splat(0.000_035)), 0),
        |_, _| Material::skin(vec3(0.08, 0.4, 0.12), 0.35),
    ));
    for cell in 0..2 {
        let (center, radii) = if cell == 0 {
            (Vec3::ZERO, vec3(0.85, 0.72, 0.85))
        } else {
            (vec3(0.75, 0.3, 0.2), vec3(0.6, 0.55, 0.6))
        };
        visit_shell_material(
            ray,
            c,
            Shell {
                center: center * 0.0001,
                radii: radii * 0.0001,
                bottom: -1.0,
                scallop: 0.0,
            },
            |_, cosine| {
                let mut cell = Material::skin(vec3(0.16, 0.56, 0.28), 0.2);
                cell.ior = 1.36;
                cell.translucency = 0.5;
                cell.transmission = (-vec3(0.3, 0.06, 0.2) / cosine.max(0.15)).exp() * 0.85;
                cell
            },
            |hit| result.add(hit),
        );
    }
    result
}
