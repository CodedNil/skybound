use crate::scene::life::Plant;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 0.25,
    lifespan_days: 2.0,
    adult_mass: 0.001,
    altitude: (500.0, 12000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 0.00035;

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

pub fn animate(c: &mut Plant) {
    c.opaque_bounds = vec3(0.0002, 0.0002, 0.0002).extend(0.0);
}

pub fn surface(p: Vec3, _: &Plant) -> Surface {
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
pub const PARTS: &[Part] = &[];

/// Its cells, for the world renderer.
// Without pigment, for it feeds on aur and not light: milky, translucent cells lit by the charge
// they hold.
pub const LOOK: Look = Look::plant(vec3(0.62, 0.7, 0.72), vec3(0.3, 0.9, 0.85));
