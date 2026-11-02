pub const BIOLOGY: Biology = Biology {
    size: 0.13,
    maturity_days: 90.0,
    lifespan_days: 5400.0,
    adult_mass: 0.3,
    altitude: (1500.0, 9000.0),
    // A little spawn at a time.
    litter: Litter::new(1, 1, [0.35, 0.35, 0.2, 0.1, 0.0, 0.0]),
    gestation_days: 15.0,
};

pub const RENDER_RADIUS: f32 = 25.0;

/// How long its feeding ribbons trail, how finely its lens is drawn to a point, and how broad the
/// lip they hang from.
///
/// Long ribbons reach further into the neighbouring streams, and a fine point turns it more readily
/// into useful air.
pub const RIBBONS: Gene = Gene(38);
pub const KEEL: Gene = Gene(39);
pub const LIP: Gene = Gene(40);

taxon!(
    Species,
    "Levitans taeniatus",
    "Drifters",
    [
        "Drifters carry a translucent yellow-green body ahead of long ribbons traced with golden veins.",
        "Several rounded gas chambers surround a darker cluster of organs, keeping the pointed front poised in the updraft.",
        "The ribbons curl through adjacent currents and collect algae and plant fragments on their sticky margins.",
        "Gentle changes in orientation regulate collection, while folded ribbons reduce strain when the animal crosses a faster stream.",
        "Their broad altitude range lets them follow suspended food upward and settle toward richer air when it thins.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
pub fn animate(c: &mut Bubbleback) {
    c.opaque_bounds = vec3(5.2, 3.5, 5.2).extend(0.0);
    for i in 0..3 {
        cloth::prepare(
            c,
            i,
            vec3((i as f32 - 1.0) * 2.1, -0.45, 4.5),
            (10.0 + i as f32 * 2.0) * (0.8 + c.gene(RIBBONS) * 0.4),
            1.4,
        );
    }
}

pub fn surface(p: Vec3, _: &Bubbleback) -> Surface {
    let mut distance = ellipsoid(p - vec3(0.0, -0.35, 0.0), vec3(0.9, 0.4, 1.1));
    for i in -2..=2 {
        let x = i as f32;
        for half in 0..2 {
            let side = if half == 0 { -1.0 } else { 1.0 };
            let center = vec3(
                x * 0.7,
                -0.5 + (x * 2.0).sin() * 0.3,
                side * (0.8 - x.abs() * 0.15),
            );
            distance = smooth_min(distance, ellipsoid(p - center, vec3(0.65, 0.8, 0.65)), 0.35);
        }
    }
    Surface::new(distance + tissue_distance(p, 3.0), 0)
}

/// A flattened lens drawn out to a pointed leading edge.
///
/// The current presses unevenly across it and turns it to face useful air; behind, it thins to a
/// lip along which the feeding ribbons are rooted, trailing clear of the body into the
/// neighbouring streams.
pub fn envelope(p: Vec3, c: &Bubbleback) -> Surface {
    let front = (-p.z / 4.6).clamp(0.0, 1.0);
    let radii = vec3(
        4.3 * (1.0 - (0.6 + c.gene(KEEL) * 0.25) * front * front),
        1.9 * (1.0 - 0.55 * front),
        4.6,
    );
    let lens = rounded(p - vec3(0.0, -0.2 + p.z * p.z * 0.03, 0.0), radii) * 0.75;
    let lip = rounded(
        p - vec3(0.0, -0.45, 3.9),
        vec3(3.0 + c.gene(LIP) * 1.0, 0.35, 0.9),
    );
    Surface::new(smooth_min(lens, lip, 0.6), 0)
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Tint and scale, shared by the preview and the world renderer.
pub const LOOK: Look = Look::drift(vec3(0.38, 0.66, 0.19), 3.8);
