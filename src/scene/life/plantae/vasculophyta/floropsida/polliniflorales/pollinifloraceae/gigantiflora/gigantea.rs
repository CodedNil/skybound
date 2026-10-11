use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 12.0,
    ..super::BIOLOGY
};

use spirv_std::glam::{Vec3, vec3};

use crate::scene::life::plantae::{frond, rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule, ellipsoid, plantae};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const RENDER_RADIUS: f32 = 30.0;

pub const TAXON: Taxon = Taxon {
    name: "Gigantiflora gigantea",
    english_name: "Giant Blooms",
    description: "Gigantaflora produces towering flowers that can approach small trees in height. \
        Extensive luminous petals surround large nectar reservoirs with a faint internal glow. \
        The scale of these flowers attracts large pollinators that cannot feed efficiently from smaller Blooms. \
        They thrive in fertile aur-rich areas where stable currents can support their substantial flowering bodies.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(14.0, 20.0, 14.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let p = plantae::sway(p, c);
    let mut shape = Surface::new(
        capsule(p, vec3(0.0, -12.0, 0.0), vec3(0.0, 7.0, 0.0), 0.7),
        0,
    );
    shape.join(ellipsoid(p - vec3(0.0, -10.0, 0.0), vec3(3.5, 2.8, 3.5)), 3);
    shape.join(rosette(p - vec3(0.0, 7.0, 0.0), 9, 10.0, 3.0, 3.2), 1);
    shape.join(
        rosette(rotate(p - vec3(0.0, 8.0, 0.0), 0.35), 7, 6.5, 2.5, 4.0),
        1,
    );
    shape.join(ellipsoid(p - vec3(0.0, 8.5, 0.0), vec3(2.2, 1.2, 2.2)), 2);
    for i in 0..4 {
        shape.join(
            frond(
                rotate(p - vec3(0.0, -5.0 + i as f32 * 2.0, 0.0), i as f32 * 2.4),
                6.0,
                1.8,
                1.2,
                4.0,
            ),
            1,
        );
    }
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(
        ray,
        c,
        surface,
        vec3(0.48, 0.2, 0.77),
        vec3(0.28, 0.85, 1.0),
    )
}
