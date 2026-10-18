use spirv_std::glam::{Vec3, vec3, vec4};

use crate::scene::life::biology::Biology;
use crate::scene::life::surface::{Ray, Surface, TraceResult};
use crate::scene::life::{CreatureInstance, ellipsoid, prepare_pose, ribbons, smooth_min, tail};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 60.0,
    lifespan_days: 900.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 40.4;

taxon!(
    Species,
    "Lucerna undulata",
    "Shimmerays",
    [
        "Shimmerays spread broad wings and luminous trailing structures through currents carrying algae and other organic particles.",
        "Their glow also draws Sparkies close enough for the mouth appendages to sweep them into the feeding stream.",
        "During the bright aur phase they rise, then descend toward the upper edge of the Veilands as the cycle dims.",
        "Food comes from living particles and organic residues carried above the ocean, while concentrated aur remains dangerous to their tissues.",
        "Seen from afar, their coordinated vertical movement forms long shifting ribbons of light over the lower Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 17.0, 2);
    c.opaque_bounds = vec3(5.0, 5.0, 24.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(1.0, 0.75, 9.0)), 0);
    hit.distance = smooth_min(hit.distance, tail(p, c), 0.3);
    for i in 0..8 {
        let z = i as f32 * 2.0 - 7.0;
        hit.join(ellipsoid(p - vec3(0.0, 0.68, z), vec3(0.65, 0.16, 0.3)), 9);
    }
    hit.join(ribbons(p, c, 2), 4);
    hit
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(
        ray,
        c,
        surface,
        vec4(0.16, 0.26, 0.41, 0.0),
        vec3(0.22, 0.82, 0.89),
        [vec4(12.0, 5.0, 0.0, 0.25), vec4(9.0, 3.0, 9.0, 0.25)],
    )
}
