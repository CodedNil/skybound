use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    capsule, prepare_pose, ribbons, rig_ribbons, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 18.0,
    lifespan_days: 600.0,
    adult_mass: 0.02,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 26.0;

taxon!(
    Species,
    "Fluitans crypticus",
    "Floating Ones",
    [
        "Floating Ones begin life among kelp, where green-brown bodies and swaying lateral folds blend with the fronds.",
        "Instead of searching open air, they wait in the narrow spaces through which feeding animals draw vegetation.",
        "A passing grazer brings the sucker within reach, allowing the parasite to leave the plant and settle permanently on the host.",
        "The trailing folds then lie against the new surface, reducing drag during travel.",
        "Dense Jungles favour this ambush life cycle, while the movement of kelp mats carries unattached young into new feeding routes.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 4.5, 9.0, 2);
    c.opaque_bounds = vec3(3.0, 3.0, 18.0).extend(0.0);
    // Tendrils off the end of the stalk, finishing the mimicry of a torn-off
    // leaf.
    rig_ribbons(c, 2, vec3(0.0, 0.1, 4.4), 0.5, 9.0);
}

/// A leaf: a cupped, lance-shaped blade with a raised midrib and paired
/// side veins, its margin lobed like the leaflets it hides among.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let along = (p.z / 4.2).clamp(-1.0, 1.0);
    let lobes = 1.0 + 0.14 * (p.z * 3.1).sin();
    let width = 1.6 * (1.0 - along * along).max(0.0).sqrt() * (1.0 - along * 0.25) * lobes;
    let cupped = p.y - p.x * p.x * 0.09 - along * along * 0.3;
    let blade = (p.x.abs() - width)
        .max(cupped.abs() - 0.07)
        .max(p.z.abs() - 4.2)
        * 0.55;
    let midrib = tapered_tail(p, vec3(0.0, 0.05, -4.3), vec3(0.0, 0.3, 4.6), 0.16, 0.05);
    let mut hit = Surface::new(smooth_min(blade, midrib, 0.12), 0);
    let mirrored = vec3(p.x.abs(), p.y, p.z);
    for i in 0..6 {
        let z = i as f32 * 1.25 - 3.2;
        let reach = 1.35 * (1.0 - (z / 4.2).powi(2)).max(0.1);
        hit.join(
            capsule(
                mirrored,
                vec3(0.05, 0.08, z),
                vec3(reach, 0.08 + reach * reach * 0.09, z + reach * 0.55),
                0.035,
            ),
            4,
        );
    }
    hit.join(ribbons(p, c, 2), 4);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.24, 0.39, 0.19, 0.0),
    vec3(0.53, 0.61, 0.25),
    [Vec4::ZERO; 2],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
