#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::GANGLION,
    animalia::physiology::MUSCLE,
    animalia::physiology::PUMP,
    organs::{Viscera, pair, site},
};
use crate::scene::{
    life::{
        CreatureInstance, Look,
        biology::Biology,
        capsule,
        genome::Gene,
        prepare_pose, ribbons, rig_ribbons, smooth_min,
        surface::{Ray, Surface, TraceResult},
        tapered_tail,
    },
    math::Power,
};
#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};
use spirv_std::glam::{Vec3, Vec4, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.04,
    maturity_days: 18.0,
    lifespan_days: 600.0,
    adult_mass: 0.02,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 26.0;

/// How closely it passes for a torn-off leaf.
///
/// How deeply lobed its blade, how far its false veins reach, how cupped it is, and how long the
/// tendrils off its stalk. Every one a different leaf, so no two hang alike among the plants they
/// mimic.
pub const LOBES: Gene = Gene(38);
pub const VEINS: Gene = Gene(39);
pub const CUP: Gene = Gene(40);
pub const TENDRILS: Gene = Gene(41);

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
    // Tendrils off the end of the stalk, finishing the mimicry of a torn-off leaf.
    let tendrils = 7.0 + c.gene(TENDRILS) * 4.0;
    rig_ribbons(c, 2, vec3(0.0, 0.1, 4.4), 0.5, tendrils);
}

/// A leaf: a cupped, lance-shaped blade with a raised midrib and paired side veins, its margin
/// lobed like the leaflets it hides among.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let along = (p.z / 4.2).clamp(-1.0, 1.0);
    let lobes = 1.0 + (0.06 + c.gene(LOBES) * 0.16) * (p.z * 3.1).sin();
    let width = 1.6 * (1.0 - along * along).max(0.0).sqrt() * (1.0 - along * 0.25) * lobes;
    let cupped = p.y - p.x * p.x * (0.04 + c.gene(CUP) * 0.1) - along * along * 0.3;
    let blade = (p.x.abs() - width)
        .max(cupped.abs() - 0.07)
        .max(p.z.abs() - 4.2)
        * 0.55;
    let midrib = tapered_tail(p, vec3(0.0, 0.05, -4.3), vec3(0.0, 0.3, 4.6), 0.16, 0.05);
    let mut hit = Surface::new(smooth_min(blade, midrib, 0.12), 0);
    let mirrored = vec3(p.x.abs(), p.y, p.z);
    for i in 0..6 {
        let z = i as f32 * 1.25 - 3.2;
        let reach = (1.1 + c.gene(VEINS) * 0.5) * (1.0 - (z / 4.2).power(2)).max(0.1);
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

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.09, -1.12), vec3(0.06, 0.06, 0.06)),
    site(PUMP, vec3(0.00, 0.09, 1.45), vec3(0.02, 0.02, 0.02)),
    site(CROP, vec3(0.00, 0.28, 4.02), vec3(0.06, 0.06, 0.06)),
    pair(MUSCLE, vec3(0.56, 0.19, 2.89), vec3(0.02, 0.02, 0.02)),
    site(BROOD, vec3(0.00, 0.28, 4.34), vec3(0.03, 0.03, 0.03)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.24, 0.39, 0.19, 0.0),
    vec3(0.53, 0.61, 0.25),
    [Vec4::ZERO; 2],
)
.skinned(vec4(1.0, 0.6, 2.0, 0.10));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Floating Ones wait inside the cover of large plants, hanging still among the fronds in
    //! the narrow ways grazers feed along, until a passing grazer brings its hide within reach.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, hang_still},
            nature::ROOST,
            physiology::{Light, Physiology},
            sanguinivora::errantia::ETHOLOGY as WANDERER,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Aurlit,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        wander: |_, s| {
            // Into the fronds first, then still.
            s.seen
                .mark(ROOST)
                .filter(|cover| cover.length() > 4.0)
                .map_or_else(hang_still, |cover| {
                    Goal::new(Activity::Drift, Some(cover), 0.3).hovering()
                })
        },
        ..WANDERER
    };
}
