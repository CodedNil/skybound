pub use super::render_wings;
use super::{Build, shredder};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::Biology,
    mouth, prepare_pose,
    surface::{Ray, Surface, TraceResult},
};
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
use spirv_std::glam::{Vec3, vec2, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    size: 0.2,
    maturity_days: 1800.0,
    lifespan_days: 30000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Ferrispina falcata",
    "Shredders",
    [
        "Shredders are long, flexible predators with two pairs of long, swept wings and a crest of six tall iron blades raked back along the spine.",
        "They bank past prey in curved cutting passes, rolling to draw the crest across it while keeping the vulnerable underside away from a struggling target, the blades laid back for speed until the moment of the cut.",
        "The crest is also their courtship: rivals and mates raise the blades to full height and spread the red-banded web between them, and the biggest, soundest crests win.",
        "Young animals hunt in loose coordinated groups that split feeding herds and exploit the openings created by a companion.",
        "Their hunting follows herd routes and open gaps between plant communities, where a long body has room to turn.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 7.0, 10.4, 0);
    c.opaque_bounds = vec3(6.0, 7.0, 30.0).extend(0.0);
    rig_tentacles(c, mouth(BUILD.length, BUILD.head), 1.0, ORAL_PAIRS);
}

/// A long-range hunter of the open herd routes: a long, slim body and a crest of six tall blades
/// raked well back, each a lance of iron drawn along its prey in the roll of a fast pass. Long,
/// narrow, swept wings carry it fast between herds.
const BUILD: Build = Build {
    length: 8.5,
    depth: 1.25,
    head: 0.7,
    blades: 6,
    height: 4.2,
    width: 1.3,
    rake: 0.45,
    web: 0.75,
};

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = shredder(p, c, BUILD);
    hit.join(
        oral_tentacles(p, c, mouth(BUILD.length, BUILD.head), 1.0, ORAL_PAIRS),
        8,
    );
    hit
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.06, -6.16), vec3(0.16, 0.20, 0.39)),
    site(PUMP, vec3(0.00, -0.19, -2.95), vec3(0.18, 0.22, 0.39)),
    site(CROP, vec3(0.00, 0.06, 1.34), vec3(0.68, 0.68, 2.19)),
    pair(BREATHER, vec3(0.29, 0.13, -3.75), vec3(0.19, 0.20, 0.20)),
    site(FLOAT, vec3(0.00, 0.19, -1.34), vec3(0.22, 0.27, 0.52)),
    pair(
        FLIGHT_MUSCLE,
        vec3(0.62, 0.13, -1.61),
        vec3(0.12, 0.12, 0.12),
    ),
    site(CISTERN, vec3(0.00, 0.06, 4.02), vec3(0.51, 0.51, 1.62)),
    site(BROOD, vec3(0.00, 0.19, 6.16), vec3(0.38, 0.38, 1.22)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.19, 0.22, 0.31, 0.0),
    vec3(0.72, 0.18, 0.12),
    [vec4(14.0, 2.6, -2.0, 0.8), vec4(9.0, 1.9, 3.4, 0.8)],
)
.hinged([vec2(1.0, 0.3), vec2(0.9, 0.25)])
.styled([vec4(1.2, 0.15, 3.0, 0.0), vec4(1.0, 0.15, 2.0, 0.0)])
.skinned(vec4(1.0, 0.6, 1.4, -0.05))
.flying(vec4(0.90, 0.32, 0.45, 1.10))
.marked(vec4(4.00, 4.00, 0.75, 0.40), vec3(0.06, 0.04, 0.05));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        ..super::super::PHYSIOLOGY
    };
}
