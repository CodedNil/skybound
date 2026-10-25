pub use super::render_wings;
use super::{Build, shredder};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    mouth, prepare_pose,
    surface::{Ray, Surface, TraceResult},
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};

pub const BIOLOGY: Biology = Biology {
    maturity_days: 1800.0,
    lifespan_days: 30000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

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
    prepare_pose(c, 7.0, 16.0, 0);
    c.opaque_bounds = vec3(6.0, 7.0, 30.0).extend(0.0);
    rig_tentacles(c, mouth(BUILD.length, BUILD.head), 1.0, BIOLOGY.oral_pairs);
}

/// A long-range hunter of the open herd routes: a long, slim body and a
/// crest of six tall blades raked well back, each a lance of iron drawn
/// along its prey in the roll of a fast pass. Long, narrow, swept wings
/// carry it fast between herds.
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
        oral_tentacles(
            p,
            c,
            mouth(BUILD.length, BUILD.head),
            1.0,
            BIOLOGY.oral_pairs,
        ),
        8,
    );
    hit
}

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
