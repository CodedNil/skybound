pub const BIOLOGY: Biology = Biology {
    size: 0.2,
    maturity_days: 1800.0,
    lifespan_days: 30000.0,
    // One or two young, rarely more.
    litter: Litter::new(1, 1, [0.6, 0.32, 0.07, 0.01, 0.0, 0.0]),
    gestation_days: 200.0,
    altitude: (2000.0, 11000.0),
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

/// How many blades its crest carries and how deep its chest: more blades cut a longer wound in one
/// pass, and a deep chest drives the wings harder, but a slim one is faster between herds.
pub const BLADES: Gene = Gene(38);
pub const CHEST: Gene = Gene(39);

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Ferrispina falcata",
    "Shredders",
    [
        "Shredders are long, flexible predators with two pairs of long, swept wings and a crest of tall iron blades raked back along the spine, five to seven of them as their parents carried.",
        "They bank past prey in curved cutting passes, rolling to draw the crest across it while keeping the vulnerable underside away from a struggling target, the blades laid back for speed until the moment of the cut.",
        "The crest is also their courtship: rivals and mates raise the blades to full height and spread the red-banded web between them, and the biggest, soundest crests win.",
        "Young animals hunt in loose coordinated groups that split feeding herds and exploit the openings created by a companion.",
        "Their hunting follows herd routes and open gaps between plant communities, where a long body has room to turn.",
    ],
);

pub fn animate(c: &mut Tetralate) {
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

pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
    let build = Build {
        blades: 5 + (c.gene(BLADES) * 2.99) as u32,
        depth: BUILD.depth * (0.9 + c.gene(CHEST) * 0.2),
        ..BUILD
    };
    let mut hit = shredder(p, c, build);
    hit.blend(
        oral_tentacles(p, c, mouth(BUILD.length, BUILD.head), 1.0, ORAL_PAIRS),
        8,
        0.1,
    );
    hit
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.19, 0.22, 0.31, 0.0),
    vec3(0.72, 0.18, 0.12),
    [vec4(14.0, 2.6, -2.0, 0.8), vec4(9.0, 1.9, 3.4, 0.8)],
)
.hinged([vec2(1.0, 0.3), vec2(0.9, 0.25)])
.stalked([vec2(0.30, 0.40), vec2(0.30, 0.40)])
.styled([vec4(1.2, 0.15, 3.0, 0.0), vec4(1.0, 0.15, 2.0, 0.0)])
.skinned(vec4(1.0, 0.6, 1.4, -0.05))
.flying(vec4(0.90, 0.32, 0.45, 1.10))
.marked(vec4(4.00, 4.00, 0.75, 0.40), vec3(0.06, 0.04, 0.05));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        ..super::super::PHYSIOLOGY
    };
}
