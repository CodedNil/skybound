use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{gape, oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::{Biology, Litter},
    ellipsoid,
    genome::Gene,
    louvres, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
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
    organs::{Function, Organ, Viscera, pair, site},
};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.33,
    maturity_days: 900.0,
    lifespan_days: 18000.0,
    adult_mass: 14.0,
    // One young, sometimes two: a stooping hunter can carry no more.
    litter: Litter::new(1, 1, [0.75, 0.22, 0.03, 0.0, 0.0, 0.0]),
    gestation_days: 120.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 41.5;

/// How long its stylet grows, how large its venom sac, how wide the fork of its tail, and how heavy
/// the iron scutes along its back.
///
/// A long stylet strikes from further, a large sac stills bigger prey, a wide fork turns the dive
/// tighter, and heavy scutes take a glancing blow.
pub const STYLET: Gene = Gene(38);
pub const VENOM_SAC: Gene = Gene(39);
pub const FORK: Gene = Gene(40);
pub const SCUTES: Gene = Gene(41);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Aculeorhynchus acuminatus",
    "Needlesharks",
    [
        "Needlesharks hunt from high above, folding their long swept wings to fall on their prey, a needle-fine stylet of iron-hardened tissue aligned ahead of the dive.",
        "The stylet is aimed at the pump or the nerve centre for a killing thrust, and as it goes in a sac at its root pumps venom down it, so even a strike that misses everything vital leaves the prey paralysed.",
        "So thin a weapon slips out as easily as it went in, and the shock of the impact is taken up by a long sheath of muscle the stylet's root slides back into, running down through the head and neck.",
        "Struck prey falls limp toward the aur, and the hunter wheels and races it down, catching it from below in a basket of long hooked mouth tentacles and carrying it up into rising air to eat on the wing, rasping it apart with the mouth beneath its head; if it is slow, the venom wears off and the prey escapes.",
        "Two forward eyes either side of the stylet's root judge each dive, a swallow-tailed fork steers it, and young needlesharks spend long hours stooping on one another in play.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 15.0).extend(0.0);
    rig_tentacles(c, MOUTH, 1.8, ORAL_PAIRS);
}

/// Where the mouth opens beneath the head, and the catching tentacles hang from it.
const MOUTH: Vec3 = vec3(0.0, -0.5, -3.1);

/// Where the stylet leaves the head.
const ROOT: Vec3 = vec3(0.0, -0.02, -4.95);

/// The body's half girth along its length: a small head, a deep chest for the wing muscles, a
/// waist, and a long tapering tail.
fn girth(z: f32) -> f32 {
    let stations = [
        vec2(-4.9, 0.25),
        vec2(-4.0, 0.55),
        vec2(-2.8, 0.75),
        vec2(-1.2, 1.05),
        vec2(1.8, 0.7),
        vec2(4.2, 0.35),
        vec2(6.5, 0.18),
        vec2(8.5, 0.1),
    ];
    let mut radius = 0.1;
    for i in 0..7 {
        let a = stations[i];
        let b = stations[i + 1];
        if z >= a.x && z < b.x {
            radius =
                a.y.lerp(b.y, ((z - a.x) / (b.x - a.x)).smoothstep(0.0, 1.0));
        }
    }
    // Rounded off at the snout.
    if z < -4.9 {
        radius = 0.0;
    }
    radius
}

/// A stooping hunter built around its stylet.
///
/// The stylet is needle-fine, iron-hardened as its lineage's blades are, thin enough to slip in and
/// out of prey without catching; its root slides back into a cushioned sheath through the head and
/// neck to take up the shock of the strike, and a venom sac swells amber beneath the head. The
/// head is a narrow swept hood with a pair of forward eyes either side of the stylet's root, and a
/// pair beneath looking down. The body is deepest at the keel where the wing muscles sit, ridged
/// along the back with small iron scutes, and drawn out into a long tail ending in a swallow-tailed
/// fork that steers the dive. Beneath the head, a rasping mouth and three pairs of long hooked
/// tentacles catch and hold what the stylet has stilled.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let grown = 0.15 + 0.85 * c.life.y * c.life.y;
    // One smooth body of revolution, deeper than wide, for the dive.
    let narrow = vec3(p.x * 1.15, (p.y - 0.05) * 0.9, p.z);
    let radial = vec2(narrow.x, narrow.y).length();
    let mut body = ((radial - girth(p.z)) * 0.75)
        .max(-4.9 - p.z)
        .max(p.z - 8.6);
    // The keel under the chest, where the wing muscles anchor.
    let keel = ellipsoid(p - vec3(0.0, -0.75, -1.3), vec3(0.32, 0.55, 1.5));
    body = louvres(smooth_min(body, keel, 0.4), p, vec3(0.9, 0.9, 3.6));
    // The hood: thin and swept, narrowing to the stylet's root.
    let ahead = ((p.z + 3.0) / -2.0).clamp(0.0, 1.0);
    let hood = rounded(
        p - vec3(0.0, 0.32, -3.9),
        vec3(0.95 * (1.0 - ahead * 0.75), 0.09, 1.15),
    );
    body = smooth_min(body, hood, 0.25);
    let mut hit = Surface::new(body, 0);
    // The stylet: needle-fine, each animal's its own length.
    let length = 8.5 * grown * (0.88 + c.gene(STYLET) * 0.24);
    let tip = ROOT + vec3(0.0, -0.08, -length);
    hit.blend(
        tapered_tail(p, ROOT + vec3(0.0, 0.0, 0.3), tip, 0.075, 0.006),
        6,
        0.04,
    );
    // The venom sac beneath the head.
    hit.blend(
        ellipsoid(
            p - vec3(0.0, -0.36, -3.95),
            vec3(0.24, 0.2, 0.38) * (0.85 + c.gene(VENOM_SAC) * 0.4),
        ),
        2,
        0.12,
    );
    // Iron scutes along the back.
    let station = ((p.z + 2.4) / 0.75).round().clamp(0.0, 8.0);
    let z = -2.4 + station * 0.75;
    let scute = rounded(
        p - vec3(0.0, girth(z) * 1.08, z),
        vec3(0.16, 0.07, 0.22) * (1.0 - station * 0.07) * (0.75 + c.gene(SCUTES) * 0.5),
    );
    hit.join(scute, 6);
    // The swallow-tailed fork.
    let side = vec3(p.x.abs(), p.y * 2.5, p.z);
    let fork = tapered_tail(
        side,
        vec3(0.05, 0.25, 8.3),
        vec3(1.1 + c.gene(FORK) * 0.8, 0.0, 12.6 - c.gene(FORK) * 0.6),
        0.22,
        0.02,
    ) / 2.5;
    hit.distance = smooth_min(hit.distance, fork, 0.2);
    // Eyes: a forward pair either side of the stylet's root to judge the dive, and a pair beneath.
    hit.join((side_of(p) - vec3(0.22, 0.1, -4.55)).length() - 0.13, 3);
    hit.join((side_of(p) - vec3(0.3, -0.32, -3.4)).length() - 0.1, 3);
    rasp(&mut hit, p, c);
    hit.blend(oral_tentacles(p, c, MOUTH, 1.8, ORAL_PAIRS), 8, 0.18);
    hit
}

/// The rasping mouth beneath the head: a round, lipped opening facing down and forward, which
/// everts as it feeds into a short cone ringed with rows of hooked, iron-hardened teeth that it
/// works against what its tentacles hold, tearing it apart a mouthful at a time. In flight it is
/// drawn in, only the lip showing.
fn rasp(hit: &mut Surface, p: Vec3, c: &CreatureInstance) {
    let axis = vec3(0.0, -0.93, -0.37);
    let across = vec3(1.0, 0.0, 0.0);
    let up = axis.cross(across);
    let q = p - MOUTH - vec3(0.0, 0.18, 0.0);
    let along = q.dot(axis);
    let radial = (q - axis * along).length();
    if radial > 0.6 || along.abs() > 0.8 {
        return;
    }
    let open = gape(c);
    // The lip, a thick ring, and the cavity inside it.
    let lip = vec2(radial - 0.2, along).length() - 0.07;
    hit.carve(vec2(radial, along - 0.05).length() - 0.16, 0.04);
    hit.blend(lip, 0, 0.06);
    // The everted rasp, pushed out through the lip, its teeth in rings round it.
    let reach = open * 0.45;
    if reach > 0.01 {
        let t = (along / reach).clamp(0.0, 1.0);
        let cone = (radial - 0.16 * (1.0 - t * 0.35))
            .max(along - reach)
            .max(-along)
            * 0.8;
        let around = q.dot(up).atan2(q.dot(across));
        let rows = ((along / 0.07).fract_gl() - 0.5).abs();
        let teeth = (around * 9.0).sin().max(0.0) * (0.5 - rows).max(0.0) * 0.08;
        hit.join(cone - teeth, 6);
    }
}

const fn side_of(p: Vec3) -> Vec3 {
    vec3(p.x.abs(), p.y, p.z)
}

/// The venom sac, whose store is spent on each use and slowly made again.
#[cfg(not(target_arch = "spirv"))]
const VENOM: Organ = Organ::new("venom sac", Function::Secretion, vec3(0.55, 0.85, 0.30));

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.0, 0.05, -4.0), vec3(0.22, 0.2, 0.3)),
    site(VENOM, vec3(0.0, -0.36, -3.95), vec3(0.2, 0.16, 0.32)),
    site(PUMP, vec3(0.0, -0.15, -2.2), vec3(0.3, 0.3, 0.35)),
    pair(BREATHER, vec3(0.35, 0.15, -2.7), vec3(0.15, 0.22, 0.4)),
    pair(FLIGHT_MUSCLE, vec3(0.42, 0.1, -1.2), vec3(0.25, 0.4, 0.6)),
    site(FLOAT, vec3(0.0, 0.5, -0.8), vec3(0.28, 0.22, 1.0)),
    site(CROP, vec3(0.0, -0.2, 0.3), vec3(0.45, 0.42, 0.85)),
    site(CISTERN, vec3(0.0, -0.1, 1.6), vec3(0.3, 0.28, 0.45)),
    site(BROOD, vec3(0.0, 0.0, 2.8), vec3(0.22, 0.22, 0.45)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.20, 0.30, 0.40, 0.0),
    vec3(0.70, 0.76, 0.82),
    [vec4(13.0, 2.6, -1.3, 0.75), vec4(5.0, 1.6, 2.2, 0.75)],
)
.hinged([vec2(0.95, 0.35), vec2(0.6, 0.2)])
.styled([vec4(1.2, 0.1, 3.0, 0.0), vec4(0.9, 0.3, 0.0, 0.0)])
.skinned(vec4(5.0, 0.4, 2.0, -0.10))
.flying(vec4(1.10, 0.30, 0.55, 1.00))
.marked(vec4(4.00, 4.00, 0.80, 0.45), vec3(0.06, 0.08, 0.12));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::animalia::physiology::{Light, Physiology};

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        speed: 75.0,
        flocks: false,
        ..super::super::PHYSIOLOGY
    };
}
