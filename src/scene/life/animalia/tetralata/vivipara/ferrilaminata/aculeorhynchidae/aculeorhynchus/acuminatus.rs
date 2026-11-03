use crate::scene::life::Tetralate;

pub const BIOLOGY: Biology = Biology {
    size: 0.33,
    maturity_days: 900.0,
    lifespan_days: 18000.0,
    adult_mass: 14.0,
    // One young, sometimes two: a stooping hunter can carry no more.
    litter: Litter::new(1, 1, [0.75, 0.22, 0.03, 0.0, 0.0, 0.0]),
    gestation_days: 120.0,
    // High above everything it hunts, for the length of its stoop.
    altitude: (4000.0, 14000.0),
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
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Species,
    "Aculeorhynchus acuminatus",
    "Needlesharks",
    [
        "Needlesharks hunt from high above, folding their long swept wings to fall on their prey, a needle-fine stylet aligned ahead of the dive, which their body grows from the iron in their diet, as all their order eats metal, and which lengthens and hardens through their life.",
        "The stylet is aimed at the pump or the nerve centre for a killing thrust, and as it goes in a sac at its root pumps venom down it, so even a strike that misses everything vital leaves the prey paralysed.",
        "The shock of the impact is taken up by a long sheath of muscle the stylet's root slides back into, running down through the head and neck.",
        "It does not let go. Small rows of barbs rake back along the shaft, so the prey slides towards the head and never off the point, and a needleshark flies on with its kill skewered ahead of it, sometimes for days, its very long mouth tentacles reaching out along the stylet to strip it and work it down the shaft to the rasping mouth beneath the head, a mouthful at a time.",
        "So it hunts only small prey, a quarter of its own weight at most, that it can fly with; anything larger would pin it to the sky. If it misjudges, or strikes something still strong enough to fight, it sheds the lance: the stylet's last hand's breadth is grown as a separate barbed head on a seam, which a twist of the sheath muscle snaps, leaving it in the victim while the hunter pulls free. The venom often finishes what the strike began, and it may come back later to scavenge, but a new head takes weeks and much iron to grow, and until then it strikes short.",
        "Two forward eyes either side of the stylet's root judge each dive, a swallow-tailed fork steers it, and young needlesharks spend long hours stooping on one another in play.",
    ],
);

pub fn animate(c: &mut Tetralate) {
    c.opaque_bounds = vec3(4.0, 4.0, 15.0).extend(0.0);
    rig_grasping(c, MOUTH, REACH, ORAL_PAIRS, GRIP);
}

/// Where the mouth opens beneath the head, and the catching tentacles hang from it.
const MOUTH: Vec3 = vec3(0.0, -0.5, -3.1);

/// How far the mouth tentacles reach, as a share of the lineage's usual reach, and how much further
/// the outer pair grows: the length of the stylet, to work what is skewered on it down to the
/// mouth, while the inner pairs stay short to hold it there.
const REACH: f32 = 1.8;
const GRIP: f32 = 1.4;

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

/// Its forward eyes and the pair beneath, seated on its own skin.
#[cfg(not(target_arch = "spirv"))]
pub const ANCHORS: u32 = 2;

/// Where eye `index` lies on its design, before it is seated on its own skin.
pub fn anchor(index: u32, _: &CreatureInstance) -> Vec4 {
    if index == 0 {
        vec3(0.22, 0.1, -4.55).extend(0.13 * SUNK)
    } else {
        vec3(0.3, -0.32, -3.4).extend(0.1 * SUNK)
    }
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
pub fn surface(p: Vec3, c: &Tetralate) -> Surface {
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
    // It grows from the iron the hunter eats, so it reaches its full length only as it gathers
    // enough over its life.
    let length = 8.5 * grown * (0.88 + c.gene(STYLET) * 0.24) * (0.6 + c.stores.x.min(1.0) * 0.4);
    let tip = ROOT + vec3(0.0, -0.08, -length);
    let shaft = tapered_tail(p, ROOT + vec3(0.0, 0.0, 0.3), tip, 0.075, 0.006);
    hit.blend(shaft, 6, 0.04);
    if shaft < 0.4 {
        // Barbs raking back along the shaft, so what it skewers slides towards the head and never
        // off the point, in four rows turned about it.
        let axis = (tip - ROOT) / (tip - ROOT).length().max(1e-4);
        let mut barbs = 1e6f32;
        for index in 0..10 {
            let along = 0.18 + index as f32 * 0.065;
            let at = ROOT.lerp(tip, along);
            let turn = index as f32 * FRAC_PI_2;
            let out = vec3(turn.cos(), turn.sin(), 0.0);
            let thick = 0.075 - along * 0.06;
            barbs = barbs.min(tapered_tail(
                p,
                at,
                at + out * thick * 1.6 - axis * thick * 3.0,
                thick * 0.45,
                0.004,
            ));
        }
        hit.join(barbs, 6);
        // The seam where the lance-head is grown separately, ready to be shed.
        let seam = ROOT.lerp(tip, 0.84);
        let q = p - seam;
        let along = q.dot(axis);
        let ring = vec2((q - axis * along).length() - 0.022, along).length() - 0.006;
        hit.carve(ring, 0.004);
    }
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
    // Measured bare, the body has no eyes set in it yet.
    if !c.wings.measuring() {
        hit.socket(
            (side_of(p) - c.seated(0, anchor(0, c).truncate())).length() - 0.13,
            0.13,
        );
        hit.socket(
            (side_of(p) - c.seated(1, anchor(1, c).truncate())).length() - 0.1,
            0.1,
        );
    }
    rasp(&mut hit, p, c);
    hit.blend(oral_grasping(p, c, MOUTH, REACH, ORAL_PAIRS, GRIP), 8, 0.18);
    hit
}

/// The rasping mouth beneath the head: a round, lipped opening facing down and forward, which
/// everts as it feeds into a short cone ringed with rows of hooked, iron-hardened teeth that it
/// works against what its tentacles hold, tearing it apart a mouthful at a time. In flight it is
/// drawn in, only the lip showing.
fn rasp(hit: &mut Surface, p: Vec3, c: &Tetralate) {
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

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[site(VENOM, vec3(0.0, -0.36, -3.95), vec3(0.2, 0.16, 0.32))];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.20, 0.30, 0.40, 0.0),
    vec3(0.70, 0.76, 0.82),
    [vec4(13.0, 2.6, -1.3, 0.75), vec4(5.0, 1.6, 2.2, 0.75)],
)
.hinged([vec2(0.95, 0.35), vec2(0.6, 0.2)])
.stalked([vec2(0.22, 0.22), vec2(0.25, 0.25)])
.styled([vec4(1.2, 0.1, 3.0, 0.0), vec4(0.9, 0.3, 0.0, 0.0)])
.skinned(vec4(5.0, 0.4, 2.0, -0.10))
.flying(vec4(1.10, 0.30, 0.55, 1.00))
.marked(vec4(4.00, 4.00, 0.80, 0.45), vec3(0.06, 0.08, 0.12));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{
        animalia::{
            mind::{Ethology, orbit},
            physiology::{Light, Physiology},
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Twilight,
        speed: 75.0,
        flocks: false,
        ..super::super::PHYSIOLOGY
    };
    /// Idle, a Needleshark soars: slow wide circles riding the rising air above the herds, every
    /// pass a look down for something small enough to fall on.
    pub const ETHOLOGY: Ethology = Ethology {
        wander: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Drift,
                Some(orbit(up, up, 80.0, s.now * 0.08 + s.wobble()) + up * 8.0),
                0.45,
            )
        },
        ..super::super::ETHOLOGY
    };
}
