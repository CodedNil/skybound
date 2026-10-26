pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance,
    animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::{BLADE_KEENNESS, BLADE_LENGTH},
    biology::Biology,
    capsule, cowl, eyes, louvres, mouth, rounded, smooth_min,
    surface::Surface,
    tapered_tail,
};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
pub mod custos;
pub mod falcata;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 25.0,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Genus,
    "Ferrispina",
    "Iron Spine",
    [
        "Members of Ferrispina are serpentine predators whose spinal blades deliver sweeping aerial attacks.",
        "Independent wing pairs turn the long body through tight arcs without requiring contact with a surface.",
        "Juvenile groups use coordinated approaches to separate prey, while mature animals can overpower larger targets alone.",
        "One branch has specialised in guarding nursery currents, exchanging wide-ranging pursuit for repeated defensive patrols.",
        "Their shared flexible spine and cutting pass distinguish these roles from the straight piercing attack of the snout-armed family.",
    ],
);

/// A crest blade: a flat, thin blade of iron-hardened tissue.
///
/// It is broad at its root and tapers to a point, its midline sweeping back ever more steeply
/// towards the tip so its whole front edge is drawn across whatever the animal rolls past. `lean`
/// tilts it from upright towards one side. Its section is a lens, keen at both edges: the keener
/// its genes made it, the thinner the lens and the finer the point. Every cut wears it, `worn`
/// chipping nicks out of its front edge and rounding its point until it is honed again.
pub fn blade(
    p: Vec3,
    root: Vec3,
    height: f32,
    rake: f32,
    width: f32,
    lean: f32,
    edge: Vec2,
) -> f32 {
    let (keen, worn) = (edge.x, edge.y);
    let q = p - root;
    let (sine, cosine) = lean.sin_cos();
    let across = q.x * cosine - q.y * sine;
    let up = q.x * sine + q.y * cosine;
    let tall = height * (1.0 - worn * 0.12);
    let t = (up / tall.max(1e-3)).clamp(0.0, 1.0);
    let midline = rake * tall * t * t;
    let slope = 2.0 * rake * t;
    let point = 0.6 + keen * 0.6;
    let half = width * 0.5 * (1.0 - t).powf(point) + width * (0.01 + worn * 0.04);
    let along = (q.z - midline) / (1.0 + slope * slope).sqrt();
    let mut outline = (along.abs() - half).max(-up).max(up - tall);
    // Nicks along the front edge where blows met bone.
    if worn > 0.05 {
        let nick = ((t * 9.0).fract_gl() - 0.5).abs() * 2.0;
        let bite = (1.0 - nick.smoothstep(0.0, 0.4)) * worn * width * 0.12;
        outline = outline.max(-(along + half - bite).max(-along));
    }
    let lens = (1.0 - (along / half).powi(2)).max(0.0);
    let thickness = width * (0.12 - keen * 0.06) * (1.0 - t * 0.7);
    let section = across.abs() - thickness * lens - 0.005;
    outline.max(section) * 0.8
}

/// How a Ferrispina is built: its body's length and depth, how many crest blades it raises and how
/// tall, broad and raked back they are, and how high the display web between them reaches up them.
#[derive(Clone, Copy)]
pub struct Build {
    pub length: f32,
    pub depth: f32,
    pub head: f32,
    pub blades: u32,
    pub height: f32,
    pub width: f32,
    pub rake: f32,
    pub web: f32,
}

impl Build {
    /// The body's half-depth at a distance `z` along it, and how high its midline runs: slim at
    /// the head, deepest through the chest where the flight muscles are, pinched at the waist
    /// before the tail.
    fn girth(self, z: f32) -> Vec2 {
        let t = ((z + self.length) / (2.0 * self.length)).clamp(0.0, 1.0);
        let chest = (-(t - 0.3) * (t - 0.3) / 0.03).exp();
        let haunch = (-(t - 0.72) * (t - 0.72) / 0.02).exp();
        let depth = self.depth * (0.5 + chest * 0.8 + haunch * 0.2) * (1.0 - t * 0.35);
        Vec2::new(depth, 0.05 - t * 0.25)
    }

    /// The top of the back at `z`.
    fn back(self, z: f32) -> f32 {
        let girth = self.girth(z);
        girth.y + girth.x
    }
}

/// The Iron Spine body plan, which every member of the genus shares.
///
/// The body is deep and narrow, flattened from side to side like a blade itself so it slips
/// through the roll of a cutting pass: slim at the hooded head, deepest through the chest where the
/// flight muscles drive both wing pairs, pinched at the waist, and drawn out into a flexible tail
/// that ends in a forked rudder of two small blades, which whips it through tight arcs. Overlapping
/// armoured scutes run down the back, each crest blade rising from a socket of bone among them, and
/// rows of plates guard the forebody's flanks; the belly is softer, ridged where it flexes. Beneath
/// the hood, iron-hardened jaws hook together, opening wide to bite pieces from what it has cut
/// down while the mouth tentacles hold it to them.
///
/// The crest is as much a display as a weapon. A web of skin stretched between the blades' roots
/// carries the animal's colours, and both are raised to their full height to court a mate or face
/// down a rival, laid back for speed in the chase. How long and keen the blades grow is inherited;
/// how worn they are is the life it has led since it last honed them.
pub fn shredder(p: Vec3, c: &CreatureInstance, build: Build) -> Surface {
    let mut hit = Surface::new(hull(p, c, build), 0);
    // The forked rudder at the tail's end.
    let end = c.tail[5].truncate();
    let back_of = c.tail[4].truncate();
    // The shader cannot use `normalize_or`, whose check for an infinite length it cannot compile.
    let heading = (end - back_of) / (end - back_of).length().max(1e-4);
    for side in 0..2 {
        let out = if side == 0 { -1.0 } else { 1.0 };
        let fin = end + heading * 1.4 + vec3(out * 0.9, 0.35, 0.0);
        hit.join(tapered_tail(p, end - heading * 0.4, fin, 0.22, 0.02), 11);
    }
    armour(p, build, &mut hit);
    crest(p, c, build, &mut hit);
    hit.join(jaws(p, c, build), 6);
    let narrow = vec3(p.x * 1.3, p.y, p.z);
    hit.socket(
        eyes(narrow, build.length, build.head) / 1.3,
        build.head * 0.16,
    );
    hit
}

/// The body itself: the deep, narrow hull with its keel and shoulders, the hooded head, the
/// louvres, and the tail.
fn hull(p: Vec3, c: &CreatureInstance, build: Build) -> f32 {
    let length = build.length;
    let girth = build.girth(p.z);
    let width = girth.x * 0.6;
    let middle = vec2(p.x / width.max(0.1), (p.y - girth.y) / girth.x.max(0.1));
    let mut body = (middle.length() - 1.0) * width.max(0.1) * 0.8;
    body = body.max(p.z - length).max(-p.z - length - 0.2);
    // A keel along the belly and softer ridges across it where it flexes.
    let keel = capsule(
        p,
        vec3(0.0, -girth.x * 0.7, -length * 0.6),
        vec3(0.0, -girth.x * 0.5, length * 0.3),
        0.18,
    );
    body = smooth_min(body, keel, 0.35);
    if p.y < girth.y {
        body += (p.z * 9.0).cos().abs() * 0.025;
    }
    // Shoulders over each pair of wing roots.
    let flank = vec3(p.x.abs(), p.y, p.z);
    for pair in 0..2 {
        let z = -length * (0.42 - pair as f32 * 0.24);
        let shoulder = rounded(
            flank - vec3(width * 0.55, build.back(z) * 0.45, z),
            vec3(0.45, 0.5, 0.9),
        );
        body = smooth_min(body, shoulder, 0.4);
    }
    // The hood, and the head beneath it.
    let narrow = vec3(p.x * 1.3, p.y, p.z);
    let skull = rounded(
        p - vec3(0.0, 0.0, -length - build.head * 0.4),
        vec3(0.45, 0.5, 1.0) * build.head * 1.2,
    );
    body = smooth_min(body, skull, 0.4);
    body = smooth_min(
        body,
        cowl(narrow, length, build.head) / 1.3,
        0.5 * build.head,
    );
    body = louvres(body, narrow, vec3(1.1, build.depth, length));
    // The tail, thick at its root and tapering, ridged along its top by a row of small spines.
    let mut whip = 1e6f32;
    for i in 0..5 {
        let t = i as f32 / 5.0;
        let from = c.tail[i].truncate();
        let to = c.tail[i + 1].truncate();
        let thick = build.depth * 0.42 * (1.0 - t * 0.75);
        whip = smooth_min(whip, tapered_tail(p, from, to, thick, thick * 0.8), 0.3);
        let spine = from.lerp(to, 0.5) + vec3(0.0, thick * 0.9, 0.0);
        whip = whip.min(tapered_tail(
            p,
            spine - vec3(0.0, thick * 0.4, 0.0),
            spine + vec3(0.0, thick * 0.6, thick),
            thick * 0.3,
            0.01,
        ));
    }
    body = smooth_min(body, whip, 0.5);
    body
}

/// Scutes overlapping down the back, and plates down the forebody's flanks.
fn armour(p: Vec3, build: Build, hit: &mut Surface) {
    let length = build.length;
    let width = build.girth(p.z).x * 0.6;
    let flank = vec3(p.x.abs(), p.y, p.z);
    let spacing = 0.5;
    let cell = (p.z / spacing).round();
    let z = cell * spacing;
    if z.abs() < length * 0.75 {
        let top = build.back(z);
        let scute = rounded(
            p - vec3(0.0, top - 0.05, z + spacing * 0.2),
            vec3(width.max(0.2) * 0.6, 0.18, spacing * 0.7),
        );
        hit.join(scute, 5);
        if z < length * 0.1 {
            let girth = build.girth(z);
            let plate = rounded(
                flank - vec3(girth.x * 0.42, girth.y + girth.x * 0.25, z + spacing * 0.15),
                vec3(0.08, girth.x * 0.28, spacing * 0.62),
            );
            hit.join(plate, 5);
        }
    }
}

/// The crest: its blades, as long and keen as its genes and as worn as its life have made them,
/// each in its socket, and the display web between them.
fn crest(p: Vec3, c: &CreatureInstance, build: Build, hit: &mut Surface) {
    let length = build.length;
    let grown = (c.life.y * c.life.y).max(0.12);
    let own = c.identity.x;
    // Its genes set how long and keen its blades grow; its life how worn they are.
    // How long its genes would grow them, as far as the iron it has gathered lets them.
    let long = (0.75 + c.gene(BLADE_LENGTH) * 0.55) * (0.45 + c.stores.x.min(1.0) * 0.55);
    let edge = vec2(c.gene(BLADE_KEENNESS), c.scars.y);
    // Laid back by speed, raised in display.
    let laid = (c.animation.w / 60.0).clamp(0.0, 1.0) * 0.8;
    let rake = build.rake + laid;
    let first = -length * 0.62;
    let spacing = length * 1.3 / build.blades as f32;
    // Only the blades rooted near here, and those raked back over it.
    let station = ((p.z - first) / spacing).round();
    for offset in 0..4 {
        let index = station + offset as f32 - 2.0;
        if index < 0.0 || index >= build.blades as f32 {
            continue;
        }
        let i = index as u32;
        let z = first + index * spacing + (variety(i, 1.0, own) - 0.5) * spacing * 0.2;
        let shoulder = (1.0 - ((z + length * 0.3) / (length * 0.95)).abs()).max(0.0);
        let height = build.height
            * long
            * (0.35 + 0.65 * shoulder)
            * (0.8 + variety(i, 2.0, own) * 0.4)
            * grown
            / (1.0 + laid * 0.6);
        let root = vec3(0.0, build.back(z) - 0.15, z);
        hit.join(
            blade(
                p,
                root,
                height,
                rake * (0.85 + variety(i, 3.0, own) * 0.3),
                build.width * (0.4 + 0.6 * grown) * (0.85 + shoulder * 0.3),
                (variety(i, 4.0, own) - 0.5) * 0.12,
                edge,
            ),
            11,
        );
        // The bony socket the blade rises from.
        hit.join(
            rounded(
                p - root - vec3(0.0, 0.05, 0.0),
                vec3(0.22, 0.2, build.width * 0.4),
            ),
            5,
        );
    }
    // A low web of skin joining the blades' roots, swept deep between them, where the
    // display colours show without crowding the blades themselves.
    let span = first + (build.blades as f32 - 1.0) * spacing;
    let along = ((p.z - first) / spacing).fract_gl();
    let shoulder = (1.0 - ((p.z + length * 0.3) / (length * 0.95)).abs()).max(0.0);
    let top = build.height
        * long
        * build.web
        * 0.32
        * grown
        * (0.4 + 0.6 * shoulder)
        * (0.35 + 0.65 * (along * TAU).cos().abs().powf(3.0))
        / (1.0 + laid);
    let base = build.back(p.z) - 0.2;
    let web = (p.x.abs() - 0.05 * (1.0 - (p.y - base) / top.max(0.1)).max(0.3))
        .max(p.y - base - top)
        .max(base - 0.3 - p.y)
        .max(first - p.z)
        .max(p.z - span);
    hit.blend(web * 0.8, 4, 0.12);
}

/// The jaws: a pair of hooked, iron-hardened mandibles beneath the hood, hinged at its back and
/// meeting in a hook in front of the mouth. They part sideways and down as far as it is gaping:
/// wide as it bites a piece from what it holds, a little as it strikes, shut in flight.
fn jaws(p: Vec3, c: &CreatureInstance, build: Build) -> f32 {
    let gape = if c.dynamics.w > 0.0 {
        c.dynamics.z
    } else {
        0.0
    };
    let mouth = mouth(build.length, build.head);
    let side = vec3(p.x.abs(), p.y, p.z);
    let head = build.head;
    let hinge = mouth + vec3(0.28, 0.05, 0.4) * head;
    let (swing, hold) = (gape * 0.7).sin_cos();
    let open = |offset: Vec3| {
        vec3(
            offset.x * hold - offset.z * swing * 0.6,
            offset.y - gape * 0.25 * head * (-offset.z).max(0.0),
            offset.z * hold + offset.x * swing * 0.3,
        )
    };
    let elbow = hinge + open(vec3(-0.05, -0.12, -0.55) * head);
    let tip = hinge + open(vec3(-0.22, -0.05, -0.95) * head);
    let mandible = tapered_tail(side, hinge, elbow, 0.13 * head, 0.1 * head).min(tapered_tail(
        side,
        elbow,
        tip,
        0.1 * head,
        0.02 * head,
    ));
    // Teeth along the inner edge.
    let tooth = ((side.z - hinge.z) / (0.12 * head)).fract_gl() - 0.5;
    let middle = hinge.lerp(tip, 0.5);
    let teeth = (vec2(tooth * 0.12 * head, side.x - middle.x).length() - 0.03 * head)
        .max(side.z - hinge.z)
        .max(tip.z - side.z)
        .max((side.y - middle.y).abs() - 0.06 * head);
    mandible.min(teeth)
}

/// A repeatable value in [0, 1) for blade `index` and trait `key` of an individual, so each blade,
/// and each animal's set of them, grows its own way.
pub fn variety(index: u32, key: f32, individual: f32) -> f32 {
    ((index as f32 * 12.9898 + key * 78.233 + individual * 311.7).sin() * 43_758.547).fract_gl()
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Iron Spines hunt in banked passes, coming in from the side so the crest is drawn along
    //! the prey's back as they roll past, cutting into its gas stores. They court by circling
    //! one another with crest and web raised, and their young hunt and play in loose groups.

    use crate::scene::life::{
        animalia::{
            mind::{Aim, Blow, Ethology, Mind, Sighting, Situation, orbit},
            nature::lead,
            physiology::Physiology,
            tetralata::vivipara::ferrilaminata::ferrispinidae::ETHOLOGY as IRON_EDGED,
        },
        behaviour::{Activity, Goal},
        organs::{Strike, Wound},
    };
    use spirv_std::glam::Vec3;

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 65.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: pass,
        court: crest,
        blow: slash,
        hunts: true,
        ..IRON_EDGED
    };

    fn pass(_: &mut Mind, s: &Situation, prey: Sighting) -> Goal {
        let distance = prey.offset.length();
        let side = s.up().cross(prey.offset).normalize_or_zero();
        let offset = if distance > 25.0 {
            side * 12.0
        } else {
            Vec3::ZERO
        };
        let activity = if distance < 30.0 {
            Activity::Strike
        } else {
            Activity::Hunt
        };
        Goal::new(activity, Some(lead(s.kind, prey) + offset), 1.3)
    }

    fn crest(s: &Situation) -> Goal {
        let up = s.up();
        Goal::new(
            Activity::Display,
            s.seen
                .mate
                .map(|(mate, _)| mate + orbit(up, mate, 10.0, s.now * 0.5)),
            0.4,
        )
    }

    /// The crest drawn along the back as it rolls past.
    #[expect(
        clippy::unnecessary_wraps,
        reason = "it fills the blow slot, which some hunters leave empty"
    )]
    fn slash(aim: &Aim) -> Option<Blow> {
        let at = Vec3::new(aim.scatter.x * 0.5, aim.extent.y * 0.6, aim.scatter.z);
        Some(Blow {
            strike: Strike {
                from: at - aim.along * aim.extent.z,
                to: at + aim.along * aim.extent.z,
                radius: aim.extent.y * 0.35,
                // A keen crest opens the gas stores; a dull one only bruises.
                force: 0.9 * aim.edge,
                wound: Wound::Slash,
            },
            stun: 0.0,
            rest: 3.0,
            cost: 0.0,
        })
    }
}
