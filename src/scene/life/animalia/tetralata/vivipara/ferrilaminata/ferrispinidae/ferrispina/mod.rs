pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, biology::Biology, eyes, smooth_min, surface::Surface, tail, torso,
};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec3, vec3};
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
/// It is broad at its root and tapers to a needle point, its midline sweeping back ever more
/// steeply towards the tip so its whole front edge is drawn across whatever the animal rolls past.
/// `lean` tilts it from upright towards one side, and its section is a lens, keen at both edges.
pub fn blade(p: Vec3, root: Vec3, height: f32, rake: f32, width: f32, lean: f32) -> f32 {
    let q = p - root;
    let (sine, cosine) = lean.sin_cos();
    let across = q.x * cosine - q.y * sine;
    let up = q.x * sine + q.y * cosine;
    let t = (up / height.max(1e-3)).clamp(0.0, 1.0);
    let midline = rake * height * t * t;
    let slope = 2.0 * rake * t;
    let half = width * 0.5 * (1.0 - t).powf(0.8) + width * 0.01;
    let along = (q.z - midline) / (1.0 + slope * slope).sqrt();
    let outline = (along.abs() - half).max(-up).max(up - height);
    let lens = (1.0 - (along / half).powi(2)).max(0.0);
    let section = across.abs() - width * 0.09 * (1.0 - t * 0.7) * lens - 0.005;
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

/// The Iron Spine body plan, which every member of the genus shares.
///
/// A long, deep and narrow body is flattened from side to side like a blade itself so it slips
/// through the roll of a cutting pass, ending in the flexible tail that whips it through tight
/// arcs; along the spine runs a crest of blades. The crest is as much a display as a weapon. A web
/// of skin stretched between the blades' roots carries the animal's colours, and both are raised to
/// their full height to court a mate or face down a rival, laid back for speed in the chase.
/// Bigger, sounder blades win mates, so no two animals' crests are alike, and the tallest stand
/// over the shoulders where the roll drives them hardest.
pub fn shredder(p: Vec3, c: &CreatureInstance, build: Build) -> Surface {
    let length = build.length;
    let narrow = vec3(p.x * 1.3, p.y, p.z);
    let mut hit = Surface::new(
        smooth_min(
            torso(narrow, vec3(1.15, build.depth, length), build.head) / 1.3,
            tail(p, c),
            0.4,
        ),
        0,
    );
    let grown = (c.life.y * c.life.y).max(0.12);
    let own = c.identity.x;
    // Laid back by speed, raised in display.
    let laid = (c.animation.w / 60.0).clamp(0.0, 1.0) * 0.8;
    let rake = build.rake + laid;
    let first = -length * 0.62;
    let spacing = length * 1.3 / build.blades as f32;
    let back = |z: f32| build.depth * (0.78 - (z / length).powi(2) * 0.55);
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
        let height =
            build.height * (0.35 + 0.65 * shoulder) * (0.8 + variety(i, 2.0, own) * 0.4) * grown
                / (1.0 + laid * 0.6);
        hit.join(
            blade(
                p,
                vec3(0.0, back(z) - 0.15, z),
                height,
                rake * (0.85 + variety(i, 3.0, own) * 0.3),
                build.width * (0.4 + 0.6 * grown) * (0.85 + shoulder * 0.3),
                (variety(i, 4.0, own) - 0.5) * 0.12,
            ),
            11,
        );
    }
    // The display web, scalloped down between the blades.
    let span = first + (build.blades as f32 - 1.0) * spacing;
    let along = ((p.z - first) / spacing).fract_gl();
    let shoulder = (1.0 - ((p.z + length * 0.3) / (length * 0.95)).abs()).max(0.0);
    let top = build.height
        * build.web
        * grown
        * (0.35 + 0.65 * shoulder)
        * (0.72 + 0.28 * (along * TAU).cos())
        / (1.0 + laid);
    let base = back(p.z) - 0.25;
    let web = (p.x.abs() - 0.04)
        .max(p.y - base - top)
        .max(base - 0.2 - p.y)
        .max(first - p.z)
        .max(p.z - span);
    hit.join(web * 0.8, 4);
    hit.join(eyes(narrow, length, build.head) / 1.3, 3);
    hit
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
