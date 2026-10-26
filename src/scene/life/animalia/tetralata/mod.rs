pub use super::render_wings;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::organs::{Function, Organ};
use crate::scene::life::{CreatureInstance, biology::Biology, smooth_min, tapered_tail};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Quat, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
pub mod specialia;
pub mod vivipara;
pub mod volitantia;

/// A cistern of water, filled at Pondplants and in cloud, that carries a tetralate through dry air.
#[cfg(not(target_arch = "spirv"))]
pub const CISTERN: Organ = Organ::new("water cistern", Function::Water, vec3(0.22, 0.46, 0.80));
/// The flight muscles at the roots of the four wings.
#[cfg(not(target_arch = "spirv"))]
pub const FLIGHT_MUSCLE: Organ =
    Organ::new("flight muscle", Function::Flight, vec3(0.52, 0.16, 0.20));

pub const BIOLOGY: Biology = super::BIOLOGY;

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, FOLLOW, INVESTIGATE, PHYSIOLOGY, PLAY, roost, shade};

/// The pair of mouth tentacles the tetralates inherited, for handling food against a moving
/// body.
pub const ORAL_PAIRS: u32 = 1;

taxon!(
    Phylum,
    "Tetralata",
    "Four Winged",
    [
        "Tetralata descend from elongated, limbless animals with two pairs of wings and a pair of muscular mouth tentacles.",
        "Wings control movement through the updraft, while the tentacles gather food, hold anchors and manipulate objects without walking limbs.",
        "A digestive crop and pulsatile nutrient channels support more concentrated feeding than the absorptive surfaces of bubblebacks.",
        "Pleated respiratory passages recover moisture, although vigorous flight still creates a need for drinking or humid air.",
        "Later lineages modify wing margins and feeding tentacles, with extra tentacle pairs and internal brooding evolving in the live-bearing branch.",
    ],
);

/// Knots along each simulated mouth tentacle.
const KNOTS: usize = 12;

/// How long a tentacle is settled for before it is drawn.
const SETTLE: f32 = 1.5;

/// Where tentacle `index` keeps its knots: the instance's cloth slots, which no four-winged animal
/// uses for cloth.
const fn slot(index: usize) -> (usize, usize) {
    (index / 3, index % 3)
}

/// The roots of a creature's mouth tentacles, a pair to each lane, set around the mouth, and how
/// far each reaches.
fn tentacle_root(mouth: Vec3, scale: f32, index: usize) -> (Vec3, f32, f32) {
    let lane = (index / 2) as f32;
    let side = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
    let root = mouth
        + vec3(
            side * (0.14 + lane * 0.09),
            (lane % 2.0 - 0.5) * 0.12,
            lane * 0.1,
        ) * scale;
    (root, scale * (1.5 - lane * 0.23), side)
}

/// Pose the mouth tentacles by simulating them.
///
/// Each is a chain of knots held at the mouth, pulled by its own muscle towards a pose reaching
/// forward and in beneath the face, but given to gravity in the animal's own frame and to the air
/// streaming past as it flies, and writhing slowly as muscle waves run down it. Settled from that
/// pose over the last second and a half, which is all the damping remembers.
pub fn rig_tentacles(c: &mut CreatureInstance, mouth: Vec3, scale: f32, pairs: u32) {
    const STEP: f32 = 1.0 / 120.0;
    let scale = scale * (0.35 + c.life.y * 0.65);
    let turn = Quat::from_array(c.rotation.to_array()).conjugate();
    let gravity = turn * vec3(0.0, -9.8, 0.0);
    // The animal flies head first, so the air streams back along it.
    let wind = vec3(0.0, 0.0, c.animation.w.max(0.0));
    for index in 0..(pairs as usize * 2).min(9) {
        let (root, reach, side) = tentacle_root(mouth, scale, index);
        let lane = (index / 2) as f32;
        let length = reach / (KNOTS - 1) as f32;
        let rest = |knot: usize, time: f32| {
            let t = knot as f32 / (KNOTS - 1) as f32;
            let writhe = (time * 1.3 + lane * 1.9 - t * 4.0).sin() * 0.25 * t;
            let curl = t * (1.2 + lane * 0.2) + writhe;
            root + vec3(
                side * (0.12 + 0.35 * t + writhe * 0.4) * reach * t,
                (-0.35 * t - 0.25 * (1.0 - curl.cos())) * reach,
                (-0.9 * curl.sin() / (1.2 + lane * 0.2)) * reach * t.max(0.15),
            )
        };
        let end = c.animation.z.max(0.0);
        let settle = |start: f32| {
            let start = start.max(0.0);
            let mut points = [Vec3::ZERO; KNOTS];
            for (knot, point) in points.iter_mut().enumerate() {
                *point = rest(knot, start);
            }
            let mut previous = points;
            let steps = ((end - start) / STEP) as usize;
            for step in 0..steps {
                let time = start + step as f32 * STEP;
                let last = previous;
                previous = points;
                for knot in 1..KNOTS {
                    let t = knot as f32 / (KNOTS - 1) as f32;
                    let velocity = (points[knot] - last[knot]) * 0.985;
                    // Muscle tone, strongest near the root where the tentacle is thickest, holds
                    // the pose against the load.
                    let tone = (rest(knot, time) - points[knot]) * (60.0 * (1.0 - t * 0.7));
                    let drag = (wind - velocity / STEP) * 1.5;
                    points[knot] += velocity + (gravity * 0.35 + drag + tone) * STEP * STEP;
                }
                for _ in 0..2 {
                    for knot in 1..KNOTS {
                        let delta = points[knot] - points[knot - 1];
                        let fix = delta * (1.0 - length / delta.length().max(1e-6));
                        if knot == 1 {
                            points[knot] -= fix;
                        } else {
                            points[knot - 1] += fix * 0.5;
                            points[knot] -= fix * 0.5;
                        }
                    }
                }
                points[0] = root;
            }
            points
        };
        // Settled from two starts on a fixed grid, crossfaded, so the pose moves smoothly from one
        // frame to the next.
        let grid = (end / SETTLE).floor();
        let weight = end / SETTLE - grid;
        let earlier = settle((grid - 1.0) * SETTLE);
        let later = settle(grid * SETTLE);
        let mut points = earlier;
        for knot in 0..KNOTS {
            points[knot] = earlier[knot].lerp(later[knot], weight);
        }
        let radius = scale * (0.055 - lane * 0.006);
        let (sheet, column) = slot(index);
        for (knot, point) in points.iter().enumerate() {
            let t = knot as f32 / (KNOTS - 1) as f32;
            c.cloth[sheet][knot][column] = point.extend(radius * (1.0 - t * 0.85));
        }
        c.previous_cloth[sheet] = c.cloth[sheet];
    }
}

/// The mouth tentacles posed by [`rig_tentacles`]: tubes tapering smoothly through their knots,
/// faintly ringed with muscle.
pub fn oral_tentacles(
    point: Vec3,
    creature: &CreatureInstance,
    mouth: Vec3,
    scale: f32,
    pairs: u32,
) -> f32 {
    let scale = scale * (0.35 + creature.life.y * 0.65);
    let bounds = (point - mouth).length() - scale * 1.7;
    if bounds > scale * 0.1 {
        return bounds;
    }
    let mut distance = 1e6f32;
    for index in 0..8 {
        if index >= pairs as usize * 2 {
            break;
        }
        let (sheet, column) = slot(index);
        let root = creature.cloth[sheet][0][column];
        if (point - root.truncate()).length() > scale * 1.6 + root.w {
            continue;
        }
        for knot in 0..KNOTS - 1 {
            let a = creature.cloth[sheet][knot][column];
            let b = creature.cloth[sheet][knot + 1][column];
            let t = knot as f32 / (KNOTS - 1) as f32;
            let rings = 1.0 + (t * PI * 14.0).sin() * 0.05;
            distance = smooth_min(
                distance,
                tapered_tail(point, a.truncate(), b.truncate(), a.w * rings, b.w),
                a.w * 0.6,
            );
        }
    }
    distance * 0.9
}

/// A tube curving through `control` from `root` to a tapering `tip`.
pub fn curved_tube(point: Vec3, root: Vec3, control: Vec3, tip: Vec3, radius: f32) -> f32 {
    let linear = (control - root) * 2.0;
    let quadratic = tip - control * 2.0 + root;
    let mut fraction = ((point - root).dot(tip - root) / (tip - root).length_squared()).saturate();
    for _ in 0..4 {
        let tangent = linear + quadratic * (2.0 * fraction);
        let delta = root + linear * fraction + quadratic * fraction * fraction - point;
        let denominator = tangent.length_squared() + delta.dot(quadratic) * 2.0;
        fraction = (fraction - delta.dot(tangent) / denominator.abs().max(1e-5)).saturate();
    }
    let center = root + linear * fraction + quadratic * fraction * fraction;
    let taper = 1.0 - fraction * 0.88;
    let folds = 1.0 + (fraction * PI * 18.0).sin() * 0.04;
    ((point - center).length() - radius * taper * folds).min(tapered_tail(
        point,
        root,
        root + linear * 0.04,
        radius,
        radius * 0.92,
    )) * 0.7
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The four-winged animals sleep clinging to plants, wings folded, and where they choose to
    //! follows the light they live in. Their young play.

    use crate::scene::life::{
        Kind,
        animalia::{
            ETHOLOGY as ANIMAL,
            mind::{Ethology, Intent, Situation, Urges, orbit},
            nature::ROOST,
            physiology::{Brood, Light, Physiology, flow_channels, grazes},
        },
        behaviour::{Activity, Goal},
    };

    /// A tetralate's young: eggs, kept as the lineage keeps them, then a larva; the young of those
    /// born live take their parents' shape from the first.
    fn young(kind: Kind, young: f32) -> u32 {
        let brood = kind.animal().map_or(Brood::Tethered, |animal| animal.brood);
        match brood {
            _ if young >= 0.7 => 0,
            Brood::Drifting if young < 0.3 => 8,
            Brood::Strings if young < 0.3 => 9,
            Brood::Carried if young < 0.3 => 10,
            Brood::Tethered if young < 0.3 => 4,
            Brood::Live | Brood::Pouch => 0,
            _ => 3,
        }
    }

    use spirv_std::glam::Vec3;

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: grazes,
        breathe: flow_channels,
        digestion: 0.004,
        speed: 12.0,
        form: young,
        hatches: true,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        sleep: roost,
        clings: true,
        ..ANIMAL
    };

    /// The young of every tetralate play a little, chasing their fellows.
    fn urges(s: &Situation, urges: &mut Urges) {
        (ANIMAL.urges)(s, urges);
        if s.drives.youth < 0.8 && s.seen.kin_count > 0 {
            urges.add(
                PLAY,
                (1.0 - s.drives.youth) * 0.25 * (1.0 - s.drives.hunger),
            );
        }
    }

    /// Chasing its fellows, mock-fighting and showing off, as tetralate young do and the
    /// live-bearers keep doing all their lives.
    pub const PLAY: Intent = Intent {
        name: "play",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Play,
                s.seen
                    .kin
                    .map(|kin| kin + orbit(up, kin, 4.0, s.now * 2.0 + s.wobble())),
                1.0,
            )
        },
    };

    /// Coming to look at something strange, the player among them: close enough to look, and no
    /// closer.
    pub const INVESTIGATE: Intent = Intent {
        name: "investigate",
        act: |_, s| {
            Goal::new(
                Activity::Investigate,
                s.seen
                    .player
                    .map(|player| player - player.normalize_or_zero() * 25.0),
                0.5,
            )
            .hovering()
        },
    };

    /// Keeping with its herd or swarm, matching their heading; taken up separately in several
    /// tetralate lineages.
    pub const FOLLOW: Intent = Intent {
        name: "follow",
        act: |_, s| {
            Goal::new(
                Activity::Drift,
                s.seen.kin.map(|kin| kin + s.seen.heading * 4.0),
                0.5,
            )
        },
    };

    /// Going to roost on a plant.
    ///
    /// Sunlit animals sleep in the shade beneath it, out of the drying glare; aur-lit ones climb
    /// into the darker air above, away from the glow; and twilight animals slip across into the
    /// dim side of the sunset band first.
    pub fn roost(s: &Situation) -> Goal {
        let shift = shade(s);
        Goal::new(
            Activity::Sleep,
            Some(s.seen.mark(ROOST).unwrap_or(Vec3::ZERO) + shift),
            0.35,
        )
        .folded(0.9)
        .hovering()
    }

    /// Where a sleeper moves to keep out of light it does not live in.
    pub fn shade(s: &Situation) -> Vec3 {
        let up = s.up();
        match s.kind.light() {
            Light::Sunlit => -up * 4.0,
            Light::Aurlit => up * 80.0,
            Light::Twilight => -s.seen.sunward * 150.0,
            Light::Anywhere => Vec3::ZERO,
        }
    }
}
