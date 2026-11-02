use crate::scene::life::Kind;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::animalia::physiology::{BREATHER, FLOAT, GANGLION, PUMP};
pub mod specialia;
pub mod vivipara;
pub mod volitantia;
pub mod young;

/// A cistern of water, filled at Pondplants and in cloud, that carries a tetralate through dry air.
#[cfg(not(target_arch = "spirv"))]
pub const CISTERN: Organ = Organ::new("water cistern", Function::Water, vec3(0.22, 0.46, 0.80));
/// The flight muscles at the roots of the four wings.
#[cfg(not(target_arch = "spirv"))]
pub const FLIGHT_MUSCLE: Organ =
    Organ::new("flight muscle", Function::Flight, vec3(0.52, 0.16, 0.20)).of(Tissue::Muscle);
/// The muscle at the base of the tail that sets it, and with it the body's heading.
#[cfg(not(target_arch = "spirv"))]
pub const TAIL_MUSCLE: Organ =
    Organ::new("tail muscle", Function::Movement, vec3(0.60, 0.22, 0.24)).of(Tissue::Muscle);
/// A rod of coiled elastic fibre down each side of the back, from the head to the tail: the
/// flight muscles pull against it, it bows with every stroke, and it springs the stroke back.
#[cfg(not(target_arch = "spirv"))]
pub const SPRING_ROD: Organ =
    Organ::new("spring rod", Function::Support, vec3(0.90, 0.86, 0.72)).of(Tissue::Spring);
/// A hoop of the same fibre round the body at each wing pair's roots, which the rods run through
/// and the flight muscles are anchored on.
#[cfg(not(target_arch = "spirv"))]
pub const GIRDLE: Organ =
    Organ::new("wing girdle", Function::Support, vec3(0.82, 0.78, 0.64)).of(Tissue::Spring);

/// The tetralate body plan.
///
/// A nerve centre in the head, breathing folds behind it, the pump below
/// them, a flight muscle at the root of every wing, gas above the crop, the cistern behind, and a
/// steering muscle at the tail's root, all hung in a frame of two spring rods and a girdle at each
/// wing pair.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[
    Part::lobe(GANGLION, vec3(-0.9, 0.2, 0.0), vec3(0.4, 0.4, 0.07)),
    Part::lobe(BREATHER, vec3(-0.62, 0.05, 0.55), vec3(0.28, 0.35, 0.09)),
    Part::lobe(PUMP, vec3(-0.5, -0.25, 0.0), vec3(0.32, 0.35, 0.09)),
    Part::shoulder(FLIGHT_MUSCLE, 0, vec3(0.38, 0.5, 0.13)),
    Part::shoulder(FLIGHT_MUSCLE, 1, vec3(0.34, 0.45, 0.11)),
    Part::lobe(FLOAT, vec3(-0.05, 0.55, 0.0), vec3(0.38, 0.3, 0.2)),
    Part::lobe(CISTERN, vec3(0.35, 0.05, 0.0), vec3(0.36, 0.33, 0.13)),
    Part::lobe(TAIL_MUSCLE, vec3(0.88, 0.0, 0.4), vec3(0.3, 0.45, 0.1)),
    Part::rod(
        SPRING_ROD,
        vec3(-0.85, 0.6, 0.22),
        vec3(0.95, 0.5, 0.22),
        0.06,
    ),
    Part::girdle(GIRDLE, 0, 0.05),
    Part::girdle(GIRDLE, 1, 0.05),
];

/// What a tetralate adds to an animal: its four wings, its tail and ribbons, and the knots of its
/// mouth tentacles, which it handles food and the world with.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Tetralate {
    pub animal: Animal,
    pub wings: Wings,
    pub rig: Rig,
    /// The knots of each mouth tentacle, laid out by [`slot`], and where they were last frame.
    pub tentacles: [[[Vec4; 3]; KNOTS]; 3],
    pub previous_tentacles: [[[Vec4; 3]; KNOTS]; 3],
    /// Whatever others have hung on it, as the Næ dress their Nætu: up to [`DRAPES`] strands of
    /// cloth, each a chain of knots its simulation left, as a lived thing and never inherited.
    pub drapes: [[Vec4; DRAPE_KNOTS]; DRAPES],
}

/// How many strands of cloth a tetralate can have hung on it, and how many knots each is
/// simulated with.
pub const DRAPES: usize = 7;
pub const DRAPE_KNOTS: usize = 8;

impl Deref for Tetralate {
    type Target = Animal;
    fn deref(&self) -> &Animal {
        &self.animal
    }
}

impl DerefMut for Tetralate {
    fn deref_mut(&mut self) -> &mut Animal {
        &mut self.animal
    }
}

impl Body for Tetralate {
    fn organism(&self) -> &CreatureInstance {
        &self.animal.organism
    }
    fn carrying(&self) -> (Vec4, Vec4) {
        self.animal.carrying()
    }
    fn sex(&self) -> f32 {
        self.animal.sex()
    }
    fn swung(&self, point: Vec3) -> Vec3 {
        // The Naeturidae's tail shafts are drawn wider than their knots.
        let kind = self.kind.x;
        let scale = if kind == Kind::Nae as u32 {
            vec3(0.2, 0.25, 1.0)
        } else if kind == Kind::Naetu as u32 {
            vec3(3.0, 3.0, 1.0)
        } else {
            Vec3::ONE
        };
        self.rig.swung(point, scale)
    }
}

impl Creature for Tetralate {
    fn animal(&self) -> &Animal {
        &self.animal
    }
}

impl Flier for Tetralate {
    fn wings(&self) -> &Wings {
        &self.wings
    }
}

impl Rigged for Tetralate {
    fn rig(&self) -> &Rig {
        &self.rig
    }
    fn rigging(&mut self) -> (&mut CreatureInstance, &mut Rig) {
        (&mut self.animal.organism, &mut self.rig)
    }
}

/// What the tetralates are born with beyond every animal's temper.
///
/// How much they play, how long a tail they grow to steer by, how far their mouth tentacles reach,
/// and how closely they keep to their fellows when they travel together.
pub const PLAYFULNESS: Gene = Gene(20);
pub const TAIL: Gene = Gene(21);
pub const REACH: Gene = Gene(22);
pub const FORMATION: Gene = Gene(23);

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
        "With nothing to put at its back, a tetralate sees all round at once: a circlet of small lensed eyes rings its head from ahead to behind, each in its own lidded socket, and a larger pair beneath watches the aur, where the light and the worst danger lie. The mouth opens beneath, among the tentacles that bring food up to it.",
        "The updraft and the gas stores carry the body, so the wings are oars and rudders rather than lifting surfaces: each grows from a short, muscular stalk on the shoulder about which the whole wing twists, square to the air to drive on the downstroke and feathered edge-on to slip back on the up, and turned to steer in a glide. A wing joined to the flank along its length, as the blood-feeders' are, could not turn so; the stalk is narrowest in hoverers and divers, broadest in the heavy cruisers.",
        "A digestive crop and pulsatile nutrient channels support more concentrated feeding than the absorptive surfaces of bubblebacks.",
        "Pleated respiratory passages recover moisture, although vigorous flight still creates a need for drinking or humid air.",
        "Later lineages modify wing margins and feeding tentacles, with extra tentacle pairs and internal brooding evolving in the live-bearing branch.",
        "They were the first animals to grow a frame: two rods of coiled elastic fibre down the back and a hoop of it at each pair of wing roots, which bend with every wingbeat and spring it back, so that a stroke costs less than its full price.",
    ],
    anatomy: ANATOMY,
);

/// Knots drawn along each mouth tentacle.
const KNOTS: usize = 12;

/// The knot halfway along a tentacle, which with its root and tip is all that is simulated.
const MIDDLE: usize = (KNOTS - 1) / 2;

/// How long a tentacle is settled for before it is drawn.
const SETTLE: f32 = 1.5;

/// Where tentacle `index` keeps its knots among the instance's tentacle knots.
const fn slot(index: usize) -> (usize, usize) {
    (index / 3, index % 3)
}

/// The roots of a creature's mouth tentacles, and how far each reaches.
///
/// The tentacles are the tetralates' arms. The first pair, which every member of the phylum
/// inherits, rises from either side of the mouth; the extra pairs the live-bearers grew follow
/// back along the throat, each further down the neck than the last and set a little wider, so
/// they reach past one another to the same work rather than crowding at the mouth.
fn tentacle_root(mouth: Vec3, scale: f32, index: usize) -> (Vec3, f32, f32) {
    let lane = (index / 2) as f32;
    let side = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
    // Each pair a little out of step with the last along the throat.
    let stagger = side * 0.04 * (lane % 2.0 - 0.5);
    let root = mouth
        + vec3(
            side * (0.16 + lane * 0.07),
            -0.05 - lane * 0.14,
            lane * 0.34 + stagger,
        ) * scale;
    (root, scale * (2.4 - lane * 0.25), side)
}

/// Pose the mouth tentacles by simulating them.
///
/// Each is a root held at the mouth, a middle and a tip, pulled by its own muscle towards a pose
/// reaching forward and in beneath the face, but given to gravity in the animal's own frame and to
/// the air streaming past as it flies, and writhing slowly as muscle waves run down it. Settled
/// from that pose over the last second and a half, which is all the damping remembers, and drawn
/// along one smooth curve through the three.
pub fn rig_tentacles(c: &mut Tetralate, mouth: Vec3, scale: f32, pairs: u32) {
    rig_grasping(c, mouth, scale, pairs, 0.0);
}

/// Pose the mouth tentacles as [`rig_tentacles`] does, the outer pair grown `grip` longer again and
/// thicker, as a lineage that works with its tentacles grows them.
pub fn rig_grasping(c: &mut Tetralate, mouth: Vec3, scale: f32, pairs: u32, grip: f32) {
    const STEP: f32 = 1.0 / 120.0;
    let scale = scale * (0.35 + c.life.y * 0.65);
    let (gravity, wind) = air_frame(c, 0.0);
    for index in 0..(pairs as usize * 2).min(9) {
        let (root, reach, side) = tentacle_root(mouth, scale, index);
        let lane = (index / 2) as f32;
        let outer = if lane < 0.5 { 1.0 + grip } else { 1.0 };
        let reach = reach * (0.8 + c.gene(REACH) * 0.4) * outer;
        let length = reach * 0.5;
        // At rest they hang forward and down beneath the head like loosely held arms, bent a
        // little at the middle, each moving slowly on its own.
        let trailing = |knot: usize, time: f32| {
            let t = knot as f32 / (KNOTS - 1) as f32;
            let writhe = (time * 0.9 + lane * 1.9 + side - t * 3.0).sin() * 0.12 * t;
            let elbow = (t * PI).sin();
            root + vec3(
                side * (0.08 + 0.18 * t + writhe) * reach * t,
                (-0.62 * t + 0.08 * elbow) * reach,
                (-0.32 * t - 0.12 * elbow + writhe * 0.5) * reach,
            )
        };
        // Feeding, the tentacles reach forward and close round the food at the mouth, holding it
        // to the jaws; otherwise they trail, writhing a little.
        let grasp = gape(c);
        let rest = |knot: usize, time: f32| {
            let t = knot as f32 / (KNOTS - 1) as f32;
            let reach_forward = root
                + vec3(
                    side * reach * (0.22 * (t * PI).sin() - 0.12 * t),
                    -reach * (0.18 * t - 0.08 * (t * PI).sin()),
                    -reach * 0.62 * t,
                );
            trailing(knot, time).lerp(reach_forward, grasp)
        };
        let end = c.animation.z.max(0.0);
        // Only the middle and the tip are simulated: a tentacle is one muscle, which bends in a
        // single smooth arc, and a chain of many knots kinks into zigzags no tentacle makes.
        let settle = |start: f32| {
            let mut points = [root, rest(MIDDLE, start), rest(KNOTS - 1, start)];
            let mut previous = points;
            let steps = ((end - start) / STEP) as usize;
            for step in 0..steps {
                let time = start + step as f32 * STEP;
                let last = previous;
                previous = points;
                for joint in 1..3 {
                    let velocity = (points[joint] - last[joint]) * 0.985;
                    // Muscle tone, strongest near the root where the tentacle is thickest, holds
                    // the pose against the load.
                    let pose = rest(if joint == 1 { MIDDLE } else { KNOTS - 1 }, time);
                    let tone = (pose - points[joint]) * if joint == 1 { 55.0 } else { 30.0 };
                    let drag = (wind - velocity / STEP) * 1.5;
                    swing(&mut points[joint], velocity, gravity + drag + tone, STEP);
                }
                tie(&mut points, length);
                tie(&mut points, length);
                points[0] = root;
            }
            points
        };
        let (earlier, later, weight) = settled(end, SETTLE, settle);
        let joints = [
            earlier[0].lerp(later[0], weight),
            earlier[1].lerp(later[1], weight),
            earlier[2].lerp(later[2], weight),
        ];
        // The drawn knots lie along the one curve through the root, the middle and the tip.
        let control = joints[1] * 2.0 - (joints[0] + joints[2]) * 0.5;
        let mut points = [Vec3::ZERO; KNOTS];
        for (knot, point) in points.iter_mut().enumerate() {
            let t = knot as f32 / (KNOTS - 1) as f32;
            *point = joints[0] * ((1.0 - t) * (1.0 - t))
                + control * (2.0 * t * (1.0 - t))
                + joints[2] * (t * t);
        }
        let radius = scale * (0.075 - lane * 0.008) * (1.0 + (outer - 1.0) * 0.7);
        let (sheet, column) = slot(index);
        for (knot, point) in points.iter().enumerate() {
            let t = knot as f32 / (KNOTS - 1) as f32;
            c.tentacles[sheet][knot][column] = point.extend(radius * (1.0 - t * 0.85));
        }
        c.previous_tentacles[sheet] = c.tentacles[sheet];
    }
}

/// The tip of mouth tentacle `index` and the knot before it, each with its thickness, for a
/// species that grows something at its tips.
pub const fn tentacle_tip(c: &Tetralate, index: usize) -> (Vec4, Vec4) {
    let (sheet, column) = slot(index);
    (
        c.tentacles[sheet][KNOTS - 1][column],
        c.tentacles[sheet][KNOTS - 2][column],
    )
}

/// The mouth tentacles posed by [`rig_tentacles`]: tubes tapering smoothly along their curves.
pub fn oral_tentacles(
    point: Vec3,
    creature: &Tetralate,
    mouth: Vec3,
    scale: f32,
    pairs: u32,
) -> f32 {
    oral_grasping(point, creature, mouth, scale, pairs, 0.0)
}

/// The mouth tentacles posed by [`rig_grasping`], the outer pair `grip` longer.
pub fn oral_grasping(
    point: Vec3,
    creature: &Tetralate,
    mouth: Vec3,
    scale: f32,
    pairs: u32,
    grip: f32,
) -> f32 {
    let scale = scale * (0.35 + creature.life.y * 0.65);
    let bounds =
        (point - mouth).length() - scale * 3.2 * (0.8 + creature.gene(REACH) * 0.4) * (1.0 + grip);
    if bounds > scale * 0.1 {
        return bounds;
    }
    let mut distance = 1e6f32;
    for index in 0..8 {
        if index >= pairs as usize * 2 {
            break;
        }
        let (sheet, column) = slot(index);
        let root = creature.tentacles[sheet][0][column];
        if (point - root.truncate()).length()
            > scale * 2.6 * (0.8 + creature.gene(REACH) * 0.4) * (1.0 + grip) + root.w
        {
            continue;
        }
        for knot in 0..KNOTS - 1 {
            let a = creature.tentacles[sheet][knot][column];
            let b = creature.tentacles[sheet][knot + 1][column];
            distance = smooth_min(
                distance,
                tapered_tail(point, a.truncate(), b.truncate(), a.w, b.w),
                a.w * 0.6,
            );
        }
    }
    distance * 0.9
}

/// How far its mouth gapes: wide as it bites, parted as it strikes, shut in flight or when it is
/// drawn alone.
pub fn gape(c: &Animal) -> f32 {
    if c.motion.w > 0.0 {
        c.motion.z.clamp(0.0, 1.0)
    } else {
        0.0
    }
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

    use super::*;
    use crate::scene::life::{
        animalia::{
            ETHOLOGY as ANIMAL,
            mind::{Ethology, Intent, Situation, Urges, orbit},
            nature::ROOST,
            physiology::{Brood, Light, Physiology, Sexes, flow_channels, grazes},
            tetralata::young::{CLUTCH, DRIFTING_EGG, EGG_STRINGS, TETHERED_EGG},
        },
        behaviour::{Activity, Goal},
    };

    /// A tetralate's young: eggs, kept as the lineage keeps them, and once hatched a small body of
    /// its parents' shape that grows into theirs; the young of those born live wear it from the
    /// first.
    fn young(kind: Kind, young: f32) -> u32 {
        let brood = kind.animal().map_or(Brood::Tethered, |animal| animal.brood);
        match brood {
            Brood::Drifting if young < 0.3 => DRIFTING_EGG,
            Brood::Strings if young < 0.3 => EGG_STRINGS,
            Brood::Carried if young < 0.3 => CLUTCH,
            Brood::Tethered if young < 0.3 => TETHERED_EGG,
            _ => 0,
        }
    }

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: grazes,
        breathe: flow_channels,
        digestion: 0.004,
        speed: 12.0,
        form: young,
        hatches: true,
        // The tetralates gather in swarms, flocks and herds where a mate is never far, and their
        // sexes parted early.
        sexes: Sexes::Separate,
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
                (1.0 - s.drives.youth)
                    * (0.1 + s.temper(PLAYFULNESS) * 0.3)
                    * (1.0 - s.drives.hunger),
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
                // Some keep tight on their fellows' tails, others travel in loose company.
                s.seen
                    .kin
                    .map(|kin| kin + s.seen.heading * (2.0 + (1.0 - s.temper(FORMATION)) * 6.0)),
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
