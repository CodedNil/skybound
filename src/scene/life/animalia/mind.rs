//! How an animal decides what to do, and how what it does descends the tree of life.
//!
//! Its temperament, the drives its body raises and what it has perceived are weighed into urges by
//! the behaviour its lineage inherits, the strongest is chosen, and the lineage carries it out.
//!
//! Every animal taxon has an [`Ethology`]: the ways its members feel urges, hunt, court, sleep and
//! idle. A taxon takes its parent's with `..super::ETHOLOGY` and replaces only what its lineage
//! does differently, often by calling the parent's way and building on it. So animals start with
//! the urges every animal feels, a lineage that took to herding or swarming adds that where it
//! arose, and a genus or a species adds its own courtship, quirks and way of killing. Plants have
//! no mind.
//!
//! Intents and the things an animal notices are open in the same way. Each [`Intent`] is declared
//! in the taxon where it arose, carrying the way its ancestors carried it out, and each
//! [`Interest`] where a lineage first took notice of it: play in the tetralates, guarding the young
//! in the Guardians, the calls of the herd in the live-bearers. Only the senses every animal has
//! are fields of [`Perception`].

use crate::scene::life::{
    Kind,
    animalia::nature::{COURT, SLEEP, WANDER},
    behaviour::{Activity, Goal},
    organs::Strike,
};
use spirv_std::glam::Vec3;
use std::fmt;

/// The settled temper of one individual, fixed from its identity: how bold, how curious, how drawn
/// to its own kind, how aggressive and how restless.
#[derive(Clone, Copy, Debug, Default)]
pub struct Temperament {
    pub boldness: f32,
    pub curiosity: f32,
    pub sociability: f32,
    pub aggression: f32,
    pub restlessness: f32,
}

impl Temperament {
    pub fn of(seed: u32) -> Self {
        let trait_of = |channel: u32| {
            let mut value = seed ^ channel.wrapping_mul(0x9e37_79b9);
            value ^= value >> 15;
            value = value.wrapping_mul(0x2c1b_3c6d);
            value ^= value >> 12;
            value = value.wrapping_mul(0x297a_2d39);
            ((value ^ (value >> 15)) >> 8) as f32 / 16_777_215.0
        };
        Self {
            boldness: trait_of(11),
            curiosity: trait_of(12),
            sociability: trait_of(13),
            aggression: trait_of(14),
            restlessness: trait_of(15),
        }
    }
}

/// Another animal, as it is seen: where it is and how it moves.
#[derive(Clone, Copy, Debug)]
pub struct Sighting {
    pub offset: Vec3,
    pub velocity: Vec3,
    /// Whether it is helpless: paralysed, or held in a web.
    pub helpless: bool,
}

/// Everything an animal knows this moment, as offsets from itself in the scene's frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct Perception {
    pub up: Vec3,
    pub wind: Vec3,
    /// Food it can eat where it is: a plant, a host, drifting matter.
    pub food: Option<Vec3>,
    /// The animal it would hunt, if it hunts.
    pub prey: Option<Sighting>,
    /// The nearest predator that could eat it, and whether it is coming.
    pub threat: Option<Sighting>,
    pub carcass: Option<Vec3>,
    /// A mate, and whether that mate is courting too.
    pub mate: Option<(Vec3, bool)>,
    /// The centre of its own kind nearby, their heading, how many there are, and how many of them
    /// are fleeing.
    pub kin: Option<Vec3>,
    pub heading: Vec3,
    pub kin_count: u32,
    pub fleeing: u32,
    pub water: Option<Vec3>,
    pub player: Option<Vec3>,
    /// The way back to the altitudes it lives at, and away from a storm.
    pub habitat: Option<Vec3>,
    pub storm: Option<Vec3>,
    /// The plant it keeps as its home, for a lineage that keeps one: a Weaver's web is strung
    /// there.
    pub home: Option<Vec3>,
    /// How bright it is here, from the aur's glow alone at the equator to the full sunburst over
    /// the poles; light is a matter of place.
    pub light: f32,
    /// Along the surface towards the sunburst.
    pub sunward: Vec3,
    /// How strongly the air rises here.
    pub updraft: f32,
    /// How damp the air is, from bone dry to the wet heart of a cloud.
    pub humidity: f32,
    /// The nearest of each thing its lineage takes notice of, beyond what every animal senses.
    pub marks: [Option<(Interest, Vec3)>; 4],
}

impl Perception {
    /// Mark something its lineage takes notice of, keeping only the nearest of each.
    pub fn note(&mut self, interest: Interest, offset: Vec3) {
        let mut free = None;
        for (index, mark) in self.marks.iter_mut().enumerate() {
            match mark {
                Some((noted, at)) if *noted == interest => {
                    if offset.length_squared() < at.length_squared() {
                        *at = offset;
                    }
                    return;
                }
                None if free.is_none() => free = Some(index),
                _ => {}
            }
        }
        if let Some(index) = free {
            self.marks[index] = Some((interest, offset));
        }
    }

    /// Where the nearest thing of an interest is, if it has noticed one.
    pub fn mark(&self, interest: Interest) -> Option<Vec3> {
        self.marks
            .iter()
            .flatten()
            .find(|(noted, _)| *noted == interest)
            .map(|&(_, at)| at)
    }
}

/// Something a lineage takes notice of, declared where it first did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interest(pub &'static str);

/// Another organism, or a call, as it comes to an animal's notice.
#[derive(Clone, Copy, Debug)]
pub struct Glimpse {
    /// The one noticing.
    pub kind: Kind,
    pub other: Kind,
    pub offset: Vec3,
    /// How far grown the other is, from newborn to grown.
    pub grown: f32,
    /// Whether it was only heard: a call from far off.
    pub heard: bool,
    /// What it is doing, as far as can be seen.
    pub activity: Activity,
}

/// What the body asks for, each from nothing to urgent.
#[derive(Clone, Copy, Debug, Default)]
pub struct Drives {
    pub hunger: f32,
    pub thirst: f32,
    pub tiredness: f32,
    pub sleepiness: f32,
    /// Pain and weakness from wounds and blood loss.
    pub hurt: f32,
    /// The urge to breed: zero in the young, the starving or the spent.
    pub desire: f32,
    /// How young it is, from newborn to grown.
    pub youth: f32,
}

/// What an animal has set itself to do, declared in the taxon where it arose with the way its
/// lineage carries it out unless a descendant's `act` does it differently.
#[derive(Clone, Copy)]
pub struct Intent {
    /// Its name, which also tells intents apart.
    pub name: &'static str,
    pub act: fn(&mut Mind, &Situation) -> Goal,
}

impl PartialEq for Intent {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Intent {}

impl fmt::Debug for Intent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.name)
    }
}

impl Default for Intent {
    fn default() -> Self {
        WANDER
    }
}

/// One animal's mind between moments: what it is doing and since when, how far through a manoeuvre
/// it is, and how long it has courted.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mind {
    pub temperament: Temperament,
    pub intent: Intent,
    pub since: f32,
    /// A hunter's progress through its attack: climbing, then striking.
    pub striking: bool,
    /// Seconds spent courting the current mate.
    pub courted: f32,
    /// A lingering unease after a fright, fading over a minute, and when it last felt afraid.
    pub alarm: f32,
    pub felt: f32,
    /// Where it was wandering to, so a wander has a course.
    pub course: Vec3,
    /// How hungry it was when it last decided.
    pub hunger: f32,
    /// How worn the tools it kills with are, from keen to blunt: each blow dulls an edge, a beak
    /// or a stylet a little, and only a lineage that tends its tools gets them back.
    pub wear: f32,
}

impl Mind {
    pub fn new(seed: u32) -> Self {
        Self {
            temperament: Temperament::of(seed),
            ..Self::default()
        }
    }

    /// Feel the urges the lineage feels, keep to the current intent unless another clearly
    /// outweighs it, and carry the intent out in the lineage's way.
    pub fn decide(
        &mut self,
        kind: Kind,
        drives: Drives,
        seen: &Perception,
        now: f32,
        individual: u32,
    ) -> Goal {
        let ethology = kind.behaviour();
        let mut situation = Situation {
            kind,
            drives,
            seen,
            temper: self.temperament,
            now,
            individual,
            threat: threat(seen),
            fear: 0.0,
            wear: self.wear,
        };
        situation.fear = (ethology.fear)(&situation, self);
        self.hunger = drives.hunger;
        let mut urges = Urges::new();
        urges.add(WANDER, 0.12 + self.temperament.restlessness * 0.1);
        (ethology.urges)(&situation, &mut urges);
        let mut best = (WANDER, f32::MIN);
        for &(intent, pull) in urges.all() {
            // Sleep, once fallen into, is not broken lightly.
            let pull = if intent != self.intent {
                pull
            } else if intent == SLEEP && drives.sleepiness > 0.15 && situation.fear < 0.5 {
                pull + 0.5
            } else {
                pull + 0.15
            };
            if pull > best.1 {
                best = (intent, pull);
            }
        }
        if best.0 != self.intent {
            self.intent = best.0;
            self.since = now;
            self.striking = false;
            if best.0 != COURT {
                self.courted = 0.0;
            }
        }
        let intent = self.intent;
        (ethology.act)(self, &situation).unwrap_or_else(|| (intent.act)(self, &situation))
    }
}

/// How threatening the nearest predator is: how close it is, and whether it is coming.
fn threat(seen: &Perception) -> f32 {
    seen.threat.map_or(0.0, |threat| {
        let closeness = (1.0 - threat.offset.length() / 300.0).clamp(0.0, 1.0);
        let coming = (-threat.velocity.normalize_or_zero())
            .dot(threat.offset.normalize_or_zero())
            .max(0.0);
        closeness * (0.6 + coming * 0.8)
    })
}

/// What a mind knows as it decides: its species, its body's drives, what it perceives, its
/// temper, and how threatened and afraid it is.
#[derive(Clone, Copy)]
pub struct Situation<'a> {
    pub kind: Kind,
    pub drives: Drives,
    pub seen: &'a Perception,
    pub temper: Temperament,
    pub now: f32,
    /// A stable number for the individual, to set it a little apart from the rest.
    pub individual: u32,
    pub threat: f32,
    pub fear: f32,
    /// How worn its weapons are; see [`Mind::wear`].
    pub wear: f32,
}

impl Situation<'_> {
    pub fn up(&self) -> Vec3 {
        self.seen.up.normalize_or_zero()
    }

    /// A phase of its own, so neighbours doing the same thing are not in lockstep.
    pub fn wobble(&self) -> f32 {
        (self.individual % 628) as f32 * 0.01
    }

    /// How hungry it is, made keener in a hunter by its aggression.
    pub fn hunger(&self, hunter: bool) -> f32 {
        self.drives.hunger * (1.0 + self.temper.aggression * 0.3 * f32::from(hunter))
    }
}

/// The urges an animal feels this moment, each with how strongly it pulls.
#[derive(Clone, Copy)]
pub struct Urges {
    list: [(Intent, f32); 32],
    count: usize,
}

impl Urges {
    pub const fn new() -> Self {
        Self {
            list: [(WANDER, 0.0); 32],
            count: 0,
        }
    }

    pub const fn add(&mut self, intent: Intent, pull: f32) {
        if self.count < self.list.len() {
            self.list[self.count] = (intent, pull);
            self.count += 1;
        }
    }

    /// Strengthen or weaken every urge of one kind, as a lineage's temper does to the urges it
    /// inherited.
    pub fn scale(&mut self, intent: Intent, factor: f32) {
        for urge in &mut self.list[..self.count] {
            if urge.0 == intent {
                urge.1 *= factor;
            }
        }
    }

    pub fn all(&self) -> &[(Intent, f32)] {
        &self.list[..self.count]
    }
}

impl Default for Urges {
    fn default() -> Self {
        Self::new()
    }
}

/// A hunter's blow as it lands, seen from inside its prey.
///
/// It holds the way the hunter is moving and where it is, in the prey's own frame, the prey's size,
/// how far its dodging throws the aim, where its pump lies, and how much of a gland's reserve the
/// hunter has to spend.
#[derive(Clone, Copy)]
pub struct Aim {
    pub along: Vec3,
    pub toward: Vec3,
    pub extent: Vec3,
    pub scatter: Vec3,
    pub pump: Vec3,
    pub nerve: Vec3,
    pub reserve: f32,
    /// How much of its edge the hunter's weapon still has, from blunted to keen.
    pub edge: f32,
}

/// What a blow does: the wound, how long it holds the prey, how long until the hunter can strike
/// again, and how much of its reserve it spends.
#[derive(Clone, Copy)]
pub struct Blow {
    pub strike: Strike,
    pub stun: f32,
    pub rest: f32,
    pub cost: f32,
}

/// How things stand between a hunter and the prey it holds, in a web or by latching on.
///
/// It holds how far apart they are, how far the prey is from the hunter's web if it keeps one, how
/// near it must be to reach, what share of the prey's mass the hunter is, and the step's length.
#[derive(Clone, Copy)]
pub struct Snare {
    pub distance: f32,
    pub from_web: Option<f32>,
    pub reach: f32,
    pub share: f32,
    pub fastened: bool,
    pub dt: f32,
}

/// What holding the prey does this step: how long it is held, how much fluid is drawn from it,
/// and whether the hunter fastens on.
#[derive(Clone, Copy, Default)]
pub struct Hold {
    pub stun: f32,
    pub drain: f32,
    pub latch: bool,
}

/// How a lineage behaves.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each is a separate habit a lineage has or lacks"
)]
pub struct Ethology {
    /// Whether it carries helpless prey off as it eats, held in its mouth tentacles.
    pub carries: bool,
    /// The urges it feels, each with its pull.
    pub urges: fn(&Situation, &mut Urges),
    /// How afraid it is, from the threat in the situation.
    pub fear: fn(&Situation, &mut Mind) -> f32,
    /// How it hunts the prey it has fixed on.
    pub hunt: fn(&mut Mind, &Situation, Sighting) -> Goal,
    /// How it courts.
    pub court: fn(&Situation) -> Goal,
    /// Where and how it sleeps.
    pub sleep: fn(&Situation) -> Goal,
    /// What it does when nothing presses.
    pub wander: fn(&mut Mind, &Situation) -> Goal,
    /// Any intent it carries out in its own way, or `None` to do it as the intent's own lineage
    /// does.
    pub act: fn(&mut Mind, &Situation) -> Option<Goal>,
    /// What it takes notice of in another, beyond what every animal senses, marked in its
    /// perception.
    pub notice: fn(&Glimpse, &mut Perception),
    /// The blow it deals, if it strikes.
    pub blow: fn(&Aim) -> Option<Blow>,
    /// What it does to prey it holds, by web or by latching on.
    pub snare: fn(&Snare) -> Option<Hold>,
    /// Whether it hunts at all, and how far beyond its reach it can strike.
    pub hunts: bool,
    pub range: f32,
    /// Whether it clings to a plant while it sleeps, and keeps a plant as its home.
    pub clings: bool,
    pub anchored: bool,
}

impl Kind {
    /// How an animal behaves; a plant, having no mind, behaves inertly.
    pub fn behaviour(self) -> Ethology {
        self.ethology().unwrap_or(Ethology::INERT)
    }
}

impl Ethology {
    /// The behaviour of life at its root: none at all. Plants and the simplest drifters go where
    /// the air takes them.
    pub const INERT: Self = Self {
        urges: |_, _| {},
        fear: |_, _| 0.0,
        hunt: |_, _, _| drift(),
        court: |_| drift(),
        sleep: |_| drift(),
        wander: |_, _| drift(),
        act: |_, _| None,
        notice: |_, _| {},
        blow: |_| None,
        snare: |_| None,
        hunts: false,
        range: 0.0,
        clings: false,
        anchored: false,
        carries: false,
    };
}

const fn drift() -> Goal {
    Goal::new(Activity::Drift, None, 0.0)
}

/// A point circling a centre, `radius` out, at angle `phase`, in the plane level with `up`.
pub fn orbit(up: Vec3, centre: Vec3, radius: f32, phase: f32) -> Vec3 {
    let across = up.cross(centre).normalize_or(up.any_orthonormal_vector());
    let along = across.cross(up).normalize_or_zero();
    (across * phase.cos() + along * phase.sin()) * radius
}

/// Display to a mate from close by, hovering, or towards `fallback` when none is in sight.
pub fn display_near(seen: &Perception, distance: f32, fallback: Option<Vec3>) -> Goal {
    Goal::new(
        Activity::Display,
        seen.mate
            .map(|(mate, _)| mate - mate.normalize_or_zero() * distance)
            .or(fallback),
        0.3,
    )
    .hovering()
}

/// Sleeping on the wing: gliding slow circles on rising air, half its nerve centre asleep.
pub fn glide_asleep(s: &Situation) -> Goal {
    let up = s.up();
    Goal::new(
        Activity::Sleep,
        Some(orbit(up, up, 40.0, s.now * 0.3)),
        0.35,
    )
    .folded(0.2)
    .hovering()
}

/// Spiralling up round a mate.
pub fn spiral(s: &Situation, radius: f32) -> Goal {
    let up = s.up();
    Goal::new(
        Activity::Display,
        s.seen
            .mate
            .map(|(mate, _)| mate + orbit(up, mate, radius, s.now * 1.5) + up * 4.0),
        0.7,
    )
}

/// Hanging motionless, as ambushers and web-keepers wait.
pub const fn hang_still() -> Goal {
    Goal::new(Activity::Rest, None, 0.0).hovering()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{
        KINDS, Kingdom,
        animalia::{
            nature::{ROOST, SEEK},
            tetralata::{
                FOLLOW,
                vivipara::{YOUNG, ferrilaminata::ferrispinidae::WHETSTONE},
            },
        },
    };

    fn seen() -> Perception {
        Perception {
            up: Vec3::Y,
            ..Perception::default()
        }
    }

    #[test]
    fn a_frightened_animal_flees_and_a_guardian_stands() {
        let threat = Sighting {
            offset: Vec3::new(20.0, 0.0, 0.0),
            velocity: Vec3::new(-30.0, 0.0, 0.0),
            helpless: false,
        };
        let perception = Perception {
            threat: Some(threat),
            ..seen()
        };
        let mut grazer = Mind::new(7);
        grazer.temperament.boldness = 0.1;
        let goal = grazer.decide(Kind::Herbivolus, Drives::default(), &perception, 0.0, 7);
        assert_eq!(goal.activity, Activity::Flee);
        assert!(
            goal.offset.expect("a way out").x < 0.0,
            "fled towards the threat"
        );
        let mut guardian = Mind::new(7);
        let mut perception = perception;
        perception.note(YOUNG, Vec3::new(-5.0, 0.0, 0.0));
        let drives = Drives {
            youth: 1.0,
            ..Drives::default()
        };
        let goal = guardian.decide(Kind::Custos, drives, &perception, 0.0, 7);
        assert_eq!(
            goal.activity,
            Activity::Guard,
            "a guardian abandoned the young"
        );
    }

    #[test]
    fn a_diver_climbs_above_its_prey_before_it_stoops() {
        let prey = Sighting {
            offset: Vec3::new(10.0, 0.0, 10.0),
            velocity: Vec3::ZERO,
            helpless: false,
        };
        let perception = Perception {
            prey: Some(prey),
            ..seen()
        };
        let hungry = Drives {
            hunger: 1.0,
            youth: 1.0,
            ..Drives::default()
        };
        let mut hunter = Mind::new(3);
        let climb = hunter.decide(Kind::Aculeorhynchus, hungry, &perception, 0.0, 3);
        assert_eq!(climb.activity, Activity::Hunt);
        assert!(
            climb.offset.expect("a climb").y > 30.0,
            "dived without height"
        );
        let below = Perception {
            prey: Some(Sighting {
                offset: Vec3::new(2.0, -40.0, 2.0),
                ..prey
            }),
            ..perception
        };
        let stoop = hunter.decide(Kind::Aculeorhynchus, hungry, &below, 1.0, 3);
        assert_eq!(stoop.activity, Activity::Strike);
        assert!(stoop.speed > 1.5, "the stoop was not a dive");
    }

    #[test]
    fn sleep_comes_with_weariness_and_a_sleeper_is_not_easily_woken() {
        let tired = Drives {
            sleepiness: 0.7,
            youth: 1.0,
            ..Drives::default()
        };
        let mut perception = Perception {
            light: 0.8,
            ..seen()
        };
        perception.note(ROOST, Vec3::X);
        let mut grazer = Mind::new(1);
        assert_eq!(
            grazer
                .decide(Kind::Herbivolus, tired, &perception, 0.0, 1)
                .activity,
            Activity::Sleep,
            "a weary animal stayed awake in full light"
        );
        let slightly = Drives {
            sleepiness: 0.3,
            ..tired
        };
        grazer.decide(Kind::Herbivolus, slightly, &perception, 1.0, 1);
        assert_eq!(grazer.intent, SLEEP, "a sleeper woke at the first change");
    }

    #[test]
    fn an_animal_far_from_its_light_goes_back_to_it() {
        let glare = Perception {
            light: 0.9,
            sunward: Vec3::Z,
            ..seen()
        };
        let mut hunter = Mind::new(5);
        let goal = hunter.decide(Kind::Crepusculofalx, Drives::default(), &glare, 0.0, 5);
        assert_eq!(hunter.intent, SEEK, "a dusk hunter stayed in the glare");
        assert!(
            goal.offset.expect("a way back").z < 0.0,
            "it flew further into the sunburst"
        );
    }

    #[test]
    fn a_resting_herd_keeps_one_in_five_awake() {
        let tired = Drives {
            sleepiness: 0.6,
            youth: 1.0,
            ..Drives::default()
        };
        let herd = Perception {
            kin: Some(Vec3::X),
            kin_count: 8,
            light: 0.8,
            ..seen()
        };
        let awake = (0..50u32)
            .filter(|&individual| {
                let mut mind = Mind::new(individual);
                mind.temperament = Temperament::default();
                mind.decide(Kind::Herbivolus, tired, &herd, 0.0, individual)
                    .activity
                    != Activity::Sleep
            })
            .count();
        assert!((5..=20).contains(&awake), "{awake} of 50 kept watch");
    }

    fn urges_of(kind: Kind, seen: &Perception) -> Urges {
        let situation = Situation {
            kind,
            drives: Drives::default(),
            seen,
            temper: Temperament::default(),
            now: 0.0,
            individual: 1,
            threat: 0.0,
            fear: 0.0,
            wear: 0.0,
        };
        let mut urges = Urges::new();
        (kind.behaviour().urges)(&situation, &mut urges);
        urges
    }

    fn follows(kind: Kind) -> bool {
        let seen = Perception {
            up: Vec3::Y,
            kin: Some(Vec3::X),
            kin_count: 6,
            ..Perception::default()
        };
        urges_of(kind, &seen)
            .all()
            .iter()
            .any(|&(intent, pull)| intent == FOLLOW && pull > 0.0)
    }

    #[test]
    fn content_live_bearers_play_and_groom_and_the_cleverest_most() {
        let seen = Perception {
            up: Vec3::Y,
            kin: Some(Vec3::X * 5.0),
            kin_count: 4,
            light: 0.3,
            ..Perception::default()
        };
        let drives = Drives {
            youth: 1.0,
            ..Drives::default()
        };
        let sociable = |kind: Kind| {
            (0..100u32)
                .filter(|&individual| {
                    let mut mind = Mind::new(individual);
                    let goal = mind.decide(kind, drives, &seen, 0.0, individual);
                    matches!(goal.activity, Activity::Play | Activity::Groom)
                })
                .count()
        };
        let hunter = sociable(Kind::Ferrispina);
        let elder = sociable(Kind::Antiquus);
        let drifter = sociable(Kind::Aerogigas);
        assert!(
            hunter > 20,
            "only {hunter} of 100 content Shredders played or groomed"
        );
        assert!(
            elder > hunter,
            "the Old Wise Ones were no more sociable than Shredders"
        );
        assert_eq!(drifter, 0, "a gas-bag played");
    }

    #[test]
    fn a_blunted_iron_spine_hones_and_a_threatened_grounder_hunkers() {
        let calm = Drives {
            youth: 1.0,
            ..Drives::default()
        };
        let mut seen = seen();
        seen.light = 0.3;
        seen.note(WHETSTONE, Vec3::X * 20.0);
        let mut spine = Mind::new(3);
        spine.wear = 0.9;
        let goal = spine.decide(Kind::Ferrispina, calm, &seen, 0.0, 3);
        assert_eq!(
            goal.activity,
            Activity::Hone,
            "a blunted Iron Spine did not hone"
        );
        let threat = Sighting {
            offset: Vec3::new(30.0, 0.0, 0.0),
            velocity: Vec3::new(-20.0, 0.0, 0.0),
            helpless: false,
        };
        let danger = Perception {
            threat: Some(threat),
            ..Perception {
                up: Vec3::Y,
                ..Perception::default()
            }
        };
        let mut grounder = Mind::new(3);
        let goal = grounder.decide(Kind::Lithosoma, calm, &danger, 0.0, 3);
        assert_ne!(
            goal.activity,
            Activity::Flee,
            "a Grounder tried to outfly a hunter"
        );
    }

    #[test]
    fn behaviour_descends_the_tree_of_life() {
        for kind in KINDS {
            assert_eq!(
                kind.ethology().is_none(),
                kind.kingdom() == Kingdom::Plantae,
                "{kind:?} has a mind it should not, or lacks one"
            );
        }
        // Herding arose in the live-bearers, and swarming in the Nibblers' family, which the
        // Sweepers left behind; the gas-bags never took to either.
        assert!(follows(Kind::Herbivolus) && follows(Kind::Antiquus));
        assert!(follows(Kind::Manduca));
        assert!(
            !follows(Kind::Purgator),
            "the Sweepers kept their family's swarming"
        );
        assert!(!follows(Kind::Aerogigas), "a gas-bag herds");
        // Only lineages that took to hunting hunt.
        assert!(Kind::Aculeorhynchus.behaviour().hunts && Kind::Sericifex.behaviour().hunts);
        assert!(!Kind::Herbivolus.behaviour().hunts && !Kind::Aerogigas.behaviour().hunts);
    }
}
