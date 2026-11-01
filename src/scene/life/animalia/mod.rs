use crate::scene::{
    life::{
        Body, CLUTCH, CreatureInstance, FULL_POUCH, Look,
        biology::Biology,
        ellipsoid,
        genome::Gene,
        rounded, smooth_min,
        surface::{Hit, Ray, Surface, TraceResult, trace, visit_paired_sheets, wing_bones},
    },
    math::Power,
};
use core::{
    f32::consts::TAU,
    ops::{Deref, DerefMut},
};
use spirv_std::glam::{FloatExt, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
pub mod bullidorsata;
#[cfg(not(target_arch = "spirv"))]
pub mod mind;
#[cfg(not(target_arch = "spirv"))]
pub mod physiology;
pub mod sanguinivora;
pub mod tetralata;

pub const BIOLOGY: Biology = super::BIOLOGY;

/// What every animal adds to an organism: how its body moved this step, what its life has done to
/// its hide short of wounding it, what it stores and carries that shows, and where it carries its
/// young.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Animal {
    pub organism: CreatureInstance,
    /// What physics did to its body this step: its wingbeat phase, how hard it is working, how
    /// wide its mouth gapes, and whether it was set at all.
    pub motion: Vec4,
    pub previous_motion: Vec4,
    /// What its life has done to its hide that is no wound: how overgrown with moss and lumaweed,
    /// and how blunted its edges and spines.
    pub hide: Vec4,
    /// What it stores and carries that shows: how much iron is laid into it, how far through
    /// carrying young it is, how many it carries, and how many eggs it has glued to its body.
    pub stores: Vec4,
    /// Its brood chamber, where it swells as its young grow: the centre in its body's frame and
    /// the chamber's size.
    pub belly: Vec4,
}

impl Deref for Animal {
    type Target = CreatureInstance;
    fn deref(&self) -> &CreatureInstance {
        &self.organism
    }
}

impl DerefMut for Animal {
    fn deref_mut(&mut self) -> &mut CreatureInstance {
        &mut self.organism
    }
}

impl Body for Animal {
    fn organism(&self) -> &CreatureInstance {
        &self.organism
    }
    fn carrying(&self) -> (Vec4, Vec4) {
        (self.belly, self.stores)
    }
}

/// Wings, as the two lineages that fly by them each grew them: how torn each is, fore left, fore
/// right, hind left and hind right, from whole to gone, and how scarred where tears healed.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Wings {
    pub torn: Vec4,
    pub scarred: Vec4,
}

/// A tail and trailing ribbons, posed by simulating them each frame, with where the tail was last
/// frame for motion vectors.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rig {
    pub tail: [Vec4; 6],
    pub previous_tail: [Vec4; 6],
    pub ribbons: [Vec4; 30],
}

impl Rig {
    /// How far a point along the tail moved since last frame with its posed chain, the tail drawn
    /// at `scale` of its knots' spacing.
    pub fn swung(&self, point: Vec3, scale: Vec3) -> Vec3 {
        let length = self.tail[5].z - self.tail[0].z;
        if point.z <= self.tail[0].z || length <= 0.01 {
            return Vec3::ZERO;
        }
        let fraction = ((point.z - self.tail[0].z) / length).clamp(0.0, 1.0) * 5.0;
        let index = (fraction as usize).min(4);
        let delta = self.previous_tail[index]
            .lerp(self.previous_tail[index + 1], fraction - index as f32)
            - self.tail[index].lerp(self.tail[index + 1], fraction - index as f32);
        delta.truncate() * scale
    }
}

/// An animal's instance, whatever its lineage adds.
pub trait Creature: Body {
    fn animal(&self) -> &Animal;
}

/// An animal that flies on wings.
pub trait Flier: Creature {
    fn wings(&self) -> &Wings;
}

/// An animal with a tail and trailing ribbons, posed each frame.
pub trait Rigged: Creature {
    fn rig(&self) -> &Rig;
    /// The organism and its rig together, to pose one from the other.
    fn rigging(&mut self) -> (&mut CreatureInstance, &mut Rig);
}

/// The temper every animal is born with, in its kingdom's part of the genome: how bold, how drawn
/// to its own kind, how aggressive, how restless, how hungry by nature, how patient, and how
/// watchful.
pub const BOLDNESS: Gene = Gene(8);
pub const SOCIABILITY: Gene = Gene(9);
pub const AGGRESSION: Gene = Gene(10);
pub const RESTLESSNESS: Gene = Gene(11);
pub const APPETITE: Gene = Gene(12);
pub const PATIENCE: Gene = Gene(13);
pub const VIGILANCE: Gene = Gene(14);
pub const CURIOSITY: Gene = Gene(15);

/// The body every animal is born with besides its size and colouring.
///
/// How long it can keep working, how quickly it answers its own steering, how true it aims, and
/// how long its wings or fins grow against the rest of its kind (a broad build makes them broad
/// too).
pub const STAMINA: Gene = Gene(16);
pub const AGILITY: Gene = Gene(17);
pub const PRECISION: Gene = Gene(18);
pub const WINGSPAN: Gene = Gene(19);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Kingdom,
    "Animalia",
    "Animals",
    [
        "Animals obtain usable energy by consuming organic material made by other organisms.",
        "Porous forms absorb food across wet surfaces, while active lineages gather it into a digestive crop and distribute nutrients through pulsatile channels.",
        "Moisture-recovery folds conserve water, although sustained flight can still force animals to seek Pondplants or humid currents.",
        "Lift comes from gas chambers, wings or attachment to another living body, and every lineage must find rising air strong enough to carry it.",
        "Direct exposure to concentrated aur remains lethal except within the carefully limited mist-skimming adaptations of the Nætu lineage.",
    ],
);

/// Render the body in its coat and its wing pairs; a pair of no span is absent.
pub fn render_wings<C: Flier>(
    ray: Ray,
    c: &C,
    surface: impl Fn(Vec3, &C) -> Surface,
    look: Look,
) -> TraceResult {
    let body = |p: Vec3, c: &C| with_wings(surface(p, c), p, c, look);
    let mut result = TraceResult::opaque(trace(ray, c, body, look));
    visit_wings(ray, c, look, |hit| result.add(hit));
    result
}

/// What bending and the years do to a winged body's skin, as displacement, laid out from where its
/// own wings are rooted, so every species gets it in the right places for its build.
///
/// The skin gathers into folds rayed round each wing root, where the flight muscles pull it every
/// stroke; into rings of creases across the neck ahead of the fore wings and the hips behind the
/// hind ones, where the body flexes; and everywhere into a fine net of crossing wrinkles. All of it
/// runs deeper with age, and in proportion to the body's own size, so a giant's folds are a
/// giant's and a sparkie's barely show.
fn hide(p: Vec3, c: &CreatureInstance, look: Look, distance: f32) -> f32 {
    let fore = look.hinges[0].extend(look.wings[0].z);
    let hind = if look.wings[1].x > 0.0 {
        look.hinges[1].extend(look.wings[1].z)
    } else {
        fore + vec3(0.0, 0.0, fore.x.max(0.2) * 1.5)
    };
    // The body's own measure: the span between its wing roots.
    let size = (hind.z - fore.z).abs().max(fore.x * 1.2).max(0.15);
    if distance.abs() > size * 0.3 {
        return 0.0;
    }
    let age = 0.45 + c.life.z * 0.8 + c.life.y * 0.25;
    let flank = vec3(p.x.abs(), p.y, p.z);
    let mut skin = 0.0f32;
    // Folds rayed round each wing root.
    for index in 0..2 {
        let root = if index == 0 { fore } else { hind };
        let offset = flank - root;
        let round = offset.length() / size;
        let near = (-round * round * 3.0).exp();
        let rays = (offset.y.atan2(offset.z) * 7.0 + round * 2.0).sin().abs();
        skin -= (1.0 - rays).power(3) * near * 0.035 * size;
        skin -= (round * 14.0).sin().abs() * near * 0.012 * size;
    }
    // Rings of creases across the neck and the hips.
    let wobble = (p.x * 2.1 / size + p.y * 1.7 / size).sin() * 0.06 * size;
    for index in 0..2 {
        let (at, every) = if index == 0 {
            (fore.z - size * 0.75, 11.0)
        } else {
            (hind.z + size * 0.8, 9.0)
        };
        let along = (p.z - at + wobble) / size;
        let near = (-along * along * 6.0).exp();
        skin -= (along * every).sin().abs().power(2) * near * 0.022 * size;
    }
    // A fine net of crossing wrinkles, finer than the folds.
    let fine = 26.0 / size;
    let first = (p.z * fine + (p.x * fine * 0.6).sin() * 1.4 + (p.y * fine * 0.5).sin()).sin();
    let second = (p.x * fine * 0.9 - p.z * fine * 0.4 + (p.y * fine * 0.7).sin() * 1.2).sin();
    skin -= ((1.0 - first.abs()).power(8) + (1.0 - second.abs()).power(8)) * 0.0035 * size;
    skin * age
}

/// A grown body with its wing frames and whatever brood it carries.
///
/// The frames grow out of the body, blended in at the roots so each wing visibly springs from the
/// flank. The young have their own forms, buds and all.
pub fn with_wings<C: Flier>(mut body: Surface, p: Vec3, creature: &C, look: Look) -> Surface {
    let c = creature.animal();
    if c.life.w > 0.0 {
        return body;
    }
    if body.material == 0 {
        body.distance += hide(p, c, look, body.distance);
    }
    let mut frame = 1e6f32;
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x > 0.0 {
            frame = frame.min(wing_bones(p, creature, look, index));
        }
    }
    let joined = smooth_min(body.distance, frame, look.wings[0].y * 0.12);
    // Iron edges are only as long as the iron its owner has gathered, laid down from the shoulder
    // outwards in uneven runs; beyond them the frame is bare skin.
    let edge = if look.edge == 6 {
        let out = (p.x.abs() / look.wings[0].x.max(0.1)).clamp(0.0, 1.0);
        let run = ((p.x.abs() * 3.1 + c.chance(1) * 17.0).sin() * 0.5 + 0.5) * 0.3;
        if out < c.stores.x * 1.1 - run { 6 } else { 0 }
    } else {
        look.edge
    };
    let mut surface = Surface::new(
        joined,
        if frame < body.distance {
            edge
        } else {
            body.material
        },
    );
    if c.kind.z != 0 && look.brood.w > 0.0 {
        brood(&mut surface, p, c, look.brood);
    }
    surface
}

/// A parent's brood, where its species carries it. A clutch is a band of eggs glued round the body,
/// two rows deep and packed unevenly, wrapping the flanks and belly but leaving the back clear for
/// the wings, as many of them as it laid; a full band holds about eighteen. A pouch is a fold of
/// skin on the back opening forwards and upwards, so nothing can fall out of it towards the aur:
/// flat when empty and swollen when full. The young in it are animals of their own, drawn where
/// they sit.
fn brood(surface: &mut Surface, p: Vec3, c: &Animal, at: Vec4) {
    let q = p - vec3(0.0, at.x, at.y);
    if c.kind.z == CLUTCH {
        let angle = (q.y / at.w).atan2(q.x / at.z);
        let sector = TAU / 14.0;
        let mut eggs = 1e6f32;
        for row in 0..2 {
            let shift = row as f32 * 0.5;
            let turn = (angle / sector - shift).round() + shift;
            // Round the belly and flanks only.
            let around = turn * sector;
            if around.sin() > 0.45 {
                continue;
            }
            // Where each egg sits is chance, so it is drawn from the individual's own seed.
            let seed = ((turn * 12.9898 + row as f32 * 4.1 + c.kind.y as f32 * 0.0137).sin()
                * 43_758.547)
                .fract_gl();
            let laid = if c.stores.w > 0.0 {
                c.stores.w / 18.0
            } else {
                1.0
            };
            if ((seed * 7.31).fract_gl()) > laid {
                continue;
            }
            let size = at.z.min(at.w) * (0.2 + seed * 0.07);
            let centre = vec3(
                around.cos() * (at.z + size * 0.6),
                around.sin() * (at.w + size * 0.6),
                (row as f32 - 0.5) * size * 2.0 + (seed - 0.5) * size * 0.6,
            );
            eggs = eggs.min(ellipsoid(q - centre, vec3(size, size, size * 1.25)));
        }
        if eggs < surface.distance {
            surface.distance = smooth_min(surface.distance, eggs, at.z * 0.03);
            surface.material = 7;
        }
        return;
    }
    let full = if c.kind.z == FULL_POUCH { 1.0 } else { 0.0 };
    let height = at.w * (0.45 + full * 0.55);
    let pouch = rounded(
        q - vec3(0.0, -at.w * 0.3, 0.0),
        vec3(at.z, height + at.w * 0.3, at.z * 1.4),
    );
    // Its mouth, a slot across the front facing forwards and up.
    let mouth = rounded(
        q - vec3(0.0, height * 0.8, -at.z * 1.25),
        vec3(at.z * 0.55, height * 0.35 + 0.02, at.z * 0.4),
    );
    surface.distance = smooth_min(surface.distance, pouch.max(-mouth), at.z * 0.25);
}

/// The wing membranes of a species' look along a ray.
pub fn visit_wings<C: Flier, F: FnMut(Hit)>(ray: Ray, c: &C, look: Look, mut emit: F) {
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x <= 0.0 {
            continue;
        }
        visit_paired_sheets(ray, c, look, index, &mut emit);
    }
}

#[cfg(not(target_arch = "spirv"))]
pub mod nature {
    //! What every animal does, from the gas-bags up.
    //!
    //! It flees what would eat it, drinks, sleeps, rests, feeds, mates, ranges about, and goes back
    //! to the light and the air it lives in. Lineages below build their own ways on these.

    use crate::scene::life::{
        Kind,
        animalia::{
            AGGRESSION, BOLDNESS, CURIOSITY, RESTLESSNESS, SOCIABILITY,
            mind::{
                Ethology, Glimpse, Intent, Interest, Mind, Perception, Sighting, Situation, Urges,
                orbit,
            },
            physiology::{ANIMAL, Light, Physiology, Take, plant},
        },
        behaviour::{Activity, Goal},
    };
    use spirv_std::glam::Vec3;

    pub const PHYSIOLOGY: Physiology = ANIMAL;

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        fear,
        hunt: chase,
        court,
        sleep,
        wander,
        notice,
        ..Ethology::INERT
    };

    /// Somewhere solid to rest by or cling to: a plant of some size.
    pub const ROOST: Interest = Interest("roost");

    /// Every animal notices the larger plants about it, to rest by.
    pub fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        if !glimpse.heard && plant(glimpse.other) && glimpse.other.biology().adult_mass >= 1.0 {
            seen.note(ROOST, glimpse.offset);
        }
    }

    /// The urges every animal feels: to escape, to get out of a storm or back to its altitude, to
    /// drink, sleep, rest and feed, to mate when its body is ready, to range about, and to go back
    /// to the light it lives in.
    fn urges(s: &Situation, urges: &mut Urges) {
        let seen = s.seen;
        let drives = s.drives;
        let hunter = s.kind.behaviour().hunts;
        urges.add(FLEE, s.fear * 1.4);
        if seen.storm.is_some() {
            urges.add(SHELTER, 0.9);
        }
        if seen.habitat.is_some() {
            urges.add(RETURN, 0.7);
        }
        if seen.water.is_some() {
            urges.add(DRINK, drives.thirst * 1.3);
        }
        // Sleep comes with weariness alone, for there is no night.
        urges.add(SLEEP, drives.sleepiness * 0.9);
        urges.add(REST, drives.tiredness * 0.9);
        if hunter {
            if seen.prey.is_some() {
                urges.add(HUNT, s.hunger(true) * 1.2 + s.temper(AGGRESSION) * 0.15);
            }
        } else if seen.food.is_some() {
            urges.add(FORAGE, s.hunger(false) * 1.1 + 0.1);
        }
        let carrion = s
            .kind
            .animal()
            .is_some_and(|animal| animal.takes != Take::Living);
        if seen.carcass.is_some() && carrion {
            urges.add(SCAVENGE, s.hunger(hunter) * 1.15 + 0.1);
        }
        if seen.mate.is_some() {
            urges.add(COURT, drives.desire);
        }
        urges.add(EXPLORE, s.temper(CURIOSITY) * s.temper(RESTLESSNESS) * 0.5);
        urges.add(SEEK, astray(s.kind, seen.light) * 0.6);
    }

    /// Fear: of a threat, less in a bold animal; of its own wounds; and of its kin's panic, which
    /// spreads through a group. A fright lingers as alarm, fading over a minute.
    fn fear(s: &Situation, mind: &mut Mind) -> f32 {
        mind.alarm *= (-(s.now - mind.felt).max(0.0) / 12.0).exp();
        mind.felt = s.now;
        let fear =
            (s.threat * (1.3 - s.temper(BOLDNESS) * 0.7) + panic(s, 1.0) + s.drives.hurt * 0.5)
                .max(mind.alarm);
        mind.alarm = mind.alarm.max(fear * 0.8);
        fear
    }

    /// How much of its kin's panic an animal catches: the share of them fleeing, more in a sociable
    /// one, and more again in a lineage given to it.
    pub fn panic(s: &Situation, contagion: f32) -> f32 {
        // Fellows fleeing are frightening, but less so with no danger in sight of its own, so a
        // fright does not feed on itself once its cause has gone.
        let seen = if s.seen.threat.is_some() { 1.0 } else { 0.5 };
        s.seen.fleeing as f32 / s.seen.kin_count.max(1) as f32
            * (0.4 + s.temper(SOCIABILITY))
            * contagion
            * seen
    }

    /// The simplest hunt: fly at the prey, leading it.
    fn chase(_: &mut Mind, s: &Situation, prey: Sighting) -> Goal {
        Goal::new(Activity::Strike, Some(lead(s.kind, prey)), 1.0)
    }

    /// Where the prey will be by the time the hunter reaches it.
    pub fn lead(kind: Kind, prey: Sighting) -> Vec3 {
        let speed = kind.speed().max(1.0);
        prey.offset + prey.velocity * (prey.offset.length() / speed).min(3.0)
    }

    /// The simplest courtship: fly alongside a mate a while.
    fn court(s: &Situation) -> Goal {
        Goal::new(
            Activity::Display,
            s.seen
                .mate
                .map(|(mate, _)| mate + s.seen.heading.normalize_or_zero() * 3.0),
            0.6,
        )
    }

    /// The simplest sleep: hang where it is, wings folded, and let the air carry it.
    const fn sleep(_: &Situation) -> Goal {
        Goal::new(Activity::Sleep, None, 0.0).folded(0.9).hovering()
    }

    /// Idling: keeping near its kind.
    fn wander(_: &mut Mind, s: &Situation) -> Goal {
        Goal::new(
            Activity::Drift,
            Some(s.seen.kin.unwrap_or(Vec3::ZERO)),
            0.25,
        )
    }

    /// Idling in the lineage's way.
    pub const WANDER: Intent = Intent {
        name: "wander",
        act: |mind, s| (s.kind.behaviour().wander)(mind, s),
    };

    /// Running from a threat, jinking as it goes so a hunter cannot simply lead it.
    pub const FLEE: Intent = Intent {
        name: "flee",
        act: |_, s| {
            let seen = s.seen;
            let up = s.up();
            let away = seen
                .threat
                .map_or_else(|| -seen.kin.unwrap_or(up), |threat| -threat.offset);
            let jink = up.cross(away).normalize_or_zero() * (s.now * 1.7 + s.wobble()).sin() * 0.6;
            let direction = (away.normalize_or_zero() + jink + up * 0.2).normalize_or_zero();
            Goal::new(Activity::Flee, Some(direction * 50.0), 1.25)
        },
    };

    /// Drinking at a Pondplant.
    pub const DRINK: Intent = Intent {
        name: "drink",
        act: |_, s| Goal::new(Activity::Drink, s.seen.water, 0.6).hovering(),
    };

    /// Asleep, where and as its lineage sleeps.
    pub const SLEEP: Intent = Intent {
        name: "sleep",
        act: |_, s| (s.kind.behaviour().sleep)(s),
    };

    /// Resting tired muscles by something solid, or among its kind.
    pub const REST: Intent = Intent {
        name: "rest",
        act: |_, s| {
            Goal::new(Activity::Rest, s.seen.mark(ROOST).or(s.seen.kin), 0.3)
                .folded(0.5)
                .hovering()
        },
    };

    /// Feeding on what it eats where it is.
    pub const FORAGE: Intent = Intent {
        name: "forage",
        act: |_, s| feed(s),
    };

    /// Stalking and striking prey in its lineage's way; with no prey in sight, or a latcher's
    /// host lost, it falls back on what food there is.
    pub const HUNT: Intent = Intent {
        name: "hunt",
        act: |mind, s| {
            s.seen
                .prey
                .map_or_else(|| feed(s), |prey| (s.kind.behaviour().hunt)(mind, s, prey))
        },
    };

    fn feed(s: &Situation) -> Goal {
        Goal::new(Activity::Feed, s.seen.food.or(s.seen.home), 0.7).hovering()
    }

    /// Making for a carcass.
    pub const SCAVENGE: Intent = Intent {
        name: "scavenge",
        act: |_, s| Goal::new(Activity::Feed, s.seen.carcass, 0.8).hovering(),
    };

    /// Courting a mate in its lineage's way.
    pub const COURT: Intent = Intent {
        name: "court",
        act: |_, s| (s.kind.behaviour().court)(s),
    };

    /// Going back to the altitudes it lives at.
    pub const RETURN: Intent = Intent {
        name: "return",
        act: |_, s| Goal::new(Activity::Return, s.seen.habitat, 0.6),
    };

    /// Getting away from a storm.
    pub const SHELTER: Intent = Intent {
        name: "shelter",
        act: |_, s| Goal::new(Activity::Flee, s.seen.storm.map(|storm| -storm), 0.9),
    };

    /// Casting about in wide arcs for rising air, as gas-bags and heavy gliders do alike.
    pub const RIDE: Intent = Intent {
        name: "ride",
        act: |_, s| {
            let up = s.up();
            Goal::new(Activity::Drift, Some(orbit(up, up, 60.0, s.now * 0.2)), 0.4)
        },
    };

    /// Ranging out on a course of its own, changed now and then.
    pub const EXPLORE: Intent = Intent {
        name: "explore",
        act: |mind, s| {
            // A new course once it has come to the end of the last: across the wind, turned
            // from the old one further the more restless it is, and further out the more curious.
            if mind.course.length_squared() < 400.0 {
                let up = s.up();
                let across = s.seen.wind.cross(up).normalize_or(Vec3::X);
                let turn = 1.0 + s.temper(RESTLESSNESS) * 2.0 + s.wobble() * 0.1;
                let (sine, cosine) = turn.sin_cos();
                let heading = across * cosine + up.cross(across) * sine;
                mind.course = heading * (100.0 + s.temper(CURIOSITY) * 200.0);
            }
            Goal::new(Activity::Drift, Some(mind.course), 0.45)
        },
    };

    /// Travelling sunward if it is too dark here, away if too bright.
    pub const SEEK: Intent = Intent {
        name: "seek",
        act: |_, s| {
            let darker = s.seen.light > band(s.kind.light()).1;
            let way = if darker {
                -s.seen.sunward
            } else {
                s.seen.sunward
            };
            Goal::new(Activity::Return, Some(way * 300.0 + s.up() * 10.0), 0.6)
        },
    };

    /// The range of light a species lives in, from dimmest to brightest.
    pub const fn band(light: Light) -> (f32, f32) {
        match light {
            Light::Sunlit => (0.45, 1.0),
            Light::Twilight => (0.12, 0.45),
            Light::Aurlit => (0.0, 0.12),
            Light::Anywhere => (0.0, 1.0),
        }
    }

    /// How far outside the light it lives in an animal has strayed, from not at all to badly.
    fn astray(kind: Kind, here: f32) -> f32 {
        let (dimmest, brightest) = band(kind.light());
        ((dimmest - here).max(here - brightest).max(0.0) * 4.0).min(1.0)
    }
}
