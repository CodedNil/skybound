//! How an animal's body works.
//!
//! What it eats and in what state, how it breathes, digests, moves and stays up, what aur it can
//! store, how it breeds and keeps its young, the light it lives in and whether it keeps with its
//! group. Every animal taxon has a [`Physiology`], taking its parent's
//! with `..super::PHYSIOLOGY` and changing what its lineage changed, as bodies and behaviour do.

use crate::scene::{
    climate::Climate,
    life::{
        Kind, Kingdom,
        organs::{Function, Organ, Viscera},
        vitals::{Vitals, Waste},
    },
};
use spirv_std::glam::vec3;

/// The pump that drives the body's fluid through its channels.
pub const PUMP: Organ = Organ::new("pump", Function::Circulation, vec3(0.72, 0.12, 0.14));

/// The nerve centre that coordinates the body and the senses.
pub const GANGLION: Organ = Organ::new(
    "nerve centre",
    Function::Coordination,
    vec3(0.86, 0.80, 0.62),
);

/// The crop, where food is held and digested.
pub const CROP: Organ = Organ::new("crop", Function::Digestion, vec3(0.75, 0.42, 0.16));

/// The breathing folds, which exchange gases and wring the moisture back out of every breath.
pub const BREATHER: Organ = Organ::new(
    "breathing folds",
    Function::Breathing,
    vec3(0.88, 0.48, 0.55),
);

/// A store of lifting gas.
pub const FLOAT: Organ = Organ::new("gas store", Function::Lift, vec3(0.62, 0.78, 0.86));

/// The muscle of the body wall, which crawls, grips and squeezes.
pub const MUSCLE: Organ = Organ::new("body muscle", Function::Movement, vec3(0.52, 0.16, 0.20));

/// The brood chamber or the gonads.
pub const BROOD: Organ = Organ::new("brood chamber", Function::Breeding, vec3(0.80, 0.68, 0.30));

/// How a lineage's body works.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each is a separate trait a lineage has or lacks"
)]
pub struct Physiology {
    /// Whether it can eat another kind, by its diet.
    pub eats: fn(Kind, Kind) -> bool,
    /// What state its food must be in.
    pub takes: Take,
    /// Whether it also gleans the detritus drifting in the air.
    pub gleans: bool,
    /// Whether it carries pollen from bloom to bloom as it feeds.
    pub pollinates: bool,
    /// Whether it lives on the fluid of a far larger host.
    pub parasite: bool,
    /// Whether it grooms larger animals, eating what has grown on their hides.
    pub grooms: bool,
    /// Whether it lays iron into its edges and armour, iron it must gather through its life.
    pub ironclad: bool,
    /// Its top speed; nothing for one that cannot move itself.
    pub speed: f32,
    /// How far it perceives.
    pub senses: f32,
    /// How fast it exchanges gases with the air about it.
    pub breathe: fn(&Climate) -> f32,
    /// How fast it digests, as a share of its mass each second.
    pub digestion: f32,
    /// Whether gas bladders hold it up, as a drifter.
    pub floats: bool,
    /// Whether it lives fastened to another, and to which kinds.
    pub attaches: bool,
    pub partner: fn(Kind, Kind) -> bool,
    /// How much aur it can store, which only the Nætu lineage can.
    pub aur: f32,
    /// How its young begin and are kept.
    pub brood: Brood,
    /// Whether it must mate to breed.
    pub mates: bool,
    /// Whether its young hatch from eggs and then fly free of whatever they were laid on.
    pub hatches: bool,
    /// The form its young take at each stage of growing up, from newborn (0) to grown (1).
    pub form: fn(Kind, f32) -> u32,
    /// The light it lives in, and goes back to when it strays.
    pub light: Light,
    /// Whether it keeps with its group, matching its neighbours' headings.
    pub flocks: bool,
    /// How far its call carries, in metres; nothing in a voiceless animal.
    pub voice: f32,
    /// Which part of a plant it eats: its fronds, unless it strips bark or digs out roots.
    pub crops: Function,
    /// Whether it lets go of its host once fed, rather than staying fastened.
    pub lets_go: bool,
}

impl Kind {
    /// An animal's top speed; nothing for a plant.
    pub fn speed(self) -> f32 {
        self.animal().map_or(0.0, |animal| animal.speed)
    }

    /// The light an animal lives in; a plant lives in any.
    pub fn light(self) -> Light {
        self.animal().map_or(Light::Anywhere, |animal| animal.light)
    }

    /// Whether an animal's diet includes another kind, whatever state it is in.
    pub fn eats(self, food: Self) -> bool {
        self != food
            && self
                .animal()
                .is_some_and(|animal| (animal.eats)(self, food))
    }

    /// Whether an animal takes its food in this state.
    pub fn takes(self, take: Take) -> bool {
        self.animal().is_some_and(|animal| animal.takes == take)
    }

    /// Whether an animal visits flowers for their nectar, carrying their pollen.
    pub fn pollinates(self) -> bool {
        self.animal().is_some_and(|animal| animal.pollinates)
    }

    /// How far an organism notices others: an animal as far as its senses reach, and a plant only
    /// what passes close by.
    pub fn senses(self) -> f32 {
        self.animal().map_or(180.0, |animal| animal.senses)
    }

    /// How far an animal's call carries; nothing for the voiceless.
    pub fn voice(self) -> f32 {
        self.animal().map_or(0.0, |animal| animal.voice)
    }
}

impl Physiology {
    /// Whether it flies under its own power, kept up by wings and the gas stored in its body.
    pub fn flies(&self) -> bool {
        self.speed > 0.0 && !self.floats && !self.attaches
    }
}

/// What state an animal's food must be in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// Alive: a plant cropped, algae strained, a host drunk from.
    Living,
    /// Killed by it, or held helpless.
    Prey,
    /// Dead: carcasses, and anything else no longer alive.
    Carrion,
}

/// How a lineage keeps its young through their first, most helpless days, so that as few as
/// possible are lost to the long fall into the aur.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brood {
    /// Born live and able to fly, if weakly, from their first hour.
    Live,
    /// Born tiny and blind into a pouch on the parent's back, opening forwards and upwards so
    /// nothing can fall from it; as they grow they make short trips out to feed, longer each time,
    /// until they no longer come back.
    Pouch,
    /// Eggs bound by a tether to living vegetation, which holds them up.
    Tethered,
    /// Eggs released one by one into the updraft, each with a crown of fine threads that the
    /// rising air holds up like a seed's down.
    Drifting,
    /// Eggs glued in a band round the parent's own body, carried and defended until they hatch.
    Carried,
    /// Eggs paid out on long sticky strings, dozens to a string, that trail behind the parent
    /// until they are cut loose to drift.
    Strings,
}

impl Brood {
    /// Whether the young begin as eggs.
    pub const fn eggs(self) -> bool {
        !matches!(self, Self::Live | Self::Pouch)
    }
}

/// The light an animal lives in.
///
/// Light is a matter of place, not time: the sunburst at each pole lights the polar skies, a band
/// of sunset lies round each, and the equatorial skies are lit only by the aur glowing from below,
/// like strong moonlight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Light {
    /// The polar skies under the sunburst.
    Sunlit,
    /// The band of sunset between the poles and the equator.
    Twilight,
    /// The equatorial dimness, lit from below by the aur.
    Aurlit,
    /// Anywhere at all.
    Anywhere,
}

/// The body every animal starts from.
///
/// It eats nothing yet, cannot move itself, breathes through its skin, absorbs what it eats
/// directly, lays eggs on living plants after mating, and grows up in its parents' shape.
pub const ANIMAL: Physiology = Physiology {
    eats: |_, _| false,
    takes: Take::Living,
    gleans: false,
    pollinates: false,
    parasite: false,
    grooms: false,
    ironclad: false,
    speed: 0.0,
    senses: 180.0,
    breathe: |_| 0.3,
    digestion: 0.01,
    floats: false,
    attaches: false,
    partner: |kind, other| kind == other,
    aur: 0.0,
    brood: Brood::Tethered,
    mates: true,
    hatches: false,
    form: |_, _| 0,
    light: Light::Anywhere,
    flocks: false,
    voice: 0.0,
    crops: Function::Gathering,
    lets_go: false,
};

/// Breathing through channels that flight itself drives air along, faster in a stronger wind.
pub fn flow_channels(climate: &Climate) -> f32 {
    0.7 + climate.wind.length() * 0.05
}

/// Whether a kind is a plant.
pub fn plant(kind: Kind) -> bool {
    kind.kingdom() == Kingdom::Plantae
}

/// A grazer's diet: any living plant.
pub fn grazes(_: Kind, food: Kind) -> bool {
    plant(food)
}

/// A hunter's diet: any animal that is no parasite and not much larger than itself.
pub fn hunts(eater: Kind, food: Kind) -> bool {
    !plant(food)
        && !food.animal().is_some_and(|animal| animal.parasite)
        && food.biology().adult_mass < eater.biology().adult_mass * 3.0
}

/// An omnivore's diet: plants, and animals far smaller than itself.
pub fn omnivore(eater: Kind, food: Kind) -> bool {
    plant(food) || food.biology().adult_mass < eater.biology().adult_mass * 0.2
}

/// Whether an animal can eat another as it is now: its diet must allow it, and the food must be in
/// the state the animal takes it in.
pub fn can_eat(eater: Kind, food: Kind, alive: bool, helpless: bool) -> bool {
    let Some(animal) = eater.animal() else {
        return false;
    };
    if eater == food {
        return false;
    }
    match animal.takes {
        Take::Living => alive && (animal.eats)(eater, food),
        Take::Prey => (animal.eats)(eater, food) && (!alive || helpless),
        Take::Carrion => !alive || (animal.eats)(eater, food),
    }
}

/// A step of an animal's life.
///
/// It digests, breathes and tires as hard as it works, dries out, grows weary, bleeds and clots,
/// holds its lift, heals, and fails if its pump, nerve centre or fluid fail it. Its organs set how
/// well each of these goes.
pub fn metabolize(body: &mut Vitals, kind: Kind, climate: Climate, effort: f32, dt: f32) -> Waste {
    let Some(nature) = kind.animal() else {
        return Waste::default();
    };
    let viscera = kind.organs();
    let mut waste = digest(
        body,
        (nature.digestion, kind.biology().adult_mass),
        &viscera,
        dt,
    );
    body.stunned = (body.stunned - dt).max(0.0);
    let breather = body.working(&viscera, Function::Breathing);
    let pump = body.working(&viscera, Function::Circulation);
    let circulation = pump * body.fluid;
    let exchange = (nature.breathe)(&climate) * breather.max(0.05);
    // A body with a pump drives its fluid in pulses; one without lets it seep.
    let delivery = if viscera.has(Function::Circulation) {
        (1.0 - body.fatigue * 0.25) * circulation
    } else {
        0.4
    };
    let demand = 0.05 + effort * 0.18 / delivery.max(0.1);
    body.breathe(climate, demand, exchange, dt);
    body.energy = (body.energy - (0.0003 + effort * 0.0015) * dt).max(0.0);
    // Breath carries water away; the folds behind the louvres wring it back out, as far as they
    // are whole, and a cistern holds a reserve.
    let drying = (0.0003 + effort * 0.0008)
        * (2.0 - breather)
        * (1.0
            - viscera
                .function(&body.organs, Function::Water)
                .unwrap_or(0.0)
                * 0.35);
    body.hydration = (body.hydration + (climate.humidity * 0.0008 - drying) * dt).clamp(0.0, 1.0);
    let muscle = body.locomotion(&viscera).max(0.2);
    body.fatigue =
        (body.fatigue + (effort * 0.008 / muscle - body.breath * 0.003) * dt).clamp(0.0, 1.0);
    body.alertness = (body.alertness - dt / WAKING).max(0.0);
    circulate(body, circulation, dt);
    body.hold_lift(&viscera, dt);
    body.heal(&viscera, 0.0004, dt);
    let ganglion = body.working(&viscera, Function::Coordination);
    let failing = pump < 0.15 || ganglion < 0.15 || body.fluid < 0.2;
    let capacity = nature.aur * body.working(&viscera, Function::AurStore);
    let endured = body.endure(kind, climate, (failing, capacity), dt);
    waste.mass += endured.mass;
    waste
}

/// How long an animal can stay awake before it must sleep, in seconds.
const WAKING: f32 = 3000.0;

/// Open wounds bleed, clotting as the fluid moves; the fluid is made again from water and food.
fn circulate(body: &mut Vitals, circulation: f32, dt: f32) {
    body.fluid = (body.fluid - body.bleeding * dt).max(0.0);
    body.bleeding *= (-dt * (0.02 + circulation * 0.06)).exp();
    body.fluid = (body.fluid + 0.0008 * body.energy * body.hydration * dt).min(1.0);
}

/// Digest what the crop or folds hold, at `rate`, keeping some of it to grow on towards a little
/// past the `adult` mass; a torn crop spills what it holds.
fn digest(body: &mut Vitals, (rate, adult): (f32, f32), viscera: &Viscera, dt: f32) -> Waste {
    let crop = body.working(viscera, Function::Digestion);
    let spilled = body.gut * (1.0 - crop) * (1.0 - (-0.01 * dt).exp());
    let spilled_aur = body.gut_aur * spilled / body.gut.max(1e-10);
    body.gut -= spilled;
    body.gut_aur -= spilled_aur;
    let amount = body
        .gut
        .min(body.mass * rate * crop * body.hydration * body.breath.max(0.0) * dt);
    let aur = body.gut_aur * amount / body.gut.max(1e-10);
    body.gut -= amount;
    body.gut_aur -= aur;
    let growth = (adult * 1.3 - body.mass).max(0.0).min(amount * 0.5);
    body.mass += growth;
    body.energy = (body.energy + amount / body.mass.max(1e-10) * 8.0).min(1.0);
    Waste {
        mass: amount - growth + spilled,
        aur: aur + spilled_aur,
    }
}

impl Vitals {
    /// How well the muscles that carry it about still work: its flight muscles, if its lineage
    /// took to the air, or else those of its body.
    pub fn locomotion(&self, viscera: &Viscera) -> f32 {
        let function = if viscera.has(Function::Flight) {
            Function::Flight
        } else {
            Function::Movement
        };
        self.working(viscera, function)
    }

    /// How hard the body can work, from its flight muscles, its breath and the fluid carrying fuel
    /// to them.
    pub fn power(&self, kind: Kind) -> f32 {
        let viscera = kind.organs();
        (self.locomotion(&viscera) * 0.7 + 0.3)
            * self.breath.clamp(0.2, 1.0)
            * self.fluid.clamp(0.3, 1.0)
            * if self.stunned > 0.0 { 0.1 } else { 1.0 }
    }

    /// How steadily the nerve centre steers, falling as it is damaged or goes too long without
    /// sleep.
    pub fn composure(&self, kind: Kind) -> f32 {
        self.working(&kind.organs(), Function::Coordination) * (0.5 + self.alertness * 0.5)
    }

    /// A night's rest, compressed: the nerve centre recovers, and so do tired muscles.
    pub fn sleep(&mut self, dt: f32) {
        self.alertness = (self.alertness + dt / 420.0).min(1.0);
        self.fatigue = (self.fatigue - dt * 0.004).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{
        KINDS,
        organs::{Strike, Wound},
    };
    use spirv_std::glam::vec3;

    #[test]
    fn digestion_conserves_biomass_and_excretes_bound_aur() {
        for kind in KINDS
            .iter()
            .filter(|kind| kind.kingdom() == Kingdom::Animalia)
        {
            let mut vitals = Vitals::new(kind.biology());
            vitals.organs[0] = 0.5;
            vitals.ingest(0.5, 0.25);
            let total = vitals.mass + vitals.gut;
            let rate = kind.animal().expect("an animal").digestion;
            let adult = kind.biology().adult_mass;
            let waste = digest(&mut vitals, (rate, adult), &kind.organs(), 10.0);
            assert!(
                (vitals.mass + vitals.gut + waste.mass - total).abs() < 1e-4,
                "biomass created by {kind:?}"
            );
            assert!(
                (vitals.gut_aur + waste.aur - 0.25).abs() < 1e-5,
                "food aur disappeared"
            );
            assert!(
                vitals.aur.abs() < 1e-5,
                "animal extracted magic from its food"
            );
        }
    }

    #[test]
    fn mist_tolerance_is_inherited_only_by_naeturus() {
        for kind in KINDS
            .iter()
            .filter(|kind| kind.kingdom() == Kingdom::Animalia)
        {
            let mut vitals = Vitals::new(kind.biology());
            vitals.metabolize(
                *kind,
                Climate::sample(vec3(0.0, 0.0, 25.0), 0.0, 0.0),
                0.0,
                1.0,
            );
            assert_eq!(
                vitals.aur > 0.0,
                matches!(kind, Kind::Nae | Kind::Naetu),
                "incorrect aur capture for {kind:?}"
            );
        }
    }

    #[test]
    fn diets_hold_together() {
        let eats = |eater: Kind, food: Kind| can_eat(eater, food, true, true);
        assert!(
            eats(Kind::Ramiphaga, Kind::Spinivinea),
            "bark grazer lost its food"
        );
        assert!(
            !eats(Kind::Ramiphaga, Kind::Polliniflora),
            "bark specialist grazes flowers"
        );
        assert!(
            eats(Kind::Aerohirudo, Kind::Naetu),
            "parasite cannot find hosts"
        );
        assert!(
            !eats(Kind::Aerogigas, Kind::Nae),
            "passive sponge became a predator"
        );
        assert!(
            eats(Kind::Nebulaphora, Kind::Polliniflora),
            "nectar feeder lost its blooms"
        );
        assert!(
            eats(Kind::Purgator, Kind::Aerohirudo),
            "cleaner cannot find parasites"
        );
        assert!(
            !eats(Kind::Purgator, Kind::Lucinani),
            "cleaner took a Glowgnome for a parasite"
        );
        assert!(
            !can_eat(Kind::Aculeorhynchus, Kind::Manduca, true, false),
            "a hunter ate living prey it had not caught"
        );
        assert!(can_eat(Kind::Aculeorhynchus, Kind::Manduca, true, true));
        for kind in KINDS {
            assert!(
                !can_eat(*kind, *kind, false, true),
                "{kind:?} eats its own kind"
            );
        }
    }

    #[test]
    fn a_lance_through_the_pump_kills_and_a_slash_across_the_back_sinks() {
        let kind = Kind::Herbivolus;
        let viscera = kind.organs();
        let pump = viscera
            .find(Function::Circulation)
            .expect("grazers have a pump");
        let climate = Climate::sample(vec3(0.0, 0.0, 1000.0), 0.0, 0.0);
        let mut speared = Vitals::new(kind.biology());
        let injury = speared.wound(
            kind,
            Strike {
                from: pump + vec3(0.0, 10.0, 0.0),
                to: pump - vec3(0.0, 1.0, 0.0),
                radius: 0.3,
                force: 1.0,
                wound: Wound::Pierce,
            },
            10.0,
        );
        assert!(injury.vital, "a lance through the pump spared it");
        for _ in 0..60 {
            speared.metabolize(kind, climate, 0.2, 1.0);
        }
        assert!(speared.health <= 0.0, "a body without its pump lived on");
        let float = viscera.find(Function::Lift).expect("grazers store gas");
        let mut slashed = Vitals::new(kind.biology());
        slashed.wound(
            kind,
            Strike {
                from: float + vec3(-6.0, 0.0, 0.0),
                to: float + vec3(6.0, 0.0, 0.0),
                radius: 0.6,
                force: 1.0,
                wound: Wound::Slash,
            },
            10.0,
        );
        for _ in 0..60 {
            slashed.metabolize(kind, climate, 0.2, 1.0);
        }
        assert!(slashed.health > 0.0, "a slash across the gas store killed");
        assert!(slashed.lift < 0.85, "a torn gas store kept its lift");
    }
}
