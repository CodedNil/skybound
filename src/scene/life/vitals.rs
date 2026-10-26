//! The state of a living body, and the rules every body lives by whatever its kingdom.
//!
//! It ages, its organs are wounded and heal, its gas stores leak and refill, the aur kills it, and
//! it dies at the end of its span. How it eats, breathes, grows and moves is its kingdom's, and
//! lives with each kingdom in the tree.

use crate::scene::{
    climate::Climate,
    life::{
        Kind, Kingdom, animalia,
        biology::Biology,
        development::DAY,
        organs::{Function, MAX, Strike, Viscera, Wound},
        plantae,
    },
};

/// The state of a living body: what it has eaten and stored, how much fluid and lift it holds, how
/// rested it is, and how whole each of its organs remains.
#[derive(Clone, Copy, Debug)]
pub struct Vitals {
    pub mass: f32,
    pub gut: f32,
    pub gut_aur: f32,
    pub energy: f32,
    pub hydration: f32,
    pub breath: f32,
    pub health: f32,
    pub fatigue: f32,
    pub aur: f32,
    pub age_days: f64,
    /// How whole each organ is, by its site in the species' viscera.
    pub organs: [f32; MAX],
    /// The circulating fluid or sap, as a share of a whole body's, and how fast open wounds are
    /// losing it each second.
    pub fluid: f32,
    pub bleeding: f32,
    /// The lift held in the gas stores, as a share of full.
    pub lift: f32,
    /// How rested the nerve centre is: it drains while awake and refills in sleep.
    pub alertness: f32,
    /// A gland's store of secretion, ready to spend, or a plant's nectar.
    pub reserve: f32,
    /// Seconds of paralysis left from venom.
    pub stunned: f32,
}

#[derive(Default)]
pub struct Waste {
    pub mass: f32,
    pub aur: f32,
}

/// What a wound did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Injury {
    /// How much of the body's organs it destroyed, summed.
    pub damage: f32,
    /// Whether it destroyed the pump or the nerve centre.
    pub vital: bool,
}

/// What both kingdoms' bodies can do, each in its own way, asked of either.
impl Kind {
    /// Whether it lives fastened to another.
    pub const fn attaches(self) -> bool {
        match (self.animal(), self.plant()) {
            (Some(animal), _) => animal.attaches,
            (_, Some(plant)) => plant.attaches,
            _ => false,
        }
    }

    /// Whether bladders, membranes or a chimney hold it up without effort.
    pub const fn floats(self) -> bool {
        match (self.animal(), self.plant()) {
            (Some(animal), _) => animal.floats,
            (_, Some(plant)) => plant.floats,
            _ => false,
        }
    }

    /// Whether it keeps company with, or lives on, another kind.
    pub fn partners(self, other: Self) -> bool {
        match (self.animal(), self.plant()) {
            (Some(animal), _) => (animal.partner)(self, other),
            (_, Some(plant)) => (plant.partner)(self, other),
            _ => false,
        }
    }

    /// The form its young take `young` of the way to grown.
    pub fn young_form(self, young: f32) -> u32 {
        match (self.animal(), self.plant()) {
            (Some(animal), _) => (animal.form)(self, young),
            (_, Some(plant)) => (plant.form)(self, young),
            _ => 0,
        }
    }
}

impl Vitals {
    pub const fn new(biology: Biology) -> Self {
        Self {
            mass: biology.adult_mass,
            gut: 0.0,
            gut_aur: 0.0,
            energy: 0.7,
            hydration: 0.8,
            breath: 1.0,
            health: 1.0,
            fatigue: 0.0,
            aur: 0.0,
            age_days: biology.maturity_days as f64,
            organs: [1.0; MAX],
            fluid: 1.0,
            bleeding: 0.0,
            lift: 1.0,
            alertness: 1.0,
            reserve: 1.0,
            stunned: 0.0,
        }
    }

    /// A grown body of a kind as the world starts it: rested and fed, any aur sacs full.
    pub fn born(kind: Kind) -> Self {
        Self {
            aur: kind.animal().map_or(0.0, |animal| animal.aur),
            ..Self::new(kind.biology())
        }
    }

    pub fn ingest(&mut self, biomass: f32, bound_aur: f32) {
        self.gut += biomass.max(0.0);
        self.gut_aur += bound_aur.max(0.0);
    }

    /// How well this body still meets a need: fully if the species has no organ for it, for then
    /// some other part of it meets that need.
    pub fn working(&self, viscera: &Viscera, function: Function) -> f32 {
        viscera.function(&self.organs, function).unwrap_or(1.0)
    }

    /// Live for a step, by the ways of the body's kingdom.
    pub fn metabolize(&mut self, kind: Kind, climate: Climate, effort: f32, dt: f32) -> Waste {
        match kind.kingdom() {
            Kingdom::Animalia => animalia::physiology::metabolize(self, kind, climate, effort, dt),
            Kingdom::Plantae => plantae::physiology::metabolize(self, kind, climate, dt),
        }
    }

    /// What every body must endure each step: it ages and, towards the end of its span, fails; the
    /// aur kills it below the mists unless it can store what it absorbs, `capacity` of it; failing
    /// vital organs kill it within moments; and a starving body consumes itself.
    pub fn endure(
        &mut self,
        kind: Kind,
        climate: Climate,
        (failing, capacity): (bool, f32),
        dt: f32,
    ) -> Waste {
        let biology = kind.biology();
        let mut waste = Waste::default();
        self.age_days += f64::from(dt.max(0.0)) / f64::from(DAY);
        if failing {
            self.health = (self.health - 0.06 * dt).max(0.0);
        }
        if climate.altitude < 0.0 {
            self.health = 0.0;
            return waste;
        } else if climate.altitude < 50.0 {
            if capacity > 0.0 {
                self.aur = (self.aur + climate.aur * 0.02 * dt).min(capacity);
            } else {
                self.health = (self.health - climate.aur * 0.03 * dt).max(0.0);
            }
        }
        if !failing {
            self.health = (self.health
                + (self.energy.min(self.hydration).min(self.breath) - 0.25) * 0.001 * dt)
                .clamp(0.0, 1.0);
        }
        if self.energy < 0.1 {
            let catabolism = (self.mass * 0.001 * dt).min(self.mass);
            self.mass -= catabolism;
            waste.mass += catabolism;
        }
        let senescence = (self.age_days as f32 / biology.lifespan_days - 0.75).max(0.0);
        self.health = (self.health - senescence * dt / DAY).max(0.0);
        if self.age_days >= f64::from(biology.lifespan_days) {
            self.health = 0.0;
        }
        waste
    }

    /// Breathing towards what the air offers, against what the body demands of it, as fast as its
    /// gas exchange allows.
    pub fn breathe(&mut self, climate: Climate, demand: f32, exchange: f32, dt: f32) {
        self.breath += (climate.pressure - demand - self.breath) * (exchange * dt).min(1.0);
    }

    /// Gas leaks from torn stores, down to what they can still hold, and is made again slowly in
    /// whole ones.
    pub fn hold_lift(&mut self, viscera: &Viscera, dt: f32) {
        let capacity = self.working(viscera, Function::Lift);
        if self.lift > capacity {
            self.lift -= (self.lift - capacity) * (1.0 - (-0.05 * dt).exp());
        } else {
            self.lift = (self.lift + 0.002 * self.energy * dt).min(capacity);
        }
    }

    /// Organs regrow at `rate` in a well-fed body, fruit only in one with energy to spare, and a
    /// gland's reserve refills.
    pub fn heal(&mut self, viscera: &Viscera, rate: f32, dt: f32) {
        let rate = rate * self.energy * self.hydration;
        let mut cost = 0.0;
        for (index, site) in viscera.sites().iter().enumerate() {
            if site.organ.function == Function::Fruit && self.energy < 0.6 {
                continue;
            }
            let regrowth = (1.0 - self.organs[index]) * rate * dt;
            self.organs[index] += regrowth;
            cost += regrowth;
        }
        self.energy = (self.energy - cost * 0.05).max(0.0);
        if viscera.has(Function::Secretion) {
            let gland = self.working(viscera, Function::Secretion);
            self.reserve = (self.reserve + 0.003 * self.energy * dt).min(gland);
        }
    }

    /// Take a wound. Organs on its path are destroyed in proportion to how squarely it reaches
    /// them, and the wound bleeds for its size against the body's, `scale` long.
    pub fn wound(&mut self, kind: Kind, strike: Strike, scale: f32) -> Injury {
        let viscera = kind.organs();
        let damage = viscera.struck(strike);
        let mut injury = Injury::default();
        for (index, site) in viscera.sites().iter().enumerate() {
            let lost = damage[index].min(self.organs[index]);
            self.organs[index] -= lost;
            injury.damage += lost;
            if matches!(
                site.organ.function,
                Function::Circulation | Function::Coordination
            ) && self.organs[index] < 0.15
            {
                injury.vital = true;
            }
        }
        if strike.wound == Wound::Drain {
            self.fluid = (self.fluid - strike.force * 0.01).max(0.0);
        }
        let size = (strike.radius + (strike.to - strike.from).length() * 0.3) / scale.max(1e-3);
        self.bleeding += strike.force * strike.wound.bleeding() * size * 0.01;
        self.health = (self.health - injury.damage * 0.08).max(0.0);
        injury
    }

    /// How far below its lift this body is pulled, from gas lost from its stores: nothing for a
    /// body with no stores, or one that never relied on them to stay up.
    pub fn sinking(&self, kind: Kind) -> f32 {
        let buoyed = match kind.kingdom() {
            Kingdom::Animalia => kind.animal().is_some_and(|animal| animal.flies()),
            Kingdom::Plantae => kind.floats(),
        };
        if buoyed && kind.organs().has(Function::Lift) {
            (1.0 - self.lift) * 3.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{KINDS, Kind};
    use spirv_std::glam::vec3;

    #[test]
    fn ancient_organisms_age_by_one_fixed_tick_and_die_at_their_lifespan() {
        let biology = Kind::Antiquus.biology();
        let mut body = Vitals::new(biology);
        let tick_days = 1.0 / 64.0 / f64::from(DAY);
        body.age_days = f64::from(biology.lifespan_days) - tick_days;
        let before = body.age_days;
        body.metabolize(
            Kind::Antiquus,
            Climate::sample(vec3(0.0, 0.0, 1000.0), 0.0, 0.0),
            0.0,
            1.0 / 64.0,
        );
        assert!((body.age_days - before - tick_days).abs() < 1e-9);
        assert!(body.health < f32::EPSILON);
    }

    #[test]
    fn pure_aur_is_lethal_to_every_kingdom() {
        for kind in KINDS {
            let mut vitals = Vitals::new(kind.biology());
            vitals.metabolize(
                *kind,
                Climate::sample(vec3(0.0, 0.0, -1.0), 0.0, 0.0),
                0.0,
                1.0,
            );
            assert!(vitals.health < 1e-5, "{kind:?} survived pure aur");
        }
    }
}
