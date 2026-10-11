pub use crate::scene::climate::Climate;
#[cfg(test)]
use spirv_std::glam::vec3;

use super::biology::{Anatomy, Biology, Digestion, Energy, Exchange, Transport};

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
    pub age: f32,
}

#[derive(Default)]
pub struct Waste {
    pub mass: f32,
    pub aur: f32,
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
            age: 0.0,
        }
    }

    pub fn ingest(&mut self, biomass: f32, bound_aur: f32) {
        self.gut += biomass.max(0.0);
        self.gut_aur += bound_aur.max(0.0);
    }

    pub fn metabolize(
        &mut self,
        biology: Biology,
        climate: Climate,
        effort: f32,
        dt: f32,
    ) -> Waste {
        let mut waste = self.digest(biology, dt);
        let exchange = match biology.organs.exchange {
            Exchange::Surface => 0.15,
            Exchange::Pores => 0.3,
            Exchange::FlowChannels => 0.7 + climate.wind.length() * 0.05,
        };
        let delivery = match biology.organs.transport {
            Transport::Diffusion => 0.4,
            Transport::Pulsatile => 1.0 - self.fatigue * 0.25,
            Transport::Vascular => 0.7 + self.hydration * 0.3,
        };
        let demand = 0.05 + effort * 0.18 / delivery.max(0.1);
        self.breath += (climate.pressure - demand - self.breath) * (exchange * dt).min(1.0);
        self.energy = (self.energy - (0.0003 + effort * 0.0015) * dt).max(0.0);
        self.hydration = (self.hydration
            + (climate.humidity * 0.0008 - 0.0003 - effort * 0.0008) * dt)
            .clamp(0.0, 1.0);
        self.fatigue = (self.fatigue + (effort * 0.008 - self.breath * 0.003) * dt).clamp(0.0, 1.0);
        if biology.anatomy == Anatomy::Plant {
            self.energy = (self.energy + climate.energy(biology.energy) * 0.003 * dt).min(1.0);
        }
        if climate.altitude < 0.0 {
            self.health = 0.0;
            self.age += dt;
            return waste;
        } else if climate.altitude < 50.0 {
            if biology.organs.aur_capacity > 0.0 {
                self.aur = (self.aur + climate.aur * 0.02 * dt).min(biology.organs.aur_capacity);
            } else {
                self.health = (self.health - climate.aur * 0.03 * dt).max(0.0);
            }
        }
        self.health = (self.health
            + (self.energy.min(self.hydration).min(self.breath) - 0.25) * 0.001 * dt)
            .clamp(0.0, 1.0);
        if self.energy < 0.1 {
            let catabolism = (self.mass * 0.001 * dt).min(self.mass);
            self.mass -= catabolism;
            waste.mass += catabolism;
        }
        self.age += dt;
        waste
    }

    fn digest(&mut self, biology: Biology, dt: f32) -> Waste {
        let rate = match biology.organs.digestion {
            Digestion::None => 0.0,
            Digestion::Absorptive => 0.01,
            Digestion::Crop => 0.004,
        };
        let amount = self
            .gut
            .min(self.mass * rate * self.hydration * self.breath.max(0.0) * dt);
        let aur = self.gut_aur * amount / self.gut.max(1e-10);
        self.gut -= amount;
        self.gut_aur -= aur;
        let growth = (biology.adult_mass * 1.3 - self.mass)
            .max(0.0)
            .min(amount * 0.5);
        self.mass += growth;
        self.energy = (self.energy + amount / self.mass.max(1e-10) * 8.0).min(1.0);
        Waste {
            mass: amount - growth,
            aur,
        }
    }

    pub fn grow(&mut self, biology: Biology, climate: Climate, nutrients: f32, dt: f32) -> f32 {
        if biology.anatomy != Anatomy::Plant || self.health <= 0.0 {
            return 0.0;
        }
        let growth = (self.mass
            * climate.energy(biology.energy)
            * self.hydration
            * self.breath.max(0.0)
            * 0.001
            * dt)
            .min(nutrients.max(0.0))
            .min((biology.adult_mass * 1.4 - self.mass).max(0.0));
        self.mass += growth;
        growth
    }
}

impl Climate {
    pub fn energy(self, source: Energy) -> f32 {
        match source {
            Energy::Food => 0.0,
            Energy::Aur => self.aur,
            Energy::Light => self.light,
            Energy::Wind => self.wind.length_squared() * 0.03,
            Energy::Lightning => self.storm * 8.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{KINDS, Kind};

    #[test]
    fn digestion_conserves_biomass_and_excretes_bound_aur() {
        for kind in KINDS {
            let biology = kind.biology();
            if biology.anatomy == Anatomy::Plant {
                continue;
            }
            let mut vitals = Vitals::new(biology);
            vitals.ingest(0.5, 0.25);
            let total = vitals.mass + vitals.gut;
            let waste = vitals.digest(biology, 10.0);
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
    fn mist_tolerance_is_inherited_only_by_naeturus_and_pure_aur_is_lethal() {
        for kind in KINDS {
            let biology = kind.biology();
            if biology.anatomy == Anatomy::Plant {
                continue;
            }
            let mut vitals = Vitals::new(biology);
            vitals.metabolize(
                biology,
                Climate::sample(vec3(0.0, 0.0, 25.0), 0.0, 0.0),
                0.0,
                1.0,
            );
            assert_eq!(
                vitals.aur > 0.0,
                matches!(kind, Kind::Nae | Kind::Naetu),
                "incorrect aur capture for {kind:?}"
            );
            vitals.metabolize(
                biology,
                Climate::sample(vec3(0.0, 0.0, -1.0), 0.0, 0.0),
                0.0,
                1.0,
            );
            assert!(vitals.health < 1e-5, "pure aur must remain lethal");
        }
    }
}
