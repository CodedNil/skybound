use super::Kind;

pub const ANCESTRAL: Biology = Biology {
    maturity_days: 90.0,
    lifespan_days: 3000.0,
    organs: Organs::POROUS,
    anatomy: Anatomy::Porous,
    wing_pairs: 0,
    oral_pairs: 0,
    feeding: Feeding::Absorption,
    energy: Energy::Food,
    reproduction: Reproduction::Eggs,
    social: Social::Solitary,
    speed: 0.0,
    perception: 180.0,
    altitude: (50.0, 6000.0),
    adult_mass: 1.0,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anatomy {
    Porous,
    Suctorial,
    Tetralate,
    Plant,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feeding {
    Absorption,
    Grazer,
    Nectar,
    Predator,
    Scavenger,
    Filter,
    Parasite,
    Cleaner,
    Producer,
    Omnivore,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Energy {
    Food,
    Aur,
    Light,
    Wind,
    Lightning,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reproduction {
    Eggs,
    LiveBirth,
    Division,
    Fragmentation,
    Spores,
    Pollination,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Social {
    Solitary,
    Swarm,
    Herd,
    Mutualist,
    Guardian,
    Weaver,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exchange {
    Surface,
    Pores,
    FlowChannels,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    Diffusion,
    Pulsatile,
    Vascular,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Digestion {
    Absorptive,
    Crop,
    None,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lift {
    Aerosol,
    Bladders,
    Wings,
    Membrane,
    Chimney,
    Attached,
}

#[derive(Clone, Copy, Debug)]
pub struct Organs {
    pub exchange: Exchange,
    pub transport: Transport,
    pub digestion: Digestion,
    pub lift: Lift,
    pub aur_capacity: f32,
}

impl Organs {
    pub const POROUS: Self = Self {
        exchange: Exchange::Pores,
        transport: Transport::Diffusion,
        digestion: Digestion::Absorptive,
        lift: Lift::Bladders,
        aur_capacity: 0.0,
    };
    pub const ANIMAL: Self = Self {
        exchange: Exchange::FlowChannels,
        transport: Transport::Pulsatile,
        digestion: Digestion::Crop,
        lift: Lift::Wings,
        ..Self::POROUS
    };
    pub const PLANT: Self = Self {
        exchange: Exchange::Surface,
        transport: Transport::Diffusion,
        digestion: Digestion::None,
        lift: Lift::Bladders,
        ..Self::POROUS
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Biology {
    pub maturity_days: f32,
    pub lifespan_days: f32,
    pub organs: Organs,
    pub anatomy: Anatomy,
    pub wing_pairs: u32,
    pub oral_pairs: u32,
    pub feeding: Feeding,
    pub energy: Energy,
    pub reproduction: Reproduction,
    pub social: Social,
    pub speed: f32,
    pub perception: f32,
    pub altitude: (f32, f32),
    /// Relative dry biomass at maturity, shared by resource transfers and reproduction.
    pub adult_mass: f32,
}

pub fn edible(consumer: Kind, resource: Kind) -> bool {
    if consumer == resource {
        return false;
    }
    match consumer {
        Kind::Lucinani => return resource == Kind::Susurrus,
        Kind::Ramiphaga => return resource == Kind::Spinivinea,
        Kind::Antiquus => return matches!(resource, Kind::Aerothallus | Kind::Laevivelum),
        Kind::Lithosoma => return resource == Kind::Stelloradix,
        _ => {}
    }
    let eater = consumer.biology();
    let food = resource.biology();
    match eater.feeding {
        Feeding::Absorption | Feeding::Filter => {
            resource == Kind::Auracoccus
                || (consumer == Kind::Lucerna && resource == Kind::Scintilla)
        }
        Feeding::Grazer => food.anatomy == Anatomy::Plant,
        Feeding::Omnivore => {
            food.anatomy == Anatomy::Plant
                || (food.anatomy == Anatomy::Tetralate && food.adult_mass < eater.adult_mass * 0.2)
        }
        Feeding::Nectar => food.reproduction == Reproduction::Pollination,
        Feeding::Predator => {
            food.anatomy != Anatomy::Plant
                && food.anatomy != Anatomy::Suctorial
                && food.adult_mass < eater.adult_mass * 3.0
        }
        Feeding::Parasite => {
            food.anatomy != Anatomy::Plant && food.adult_mass > eater.adult_mass * 4.0
        }
        Feeding::Cleaner => food.feeding == Feeding::Parasite,
        Feeding::Scavenger | Feeding::Producer => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::KINDS;

    #[test]
    fn descendant_body_plans_and_food_web_remain_consistent() {
        for kind in KINDS {
            let biology = kind.biology();
            assert!(
                biology.maturity_days > 0.0 && biology.lifespan_days > biology.maturity_days,
                "invalid life history for {kind:?}"
            );
            assert!(
                biology.altitude.0 < biology.altitude.1,
                "invalid habitat for {kind:?}"
            );
            assert!(biology.adult_mass > 0.0, "invalid biomass for {kind:?}");
            if biology.anatomy == Anatomy::Tetralate {
                assert_eq!(
                    biology.wing_pairs, 2,
                    "lost ancestral wing pair in {kind:?}"
                );
                assert!(
                    biology.oral_pairs > 0,
                    "missing ancestral mouth palps in {kind:?}"
                );
            }
            if biology.anatomy == Anatomy::Plant {
                assert_eq!(biology.wing_pairs, 0, "plant has animal wings");
                assert!(
                    biology.energy != Energy::Food,
                    "producer needs an energy source"
                );
            }
        }
        assert!(
            edible(Kind::Ramiphaga, Kind::Spinivinea),
            "bark grazer lost its food source"
        );
        assert!(
            !edible(Kind::Ramiphaga, Kind::Polliniflora),
            "bark specialist grazes flowers"
        );
        assert!(
            edible(Kind::Aerohirudo, Kind::Naetu),
            "parasite cannot find hosts"
        );
        assert!(
            !edible(Kind::Aerogigas, Kind::Nae),
            "passive sponge became a predator"
        );
    }
}
