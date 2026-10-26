//! How a plant's body works.
//!
//! What powers it, how it holds itself up, how it propagates, and what it carries sap through.
//! Plants have no mind; they grow, flower, fruit and drift. Every plant
//! taxon has a [`Physiology`], taking its parent's with `..super::PHYSIOLOGY` and changing what its
//! lineage changed.

use crate::scene::{
    climate::Climate,
    life::{
        Kind,
        organs::{Function, Organ},
        plantae::young::{BUD, SPORE, SPROUT},
        vitals::{Vitals, Waste},
    },
};
use spirv_std::glam::vec3;

/// A frond, sail or leaf, which gathers what the plant lives on, and which grazers eat.
pub const FROND: Organ = Organ::new("frond", Function::Gathering, vec3(0.25, 0.62, 0.30));

/// The stem or trunk, carrying sap and holding the plant together; its bark is what bark-eaters
/// strip.
pub const STEM: Organ = Organ::new("stem", Function::Sap, vec3(0.48, 0.36, 0.22));

/// The hanging roots, which draw water and aur from the air below.
pub const ROOT: Organ = Organ::new("hanging root", Function::Drawing, vec3(0.62, 0.55, 0.40));

/// A bladder of lifting gas.
pub const BLADDER: Organ = Organ::new("gas bladder", Function::Lift, vec3(0.62, 0.78, 0.86));

/// How a lineage's body works.
#[derive(Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each is a separate habit a lineage has or lacks"
)]
pub struct Physiology {
    /// How much energy the climate gives it: light, aur, wind or lightning, by what it lives on.
    pub powered: fn(&Climate) -> f32,
    /// How it makes the next generation.
    pub propagation: Propagation,
    /// Whether bladders, membranes or a chimney hold it up in the air.
    pub floats: bool,
    /// Whether it lives fastened to another, and to which kinds.
    pub attaches: bool,
    pub partner: fn(Kind, Kind) -> bool,
    /// Whether it draws sap through vessels, as woody plants do, rather than letting it seep.
    pub vascular: bool,
    /// The form its young take at each stage of growing, from seedling (0) to grown (1).
    pub form: fn(Kind, f32) -> u32,
    /// Whether it feeds the host it grows on a share of what it makes.
    pub feeds_host: bool,
    /// How many grow joined together in one stand, as kelp grows in forests of tangled strands; a
    /// plant whose spores and seeds scatter on the wind grows alone, wherever one came to rest.
    pub stand: usize,
}

/// How a plant makes the next generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Propagation {
    /// Releases spores that drift on the updraft.
    Spores,
    /// Splits in two.
    Division,
    /// Breaks into pieces that each grow into a whole.
    Fragmentation,
    /// Sets seed once a visitor has brought it pollen from another of its kind.
    Pollination,
}

/// The body every plant starts from: one powered by the dispersed aur, held up by gas bladders,
/// spreading by spores and letting its sap seep.
pub const PLANT: Physiology = Physiology {
    powered: |climate| climate.aur,
    propagation: Propagation::Spores,
    floats: true,
    attaches: false,
    partner: |_, _| false,
    vascular: false,
    form,
    feeds_host: false,
    stand: 1,
};

/// A plant's young: a spore or seed under its sail, then a sprouting seedling; a dividing plant's
/// young are two halves pulling apart.
fn form(kind: Kind, young: f32) -> u32 {
    if young >= 0.7 {
        return 0;
    }
    match kind.plant().map(|plant| plant.propagation) {
        Some(Propagation::Division) => BUD,
        Some(Propagation::Spores | Propagation::Pollination) if young < 0.2 => SPORE,
        _ => SPROUT,
    }
}

/// An epiphyte's partner: anything at least twenty times its size to grow on.
pub fn epiphyte(kind: Kind, host: Kind) -> bool {
    host.biology().adult_mass > kind.biology().adult_mass * 20.0
}

/// A step of a plant's life.
///
/// It breathes across its surface, makes food from what powers it in proportion to its fronds,
/// draws water through its roots, fills its blooms with nectar in the light, holds its lift, heals,
/// and dies if its stem is severed.
pub fn metabolize(body: &mut Vitals, kind: Kind, climate: Climate, dt: f32) -> Waste {
    let Some(nature) = kind.plant() else {
        return Waste::default();
    };
    let viscera = kind.organs();
    let delivery = if nature.vascular {
        0.7 + body.hydration * 0.3
    } else {
        0.4
    };
    body.breathe(climate, 0.05 / delivery.max(0.1), 0.15, dt);
    body.energy = (body.energy - 0.0003 * dt).max(0.0);
    let drying = 0.0003
        * (1.0
            - viscera
                .function(&body.organs, Function::Water)
                .unwrap_or(0.0)
                * 0.35);
    body.hydration = (body.hydration + (climate.humidity * 0.0008 - drying) * dt).clamp(0.0, 1.0);
    let frond = body.working(&viscera, Function::Gathering);
    body.energy = (body.energy + (nature.powered)(&climate) * frond * 0.003 * dt).min(1.0);
    if viscera.has(Function::Nectar) {
        body.reserve = (body.reserve
            + climate.light * body.working(&viscera, Function::Nectar) * 0.002 * dt)
            .min(1.0);
    }
    body.hold_lift(&viscera, dt);
    body.heal(&viscera, 0.002, dt);
    let failing = body.working(&viscera, Function::Sap) < 0.1;
    body.endure(kind, climate, (failing, 0.0), dt)
}

/// Growth in mass, from what powers the plant and the nutrients in the air about it, towards a
/// little past its adult size; hanging roots draw water from the humid air below. Returns the
/// nutrients taken.
pub fn grow(body: &mut Vitals, kind: Kind, climate: Climate, nutrients: f32, dt: f32) -> f32 {
    let Some(nature) = kind.plant() else {
        return 0.0;
    };
    if body.health <= 0.0 {
        return 0.0;
    }
    let viscera = kind.organs();
    let growth = (body.mass
        * (nature.powered)(&climate)
        * body.working(&viscera, Function::Gathering)
        * body.hydration
        * body.breath.max(0.0)
        * 0.001
        * dt)
        .min(nutrients.max(0.0))
        .min((kind.biology().adult_mass * 1.4 - body.mass).max(0.0));
    body.mass += growth;
    body.hydration = (body.hydration
        + climate.humidity * body.working(&viscera, Function::Drawing) * 0.0005 * dt)
        .min(1.0);
    growth
}

#[cfg(test)]
mod tests {
    use crate::scene::life::{KINDS, Kingdom};

    #[test]
    fn every_plant_has_a_source_of_energy_and_no_mind() {
        for kind in KINDS
            .iter()
            .filter(|kind| kind.kingdom() == Kingdom::Plantae)
        {
            assert!(kind.plant().is_some() && kind.animal().is_none());
            assert!(kind.ethology().is_none(), "{kind:?} is a plant with a mind");
        }
    }
}
