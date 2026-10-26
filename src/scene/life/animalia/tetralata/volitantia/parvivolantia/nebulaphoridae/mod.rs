pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod nebulaphora;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.1,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Nebulaphoridae",
    "Mistings",
    [
        "Members of this family are gentle nectar feeders with broad translucent membrane wings.",
        "The two wing pairs hold a slow approach beside a flower whose stem may be moving in another current.",
        "Flexible mouth appendages reach nectar while the body brushes pollen-bearing tissues.",
        "Coloured membranes and fine veins remain visible in scattered aur light, helping nearby animals recognise a visiting pollinator.",
        "Their journeys connect isolated flowers across gaps that pollen alone would rarely cross.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Mistings court spiralling up together like a pair of falling petals reversed, and come to
    //! look at anything new. Too small to cross the dry air between Pondplants, they drink what
    //! the cloud gives: a thirsty Misting in wet air hangs still and sips the beads of dew that
    //! gather on its wings.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges, spiral},
            physiology::Physiology,
            tetralata::{ETHOLOGY as TETRALATE, INVESTIGATE},
        },
        behaviour::{Activity, Goal},
        plantae::physiology::Propagation,
    };

    /// Hanging still in wet air, sipping the dew from its wings.
    const SIP_DEW: Intent = Intent {
        name: "sip dew",
        act: |_, _| Goal::new(Activity::Drink, None, 0.0).folded(0.4).hovering(),
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: |_, food| {
            food.plant()
                .is_some_and(|plant| plant.propagation == Propagation::Pollination)
        },
        pollinates: true,
        speed: 5.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        court: |s| spiral(s, 2.0),
        ..TETRALATE
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.player.is_some() {
            urges.add(INVESTIGATE, s.temper.curiosity * 0.6);
        }
        if s.seen.humidity > 0.7 {
            urges.add(SIP_DEW, s.drives.thirst * 1.3);
        }
    }
}
