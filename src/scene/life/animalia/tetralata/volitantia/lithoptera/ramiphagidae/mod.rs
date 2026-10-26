pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod ramiphaga;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 2.0,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Ramiphagidae",
    "Branch Eaters",
    [
        "Members of this family are armoured herbivores with reinforced mouth structures for woody plant tissue.",
        "Compact wings hold the body close to a drifting framework while the feeding appendages secure a short-lived grip.",
        "Tough outer coverings reduce damage from charged thorns and scraping branches.",
        "Their feeding opens plant material that softer grazers cannot use, releasing nutrients through fragments and waste.",
        "Dependence on woody vegetation makes dense Bramble communities more suitable than open columns of suspended algae.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Bark-strippers feed by girdling: they work round and round a stem as they strip it, so a
    //! feeding one spirals slowly up its food plant rather than gnawing one spot bare.

    use crate::scene::life::{
        Kind,
        animalia::{
            mind::{Ethology, Mind, Situation, orbit},
            nature::FORAGE,
            physiology::Physiology,
            tetralata::ETHOLOGY as TETRALATE,
        },
        behaviour::{Activity, Goal},
        organs::Function,
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        crops: Function::Sap,
        speed: 3.0,
        eats: |_, food| food == Kind::Spinivinea,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology { act, ..TETRALATE };

    /// Spiralling round the stem as it strips it.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        if mind.intent == FORAGE
            && let Some(stem) = s.seen.food
        {
            let up = s.up();
            let round = orbit(up, stem, 1.5, s.now * 0.4 + s.wobble());
            let climb = up * ((s.now * 0.05 + s.wobble()).sin() * 2.0);
            return Some(Goal::new(Activity::Feed, Some(stem + round + climb), 0.4).hovering());
        }
        (TETRALATE.act)(mind, s)
    }
}
