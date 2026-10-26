pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod manduca;
pub mod purgator;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.02,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Family,
    "Manducidae",
    "Nibblers",
    [
        "Members of this family are compact four-winged animals that obtain food from organic debris.",
        "Free-flying scavengers intercept particles in swarms, while cleaners search the surfaces of larger animals.",
        "The same grasping mouth appendages serve either to retain a drifting fragment or to pick material from a host.",
        "Frequent short wing strokes allow precise movement around an unstable food source.",
        "Their feeding returns waste and parasites to the food web before that material is lost into lower, more dangerous air.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The Manducidae took to swarming: a scrap of food is found by many eyes, and a predator faces
    //! a cloud rather than one animal. Fear runs through a swarm all at once, the murmuration
    //! that makes it hard to follow, and the boldest gang together to harry a hunter.

    use crate::scene::life::{
        animalia::{
            BOLDNESS, SOCIABILITY,
            mind::{Ethology, Intent, Mind, Situation, Urges, orbit},
            nature::panic,
            physiology::Physiology,
            tetralata::{ETHOLOGY as TETRALATE, FOLLOW},
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        flocks: true,
        speed: 9.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges: swarm,
        fear: murmuration,
        ..TETRALATE
    };

    pub fn swarm(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.kin.is_some() {
            urges.add(FOLLOW, 0.25 + s.temper(SOCIABILITY) * 0.35);
        }
        if s.seen.threat.is_some() && s.seen.kin_count >= 3 && s.temper(BOLDNESS) > 0.55 {
            urges.add(MOB, s.threat * s.temper(BOLDNESS) * 1.2);
        }
    }

    /// Ganging up with the swarm to harry a predator, circling it.
    const MOB: Intent = Intent {
        name: "mob",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Guard,
                s.seen
                    .threat
                    .map(|threat| threat.offset + orbit(up, threat.offset, 8.0, s.now)),
                1.1,
            )
        },
    };

    /// The swarm's panic runs through it twice as fast as through other animals.
    fn murmuration(s: &Situation, mind: &mut Mind) -> f32 {
        (TETRALATE.fear)(s, mind) + panic(s, 1.0)
    }
}
