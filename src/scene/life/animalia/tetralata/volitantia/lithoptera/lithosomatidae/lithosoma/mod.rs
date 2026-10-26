pub use super::{BIOLOGY, render_wings};

pub mod grave;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Lithosoma",
    "Stone Body",
    [
        "Members of Lithosoma are dense herbivores whose outer tissues accumulate a protective mineral crust.",
        "Broad wings act mainly as brakes against descent, with strong regional updrafts carrying much of the body's weight.",
        "Mouth structures gather Root-Star tissue from below while the animal avoids the mist into which the plant's roots extend.",
        "Their heavy form tolerates rough vegetation but limits movement across weak or sharply sheared currents.",
        "Feeding and mineral replenishment therefore keep them close to dependable lower-Garden air columns.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Grounders are too heavy to roost: they sleep riding a rising column on their saucer, and
    //! spend their waking hours finding columns strong enough to carry them. They are too slow to
    //! outfly anything, and do not try: threatened, a Grounder draws in and goes stone-still,
    //! trusting the mineral armour that Iron Spines hone their blades on.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges, glide_asleep},
            nature::{FLEE, RIDE},
            tetralata::ETHOLOGY as TETRALATE,
        },
        behaviour::{Activity, Goal},
    };

    /// Drawn in and still, armour turned to the threat.
    const HUNKER: Intent = Intent {
        name: "hunker",
        act: |_, _| Goal::new(Activity::Rest, None, 0.0).folded(1.0).hovering(),
    };

    pub const ETHOLOGY: Ethology = Ethology {
        urges,
        sleep: glide_asleep,
        clings: false,
        ..TETRALATE
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (TETRALATE.urges)(s, urges);
        if s.seen.updraft < 0.5 {
            urges.add(RIDE, 0.3);
        }
        urges.scale(FLEE, 0.15);
        urges.add(HUNKER, s.fear * 1.3);
    }
}
