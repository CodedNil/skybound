pub use super::render_wings;
use crate::scene::life::biology::Biology;
pub mod errantia;
pub mod volantia;
pub mod young;

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.05,
    ..super::BIOLOGY
};

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, HOST, PHYSIOLOGY, WATERHOLE, hosts};

taxon!(
    Phylum,
    "Sanguinivora",
    "Blood Eaters",
    [
        "Members of this phylum descend from soft-bodied animals with a muscular feeding sucker and paired mouth appendages.",
        "Fluid drawn from living hosts supplies both nutrients and water, allowing the feeding end to serve as an anchor as well as a mouth.",
        "One branch remains wingless and settles for long periods, while another has evolved a single pair of wings for repeated visits.",
        "These wings arose separately from the two ancestral pairs of Tetralata.",
        "Grooming mimicry and plant mutualism developed from brief, restrained feeding, giving some descendants a lasting place in their hosts' lives.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The blood-feeders hunt by fastening onto a far larger host and drawing its fluid, leaving
    //! its organs whole so it lives to carry them.

    use crate::scene::life::{
        Kind,
        animalia::{
            ETHOLOGY as ANIMAL,
            mind::{
                Ethology, Glimpse, Hold, Interest, Mind, Perception, Sighting, Situation, Snare,
            },
            nature::lead,
            physiology::{Physiology, flow_channels, plant},
            sanguinivora::young::LARVA,
        },
        behaviour::{Activity, Goal},
        organs::Function,
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: hosts,
        parasite: true,
        breathe: flow_channels,
        digestion: 0.004,
        form: |_, young| if young < 0.7 { LARVA } else { 0 },
        ..super::super::PHYSIOLOGY
    };

    /// A blood-feeder's hosts: animals at least four times its size.
    pub fn hosts(eater: Kind, host: Kind) -> bool {
        !plant(host) && host.biology().adult_mass > eater.biology().adult_mass * 4.0
    }

    pub const ETHOLOGY: Ethology = Ethology {
        hunt: approach,
        snare: latch,
        notice,
        hunts: true,
        ..ANIMAL
    };

    /// A host, wherever one is in sight.
    pub const HOST: Interest = Interest("host");
    /// A plant animals come to drink at, where hosts crowd.
    pub const WATERHOLE: Interest = Interest("waterhole");

    /// Every blood-feeder watches for hosts, and for the places they gather to drink.
    pub fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (ANIMAL.notice)(glimpse, seen);
        if glimpse.heard {
            return;
        }
        if hosts(glimpse.kind, glimpse.other) {
            seen.note(HOST, glimpse.offset);
        }
        if plant(glimpse.other) && glimpse.other.organs().has(Function::Water) {
            seen.note(WATERHOLE, glimpse.offset);
        }
    }

    /// Closing on a host to fasten on.
    fn approach(_: &mut Mind, s: &Situation, host: Sighting) -> Goal {
        Goal::new(Activity::Attach, Some(lead(s.kind, host)), 0.9)
    }

    /// Once within reach it fastens on and drinks, more from a host it is large against.
    fn latch(snare: &Snare) -> Option<Hold> {
        (snare.distance < snare.reach).then_some(Hold {
            stun: 0.0,
            drain: snare.dt * snare.share * 60.0,
            latch: true,
        })
    }
}
