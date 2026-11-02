#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::animalia::physiology::{GANGLION, PUMP};
pub mod errantia;
pub mod volantia;

/// The muscular sucker at the feeding end, which draws a host's fluid and holds the body fast
/// to it.
#[cfg(not(target_arch = "spirv"))]
pub const SUCKER: Organ =
    Organ::new("sucker", Function::Handling, vec3(0.66, 0.24, 0.30)).of(Tissue::Muscle);

/// The blood-feeder body plan: frameless, all muscle and fluid, the sucker filling the feeding
/// end, the nerve centre close behind it and the pump over the crop.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[
    Part::lobe(SUCKER, vec3(-0.95, -0.1, 0.0), vec3(0.6, 0.6, 0.08)),
    Part::lobe(GANGLION, vec3(-0.78, 0.25, 0.0), vec3(0.35, 0.35, 0.06)),
    Part::lobe(PUMP, vec3(-0.3, 0.3, 0.0), vec3(0.3, 0.3, 0.1)),
];

pub const BIOLOGY: Biology = Biology {
    adult_mass: 0.05,
    ..super::BIOLOGY
};

/// What a blood-feeder adds to an animal: the wings its flying lineages grew, and the tail and
/// trailing ribbons some of them pose.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BloodFeeder {
    pub animal: Animal,
    pub wings: Wings,
    pub rig: Rig,
}

impl Deref for BloodFeeder {
    type Target = Animal;
    fn deref(&self) -> &Animal {
        &self.animal
    }
}

impl DerefMut for BloodFeeder {
    fn deref_mut(&mut self) -> &mut Animal {
        &mut self.animal
    }
}

impl Body for BloodFeeder {
    fn organism(&self) -> &CreatureInstance {
        &self.animal.organism
    }
    fn carrying(&self) -> (Vec4, Vec4) {
        self.animal.carrying()
    }
    fn sex(&self) -> f32 {
        self.animal.sex()
    }
    fn swung(&self, point: Vec3) -> Vec3 {
        self.rig.swung(point, Vec3::ONE)
    }
}

impl Creature for BloodFeeder {
    fn animal(&self) -> &Animal {
        &self.animal
    }
}

impl Flier for BloodFeeder {
    fn wings(&self) -> &Wings {
        &self.wings
    }
}

impl Rigged for BloodFeeder {
    fn rig(&self) -> &Rig {
        &self.rig
    }
    fn rigging(&mut self) -> (&mut CreatureInstance, &mut Rig) {
        (&mut self.animal.organism, &mut self.rig)
    }
}

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
        "They have no frame: the body is a muscular sac kept in shape by its own fluid, which is why it can stretch so far round a meal.",
        "Grooming mimicry and plant mutualism developed from brief, restrained feeding, giving some descendants a lasting place in their hosts' lives.",
    ],
    anatomy: ANATOMY,
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The blood-feeders hunt by fastening onto a far larger host and drawing its fluid, leaving
    //! its organs whole so it lives to carry them.

    use super::*;
    use crate::scene::life::{
        Kind,
        animalia::{
            ETHOLOGY as ANIMAL,
            mind::{
                Ethology, Glimpse, Hold, Interest, Mind, Perception, Sighting, Situation, Snare,
            },
            nature::lead,
            physiology::{Physiology, flow_channels, plant},
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: hosts,
        parasite: true,
        breathe: flow_channels,
        digestion: 0.004,
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
