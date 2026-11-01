use crate::scene::life::{
    Body, CreatureInstance,
    animalia::{Animal, Creature},
    biology::Biology,
};
use core::ops::{Deref, DerefMut};
use spirv_std::glam::Vec4;
pub mod caelomorpha;

pub const BIOLOGY: Biology = super::BIOLOGY;

/// What a gas-bag adds to an animal: the knots of the feeding curtains it trails, three sheets of
/// them, and where they were last frame.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Bubbleback {
    pub animal: Animal,
    pub curtains: [[[Vec4; 3]; 12]; 3],
    pub previous_curtains: [[[Vec4; 3]; 12]; 3],
}

impl Deref for Bubbleback {
    type Target = Animal;
    fn deref(&self) -> &Animal {
        &self.animal
    }
}

impl DerefMut for Bubbleback {
    fn deref_mut(&mut self) -> &mut Animal {
        &mut self.animal
    }
}

impl Body for Bubbleback {
    fn organism(&self) -> &CreatureInstance {
        &self.animal.organism
    }
    fn carrying(&self) -> (Vec4, Vec4) {
        self.animal.carrying()
    }
}

impl Creature for Bubbleback {
    fn animal(&self) -> &Animal {
        &self.animal
    }
}

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Phylum,
    "Bullidorsata",
    "Bubblebacks",
    [
        "Bubblebacks descend from porous animals whose moist outer tissues both collect and digest suspended food.",
        "Several gas bladders distribute lift through the body, so local damage need not send the whole animal toward the aur ocean.",
        "Gelatinous organs exchange nutrients over short distances without the concentrated digestive crop of active flyers.",
        "Chamber inflation and flexible membranes control orientation while the planetary updraft supplies most movement.",
        "Their descendants divide the sky by arranging the same buoyant and absorptive tissues into compact balloons, trailing leaves or enormous feeding canopies.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
pub mod nature {
    //! The gas-bags have no muscle to fly with and no nerve centre to plan with: they drift, sleep
    //! slack, and breed by releasing spawn among their kind.

    use crate::scene::life::{
        Kind,
        animalia::{
            ETHOLOGY as ANIMAL,
            mind::{Ethology, Situation},
            physiology::Physiology,
        },
        behaviour::{Activity, Goal},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        eats: |_, food| food == Kind::Auracoccus,
        floats: true,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        court: spawn,
        sleep: torpor,
        ..ANIMAL
    };

    /// Spawning: drifting into the thick of its kind to release.
    pub const fn spawn(s: &Situation) -> Goal {
        Goal::new(Activity::Display, s.seen.kin, 0.2)
    }

    /// Torpor: going slack and letting the air carry it.
    pub const fn torpor(_: &Situation) -> Goal {
        Goal::new(Activity::Sleep, None, 0.0)
    }
}
