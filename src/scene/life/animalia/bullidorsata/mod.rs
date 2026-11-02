use crate::scene::life::{
    Body, CreatureInstance,
    animalia::{Animal, Creature},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::{BREATHER, CROP, FLOAT, MUSCLE},
    organs::{Function, Organ, Part, Tissue},
};
use core::ops::{Deref, DerefMut};
use spirv_std::glam::Vec4;
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec3;
pub mod caelomorpha;

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
    fn sex(&self) -> f32 {
        self.animal.sex()
    }
}

impl Creature for Bubbleback {
    fn animal(&self) -> &Animal {
        &self.animal
    }
}

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

/// The gelatinous digestive tissue that stands in for a crop, spread through the body's core.
#[cfg(not(target_arch = "spirv"))]
pub const GEL: Organ = Organ::new("digestive gel", Function::Digestion, vec3(0.80, 0.62, 0.42));
/// The muscle of the bladder walls, which squeezes gas from chamber to chamber to tilt and trim.
#[cfg(not(target_arch = "spirv"))]
pub const CHAMBER_MUSCLE: Organ =
    Organ::new("chamber wall", Function::Movement, vec3(0.62, 0.30, 0.34)).of(Tissue::Muscle);
/// Elastic hoops of spring fibre that hold the great bladder open against the air, as a balloon's
/// ribs would, and let it give rather than burst.
#[cfg(not(target_arch = "spirv"))]
pub const HOOP: Organ =
    Organ::new("bladder hoop", Function::Support, vec3(0.86, 0.84, 0.74)).of(Tissue::Spring);

/// The bubbleback body plan.
///
/// One great bladder over a core of digestive gel, smaller bladders round it, breathing folds at
/// the sides, the chamber walls' muscle in place of a body wall's, and three hoops holding the
/// great bladder open.
#[cfg(not(target_arch = "spirv"))]
pub const ANATOMY: &[Part] = &[
    Part::lobe(GEL, vec3(0.0, -0.4, 0.0), vec3(0.45, 0.35, 0.3)).replacing(CROP),
    Part::rod(
        CHAMBER_MUSCLE,
        vec3(-0.55, 0.1, 0.8),
        vec3(0.55, 0.1, 0.8),
        0.12,
    )
    .replacing(MUSCLE),
    Part::lobe(FLOAT, vec3(-0.1, 0.35, 0.0), vec3(0.55, 0.5, 0.4)),
    Part::lobe(FLOAT, vec3(0.45, 0.2, 0.5), vec3(0.3, 0.35, 0.22)),
    Part::lobe(FLOAT, vec3(-0.6, 0.15, 0.5), vec3(0.28, 0.3, 0.18)),
    Part::lobe(BREATHER, vec3(-0.35, -0.25, 0.6), vec3(0.2, 0.25, 0.12)),
    Part::hoop(HOOP, -0.45, 0.05),
    Part::hoop(HOOP, 0.0, 0.05),
    Part::hoop(HOOP, 0.45, 0.05),
];

taxon!(
    Phylum,
    "Bullidorsata",
    "Bubblebacks",
    [
        "Bubblebacks descend from porous animals whose moist outer tissues both collect and digest suspended food.",
        "Several gas bladders distribute lift through the body, so local damage need not send the whole animal toward the aur ocean.",
        "Gelatinous organs exchange nutrients over short distances without the concentrated digestive crop of active flyers.",
        "Chamber inflation and flexible membranes control orientation while the planetary updraft supplies most movement, and hoops of spring fibre, grown apart from any other lineage's frame, hold the great bladder open and let it give under a blow rather than burst.",
        "Their descendants divide the sky by arranging the same buoyant and absorptive tissues into compact balloons, trailing leaves or enormous feeding canopies.",
    ],
    anatomy: ANATOMY,
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
