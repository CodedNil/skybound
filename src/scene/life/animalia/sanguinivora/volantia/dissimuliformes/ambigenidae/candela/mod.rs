pub use super::{BIOLOGY, render_wings};

pub mod lucens;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Candela",
    "Candle",
    [
        "Members of Candela are luminous sap feeders whose association with woody hosts has become mutualistic.",
        "Their small bodies fit within cavities that protect them from drying, predators and the strongest crosswinds.",
        "Sap carries both their food and the chemical signals that coordinate groups around the host.",
        "A single pair of wings can produce useful collective force when many animals respond together.",
        "Light, controlled feeding and coordinated movement distinguish this lineage from relatives that only disguise the cost of parasitism.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Glowgnomes court with the lamps they carry, hanging together in the dark of the wood. As
    //! the light fails they come home to their Whisperwood and gather on its trunk, lamps lit, so
    //! the tree glows to draw the visitors that feed it.

    use crate::scene::life::{
        Kind,
        animalia::{
            mind::{
                Ethology, Glimpse, Intent, Interest, Perception, Situation, Urges, display_near,
                orbit,
            },
            nature::ROOST,
            sanguinivora::ETHOLOGY as BLOOD_FEEDER,
        },
        behaviour::{Activity, Goal},
    };

    /// Its Whisperwood.
    const TREE: Interest = Interest("tree");

    /// Lamp lit, on the trunk of its tree.
    const LAMPLIGHT: Intent = Intent {
        name: "lamplight",
        act: |_, s| {
            let up = s.up();
            Goal::new(
                Activity::Display,
                s.seen
                    .mark(TREE)
                    .map(|tree| tree + orbit(up, tree, 2.5, s.wobble() * 9.0)),
                0.4,
            )
            .hovering()
        },
    };

    fn notice(glimpse: &Glimpse, seen: &mut Perception) {
        (BLOOD_FEEDER.notice)(glimpse, seen);
        if !glimpse.heard && glimpse.other == Kind::Susurrus {
            seen.note(TREE, glimpse.offset);
        }
    }

    fn urges(s: &Situation, urges: &mut Urges) {
        (BLOOD_FEEDER.urges)(s, urges);
        if s.seen.mark(TREE).is_some() {
            urges.add(LAMPLIGHT, ((0.2 - s.seen.light) * 4.0).max(0.0));
        }
    }

    pub const ETHOLOGY: Ethology = Ethology {
        court: glow,
        urges,
        notice,
        ..BLOOD_FEEDER
    };

    fn glow(s: &Situation) -> Goal {
        display_near(s.seen, 2.0, s.seen.mark(ROOST))
    }
}
