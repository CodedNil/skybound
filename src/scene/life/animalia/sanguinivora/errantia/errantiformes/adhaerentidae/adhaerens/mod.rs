pub use super::{BIOLOGY, render_wings};

pub mod tenax;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Adhaerens",
    "Clinger",
    [
        "Members of Adhaerens are fluid-feeding parasites that attach permanently to larger animals.",
        "Windborne juveniles expose a broad sucker when they brush against a suitable host.",
        "After attachment, the flattened body grows close to the skin and develops a surface resembling the surrounding fur.",
        "Feeding proceeds slowly enough to reduce irritation, while the host supplies transport through otherwise dangerous open air.",
        "Their life cycle depends on releasing young where passing animals are more common than empty current.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! The young of the Attached Ones do not search the open air for a host: they wait where hosts
    //! must come, hanging in the still air beside a Pondplant until a herd crowds in to drink.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Intent, Situation, Urges},
            sanguinivora::{ETHOLOGY as WANDERER, HOST, WATERHOLE},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology { urges, ..WANDERER };

    /// Hanging beside a drinking-plant, waiting for a herd.
    const WAIT: Intent = Intent {
        name: "wait at the water",
        act: |_, s| {
            let beside = s
                .seen
                .mark(WATERHOLE)
                .map(|water| water - water.normalize_or_zero() * 6.0);
            Goal::new(Activity::Rest, beside, 0.4).hovering()
        },
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (WANDERER.urges)(s, urges);
        if s.seen.mark(WATERHOLE).is_some() && s.seen.mark(HOST).is_none() {
            urges.add(WAIT, 0.45);
        }
    }
}
