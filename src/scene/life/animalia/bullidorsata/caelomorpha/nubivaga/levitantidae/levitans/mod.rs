pub use super::BIOLOGY;

pub mod taeniatus;

#[cfg(not(target_arch = "spirv"))]
pub use super::PHYSIOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

taxon!(
    Genus,
    "Levitans",
    "Levitator",
    [
        "Drifters are flattened bubblebacks with a pointed leading edge and long, flexible feeding ribbons.",
        "Rounded gas chambers support the central organs while the ribbons spread through neighbouring streams of air.",
        "Unequal drag across the body turns it gently toward useful light and food-bearing currents.",
        "A ribbon can fold against the body in rough air, reducing the chance that opposing currents tear it away.",
        "This narrow, yielding form allows them to travel between open Gardens and the lower reaches of the Highlands.",
    ],
);

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! Drifters trawl. A hungry one turns side-on to the wind so its long ribbons sweep across the
    //! neighbouring streams of air, and in rough air it furls them against its body before the
    //! crossing currents can tear them away.

    use crate::scene::life::{
        animalia::{
            bullidorsata::ETHOLOGY as GAS_BAG,
            mind::{Ethology, Intent, Situation, Urges},
        },
        behaviour::{Activity, Goal},
    };

    pub const ETHOLOGY: Ethology = Ethology { urges, ..GAS_BAG };

    /// Lying across the wind, ribbons spread through the streams beside it.
    const TRAWL: Intent = Intent {
        name: "trawl",
        act: |_, s| {
            let across = s.up().cross(s.seen.wind).normalize_or_zero();
            Goal::new(Activity::Feed, Some(across * 20.0), 0.2)
        },
    };

    /// Ribbons folded flat against the body, riding out the rough air.
    const FURL: Intent = Intent {
        name: "furl",
        act: |_, _| Goal::new(Activity::Rest, None, 0.0).folded(1.0).hovering(),
    };

    fn urges(s: &Situation, urges: &mut Urges) {
        (GAS_BAG.urges)(s, urges);
        let current = s.seen.wind.length();
        urges.add(TRAWL, s.drives.hunger * 0.9);
        if current > 12.0 || s.seen.storm.is_some() {
            urges.add(FURL, 0.5 + (current - 12.0).max(0.0) * 0.05);
        }
    }
}
