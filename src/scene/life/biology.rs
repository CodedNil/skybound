//! What every living thing has, plant or animal.
//!
//! A span of life to live out, a time to grow up in, a size, and a band of air it lives in.
//! Everything else about a body differs between the
//! kingdoms, and lives with each kingdom in the tree.

/// Life at its root, which every taxon refines with `..super::BIOLOGY`.
pub const ANCESTRAL: Biology = Biology {
    maturity_days: 90.0,
    lifespan_days: 3000.0,
    adult_mass: 1.0,
    altitude: (50.0, 6000.0),
    size: 1.0,
};

/// A life history.
#[derive(Clone, Copy, Debug)]
pub struct Biology {
    /// Days from birth to being grown.
    pub maturity_days: f32,
    /// Days from birth to death of old age.
    pub lifespan_days: f32,
    /// Relative dry biomass when grown, shared by resource transfers and reproduction.
    pub adult_mass: f32,
    /// The band of altitudes it lives in, in metres above the aur.
    pub altitude: (f32, f32),
    /// How much larger or smaller than its drawn design the grown body is. A species' surface and
    /// organs are laid out in the design's own frame, and the body grows to this scale of it.
    pub size: f32,
}

#[cfg(test)]
mod tests {
    use crate::scene::life::KINDS;

    #[test]
    fn every_life_history_is_consistent() {
        for kind in KINDS {
            let biology = kind.biology();
            assert!(
                biology.maturity_days > 0.0 && biology.lifespan_days > biology.maturity_days,
                "invalid life history for {kind:?}"
            );
            assert!(
                biology.altitude.0 < biology.altitude.1,
                "invalid habitat for {kind:?}"
            );
            assert!(biology.adult_mass > 0.0, "invalid biomass for {kind:?}");
            assert!(biology.size > 0.0, "invalid size for {kind:?}");
        }
    }
}
