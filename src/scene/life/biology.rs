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
    litter: Litter::new(1, 1, [0.6, 0.3, 0.1, 0.0, 0.0, 0.0]),
    // A plant sets its seed or buds within minutes; every animal lineage sets its own carrying.
    gestation_days: 0.001,
};

/// How many young are born, or eggs laid, or spores cast, at a time.
///
/// Not every number is as likely as every other. Each species has a run of numbers it can bear,
/// from `first` up by `step`, and a weight for each: most bear near the commonest number, a few
/// more or fewer, and now and then far more, as litters and clutches run among living things. An
/// individual's fecundity skews its own chances towards the larger numbers or the smaller.
#[derive(Clone, Copy, Debug)]
pub struct Litter {
    pub first: u32,
    pub step: u32,
    pub weights: [f32; 6],
}

impl Litter {
    pub const fn new(first: u32, step: u32, weights: [f32; 6]) -> Self {
        Self {
            first,
            step,
            weights,
        }
    }
}

// Drawn only by the simulation, on the CPU.
#[cfg(not(target_arch = "spirv"))]
impl Litter {
    /// The chance of each number, skewed by `fecundity` in [0, 1]: a fecund individual's chances
    /// lean towards the larger litters, a barren one's towards the smaller.
    pub fn chances(self, fecundity: f32) -> [f32; 6] {
        let lean = (fecundity - 0.5) * 1.2;
        let mut chances = [0.0; 6];
        let mut total = 0.0;
        for (index, chance) in chances.iter_mut().enumerate() {
            *chance = self.weights[index] * (lean * index as f32).exp();
            total += *chance;
        }
        for chance in &mut chances {
            *chance /= total.max(1e-6);
        }
        chances
    }

    /// How many are born for a draw `roll` in [0, 1).
    pub fn draw(self, roll: f32, fecundity: f32) -> u32 {
        let mut left = roll;
        let chances = self.chances(fecundity);
        for (index, chance) in chances.iter().enumerate() {
            if left < *chance {
                return self.first + self.step * index as u32;
            }
            left -= chance;
        }
        let last = chances
            .iter()
            .rposition(|chance| *chance > 0.0)
            .unwrap_or(0);
        self.first + self.step * last as u32
    }

    /// The number born on average, for a typical individual.
    pub fn mean(self) -> f32 {
        let chances = self.chances(0.5);
        (0..6)
            .map(|index| chances[index] * (self.first + self.step * index as u32) as f32)
            .sum()
    }
}

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
    /// How many it bears at a time.
    pub litter: Litter,
    /// Days from conception to birth, or to laying for those that lay eggs.
    pub gestation_days: f32,
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
            let litter = biology.litter;
            assert!(
                litter.first > 0 && litter.step > 0 && litter.weights.iter().all(|w| *w >= 0.0),
                "invalid litter for {kind:?}"
            );
            assert!(
                (litter.weights.iter().sum::<f32>() - 1.0).abs() < 0.02,
                "litter weights for {kind:?} do not sum to one"
            );
            assert!(
                biology.gestation_days > 0.0,
                "invalid gestation for {kind:?}"
            );
        }
    }
}
