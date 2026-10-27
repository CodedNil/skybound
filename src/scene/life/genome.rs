//! What every living thing inherits: its genome.
//!
//! An organism is born with a set of genes, each a value from zero to one, half from each parent
//! where it had two, blended a little and changed a little in the copying. What a gene does is
//! declared in the taxon where it arose, as a [`Gene`] naming its place in the genome: the few at
//! the root shape every body, each kingdom reads its own range, and each lineage the places left
//! for its rank. So a Ferrispina's spines are as long and keen as its parents' were, a Naetu as
//! bold or as shy, and a stand of kelp still looks like the strand it broke from.
//!
//! Places are shared out by rank, so no two genes on one line of descent ever read the same place:
//! the root `0..8`, a kingdom `8..20`, a phylum `20..26`, a class `26..30`, an order `30..32`, a
//! family `32..36`, a genus `36..38`, and a species `38..48`, so each species has room for traits
//! of its own. Plants, whose bodies are all form, read `8..32` for it.
//!
//! What an organism makes of its genes through its life, its scars, what it has learned and how
//! its wings have been torn, is not inherited; that is kept with its body and its mind.

use crate::scene::life::CreatureInstance;
use spirv_std::glam::Vec4;

/// How many genes every genome holds.
pub const GENES: usize = 48;

/// A gene: one place in the genome and what the taxon that declares it reads there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gene(pub u32);

/// How large it grows against others of its kind.
pub const SIZE: Gene = Gene(0);

/// Its colouring: the hue its pigment turns towards, how dark or pale it is, and how boldly
/// patterned.
pub const HUE: Gene = Gene(1);
pub const SHADE: Gene = Gene(2);
pub const PATTERN: Gene = Gene(3);

/// Its build: heavy or light in the forebody, and the set of its back.
pub const BUILD: Gene = Gene(4);
pub const ARCH: Gene = Gene(5);

/// How many young it bears at a time against others of its kind; see [`Litter`].
///
/// [`Litter`]: crate::scene::life::biology::Litter
pub const FECUNDITY: Gene = Gene(6);

impl CreatureInstance {
    /// A value in [0, 1) that fell out by chance as this individual grew, not inherited: where a
    /// crystal formed or an egg was glued, how long each finger of a cage came out. Each `salt`
    /// draws another. Seed zero is the reference specimen, so it too is repeatable.
    pub fn chance(&self, salt: u32) -> f32 {
        let mut x = self.kind.y ^ salt.wrapping_mul(0x9e37_79b9);
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        (x >> 8) as f32 / 16_777_216.0
    }

    /// The value of one of its genes, as the renderer reads it.
    pub fn gene(&self, gene: Gene) -> f32 {
        let packed = self.genes[(gene.0 / 4) as usize];
        match gene.0 % 4 {
            0 => packed.x,
            1 => packed.y,
            2 => packed.z,
            _ => packed.w,
        }
    }
}

/// An organism's genes.
#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Genome(pub [f32; GENES]);

#[cfg(not(target_arch = "spirv"))]
impl Genome {
    /// The reference specimen's genes, each midway, as the previews draw a species.
    pub const TYPICAL: Self = Self([0.5; GENES]);

    /// Genes for one of a wild population, fixed by a seed.
    pub fn wild(seed: u64) -> Self {
        let mut genes = [0.0; GENES];
        for (locus, gene) in genes.iter_mut().enumerate() {
            *gene = unit(seed, locus as u64);
        }
        Self(genes)
    }

    /// A relative of this genome: the same family, each gene drawn `spread` of the way towards
    /// another of the population, so a herd or a stand looks related without being alike.
    pub fn kin(self, seed: u64, spread: f32) -> Self {
        let other = Self::wild(seed);
        let mut genes = self.0;
        for (gene, stranger) in genes.iter_mut().zip(other.0) {
            *gene += (stranger - *gene) * spread;
        }
        Self(genes)
    }

    /// The genes of a young one of two parents, or of one when it had only one: each gene taken
    /// mostly from one parent or the other, blended a little with the other's, and changed a little
    /// in the copying.
    pub fn child(mother: Self, father: Self, seed: u64) -> Self {
        let mut genes = [0.0; GENES];
        for (locus, gene) in genes.iter_mut().enumerate() {
            let (from, other) = if unit(seed, locus as u64) < 0.5 {
                (mother.0[locus], father.0[locus])
            } else {
                (father.0[locus], mother.0[locus])
            };
            let change = (unit(seed, locus as u64 + 64) - 0.5) * 0.12;
            *gene = (from * 0.8 + other * 0.2 + change).clamp(0.0, 1.0);
        }
        Self(genes)
    }

    /// The same genome with one gene set otherwise.
    pub const fn with(mut self, gene: Gene, value: f32) -> Self {
        self.0[gene.0 as usize] = value;
        self
    }

    pub const fn get(self, gene: Gene) -> f32 {
        self.0[gene.0 as usize]
    }

    /// The genes packed for the renderer.
    pub fn packed(self) -> [Vec4; GENES / 4] {
        let mut packed = [Vec4::ZERO; GENES / 4];
        for (index, four) in packed.iter_mut().enumerate() {
            *four = Vec4::from_slice(&self.0[index * 4..index * 4 + 4]);
        }
        packed
    }
}

#[cfg(not(target_arch = "spirv"))]
impl Default for Genome {
    fn default() -> Self {
        Self::TYPICAL
    }
}

/// A number from zero to one fixed by a seed and a salt.
#[cfg(not(target_arch = "spirv"))]
fn unit(seed: u64, salt: u64) -> f32 {
    let mut x = seed ^ salt.wrapping_mul(0xd6e8_feb8_6659_fd93);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_take_after_their_parents() {
        let mother = Genome::wild(1);
        let father = Genome::wild(2);
        let child = Genome::child(mother, father, 3);
        let stranger = Genome::wild(4);
        let distance =
            |a: Genome, b: Genome| -> f32 { a.0.iter().zip(b.0).map(|(x, y)| (x - y).abs()).sum() };
        let parents = distance(child, mother).min(distance(child, father));
        assert!(
            parents < distance(child, stranger),
            "a child no more like its parents"
        );
        assert_eq!(
            child,
            Genome::child(mother, father, 3),
            "inheritance is not repeatable"
        );
        let family = mother.kin(9, 0.2);
        assert!(distance(family, mother) < distance(stranger, mother));
    }
}
