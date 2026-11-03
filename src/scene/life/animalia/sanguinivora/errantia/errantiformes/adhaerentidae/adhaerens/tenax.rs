use crate::scene::life::BloodFeeder;
#[cfg(not(target_arch = "spirv"))]
pub use nature::ETHOLOGY;

pub const BIOLOGY: Biology = Biology {
    size: 0.07,
    maturity_days: 15.0,
    lifespan_days: 900.0,
    adult_mass: 0.03,
    // A few eggs left on a host's skin.
    litter: Litter::new(2, 1, [0.25, 0.3, 0.2, 0.13, 0.08, 0.04]),
    gestation_days: 3.0,
    altitude: (1500.0, 10000.0),
};

pub const RENDER_RADIUS: f32 = 20.0;

/// How broad its suckers, how many rings its body, and how restlessly it writhes.
///
/// Broad suckers hold through a host's hardest turn, and a many-ringed, restless body creeps faster
/// to a fresh place to feed.
pub const SUCKERS: Gene = Gene(38);
pub const SEGMENTS: Gene = Gene(39);
pub const WRITHE: Gene = Gene(40);

taxon!(
    Species,
    "Adhaerens tenax",
    "Attached Ones",
    [
        "Attached Ones begin as flattened, segmented juveniles carried between animals by the updraft.",
        "A broad sucker secures the first suitable host and becomes the centre of a permanent feeding attachment.",
        "As the body grows, its colours and surface texture resemble the host's fur, concealing the parasite from casual grooming.",
        "Small, sustained meals limit the disturbance that would provoke removal or attract Sweepers.",
        "Dense herds and shared drinking sites offer the young better chances of attachment than the sparse air between feeding communities.",
    ],
);

pub fn animate(c: &mut BloodFeeder) {
    c.opaque_bounds = vec3(1.3, 1.1, 4.2).extend(0.0);
}

/// How far its writhing carries it aside at `z` along it, growing toward its tail.
fn sway(z: f32, c: &CreatureInstance) -> f32 {
    (c.animation.z * 1.5 + z * 0.9).sin() * (0.03 + c.gene(WRITHE) * 0.04) * (z + 3.2).max(0.0)
}

/// Its row of cup eyes, each seated on its own skin: where the first lies, how far apart they
/// are, and how many there are to a side.
const EYES: (Vec3, f32, u32) = (vec3(0.12, 0.13, -2.8), 0.2, 3);

#[cfg(not(target_arch = "spirv"))]
pub const ANCHORS: u32 = 3;

#[cfg(not(target_arch = "spirv"))]
pub fn anchor(index: u32, c: &CreatureInstance) -> Vec4 {
    let at = spot(EYES.0, EYES.1, index);
    // Where its writhing has carried that part of it, the cup's centre on the skin.
    (at + vec3(sway(at.z, c), 0.0, 0.0)).extend(0.0)
}

/// A flattened, ringed leech, narrow at the head and broad behind, gripping with a small sucker
/// around its mouth and a broad disc at its tail.
pub fn surface(p: Vec3, c: &BloodFeeder) -> Surface {
    let sway = sway(p.z, c);
    let grip = 0.85 + c.gene(SUCKERS) * 0.3;
    let q = p - vec3(sway, 0.0, 0.0);
    let mut hit = Surface::new(
        annulated(
            q,
            vec3(0.0, 0.0, -3.0),
            vec3(0.0, -0.05, 2.9),
            vec3(0.32, 0.9, 0.55),
            14.0 + (c.gene(SEGMENTS) * 6.0).floor(),
            0.07,
            0.48,
        ),
        0,
    );
    hit.blend(
        sucker(
            q,
            vec3(0.0, -0.1, -3.05),
            vec3(0.0, -0.45, -0.9).normalize(),
            0.36 * grip,
        ),
        8,
        0.08,
    );
    hit.blend(
        sucker(
            q,
            vec3(0.0, -0.3, 3.0),
            vec3(0.0, -0.9, 0.45).normalize(),
            0.7 * grip,
        ),
        4,
        0.08,
    );
    // Measured bare, the body has no eyes set in it yet.
    if !c.wings.measuring() {
        hit.pit(seated_spots(q, c, 0, EYES, 0.04), 0.04);
    }
    hit
}

/// The organs that are its own, beside the body plan its lineage gives it; see [`Part`].
#[cfg(not(target_arch = "spirv"))]
pub const PARTS: &[Part] = &[];

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.39, 0.29, 0.21, 0.0),
    vec3(0.69, 0.54, 0.32),
    [Vec4::ZERO; 2],
)
.skinned(vec4(3.0, 1.0, 2.0, 0.15));

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{
        animalia::mind::{Ethology, Interest},
        behaviour::{Activity, Goal},
    };

    /// When a cleaner comes near its host, an Attached One stops feeding and lies flat and still,
    /// its colours matched to the hide, and is passed over.
    pub const ETHOLOGY: Ethology = Ethology {
        act: |mind, s| {
            if s.seen
                .mark(CLEANER)
                .is_some_and(|cleaner| cleaner.length() < 25.0)
            {
                return Some(Goal::new(Activity::Rest, None, 0.0).folded(1.0).hovering());
            }
            (super::super::ETHOLOGY.act)(mind, s)
        },
        notice: |glimpse, seen| {
            (super::super::ETHOLOGY.notice)(glimpse, seen);
            if !glimpse.heard && glimpse.other.animal().is_some_and(|animal| animal.grooms) {
                seen.note(CLEANER, glimpse.offset);
            }
        },
        ..super::super::ETHOLOGY
    };

    /// A cleaner, which would pick it off its host.
    const CLEANER: Interest = Interest("cleaner");
}
