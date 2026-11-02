#![allow(
    hidden_glob_reexports,
    reason = "every taxon inherits all its parent has, and anything it defines itself overrides it"
)]
#![allow(
    clippy::wildcard_imports,
    reason = "a taxon's behaviour sees all its taxon has, as the taxon sees all its parent has"
)]
#[cfg(not(target_arch = "spirv"))]
pub use crate::scene::life::organs::{Function, Organ, Part, Tissue, pair, site};
use crate::scene::{
    life::{
        animalia::{
            Flier, Rigged,
            bullidorsata::caelomorpha::gel,
            tetralata::vivipara::ferrilaminata::naeturidae::naeturus::{nae, naetu},
            visit_wings, with_wings,
        },
        plantae::plant_material,
        surface::coat,
    },
    planet::GRAVITY,
};
/// The vocabulary every taxon draws, grows and describes its bodies in, which it inherits with
/// all else its parents have.
pub use crate::scene::{
    life::{
        biology::{Biology, Litter},
        genome::Gene,
        surface::{Hit, Material, Ray, Surface, TraceResult, trace_material},
    },
    math::Power,
};
pub use animalia::{bullidorsata::Bubbleback, sanguinivora::BloodFeeder, tetralata::Tetralate};
pub use core::f32::consts::{FRAC_PI_2, PI, TAU};
pub use plantae::Plant;
pub use spirv_std::glam::{
    FloatExt, Quat, UVec4, Vec2, Vec3, Vec3Swizzles, Vec4, vec2, vec3, vec4,
};
#[cfg(target_arch = "spirv")]
pub use spirv_std::num_traits::Float;

/// The most organs a body has, and a cutaway shows.
pub const ORGAN_SLOTS: usize = 20;

/// What every organism drawn has, plant or animal: where it is and how it moves, what it is, how
/// far through its life, the genes it was born with, and the wounds and scars its life has left.
///
/// Each lineage's instance holds this and adds what that lineage alone has, as the lineage itself
/// arose from its parent: a [`plantae::Plant`] the way it bends and the water it holds, an
/// [`animalia::Animal`] its wingbeat and stores, and so on down the tree. Each derefs to the one
/// it extends, so `c.kind` reads the same on any of them, and a field a lineage lacks is simply
/// not there.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CreatureInstance {
    pub position: Vec4,
    pub opaque_bounds: Vec4,
    pub rotation: Vec4,
    pub animation: Vec4,
    pub previous_animation: Vec4,
    pub previous_position: Vec4,
    pub previous_rotation: Vec4,
    pub kind: UVec4,
    pub life: Vec4, // Age in days, juvenile development, senescence, developmental form.
    pub identity: Vec4, // The first four genes; seed zero is the reference specimen.
    /// Every gene it was born with; see [`genome`].
    pub genes: [Vec4; genome::GENES / 4],
    pub proportions: Vec4, // Body scale and wing development.
    pub previous_proportions: Vec4,
    /// Every wound its body carries, open or long healed to a scar; see [`Wound`].
    pub wounds: [Wound; WOUNDS],
    /// For a cutaway, its organs: each where it lies and whether it is one of a pair, how large it
    /// is and whether it glows, and its colour. `kind.w` holds how many there are; a creature with
    /// none is drawn whole.
    pub organs: [[Vec4; 3]; ORGAN_SLOTS],
}

/// How many wounds and scars a body shows at once; a new one replaces the slightest.
pub const WOUNDS: usize = 6;

/// A wound, open or healed, anywhere on any living body.
///
/// Plants and animals are wounded alike, by bites, cuts, punctures, tears, burns and crushing, and
/// a scar is only a wound that has healed: it keeps its place, its size and the shape of whatever
/// made it, and as it heals its raw, open tissue closes over, an animal's to pale, puckered scar
/// and a plant's to a ridge of callus.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Wound {
    /// Where it is in the body's own frame, and how large; zero size is no wound.
    pub site: Vec4,
    /// How far it has healed, from open at zero to a scar at one; what made it, as a
    /// [`Injury`] number; the bearing of a cut or tear across the body; and how deep it went.
    pub state: Vec4,
}

/// What made a wound, which sets how it harms the body and the shape it leaves.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Injury {
    /// A lance, stylet or thorn driven deep along a line: a small, deep puncture.
    Pierce = 0,
    /// A blade drawn across the body: a long, straight, shallow cut.
    Slash = 1,
    /// Membrane or flesh ripped apart: a ragged tear.
    Tear = 2,
    /// Jaws or a rasp tearing out a mouthful: the arc of a bite's teeth.
    Bite = 3,
    /// Acid, or the aur, eating into the surface: a broad burn.
    Burn = 4,
    /// A blow that breaks nothing open: a crushed, bruised patch.
    Crush = 5,
    /// A parasite's sucker drawing fluid, leaving the organs whole: a small, round mark.
    Drain = 6,
}

impl Wound {
    pub const NONE: Self = Self {
        site: Vec4::ZERO,
        state: Vec4::ZERO,
    };

    pub const fn open(at: Vec3, size: f32, injury: Injury, bearing: f32, depth: f32) -> Self {
        Self {
            site: Vec4::new(at.x, at.y, at.z, size),
            state: Vec4::new(0.0, injury as u32 as f32, bearing, depth),
        }
    }

    /// How large it is; zero for no wound.
    pub fn size(&self) -> f32 {
        self.site.w
    }

    /// How far it has healed.
    pub fn healed(&self) -> f32 {
        self.state.x
    }

    pub fn injury(&self) -> u32 {
        self.state.y as u32
    }
}

/// What every lineage's instance can be asked, whatever it adds.
///
/// That is the organism it extends, and what that lineage alone has that every body's drawing
/// needs to know of. A lineage answers for what it has; the rest are nothing.
pub trait Body {
    fn organism(&self) -> &CreatureInstance;

    /// Where a body carrying young swells, round its brood chamber (centre and size), and how far
    /// through its carrying it is and how many it carries; zero for none.
    fn carrying(&self) -> (Vec4, Vec4) {
        (Vec4::ZERO, Vec4::ZERO)
    }

    /// How far a point on the body moved since the last frame beyond the body's own motion, as a
    /// swinging tail does.
    fn swung(&self, _point: Vec3) -> Vec3 {
        Vec3::ZERO
    }

    /// Which sex it is, as it is drawn: zero for a body with no sexes or both at once, one for the
    /// sex that bears the young and two for the other. Plants have none; animals carry theirs.
    fn sex(&self) -> f32 {
        0.0
    }
}

impl Body for CreatureInstance {
    fn organism(&self) -> &CreatureInstance {
        self
    }
}

/// An organism as it is drawn, in whichever lineage's instance it is drawn with.
#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy)]
#[expect(
    clippy::large_enum_variant,
    reason = "each is copied whole into its lineage's GPU buffer, so boxing would only add a step"
)]
pub enum Drawn {
    Plant(Plant),
    Bubbleback(Bubbleback),
    BloodFeeder(BloodFeeder),
    Tetralate(Tetralate),
}

#[cfg(not(target_arch = "spirv"))]
impl Drawn {
    pub fn new(position: Vec4, rotation: Vec4, animation: Vec4, kind: u32) -> Self {
        Self::specimen(
            position,
            rotation,
            animation,
            kind,
            KINDS[kind as usize].biology().maturity_days * 1.25,
            0,
        )
    }

    /// One of a species as the previews draw it: the reference specimen for seed zero, and for
    /// any other seed one of a wild population with genes of its own.
    pub fn specimen(
        position: Vec4,
        rotation: Vec4,
        animation: Vec4,
        kind: u32,
        age_days: f32,
        seed: u32,
    ) -> Self {
        let genes = if seed == 0 {
            genome::Genome::TYPICAL
        } else {
            genome::Genome::wild(u64::from(seed))
        };
        Self::born(position, rotation, animation, kind, age_days, seed, genes)
    }

    /// An individual with the genes it was born with, in its lineage's instance, posed as its
    /// species poses itself.
    pub fn born(
        position: Vec4,
        rotation: Vec4,
        animation: Vec4,
        kind: u32,
        age_days: f32,
        seed: u32,
        genes: genome::Genome,
    ) -> Self {
        let species = KINDS[kind as usize];
        let mut organism = CreatureInstance {
            position,
            rotation,
            animation,
            previous_animation: animation,
            previous_position: position,
            previous_rotation: rotation,
            kind: UVec4::new(kind, 0, 0, 0),
            ..CreatureInstance::default()
        };
        development::prepare(&mut organism, species.biology(), age_days, seed, genes);
        organism.previous_proportions = organism.proportions;
        organism.previous_animation = organism.animation;
        let animal = |organism| animalia::Animal {
            organism,
            ..animalia::Animal::default()
        };
        let mut drawn = match layout_of(kind) {
            TETRALATE => Self::Tetralate(Tetralate {
                animal: animal(organism),
                ..Tetralate::default()
            }),
            BLOOD_FEEDER => Self::BloodFeeder(BloodFeeder {
                animal: animal(organism),
                ..BloodFeeder::default()
            }),
            BUBBLEBACK => Self::Bubbleback(Bubbleback {
                animal: animal(organism),
                ..Bubbleback::default()
            }),
            _ => Self::Plant(Plant {
                organism,
                ..Plant::default()
            }),
        };
        animate(&mut drawn);
        let look = species.look();
        let c = drawn.organism_mut();
        if c.life.w > 0.0 {
            // Eggs and spores are sized from the body alone and not the adult's wingspan.
            c.opaque_bounds =
                juvenile_bounds(c.life.w as u32, c.opaque_bounds.truncate()).extend(0.0);
        } else if look.family == COAT {
            // The bounds must hold the wings' frames at full stretch and stroke, or tracing
            // clips them.
            for index in 0..2 {
                let wing = look.wings[index];
                if wing.x > 0.0 {
                    c.opaque_bounds = c
                        .opaque_bounds
                        .truncate()
                        .max(vec3(
                            look.hinges[index].x + wing.x * 1.1,
                            wing.x * 0.5,
                            wing.z.abs() + wing.y * 1.6 + wing.x * 0.35,
                        ))
                        .extend(0.0);
                }
            }
        }
        // A pouch is part of a grown parent's body, empty or not; and its brood, wherever it is
        // carried, must lie within its bounds.
        if look.brood.w > 0.0 && c.life.w <= 0.0 {
            if species
                .animal()
                .is_some_and(|animal| animal.brood == animalia::physiology::Brood::Pouch)
                && c.kind.z == 0
            {
                c.kind.z = EMPTY_POUCH;
            }
            c.opaque_bounds = c
                .opaque_bounds
                .truncate()
                .max(vec3(
                    look.brood.z * 1.4,
                    look.brood.x.abs() + look.brood.w * 1.6,
                    look.brood.y.abs() + look.brood.z * 1.6,
                ))
                .extend(0.0);
        }
        // The bounds are laid out in the design's frame, and the body drawn at its species' size.
        let size = species.biology().size;
        let design = (c.position.w / size).max(c.opaque_bounds.truncate().length());
        c.position.w = design * development::scale(c).max_element().max(size);
        match &mut drawn {
            Self::Tetralate(c) => c.rig.previous_tail = c.rig.tail,
            Self::BloodFeeder(c) => c.rig.previous_tail = c.rig.tail,
            _ => {}
        }
        drawn
    }

    /// The tetralate it is, if it is one.
    pub const fn tetralate(self) -> Option<Tetralate> {
        match self {
            Self::Tetralate(c) => Some(c),
            _ => None,
        }
    }

    /// The gas-bag it is, if it is one.
    pub const fn bubbleback(self) -> Option<Bubbleback> {
        match self {
            Self::Bubbleback(c) => Some(c),
            _ => None,
        }
    }

    /// What every organism has, whatever its lineage adds.
    pub fn organism(&self) -> &CreatureInstance {
        match self {
            Self::Plant(c) => c,
            Self::Bubbleback(c) => c,
            Self::BloodFeeder(c) => c,
            Self::Tetralate(c) => c,
        }
    }

    pub fn organism_mut(&mut self) -> &mut CreatureInstance {
        match self {
            Self::Plant(c) => c,
            Self::Bubbleback(c) => c,
            Self::BloodFeeder(c) => c,
            Self::Tetralate(c) => c,
        }
    }

    /// What an animal adds; none for a plant.
    pub fn animal_mut(&mut self) -> Option<&mut animalia::Animal> {
        match self {
            Self::Plant(_) => None,
            Self::Bubbleback(c) => Some(c),
            Self::BloodFeeder(c) => Some(c),
            Self::Tetralate(c) => Some(c),
        }
    }

    /// Wings, for the lineages that fly on them.
    pub const fn wings_mut(&mut self) -> Option<&mut animalia::Wings> {
        match self {
            Self::BloodFeeder(c) => Some(&mut c.wings),
            Self::Tetralate(c) => Some(&mut c.wings),
            _ => None,
        }
    }

    /// Mark what an animal is carrying: `expecting` young, `gestation` of the way to birth,
    /// swelling it round its brood chamber.
    pub fn carrying(&mut self, expecting: u32, gestation: f32) {
        let kind = KINDS[self.organism().kind.x as usize];
        let Some(animal) = self.animal_mut() else {
            return;
        };
        if let Some(chamber) = kind
            .organs()
            .sites()
            .iter()
            .find(|site| site.organ.function == organs::Function::Breeding)
        {
            animal.belly = chamber.centre.extend(chamber.radii.max_element());
        }
        animal.stores.y = if expecting > 0 {
            gestation.clamp(0.0, 1.0)
        } else {
            0.0
        };
        animal.stores.z = expecting as f32;
    }

    /// Carry where it was last frame over from `old`, its pose then, for motion vectors: its
    /// place, turn and stroke, and whatever its lineage poses besides.
    pub fn follow(&mut self, old: &Self) {
        let then = *old.organism();
        let c = self.organism_mut();
        c.previous_position = then.position;
        c.previous_rotation = then.rotation;
        c.previous_animation = then.animation;
        c.previous_proportions = then.proportions;
        match (self, old) {
            (Self::Tetralate(c), Self::Tetralate(old)) => {
                c.animal.previous_motion = old.animal.motion;
                c.rig.previous_tail = old.rig.tail;
                c.previous_tentacles = old.tentacles;
            }
            (Self::BloodFeeder(c), Self::BloodFeeder(old)) => {
                c.animal.previous_motion = old.animal.motion;
                c.rig.previous_tail = old.rig.tail;
            }
            (Self::Bubbleback(c), Self::Bubbleback(old)) => {
                c.animal.previous_motion = old.animal.motion;
                c.previous_curtains = old.curtains;
            }
            (Self::Plant(c), Self::Plant(old)) => c.previous_bend = old.bend,
            _ => {}
        }
    }

    /// The body's surface at a point in its own frame.
    pub fn surface(&self, point: Vec3) -> Surface {
        let c = self.organism();
        if c.life.w > 0.0 {
            return development::juvenile_surface(point, c);
        }
        let kind = c.kind.x;
        match self {
            Self::Plant(c) => plant_surface(kind, point, c),
            Self::Bubbleback(c) => bubbleback_surface(kind, point, c),
            Self::BloodFeeder(c) => blood_feeder_surface(kind, point, c),
            Self::Tetralate(c) => tetralate_surface(kind, point, c),
        }
    }

    /// Everything of the body along a ray: its solid body, and its thin membranes and envelopes in
    /// front of it.
    pub fn render(&self, ray: Ray) -> surface::TraceResult {
        let solid = self.render_opaque(ray);
        let mut result = surface::TraceResult::opaque(solid);
        let thin = Ray {
            max_distance: if solid.is_hit() {
                solid.distance()
            } else {
                ray.max_distance
            },
            ..ray
        };
        let kind = self.organism().kind.x;
        match self {
            Self::Tetralate(c) => visit_tetralate_thin(kind, thin, c, |hit| result.add(hit)),
            Self::BloodFeeder(c) => visit_blood_feeder_thin(kind, thin, c, |hit| result.add(hit)),
            Self::Bubbleback(c) => visit_bubbleback_thin(kind, thin, c, |hit| result.add(hit)),
            Self::Plant(_) => {}
        }
        result
    }

    /// The solid body along a ray.
    pub fn render_opaque(&self, ray: Ray) -> Hit {
        let kind = self.organism().kind.x;
        match self {
            Self::Plant(c) => render_plant(kind, ray, c),
            Self::Bubbleback(c) => render_bubbleback(kind, ray, c),
            Self::BloodFeeder(c) => render_blood_feeder(kind, ray, c),
            Self::Tetralate(c) => render_tetralate(kind, ray, c),
        }
    }
}

/// The bounds of an egg or spore, from the half extents of the adult's body: a string of eggs
/// trailing far behind, an egg round, and a spore in the plant's own bounds.
#[cfg(not(target_arch = "spirv"))]
fn juvenile_bounds(form: u32, body: Vec3) -> Vec3 {
    let size = body.max_element();
    match form {
        animalia::tetralata::young::EGG_STRINGS => vec3(size * 0.6, size * 0.6, size * 1.5),
        plantae::young::SPORE => body,
        _ => Vec3::splat(size * 0.6),
    }
}

fn prepare_pose<C: Rigged>(creature: &mut C, length: f32, tail: f32, ribbons: usize) {
    let (c, rig) = creature.rigging();
    // As long as its genes have it, and growing with it.
    let tail = tail * (0.35 + c.life.y * 0.65) * (0.8 + c.gene(animalia::tetralata::TAIL) * 0.4);
    for i in 0..6 {
        let t = i as f32 / 5.0;
        rig.tail[i] = vec3(
            (c.animation.z * 2.0 - t * 4.0).sin() * t * 1.3,
            -t * 0.6,
            length * 0.6 + t * tail,
        )
        .extend(0.0);
    }
    for i in 0..ribbons {
        let lane = i as f32 - (ribbons as f32 - 1.0) * 0.5;
        rig.ribbons[i * 6] = vec3(lane * 0.8, -3.9, 0.0).extend(0.0);
        for j in 1..6 {
            let t = j as f32 / 5.0;
            rig.ribbons[i * 6 + j] = vec3(
                lane * (1.0 + t * 3.5) + (c.animation.z * 1.6 - t * 5.0 + lane).sin() * t,
                (-1.5 - (t * 3.0).sin() * 1.6) * 3.0,
                t * (length + 12.0),
            )
            .extend(0.0);
        }
    }
}

/// A taxon's rank in the tree of life, from all life at the root down to a single species.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    /// All life.
    Root,
    /// Animals or plants.
    Kingdom,
    /// A body plan.
    Phylum,
    /// A major branch of a body plan.
    Class,
    /// A way of life within a class.
    Order,
    /// Close kin sharing a way of life.
    Family,
    /// Closer kin still, near enough to share a name.
    Genus,
    /// One kind of organism, which breeds with its own.
    Species,
}

#[derive(Clone, Copy)]
pub struct Taxon {
    pub name: &'static str,
    pub english_name: &'static str,
    pub description: &'static str,
    pub rank: Rank,
    pub parent: Option<&'static Self>,
    /// The parts of the body plan that arose in this taxon, or that it changed or lost; see
    /// [`organs::plan`].
    #[cfg(not(target_arch = "spirv"))]
    pub anatomy: &'static [organs::Part],
}

impl Taxon {
    /// Whether this taxon is `ancestor` or descends from it.
    pub fn descends_from(&self, ancestor: &Self) -> bool {
        let mut taxon = Some(self);
        while let Some(current) = taxon {
            if current.name == ancestor.name {
                return true;
            }
            taxon = current.parent;
        }
        false
    }
}

/// Taxonomic modules inherit their parent from the enclosing module, and with it everything the
/// parent has. Write each description
/// sentence as a separate string; they are joined with single spaces at compile time.
macro_rules! taxon {
    ($rank:ident, $name:expr, $english:expr, [$first:literal $(, $rest:literal)* $(,)?], anatomy: $anatomy:expr $(,)?) => {
        taxon!(@make $rank, $name, $english, concat!($first $(, " ", $rest)*), $anatomy);
    };
    ($rank:ident, $name:expr, $english:expr, [$first:literal $(, $rest:literal)* $(,)?] $(,)?) => {
        taxon!(@make $rank, $name, $english, concat!($first $(, " ", $rest)*), &[]);
    };
    ($rank:ident, $name:expr, $english:expr, $description:expr, anatomy: $anatomy:expr $(,)?) => {
        taxon!(@make $rank, $name, $english, $description, $anatomy);
    };
    ($rank:ident, $name:expr, $english:expr, $description:expr $(,)?) => {
        taxon!(@make $rank, $name, $english, $description, &[]);
    };
    (@make $rank:ident, $name:expr, $english:expr, $description:expr, $anatomy:expr) => {
        taxon!(@inherit $rank);
        pub const TAXON: $crate::scene::life::Taxon = $crate::scene::life::Taxon {
            name: $name,
            english_name: $english,
            description: $description,
            rank: $crate::scene::life::Rank::$rank,
            parent: taxon!(@parent $rank),
            #[cfg(not(target_arch = "spirv"))]
            anatomy: $anatomy,
        };
    };
    // Every taxon below the root takes all its parent has, as its own unless it defines its own:
    // its biology, physiology and ethology, its genes and organs, its shared code. A taxon names
    // only what changed where it arose.
    (@inherit Root) => {};
    (@inherit $rank:ident) => {
        pub use super::*;
    };
    (@parent Root) => { None };
    (@parent $rank:ident) => { Some(&super::TAXON) };
}

pub mod animalia;
#[cfg(not(target_arch = "spirv"))]
pub mod behaviour;
pub mod biology;
pub mod development;
pub mod genome;
#[cfg(not(target_arch = "spirv"))]
pub mod organs;
#[cfg(not(target_arch = "spirv"))]
pub mod physics;
pub mod plantae;
pub mod surface;
#[cfg(not(target_arch = "spirv"))]
pub mod vitals;

pub const BIOLOGY: biology::Biology = biology::ANCESTRAL;

taxon!(
    Root,
    "Vita",
    "Life",
    [
        "Life above the aur ocean depends on energy and moisture rising through a world without solid ground.",
        "Plants gather dispersed aur, its scattered light, wind deformation or electrical discharges, while animals obtain food from the resulting living tissues.",
        "Buoyancy, drag, powered flight and attachment keep organisms within usable air from the sunburst-lit poles, through the sunset band, to the equator lit only by the aur's glow from below.",
        "Water and minerals arrive in dangerous geyser-fed currents, making the edges of those plumes valuable habitats even when the jets themselves are lethal.",
        "Evolution repeatedly favours water conservation, secure living attachments and ways to avoid losing nutrients or young into the ocean below.",
    ],
);

macro_rules! species {
    (
        tetralates { $($t:ident: $($tp:ident)::+;)* }
        blood_feeders { $($b:ident: $($bp:ident)::+;)* }
        bubblebacks { $($g:ident: $($gp:ident)::+;)* }
        plants { $($plant:ident: $($plant_path:ident)::+;)* }
    ) => {
        species!(@all $($t: $($tp)::+;)* $($b: $($bp)::+;)* $($g: $($gp)::+;)* $($plant: $($plant_path)::+;)*);
        impl Kind {
            /// The kingdom the species belongs to.
            pub const fn kingdom(self) -> Kingdom {
                match self {
                    $(Self::$t => Kingdom::Animalia,)*
                    $(Self::$b => Kingdom::Animalia,)*
                    $(Self::$g => Kingdom::Animalia,)*
                    $(Self::$plant => Kingdom::Plantae,)*
                }
            }
            /// The instance its lineage draws it with.
            pub const fn layout(self) -> u32 {
                layout_of(self as u32)
            }
            /// How an animal behaves, as it inherits it down the tree of life; plants have none.
            #[cfg(not(target_arch = "spirv"))]
            pub const fn ethology(self) -> Option<animalia::mind::Ethology> {
                match self {
                    $(Self::$t => Some($($tp)::+::ETHOLOGY),)*
                    $(Self::$b => Some($($bp)::+::ETHOLOGY),)*
                    $(Self::$g => Some($($gp)::+::ETHOLOGY),)*
                    $(Self::$plant => None,)*
                }
            }
            /// How an animal's body works, as it inherits it down the tree of life.
            #[cfg(not(target_arch = "spirv"))]
            pub const fn animal(self) -> Option<animalia::physiology::Physiology> {
                match self {
                    $(Self::$t => Some($($tp)::+::PHYSIOLOGY),)*
                    $(Self::$b => Some($($bp)::+::PHYSIOLOGY),)*
                    $(Self::$g => Some($($gp)::+::PHYSIOLOGY),)*
                    $(Self::$plant => None,)*
                }
            }
            /// How a plant's body works, as it inherits it down the tree of life.
            #[cfg(not(target_arch = "spirv"))]
            pub const fn plant(self) -> Option<plantae::physiology::Physiology> {
                match self {
                    $(Self::$t => None,)*
                    $(Self::$b => None,)*
                    $(Self::$g => None,)*
                    $(Self::$plant => Some($($plant_path)::+::PHYSIOLOGY),)*
                }
            }
        }
        /// The instance a species is drawn with, by its index: [`TETRALATE`], [`BLOOD_FEEDER`],
        /// [`BUBBLEBACK`] or [`PLANT_LAYOUT`].
        pub const fn layout_of(kind: u32) -> u32 {
            $(if kind == Kind::$t as u32 { return TETRALATE; })*
            $(if kind == Kind::$b as u32 { return BLOOD_FEEDER; })*
            $(if kind == Kind::$g as u32 { return BUBBLEBACK; })*
            PLANT_LAYOUT
        }
        /// The body of a tetralate of species `kind`. When `kind` is known as the shader is built,
        /// as it is for each species' own pipeline, only that species' body is compiled in.
        pub fn tetralate_surface(kind: u32, point: Vec3, c: &animalia::tetralata::Tetralate) -> Surface {
            $(if kind == Kind::$t as u32 { return $($tp)::+::surface(point, c); })*
            Surface::new(1e6, 0)
        }
        /// The body of a blood-feeder of species `kind`; see [`tetralate_surface`].
        pub fn blood_feeder_surface(kind: u32, point: Vec3, c: &animalia::sanguinivora::BloodFeeder) -> Surface {
            $(if kind == Kind::$b as u32 { return $($bp)::+::surface(point, c); })*
            Surface::new(1e6, 0)
        }
        /// The organs of a gas-bag of species `kind`; see [`tetralate_surface`].
        pub fn bubbleback_surface(kind: u32, point: Vec3, c: &animalia::bullidorsata::Bubbleback) -> Surface {
            $(if kind == Kind::$g as u32 { return $($gp)::+::surface(point, c); })*
            Surface::new(1e6, 0)
        }
        /// The clear envelope of a gas-bag of species `kind`.
        pub fn bubbleback_envelope(kind: u32, point: Vec3, c: &animalia::bullidorsata::Bubbleback) -> Surface {
            $(if kind == Kind::$g as u32 { return $($gp)::+::envelope(point, c); })*
            Surface::new(1e6, 0)
        }
        /// The body of a plant of species `kind`; see [`tetralate_surface`].
        pub fn plant_surface(kind: u32, point: Vec3, c: &plantae::Plant) -> Surface {
            $(if kind == Kind::$plant as u32 { return $($plant_path)::+::surface(point, c); })*
            Surface::new(1e6, 0)
        }
        /// Pose a newly made instance as its species poses itself.
        #[cfg(not(target_arch = "spirv"))]
        fn animate(drawn: &mut Drawn) {
            match drawn {
                Drawn::Tetralate(c) => {
                    $(if c.kind.x == Kind::$t as u32 { $($tp)::+::animate(c); })*
                }
                Drawn::BloodFeeder(c) => {
                    $(if c.kind.x == Kind::$b as u32 { $($bp)::+::animate(c); })*
                }
                Drawn::Bubbleback(c) => {
                    $(if c.kind.x == Kind::$g as u32 { $($gp)::+::animate(c); })*
                }
                Drawn::Plant(c) => {
                    $(if c.kind.x == Kind::$plant as u32 { $($plant_path)::+::animate(c); })*
                }
            }
        }
    };
    (@all $($id:ident: $($path:ident)::+;)*) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Kind { $($id),* }
        pub const KINDS: &[Kind] = &[$(Kind::$id),*];
        impl Kind {
            pub const fn render_radius(self) -> f32 {
                match self { $(Self::$id => $($path)::+::RENDER_RADIUS * $($path)::+::BIOLOGY.size,)* }
            }
            pub const fn biology(self) -> biology::Biology {
                match self { $(Self::$id => $($path)::+::BIOLOGY,)* }
            }
            pub const fn look(self) -> Look {
                match self { $(Self::$id => $($path)::+::LOOK,)* }
            }
            /// Where the species' organs lie in its body: its lineage's body plan, laid out in
            /// its own build.
            #[cfg(not(target_arch = "spirv"))]
            pub fn organs(self) -> &'static organs::Viscera {
                organs::anatomy(self)
            }
            /// The parts of the body plan that are the species' own.
            #[cfg(not(target_arch = "spirv"))]
            pub const fn parts(self) -> &'static [organs::Part] {
                match self { $(Self::$id => $($path)::+::PARTS,)* }
            }
            /// The species' place in the tree of life.
            #[cfg(not(target_arch = "spirv"))]
            pub const fn taxon(self) -> &'static Taxon {
                match self { $(Self::$id => &$($path)::+::TAXON,)* }
            }
        }
        /// The look of a species by its index.
        pub const fn look_of(kind: u32) -> Look {
            match kind { $(kind if kind == Kind::$id as u32 => $($path)::+::LOOK,)* _ => Look::OWN }
        }
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES: &[&Taxon] = &[$(&$($path)::+::TAXON),*];
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES_MODULES: &[&str] = &[$(stringify!($($path)::+)),*];
        #[cfg(not(target_arch = "spirv"))]
        pub fn behave(kind: Kind, senses: &behaviour::Senses) -> behaviour::Controls {
            behaviour::organism(senses, kind)
        }
    };
}

/// The two kingdoms of life, which part at the root of the tree: animals, which eat and move and
/// decide, and plants, which make their own food and drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kingdom {
    /// Animals, which eat what other life makes.
    Animalia,
    /// Plants, which make their own food from light, aur, wind or lightning.
    Plantae,
}

species! {
    tetralates {
        Lucerna: animalia::tetralata::specialia::detritivora::lucernidae::lucerna::undulata;
        Manduca: animalia::tetralata::specialia::detritivora::manducidae::manduca::gregaria;
        Sericifex: animalia::tetralata::specialia::rapaciformes::sericificidae::sericifex::textor;
        Scintilla: animalia::tetralata::volitantia::parvivolantia::scintillidae::scintilla::scintillans;
        Nebulaphora: animalia::tetralata::volitantia::parvivolantia::nebulaphoridae::nebulaphora::vitrea;
        Ramiphaga: animalia::tetralata::volitantia::lithoptera::ramiphagidae::ramiphaga::corticivora;
        Lithosoma: animalia::tetralata::volitantia::lithoptera::lithosomatidae::lithosoma::grave;
        Purgator: animalia::tetralata::specialia::detritivora::manducidae::purgator::mutualis;
        Sputator: animalia::tetralata::specialia::rapaciformes::sericificidae::sputator::acidus;
        Aculeorhynchus: animalia::tetralata::vivipara::ferrilaminata::aculeorhynchidae::aculeorhynchus::acuminatus;
        Crepusculofalx: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::crepusculofalx::venatrix;
        Ferrispina: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::ferrispina::falcata;
        Herbivolus: animalia::tetralata::vivipara::grandivora::herbivolidae::herbivolus::migratorius;
        Umbravola: animalia::tetralata::vivipara::grandivora::herbivolidae::umbravola::tacita;
        Custos: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::ferrispina::custos;
        Naetu: animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::naetu;
        Antiquus: animalia::tetralata::vivipara::grandivora::vocigenidae::vocigenus::antiquus;
        Nae: animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::nae;
    }
    blood_feeders {
        Haematozoon: animalia::sanguinivora::errantia::errantiformes::haematozoidae::haematozoon::invasor;
        Adhaerens: animalia::sanguinivora::errantia::errantiformes::adhaerentidae::adhaerens::tenax;
        Fluitans: animalia::sanguinivora::errantia::errantiformes::adhaerentidae::fluitans::crypticus;
        Ambigena: animalia::sanguinivora::volantia::dissimuliformes::ambigenidae::ambigena::fallax;
        Aerohirudo: animalia::sanguinivora::volantia::volucriformes::aerohirudinidae::aerohirudo::membranacea;
        Ptilohirudo: animalia::sanguinivora::volantia::volucriformes::aerohirudinidae::ptilohirudo::pennata;
        Lucinani: animalia::sanguinivora::volantia::dissimuliformes::ambigenidae::candela::lucens;
    }
    bubblebacks {
        Aerogigas: animalia::bullidorsata::caelomorpha::aerogigantiformes::aerogigantidae::aerogigas::auratus;
        Levitans: animalia::bullidorsata::caelomorpha::nubivaga::levitantidae::levitans::taeniatus;
        Nubivesica: animalia::bullidorsata::caelomorpha::nubivaga::nubivesicidae::nubivesica::caerulea;
    }
    plants {
        Auracoccus: plantae::pneumatophyta::micrococcopsida::auracoccales::auracoccaceae::auracoccus::communis;
        Luminiflora: plantae::pneumatophyta::micrococcopsida::luminiflorales::luminifloraceae::luminiflora::symbiotica;
        Aerothallus: plantae::pneumatophyta::macrophyllopsida::aerothallales::aerothallaceae::aerothallus::vittatus;
        Laevivelum: plantae::pneumatophyta::macrophyllopsida::aerothallales::laevivelaceae::laevivelum::laeve;
        Lacufolium: plantae::pneumatophyta::macrophyllopsida::aerothallales::laevivelaceae::lacufolium::concavum;
        Flexilora: plantae::pneumatophyta::macrophyllopsida::flexilorales::flexiloraceae::flexilora::spiralis;
        Chromacorallum: plantae::pneumatophyta::macrophyllopsida::flexilorales::chromacorallaceae::chromacorallum::versicolor;
        Aurocorallum: plantae::pneumatophyta::macrophyllopsida::flexilorales::chromacorallaceae::aurocorallum::aureum;
        Spinivinea: plantae::vasculophyta::lignopsida::lignoradicales::spinivineaceae::spinivinea::volitans;
        Stelloradix: plantae::vasculophyta::lignopsida::lignoradicales::stelloradicaceae::stelloradix::pendula;
        Susurrus: plantae::vasculophyta::lignopsida::susurrales::susurraceae::susurrus::tubulatus;
        Tempestivaga: plantae::vasculophyta::lignopsida::tempestivagales::tempestivagaceae::tempestivaga::fulgurivora;
        Polliniflora: plantae::vasculophyta::floropsida::polliniflorales::pollinifloraceae::polliniflora::nectarifera;
        Gigantiflora: plantae::vasculophyta::floropsida::polliniflorales::pollinifloraceae::gigantiflora::gigantea;
        Pomum: plantae::vasculophyta::floropsida::pomales::pomaceae::pomum::iridescens;
        Aeromuscus: plantae::chlorophyta::muscopsida::aeromuscales::aeromuscaceae::aeromuscus::velutinus;
    }
}

/// How a species is coloured and what thin surfaces it carries.
///
/// It is read by both its preview and the world renderer: a family of materials, a base pigment
/// (with its glow), an accent, and up to two wing pairs.
#[derive(Clone, Copy)]
pub struct Look {
    pub family: u32,
    pub pigment: Vec4,
    pub accent: Vec3,
    pub wings: [Vec4; 2],
    /// Where each wing pair's shoulder is: out on the flank and up it, so the wing turns there
    /// rather than about the body's midline.
    pub hinges: [Vec2; 2],
    /// Each wing pair's stalk: how much of the chord its root keeps where it meets the shoulder,
    /// and how far the wing twists about its spar through each stroke, in radians.
    pub stalks: [Vec2; 2],
    /// Each wing pair's style; see [`surface::PLAIN_WING`].
    pub styles: [Vec4; 2],
    /// How heavy each wing pair's frame of bones is: one for a working flyer, much less for wings
    /// carried on fine veins alone.
    pub frames: [f32; 2],
    /// The body's skin: its kind of relief (zero for plain wrinkled skin), how deep and how fine
    /// that relief is, and how much duller (positive) or glossier than plain skin.
    pub skin: Vec4,
    /// How the wings beat: strokes a second, how deep, how far the hand flexes at the wrist and
    /// how far its motion lags the arm's.
    pub flight: Vec4,
    /// The membranes' markings (kind, density, strength) and how far out along the span the wrist
    /// lies; and the markings' second colour.
    pub pattern: Vec4,
    pub second: Vec3,
    /// Where a parent carries its brood, in its own frame: the height and station along the body
    /// of the band of eggs glued round it, or of the pouch on its back, and that band's or
    /// pouch's half width and half height.
    pub brood: Vec4,
    /// What the wings' frames are made of: bone (1), or iron (6) for hunters that cut with their
    /// wing edges.
    pub edge: u32,
}

/// What a parent is carrying, in `kind.z`: a clutch of eggs glued round its body, young in its
/// pouch, or a pouch with no young in it.
pub const CLUTCH: u32 = 1;
pub const FULL_POUCH: u32 = 2;
pub const EMPTY_POUCH: u32 = 3;

/// Animals in a living coat, with membrane or feathered wings.
pub const COAT: u32 = 0;

/// A tetralate's wing stalk unless its species says otherwise: a root a third of the chord wide,
/// twisting a third of a radian through the stroke.
pub const STALK: Vec2 = Vec2::new(0.33, 0.35);

/// Plants, in bark, leaf and glowing tissue.
pub const PLANT: u32 = 1;

/// Drifting gas-bags: gel round their organs inside a clear envelope.
pub const DRIFT: u32 = 2;

/// The player and their mount, drawn by their own tracers.
pub const OWN: u32 = 3;

impl Look {
    pub const fn coat(pigment: Vec4, accent: Vec3, wings: [Vec4; 2]) -> Self {
        Self {
            family: COAT,
            pigment,
            accent,
            wings,
            hinges: [Vec2::ZERO; 2],
            stalks: [STALK; 2],
            styles: [surface::PLAIN_WING; 2],
            frames: [1.0; 2],
            skin: Vec4::new(0.0, 1.0, 1.0, 0.0),
            flight: Vec4::new(0.7, 0.3, 0.35, 1.0),
            pattern: Vec4::new(0.0, 4.0, 0.5, 0.42),
            second: Vec3::new(0.2, 0.15, 0.12),
            brood: Vec4::ZERO,
            edge: 1,
        }
    }

    /// The same look, with its wing pairs hinged at these shoulders.
    pub const fn hinged(self, hinges: [Vec2; 2]) -> Self {
        Self { hinges, ..self }
    }

    /// The same look, with its wing pairs on these stalks; see [`Look::stalks`].
    pub const fn stalked(self, stalks: [Vec2; 2]) -> Self {
        Self { stalks, ..self }
    }

    /// The same look, with its wing pairs in these styles.
    pub const fn styled(self, styles: [Vec4; 2]) -> Self {
        Self { styles, ..self }
    }

    /// The same look, in this skin; see [`Look::skin`].
    pub const fn skinned(self, skin: Vec4) -> Self {
        Self { skin, ..self }
    }

    /// The same look, with its wings beating this way; see [`Look::flight`].
    pub const fn flying(self, flight: Vec4) -> Self {
        Self { flight, ..self }
    }

    /// The same look, with its membranes marked this way in this second colour;
    /// see [`Look::pattern`].
    pub const fn marked(self, pattern: Vec4, second: Vec3) -> Self {
        Self {
            pattern,
            second,
            ..self
        }
    }

    /// The same look, carrying its brood here; see [`Look::brood`].
    pub const fn brooding(self, brood: Vec4) -> Self {
        Self { brood, ..self }
    }

    /// The same look, its wings' frames made of this; see [`Look::edge`].
    pub const fn edged(self, edge: u32) -> Self {
        Self { edge, ..self }
    }

    /// The same look, with its wings' frames this heavy.
    pub const fn framed(self, frames: [f32; 2]) -> Self {
        Self { frames, ..self }
    }

    pub const fn plant(pigment: Vec3, glow: Vec3) -> Self {
        Self {
            family: PLANT,
            pigment: Vec4::new(pigment.x, pigment.y, pigment.z, 0.0),
            accent: glow,
            wings: [Vec4::ZERO; 2],
            hinges: [Vec2::ZERO; 2],
            stalks: [STALK; 2],
            styles: [surface::PLAIN_WING; 2],
            frames: [1.0; 2],
            skin: Vec4::new(0.0, 1.0, 1.0, 0.0),
            flight: Vec4::new(0.7, 0.3, 0.35, 1.0),
            pattern: Vec4::new(0.0, 4.0, 0.5, 0.42),
            second: Vec3::new(0.2, 0.15, 0.12),
            brood: Vec4::ZERO,
            edge: 1,
        }
    }

    /// A gas-bag's tint and the scale of its body.
    pub const fn drift(tint: Vec3, scale: f32) -> Self {
        Self {
            family: DRIFT,
            pigment: Vec4::new(tint.x, tint.y, tint.z, scale),
            accent: tint,
            wings: [Vec4::ZERO; 2],
            hinges: [Vec2::ZERO; 2],
            stalks: [STALK; 2],
            styles: [surface::PLAIN_WING; 2],
            frames: [1.0; 2],
            skin: Vec4::new(0.0, 1.0, 1.0, 0.0),
            flight: Vec4::new(0.7, 0.3, 0.35, 1.0),
            pattern: Vec4::new(0.0, 4.0, 0.5, 0.42),
            second: Vec3::new(0.2, 0.15, 0.12),
            brood: Vec4::ZERO,
            edge: 1,
        }
    }

    pub const OWN: Self = Self {
        family: OWN,
        pigment: Vec4::ZERO,
        accent: Vec3::ZERO,
        wings: [Vec4::ZERO; 2],
        hinges: [Vec2::ZERO; 2],
        stalks: [STALK; 2],
        styles: [surface::PLAIN_WING; 2],
        frames: [1.0; 2],
        skin: Vec4::new(0.0, 1.0, 1.0, 0.0),
        flight: Vec4::new(0.7, 0.3, 0.35, 1.0),
        pattern: Vec4::new(0.0, 4.0, 0.5, 0.42),
        second: Vec3::new(0.2, 0.15, 0.12),
        brood: Vec4::ZERO,
        edge: 1,
    };
}

/// How many of the world's creatures are drawn at once, besides the player and their mount: those
/// that cover the most of the screen.
pub const HERD: usize = 128;

/// The instance each lineage is drawn with, as [`layout_of`] tells them apart: a plant, a gas-bag,
/// a blood-feeder or a tetralate. The world keeps one buffer of each.
pub const PLANT_LAYOUT: u32 = 0;
pub const BUBBLEBACK: u32 = 1;
pub const BLOOD_FEEDER: u32 = 2;
pub const TETRALATE: u32 = 3;

/// An egg or a spore, drawn as such in its own materials: an egg's shell is not an adult's fin.
fn egg(ray: Ray, c: &CreatureInstance) -> Hit {
    let mut hit = development::render_juvenile(ray, c).opaque;
    hit.transmittance = Vec3::ZERO;
    hit
}

/// A winged animal's solid body in its coat, its wings' frames grown from its flanks.
fn coated<C: Flier, F: Fn(Vec3, &C) -> Surface>(kind: u32, ray: Ray, c: &C, surface: F) -> Hit {
    let look = look_of(kind);
    let grain = (c.organism().position.w * 0.15).max(0.7);
    let blunted = c.animal().hide.y;
    let body = move |p: Vec3, c: &C| with_wings(surface(p, c), p, c, look);
    let mut hit = trace_material(ray, c, body, move |p, id| {
        let mut m = coat(p / grain, id, look);
        // Grown iron is as bright as its last honing left it: keen and polished on an animal that
        // hones often, dull and scratched on one worn by its life.
        if id == 11 {
            m.roughness = (m.roughness * (0.45 + blunted * 1.2)).clamp(0.08, 0.9);
        }
        m
    });
    hit.transmittance = Vec3::ZERO;
    hit
}

/// The solid body of a tetralate of species `kind`: the player and their mount by their own
/// tracers, every other in its coat.
pub fn render_tetralate(kind: u32, ray: Ray, c: &Tetralate) -> Hit {
    if c.life.w > 0.0 {
        return egg(ray, c);
    }
    if kind == Kind::Nae as u32 {
        return nae::render_opaque(ray, c);
    }
    if kind == Kind::Naetu as u32 {
        return naetu::render_opaque(ray, c);
    }
    coated(kind, ray, c, move |p, c| tetralate_surface(kind, p, c))
}

/// The solid body of a blood-feeder of species `kind`.
pub fn render_blood_feeder(kind: u32, ray: Ray, c: &BloodFeeder) -> Hit {
    if c.life.w > 0.0 {
        return egg(ray, c);
    }
    coated(kind, ray, c, move |p, c| blood_feeder_surface(kind, p, c))
}

/// The organs of a gas-bag of species `kind`, in their gel.
pub fn render_bubbleback(kind: u32, ray: Ray, c: &Bubbleback) -> Hit {
    if c.life.w > 0.0 {
        return egg(ray, c);
    }
    let look = look_of(kind);
    let mut hit = trace_material(
        ray,
        c,
        move |p, c| bubbleback_surface(kind, p, c),
        move |p, _| gel(p, look.pigment.w * 0.45, look.accent),
    );
    hit.transmittance = Vec3::ZERO;
    hit
}

/// The solid body of a plant of species `kind`, bent as the air bends it.
pub fn render_plant(kind: u32, ray: Ray, c: &Plant) -> Hit {
    if c.life.w > 0.0 {
        return egg(ray, c);
    }
    let look = look_of(kind);
    let mut hit = trace_material(
        ray,
        c,
        move |p, c| {
            let mut hit = plant_surface(kind, plantae::bent(p, c), c);
            if c.tether.w > 0.0 {
                hit.join(plantae::silk(p, c.tether), plantae::SILK);
            }
            hit
        },
        move |p, id| {
            if id == plantae::SILK {
                plantae::silk_material()
            } else {
                plant_material(p, id, look.pigment.truncate(), look.accent)
            }
        },
    );
    hit.transmittance = Vec3::ZERO;
    hit
}

/// The player or their mount along a ray, by their own tracers alone, so a pass that draws only
/// them does not carry every species' body.
pub fn render_player(ray: Ray, c: &Tetralate) -> Hit {
    if c.kind.x == Kind::Nae as u32 {
        nae::render_opaque(ray, c)
    } else {
        naetu::render_opaque(ray, c)
    }
}

/// The thin surfaces of the player or their mount; see [`render_player`].
pub fn visit_player_thin<F: FnMut(Hit)>(ray: Ray, c: &Tetralate, emit: F) {
    if c.kind.x == Kind::Nae as u32 {
        nae::visit_thin(ray, c, emit);
    } else {
        naetu::visit_thin(ray, c, emit);
    }
}

/// The thin, see-through surfaces of a tetralate along a ray: its wing membranes.
pub fn visit_tetralate_thin<F: FnMut(Hit)>(kind: u32, ray: Ray, c: &Tetralate, emit: F) {
    if c.life.w > 0.0 {
        return;
    }
    if kind == Kind::Nae as u32 || kind == Kind::Naetu as u32 {
        visit_player_thin(ray, c, emit);
    } else {
        visit_wings(ray, c, look_of(kind), emit);
    }
}

/// The wing membranes of a blood-feeder along a ray.
pub fn visit_blood_feeder_thin<F: FnMut(Hit)>(kind: u32, ray: Ray, c: &BloodFeeder, emit: F) {
    if c.life.w <= 0.0 {
        visit_wings(ray, c, look_of(kind), emit);
    }
}

/// A gas-bag's clear envelope along a ray, tinted by its gel.
pub fn visit_bubbleback_thin<F: FnMut(Hit)>(kind: u32, ray: Ray, c: &Bubbleback, mut emit: F) {
    if c.life.w > 0.0 {
        return;
    }
    let look = look_of(kind);
    let tint = look.accent;
    let mut hit = trace_material(
        ray,
        c,
        move |p, c| bubbleback_envelope(kind, p, c),
        move |p, _| {
            let mut skin = gel(p, look.pigment.w, tint);
            skin.transmission = (-(Vec3::ONE - tint) * 0.2).exp() * 0.9;
            skin
        },
    );
    if hit.is_hit() {
        hit.transmittance = (-(Vec3::ONE - tint) * 0.2).exp() * 0.9;
        hit.part = 2;
        emit(hit);
    }
}

pub fn ellipsoid(p: Vec3, radius: Vec3) -> f32 {
    let exterior = (p.abs() - radius).max_element();
    if exterior > radius.min_element() * 0.2 {
        return exterior;
    }
    ((p / radius).length() - 1.0) * radius.min_element()
}

/// A rounded ellipsoid whose distance stays smooth all around it.
///
/// It is for parts blended into one another: `ellipsoid` switches to a box bound away from the
/// surface, whose corners blending and soft shadows pick up as facets. Slightly overstated near the
/// ends of long ones, so it is scaled back.
pub fn rounded(p: Vec3, radius: Vec3) -> f32 {
    let k0 = (p / radius).length();
    let k1 = (p / (radius * radius)).length();
    if k1 < 1e-6 {
        return -radius.min_element();
    }
    k0 * (k0 - 1.0) / k1 * 0.85
}

pub fn capsule(p: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let axis = b - a;
    (p - a - axis * ((p - a).dot(axis) / axis.length_squared()).saturate()).length() - radius
}

pub fn tapered_tail(p: Vec3, a: Vec3, b: Vec3, start: f32, end: f32) -> f32 {
    let axis = b - a;
    let fraction = ((p - a).dot(axis) / axis.length_squared()).saturate();
    ((p - a - axis * fraction).length() - start.lerp(end, fraction)) * 0.95
}

/// A ringed body along an axis from `a` to `b`.
///
/// Its radius swells from `start` to its broadest, `middle`, a third of the way along, then narrows
/// to `end`; it is ringed by `rings` shallow grooves `groove` deep as a share of the radius, as
/// worms, leeches and the abdomens of small flyers are. Its cross-section is squashed to `flat` of
/// its width vertically.
pub fn annulated(
    p: Vec3,
    a: Vec3,
    b: Vec3,
    radii: Vec3,
    rings: f32,
    groove: f32,
    flat: f32,
) -> f32 {
    let axis = b - a;
    let t = ((p - a).dot(axis) / axis.length_squared()).saturate();
    let swell = if t < 0.35 {
        radii.x.lerp(radii.y, (t / 0.35).smoothstep(0.0, 1.0))
    } else {
        radii
            .y
            .lerp(radii.z, ((t - 0.35) / 0.65).smoothstep(0.0, 1.0))
    };
    let ring = (t * rings * TAU).cos() * 0.5 + 0.5;
    let radius = swell * (1.0 - groove * ring * ring);
    let offset = p - a - axis * t;
    let squashed = vec3(offset.x, offset.y / flat, offset.z);
    (squashed.length() - radius) * flat.min(1.0) * 0.8
}

/// A sucking disc facing along `facing` from `centre`: a thick rim around a shallow cup.
pub fn sucker(p: Vec3, centre: Vec3, facing: Vec3, radius: f32) -> f32 {
    let q = p - centre;
    let along = q.dot(facing);
    let across = (q - facing * along).length();
    let rim = vec2(across - radius * 0.78, along).length() - radius * 0.24;
    let disc = ellipsoid(
        vec3(across, along + radius * 0.25, 0.0),
        vec3(radius, radius * 0.4, radius),
    );
    let cup = ellipsoid(
        vec3(across, along - radius * 0.12, 0.0),
        vec3(radius * 0.72, radius * 0.32, radius * 0.72),
    );
    smooth_min(rim, disc.max(-cup), radius * 0.15)
}

/// Small domed eyes, a pair to each of `rows` along the head, as leeches and many-eyed grazers
/// carry them.
pub fn eyespots(p: Vec3, first: Vec3, spacing: f32, rows: u32, radius: f32) -> f32 {
    let q = vec3(p.x.abs(), p.y, p.z);
    let mut d = 1e6f32;
    for i in 0..rows {
        let f = i as f32;
        d = d.min(
            (q - first - vec3(-f * spacing * 0.15, -f * spacing * 0.1, f * spacing)).length()
                - radius * (1.0 - f * 0.12),
        );
    }
    d
}

pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).saturate();
    b.lerp(a, h) - k * h * (1.0 - h)
}

/// A tetralate body, shaped by a life spent gliding in updrafts with no ground to stand on.
///
/// A lifting body, flattened and widest at the wing roots, is drawn out towards the tail, with gas
/// stored in a soft ridge along the spine, clear of the breathing passages. There is no neck or
/// skull: the forebody spreads into a broad, thin cowl whose rim cleaves the oncoming air and whose
/// underside funnels drifting food back to a slit mouth beneath it. Behind the cowl, slanted
/// louvres open onto the pleated breathing passages, ventilated by flight itself and wringing the
/// water back out of every breath.
pub fn torso(p: Vec3, size: Vec3, head: f32) -> f32 {
    let along = (p.z / size.z).clamp(-1.0, 1.0);
    let girth = 1.0 + 0.12 * (1.0 - (along + 0.3).abs() * 1.6).max(-0.6);
    let body = rounded(
        p - vec3(0.0, p.z * 0.03, 0.0),
        size * vec3(1.12 * girth, 0.82 * girth, 1.0),
    );
    let ridge = capsule(
        p,
        vec3(0.0, size.y * 0.6, -size.z * 0.55),
        vec3(0.0, size.y * 0.4, size.z * 0.7),
        size.y * 0.2,
    );
    let body = smooth_min(body, ridge, size.y * 0.35);
    let d = smooth_min(body, cowl(p, size.z, head), 0.9 * head + size.y * 0.1);
    louvres(slit(d, p, size.z, head), p, size)
}

/// The cowl ahead of a tetralate body `length` long: broad and thin, its rim lifting towards the
/// edges, a little dihedral for stability.
pub fn cowl(p: Vec3, length: f32, head: f32) -> f32 {
    let q = p - cowl_centre(length, head);
    let lift = q.x * q.x / (2.2 * head).power(2) * 0.35 * head;
    rounded(q - vec3(0.0, lift, 0.0), vec3(2.2, 0.55, 2.0) * head)
}

/// A body with the slit of its mouth cut beneath the cowl.
pub fn slit(body: f32, p: Vec3, length: f32, head: f32) -> f32 {
    let slit = rounded(
        p - mouth(length, head) - vec3(0.0, -0.02 * head, 0.0),
        vec3(1.1, 0.12, 0.35) * head,
    );
    body.max(-slit)
}

/// A body with its breathing louvres set into each flank behind the cowl.
///
/// Over the pleated breathing passages lies a patch of overlapping blades, each rising gently from
/// front to back to a lip that overhangs the slit behind it, as a louvred shutter does, so that air
/// is drawn in under the lips and the damp breath let out the same way is wrung dry against the
/// blades.
///
/// The blades follow the body's own surface rather than cutting into it: the slits run only a
/// little way in, they bow and slant with the flank, and the whole patch fades out at its edges.
pub fn louvres(body: f32, p: Vec3, size: Vec3) -> f32 {
    let flank = vec3(p.x.abs(), p.y, p.z);
    let reach = size.z * 0.25;
    let up = flank.y / (size.y * 0.42);
    // Along the patch, slanted back towards the top and bowed, so the slits curve with the flank.
    let along = (flank.z + size.z * 0.5 + flank.y * 0.35 + up * up * reach * 0.12) / reach;
    let patch = 1.0 - along * along - up * up;
    let thickness = size.x.min(size.y);
    if patch <= 0.0 || body > thickness * 0.3 {
        return body;
    }
    let outward = ((flank.x / size.x - 0.2) / 0.4).clamp(0.0, 1.0);
    let mask = patch.min(0.35) / 0.35 * outward * outward * (3.0 - 2.0 * outward);
    let count = 5.0;
    let phase = f32::midpoint(along, 1.0) * count;
    let blade = phase - phase.floor();
    let spacing = reach * 2.0 / count;
    // Each blade swells towards its rear lip, then drops into the next slit.
    let swell = blade * blade * (3.0 - 2.0 * blade) * spacing * 0.22 * mask;
    let shingled = body - swell;
    let slit = (1.0 - blade).min(blade * 3.0) * spacing;
    let width = spacing * 0.14 * mask;
    let depth = spacing * 0.7 * mask;
    let groove = (slit - width).max(-(body + depth));
    shingled.max(-groove)
}

/// The centre of a tetralate's cowl, ahead of a body `length` long.
fn cowl_centre(length: f32, head: f32) -> Vec3 {
    vec3(0.0, 0.25 * head, -length - 0.9 * head)
}

/// Where a tetralate's slit mouth opens beneath its cowl, and its mouth tentacles with it.
pub fn mouth(length: f32, head: f32) -> Vec3 {
    cowl_centre(length, head) + vec3(0.0, -0.55 * head, 0.25 * head)
}

pub fn tail<C: Rigged>(p: Vec3, creature: &C) -> f32 {
    let c = creature.rig();
    let mut d = 1e6f32;
    for i in 0..5 {
        let t = (i + 1) as f32 / 5.0;
        d = smooth_min(
            d,
            tapered_tail(
                p,
                c.tail[i].truncate(),
                c.tail[i + 1].truncate(),
                1.1 * (1.2 - t) + 0.16,
                1.1 * (1.0 - t) + 0.16,
            ),
            0.35,
        );
    }
    d
}

pub fn feathers<C: Rigged>(p: Vec3, creature: &C, width: f32) -> f32 {
    let c = creature.rig();
    let mut distance = 1e6f32;
    for index in 1..6 {
        let reach = width * (1.0 - index as f32 / 5.0 * 0.65);
        let knot = c.tail[index].truncate();
        let point = vec3(
            (p.x - knot.x).abs() - reach * 0.5,
            p.y - knot.y + 0.2,
            p.z - knot.z - 2.3,
        );
        let swept = vec3(
            point.x * 0.8 - point.z * 0.6,
            point.y,
            point.x * 0.6 + point.z * 0.8,
        );
        // A shaft along its length, thinning to the tip; a vane either side of it, widest at the
        // middle and arched across its width; and the tip curling down under the air's load.
        let t = ((swept.z + 4.5) / 9.0).clamp(0.0, 1.0);
        let half = reach * 0.32 * (t * (1.0 - t) * 4.0).sqrt().max(0.08);
        let droop = swept.y + 0.35 * t * t;
        let arch = -0.12 * half * (1.0 - (swept.x / half).power(2)).max(0.0);
        let vane = (swept.x.abs() - half)
            .max((droop - arch).abs() - 0.035)
            .max(swept.z.abs() - 4.5)
            * 0.6;
        let shaft = (vec2(swept.x, droop).length() - (0.09 - 0.07 * t)).max(swept.z.abs() - 4.5);
        distance = distance.min(vane.min(shaft));
    }
    distance
}

/// Eyes for open air lit from below.
///
/// A row of small eyes around the cowl's rim, from ahead round to behind, watches every bearing at
/// once, for in open sky a threat can come from anywhere and there is no ground to put at one's
/// back; a larger pair beneath looks down at the glowing aur, where both the light and the
/// deadliest danger lie.
pub fn eyes(p: Vec3, length: f32, head: f32) -> f32 {
    let q = vec3(p.x.abs(), p.y, p.z);
    let centre = cowl_centre(length, head);
    let mut d = 1e6f32;
    for i in 0..4 {
        let angle = 0.5 + i as f32 * 0.42;
        let x = angle.sin() * 2.05 * head;
        // On the rim's edge, bulging out past it, so each sees level and down as well as up.
        let rim = centre
            + vec3(
                x * 1.0,
                x * x / (2.2 * head).power(2) * 0.35 * head,
                -angle.cos() * 2.0 * head,
            );
        d = d.min((q - rim).length() - 0.11 * head);
    }
    let below = centre + vec3(0.68 * head, -0.62 * head, 0.35 * head);
    d.min((q - below).length() - 0.21 * head)
}

/// Gravity and the air streaming past, both in a creature's own frame, the air at no less than
/// `least`: it flies head first, so the stream runs back along it at its airspeed.
pub fn air_frame(c: &CreatureInstance, least: f32) -> (Vec3, Vec3) {
    let turn = Quat::from_array(c.rotation.to_array()).conjugate();
    (
        turn * vec3(0.0, -GRAVITY, 0.0),
        vec3(0.0, 0.0, c.animation.w.max(least)),
    )
}

/// A pose simulated up to `end`, smooth from one frame to the next though each is simulated afresh.
///
/// `settle` runs from two starts on a fixed grid `span` apart, and the later fades in as the
/// earlier ages out. Damping forgets anything older than a span, and in the world the clock runs
/// for hours. Returns the earlier pose, the later, and how far the later has faded in.
pub fn settled<T>(end: f32, span: f32, settle: impl Fn(f32) -> T) -> (T, T, f32) {
    let window = (end / span).floor();
    (
        settle(((window - 1.0) * span).max(0.0)),
        settle(window * span),
        end / span - window,
    )
}

/// Move a knot on by Verlet integration: on at its own `velocity` per step and by `force` over a
/// step `step` long.
pub fn swing(knot: &mut Vec3, velocity: Vec3, force: Vec3, step: f32) {
    *knot += velocity + force * step * step;
}

/// Hold a chain's knots `length` apart, the root fixed where it is tied and every other knot
/// sharing each correction with its neighbour.
pub fn tie<const N: usize>(knots: &mut [Vec3; N], length: f32) {
    for knot in 1..N {
        let delta = knots[knot] - knots[knot - 1];
        let fix = delta * (1.0 - length / delta.length().max(1e-6));
        if knot == 1 {
            knots[knot] -= fix;
        } else {
            knots[knot - 1] += fix * 0.5;
            knots[knot] -= fix * 0.5;
        }
    }
}

/// Pose trailing strands by simulating them.
///
/// `count` chains hang side by side from `root`, `spacing` apart and `length` long, passive but for
/// a little stiffness, swinging under gravity in the animal's own frame and streaming back in the
/// air it flies through.
pub fn rig_ribbons<C: Rigged>(
    creature: &mut C,
    count: usize,
    root: Vec3,
    spacing: f32,
    length: f32,
) {
    const STEP: f32 = 1.0 / 120.0;
    let (c, rig) = creature.rigging();
    let (gravity, flow) = air_frame(c, 2.0);
    let end = c.animation.z.max(0.0);
    let segment = length / 5.0;
    for chain in 0..count.min(5) {
        let lane = chain as f32 - (count as f32 - 1.0) * 0.5;
        let anchor = root + vec3(lane * spacing, 0.0, 0.0);
        let settle = |start: f32| {
            let mut points = [Vec3::ZERO; 6];
            for (knot, point) in points.iter_mut().enumerate() {
                *point = anchor + vec3(0.0, -0.4, 0.9) * (segment * knot as f32);
            }
            let mut previous = points;
            for step in 0..((end - start) / STEP) as usize {
                let time = start + step as f32 * STEP;
                let last = previous;
                previous = points;
                for knot in 1..6 {
                    let velocity = (points[knot] - last[knot]) * 0.99;
                    let gust = vec3(
                        (time * 1.1 + lane * 1.7 + knot as f32 * 0.6).sin(),
                        (time * 0.8 + lane).cos() * 0.5,
                        0.0,
                    ) * 2.0;
                    let drag = (flow + gust - velocity / STEP) * 0.8;
                    swing(&mut points[knot], velocity, gravity + drag, STEP);
                }
                tie(&mut points, segment);
                tie(&mut points, segment);
                points[0] = anchor;
            }
            points
        };
        let (earlier, later, weight) = settled(end, 2.0, settle);
        for knot in 0..6 {
            rig.ribbons[chain * 6 + knot] = earlier[knot]
                .lerp(later[knot], weight)
                .extend(length * 0.02);
        }
    }
    c.opaque_bounds = c
        .opaque_bounds
        .truncate()
        .max(root.abs() + Vec3::splat(length * 1.05 + spacing * count as f32))
        .extend(0.0);
}

/// Trailing strands posed by [`rig_ribbons`]: smooth, tapering, a little flattened, blended through
/// their knots.
pub fn ribbons<C: Rigged>(p: Vec3, creature: &C, count: usize) -> f32 {
    let c = creature.rig();
    let flatten = vec3(1.0, 1.6, 1.0);
    let mut distance = 1e6f32;
    for chain in 0..count {
        let width = c.ribbons[chain * 6].w.max(0.05);
        for knot in 0..5 {
            let from = c.ribbons[chain * 6 + knot].truncate();
            let to = c.ribbons[chain * 6 + knot + 1].truncate();
            let along = knot as f32 / 5.0;
            let start = width * (1.0 - along * 0.75);
            let end = width * (1.0 - (along + 0.2) * 0.75);
            distance = smooth_min(
                distance,
                tapered_tail(p * flatten, from * flatten, to * flatten, start, end) / 1.6,
                start * 0.8,
            );
        }
    }
    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::Quat;

    #[test]
    fn anisotropic_distance_bounds_keep_surfaces_and_signed_interiors() {
        let radius = vec3(20.0, 0.02, 2.0);
        assert!(ellipsoid(Vec3::ZERO, radius) < 0.0);
        for index in 1..100 {
            let direction = vec3(
                (index as f32 * 1.17).sin(),
                (index as f32 * 0.73).cos(),
                0.4,
            )
            .normalize();
            let surface = direction * radius;
            assert!(ellipsoid(surface, radius).abs() < 0.00001);
            for offset in [0.01, 0.1, 1.0, 10.0] {
                let normal = (direction / radius).normalize();
                let point = surface + normal * offset;
                let distance = ellipsoid(point, radius);
                assert!(
                    distance > 0.0 && distance <= offset + 0.0001,
                    "non-conservative ellipsoid distance {distance} > {offset}"
                );
            }
        }
        let point = vec3(30.0, 0.01, 0.0);
        let previous = ((point / radius).length() - 1.0) * radius.min_element();
        assert!(
            ellipsoid(point, radius) > previous * 1.5,
            "flat details must not force the old microscopic steps"
        );
    }

    #[test]
    fn species_have_complete_taxonomic_paths() {
        for (kind, taxon) in SPECIES.iter().enumerate() {
            assert_eq!(taxon.rank, Rank::Species, "renderable taxa must be species");
            let genus = taxon.parent.expect("species genus");
            assert_eq!(genus.rank, Rank::Genus);
            assert_eq!(genus.parent.expect("genus family").rank, Rank::Family);
            let epithet = if matches!(KINDS[kind], Kind::Nae | Kind::Naetu) {
                taxon.name.to_lowercase()
            } else {
                taxon
                    .name
                    .strip_prefix(&format!("{} ", genus.name))
                    .expect("species genus prefix")
                    .to_owned()
            };
            assert!(epithet.bytes().all(|letter| letter.is_ascii_lowercase()));
            let module = SPECIES_MODULES[kind].split_whitespace().collect::<String>();
            assert!(module.ends_with(&format!("::{}::{epithet}", genus.name.to_lowercase())));
            let mut ancestor = *taxon;
            loop {
                assert!(
                    !ancestor.description.trim().is_empty(),
                    "missing description for {}",
                    ancestor.name
                );
                let Some(parent) = ancestor.parent else { break };
                assert_eq!(
                    parent.rank as u8 + 1,
                    ancestor.rank as u8,
                    "missing taxonomic rank"
                );
                if ancestor.rank == Rank::Family {
                    let suffix = if module.starts_with("plantae::") {
                        "aceae"
                    } else {
                        "idae"
                    };
                    assert!(
                        ancestor.name.ends_with(suffix),
                        "invalid family suffix: {}",
                        ancestor.name
                    );
                }
                ancestor = parent;
            }
        }
    }

    #[test]
    fn every_species_renders_finite_hits_and_respects_depth_limits() {
        for (kind, taxon) in SPECIES.iter().enumerate() {
            let radius = KINDS[kind].render_radius();
            for panel in [0, 2, 3, 5, 8] {
                let c = Drawn::specimen(
                    Vec3::ZERO.extend(radius),
                    Vec4::W,
                    Vec4::ZERO,
                    kind as u32,
                    development::grid_age(KINDS[kind].biology(), panel),
                    53,
                );
                let origin = vec3(0.0, 0.0, -radius * 1.7);
                let mut hits = 0;
                for x in -5..=5 {
                    for y in -5..=5 {
                        let direction = vec3(x as f32 * 0.07, y as f32 * 0.07, 1.0).normalize();
                        let ray = Ray {
                            origin,
                            direction,
                            light: Vec3::Y,
                            max_distance: 1e6,
                            footprint: 0.001,
                            ambient: vec3(0.30, 0.36, 0.45),
                            radiance: Vec3::ONE,
                        };
                        let hit = c.render(ray).front();
                        if hit.is_hit() {
                            hits += 1;
                            assert!(
                                hit.color_depth.is_finite() && hit.normal.is_finite(),
                                "invalid hit for {}",
                                taxon.name
                            );
                            assert!((hit.normal.length() - 1.0).abs() < 0.001, "invalid normal");
                            assert!(
                                hit.distance() > 0.0 && hit.distance() < radius * 3.0,
                                "invalid depth"
                            );
                            assert!(
                                !c.render(Ray {
                                    max_distance: hit.distance() - 0.1,
                                    ..ray
                                })
                                .is_hit(),
                                "trace ignored the depth limit"
                            );
                        }
                    }
                }
                assert!(
                    hits > 0,
                    "empty render for {} at lifecycle panel {panel}",
                    taxon.name
                );
            }
        }
    }

    #[test]
    fn transformed_species_keep_depth_and_motion_consistent() {
        let radius = Kind::Naetu.render_radius();
        let rotation = Quat::from_rotation_y(0.7);
        let offset = vec3(100.0, -200.0, 300.0);
        let c = Drawn::new(
            offset.extend(radius),
            rotation.to_array().into(),
            Vec4::ZERO,
            Kind::Naetu as u32,
        );
        let ray = Ray {
            origin: offset + rotation * vec3(0.0, 0.0, -100.0),
            direction: rotation * Vec3::Z,
            light: Vec3::Y,
            max_distance: 1e6,
            footprint: 0.001,
            ambient: vec3(0.30, 0.36, 0.45),
            radiance: Vec3::ONE,
        };
        let hit = c.render(ray).front();
        assert!(hit.is_hit(), "central ray missed transformed Naetu");
        assert!(
            hit.previous_position
                .distance(ray.origin + ray.direction * hit.distance())
                < 0.001,
            "stationary specimen produced motion"
        );
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use std::{hint::black_box, time::Instant};

    /// How long tracing a grid of rays through a posed creature takes, in microseconds per ray.
    fn time(kind: Kind) -> f64 {
        let radius = kind.render_radius();
        let drawn = Drawn::specimen(
            Vec4::new(0.0, 0.0, 0.0, radius),
            Vec4::W,
            Vec4::new(0.0, 0.0, 1.5, 8.0),
            kind as u32,
            kind.biology().maturity_days * 1.5,
            7,
        );
        let mut rays = 0;
        let start = Instant::now();
        for y in 0..48 {
            for x in 0..48 {
                let aim = vec3(
                    (x as f32 / 47.0 - 0.5) * 1.6,
                    (y as f32 / 47.0 - 0.5) * 1.6,
                    -1.0,
                );
                let ray = Ray {
                    origin: Vec3::Z * radius * 2.5,
                    direction: aim.normalize(),
                    light: vec3(0.3, 1.0, 0.2).normalize(),
                    ambient: Vec3::splat(0.3),
                    radiance: Vec3::ONE,
                    max_distance: 1.0e6,
                    footprint: 0.002,
                };
                black_box(drawn.render(ray));
                rays += 1;
            }
        }
        start.elapsed().as_secs_f64() * 1.0e6 / f64::from(rays)
    }

    #[test]
    #[ignore = "a measurement, run by hand"]
    fn print_tracing_costs() {
        for &kind in KINDS {
            println!("{kind:?} {:.1} us/ray", time(kind));
        }
    }
}
