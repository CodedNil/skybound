use crate::scene::life::{
    animalia::{
        bullidorsata::caelomorpha::{
            aerogigantiformes::aerogigantidae::aerogigas::auratus,
            gel,
            nubivaga::{levitantidae::levitans::taeniatus, nubivesicidae::nubivesica::caerulea},
        },
        tetralata::vivipara::ferrilaminata::naeturidae::naeturus::{nae, naetu},
        visit_wings, with_wings,
    },
    plantae::plant_material,
    surface::{Hit, Ray, Surface, TraceResult, coat, trace_material},
};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Quat, UVec4, Vec2, Vec3, Vec4, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

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
    pub identity: Vec4, // Stable individual traits; seed zero is the reference specimen.
    pub proportions: Vec4, // Body scale and wing development.
    pub previous_proportions: Vec4,
    pub tail: [Vec4; 6],
    pub previous_tail: [Vec4; 6],
    pub ribbons: [Vec4; 30],
    pub cloth: [[[Vec4; 3]; 12]; 3],
    pub previous_cloth: [[[Vec4; 3]; 12]; 3],
}

#[cfg(not(target_arch = "spirv"))]
impl CreatureInstance {
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

    pub fn specimen(
        position: Vec4,
        rotation: Vec4,
        animation: Vec4,
        kind: u32,
        age_days: f32,
        seed: u32,
    ) -> Self {
        let mut c = Self {
            position,
            rotation,
            animation,
            previous_animation: animation,
            previous_position: position,
            previous_rotation: rotation,
            kind: UVec4::new(kind, 0, 0, 0),
            ..Self::default()
        };
        development::prepare(&mut c, KINDS[kind as usize].biology(), age_days, seed);
        c.previous_proportions = c.proportions;
        c.previous_animation = c.animation;
        prepare_instance(&mut c);
        // The bounds must hold the wings' frames at full stretch and stroke,
        // or tracing clips them.
        let look = KINDS[kind as usize].look();
        if look.family == COAT {
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
        // Eggs and newborns are round, whatever shape the adult's bounds,
        // and a string of eggs trails out long behind.
        let form = c.life.w as u32;
        if form >= 4 && form != 5 && form != 6 && form != 7 {
            let mean = c.opaque_bounds.truncate().element_sum() / 3.0;
            let length = if form == 9 {
                c.opaque_bounds.z.max(mean) * 1.5
            } else {
                mean
            };
            c.opaque_bounds = vec3(mean, mean, length).extend(0.0);
        }
        // A pouch is part of a grown parent's body, empty or not; and its
        // brood, wherever it is carried, must lie within its bounds.
        if look.brood.w > 0.0 && c.life.w <= 0.0 {
            if KINDS[kind as usize].biology().brood == biology::Brood::Pouch && c.kind.z == 0 {
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
        c.previous_tail = c.tail;
        c.position.w = c.position.w.max(c.opaque_bounds.truncate().length());
        c.position.w *= development::scale(&c).max_element().max(1.0);
        c
    }
}

fn prepare_pose(c: &mut CreatureInstance, length: f32, tail: f32, ribbons: usize) {
    let tail = tail * (0.35 + c.life.y * 0.65);
    for i in 0..6 {
        let t = i as f32 / 5.0;
        c.tail[i] = vec3(
            (c.animation.z * 2.0 - t * 4.0).sin() * t * 1.3,
            -t * 0.6,
            length * 0.6 + t * tail,
        )
        .extend(0.0);
    }
    for i in 0..ribbons {
        let lane = i as f32 - (ribbons as f32 - 1.0) * 0.5;
        c.ribbons[i * 6] = vec3(lane * 0.8, -3.9, 0.0).extend(0.0);
        for j in 1..6 {
            let t = j as f32 / 5.0;
            c.ribbons[i * 6 + j] = vec3(
                lane * (1.0 + t * 3.5) + (c.animation.z * 1.6 - t * 5.0 + lane).sin() * t,
                (-1.5 - (t * 3.0).sin() * 1.6) * 3.0,
                t * (length + 12.0),
            )
            .extend(0.0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Root,
    Kingdom,
    Phylum,
    Class,
    Order,
    Family,
    Genus,
    Species,
}

#[derive(Clone, Copy)]
pub struct Taxon {
    pub name: &'static str,
    pub english_name: &'static str,
    pub description: &'static str,
    pub rank: Rank,
    pub parent: Option<&'static Self>,
}

/// Taxonomic modules inherit their parent from the enclosing module.
/// Write each description sentence as a separate string; they are joined with
/// single spaces at compile time.
macro_rules! taxon {
    ($rank:ident, $name:expr, $english:expr, [$first:literal $(, $rest:literal)* $(,)?] $(,)?) => {
        taxon!($rank, $name, $english, concat!($first $(, " ", $rest)*));
    };
    ($rank:ident, $name:expr, $english:expr, $description:expr $(,)?) => {
        pub const TAXON: $crate::scene::life::Taxon = $crate::scene::life::Taxon {
            name: $name,
            english_name: $english,
            description: $description,
            rank: $crate::scene::life::Rank::$rank,
            parent: taxon!(@parent $rank),
        };
    };
    (@parent Root) => { None };
    (@parent $rank:ident) => { Some(&super::TAXON) };
}

pub mod anatomy;
pub mod animalia;
pub mod behaviour;
pub mod biology;
pub mod development;
pub mod physiology;
pub mod plantae;
pub mod surface;

pub const BIOLOGY: biology::Biology = biology::ANCESTRAL;

taxon!(
    Root,
    "Vita",
    "Life",
    [
        "Life above the aur ocean depends on energy and moisture rising through a world without solid ground.",
        "Plants gather dispersed aur, its scattered light, wind deformation or electrical discharges, while animals obtain food from the resulting living tissues.",
        "Buoyancy, drag, powered flight and attachment keep organisms within usable air as the forty-minute aur cycle changes light and lift.",
        "Water and minerals arrive in dangerous geyser-fed currents, making the edges of those plumes valuable habitats even when the jets themselves are lethal.",
        "Evolution repeatedly favours water conservation, secure living attachments and ways to avoid losing nutrients or young into the ocean below.",
    ],
);

macro_rules! species {
    ($($id:ident: $($path:ident)::+;)*) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Kind { $($id),* }
        pub const KINDS: &[Kind] = &[$(Kind::$id),*];
        impl Kind {
            pub const fn render_radius(self) -> f32 {
                match self { $(Self::$id => $($path)::+::RENDER_RADIUS,)* }
            }
            pub const fn biology(self) -> biology::Biology {
                match self { $(Self::$id => $($path)::+::BIOLOGY,)* }
            }
            pub const fn look(self) -> Look {
                match self { $(Self::$id => $($path)::+::LOOK,)* }
            }
        }
        /// The look of a species by its index.
        pub const fn look_of(kind: u32) -> Look {
            match kind { $(kind if kind == Kind::$id as u32 => $($path)::+::LOOK,)* _ => Look::OWN }
        }
        /// The body of any creature: one distance function switching on its
        /// species, so the world's tracer is compiled once for all of them.
        pub fn any_surface(point: Vec3, c: &CreatureInstance) -> Surface {
            if c.life.w > 0.0 { return development::juvenile_surface(point, c); }
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::surface(point, c),)* _ => Surface::new(1e6, 0) }
        }
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES: &[&Taxon] = &[$(&$($path)::+::TAXON),*];
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES_MODULES: &[&str] = &[$(stringify!($($path)::+)),*];
        pub fn behave(kind: Kind, senses: &behaviour::Senses) -> behaviour::Controls {
            behaviour::organism(senses, kind.biology())
        }
        pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
            if c.life.w > 0.0 { return development::render_juvenile(ray, c); }
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::render(ray, c),)* _ => TraceResult::MISS }
        }
        #[cfg(not(target_arch = "spirv"))]
        pub fn body_surface(point: Vec3, c: &CreatureInstance) -> crate::scene::life::surface::Surface {
            if c.life.w > 0.0 { return development::juvenile_surface(point, c); }
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::surface(point, c),)* _ => crate::scene::life::surface::Surface::new(1e6, 0) }
        }
        #[cfg(not(target_arch = "spirv"))]
        fn prepare_instance(c: &mut CreatureInstance) {
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::animate(c),)* _ => {} }
        }
    };
}
species! {
    Lucerna: animalia::tetralata::specialia::detritivora::lucernidae::lucerna::undulata;
    Manduca: animalia::tetralata::specialia::detritivora::manducidae::manduca::gregaria;
    Sericifex: animalia::tetralata::specialia::rapaciformes::sericificidae::sericifex::textor;
    Scintilla: animalia::tetralata::volitantia::parvivolantia::scintillidae::scintilla::scintillans;
    Nebulaphora: animalia::tetralata::volitantia::parvivolantia::nebulaphoridae::nebulaphora::vitrea;
    Ramiphaga: animalia::tetralata::volitantia::lithoptera::ramiphagidae::ramiphaga::corticivora;
    Lithosoma: animalia::tetralata::volitantia::lithoptera::lithosomatidae::lithosoma::grave;
    Aerogigas: animalia::bullidorsata::caelomorpha::aerogigantiformes::aerogigantidae::aerogigas::auratus;
    Levitans: animalia::bullidorsata::caelomorpha::nubivaga::levitantidae::levitans::taeniatus;
    Nubivesica: animalia::bullidorsata::caelomorpha::nubivaga::nubivesicidae::nubivesica::caerulea;
    Haematozoon: animalia::sanguinivora::errantia::errantiformes::haematozoidae::haematozoon::invasor;
    Adhaerens: animalia::sanguinivora::errantia::errantiformes::adhaerentidae::adhaerens::tenax;
    Fluitans: animalia::sanguinivora::errantia::errantiformes::adhaerentidae::fluitans::crypticus;
    Ambigena: animalia::sanguinivora::volantia::dissimuliformes::ambigenidae::ambigena::fallax;
    Aerohirudo: animalia::sanguinivora::volantia::volucriformes::aerohirudinidae::aerohirudo::membranacea;
    Ptilohirudo: animalia::sanguinivora::volantia::volucriformes::aerohirudinidae::ptilohirudo::pennata;
    Purgator: animalia::tetralata::specialia::detritivora::manducidae::purgator::mutualis;
    Sputator: animalia::tetralata::specialia::rapaciformes::sericificidae::sputator::acidus;
    Aculeorhynchus: animalia::tetralata::vivipara::ferrilaminata::aculeorhynchidae::aculeorhynchus::acuminatus;
    Crepusculofalx: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::crepusculofalx::venatrix;
    Ferrispina: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::ferrispina::falcata;
    Herbivolus: animalia::tetralata::vivipara::grandivora::herbivolidae::herbivolus::migratorius;
    Umbravola: animalia::tetralata::vivipara::grandivora::herbivolidae::umbravola::tacita;
    Lucinani: animalia::sanguinivora::volantia::dissimuliformes::ambigenidae::candela::lucens;
    Custos: animalia::tetralata::vivipara::ferrilaminata::ferrispinidae::ferrispina::custos;
    Naetu: animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::naetu;
    Antiquus: animalia::tetralata::vivipara::grandivora::vocigenidae::vocigenus::antiquus;
    Nae: animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::nae;
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

/// How a species is coloured and what thin surfaces it carries.
///
/// It is read by both its preview and the world renderer: a family of
/// materials, a base pigment (with its glow), an accent, and up to two wing
/// pairs.
#[derive(Clone, Copy)]
pub struct Look {
    pub family: u32,
    pub pigment: Vec4,
    pub accent: Vec3,
    pub wings: [Vec4; 2],
    /// Where each wing pair's shoulder is: out on the flank and up it, so
    /// the wing turns there rather than about the body's midline.
    pub hinges: [Vec2; 2],
    /// Each wing pair's style; see [`surface::PLAIN_WING`].
    pub styles: [Vec4; 2],
    /// How heavy each wing pair's frame of bones is: one for a working
    /// flyer, much less for wings carried on fine veins alone.
    pub frames: [f32; 2],
    /// The body's skin: its kind of relief (zero for plain wrinkled skin),
    /// how deep and how fine that relief is, and how much duller (positive)
    /// or glossier than plain skin.
    pub skin: Vec4,
    /// How the wings beat: strokes a second, how deep, how far the hand
    /// flexes at the wrist and how far its motion lags the arm's.
    pub flight: Vec4,
    /// The membranes' markings (kind, density, strength) and how far out
    /// along the span the wrist lies; and the markings' second colour.
    pub pattern: Vec4,
    pub second: Vec3,
    /// Where a parent carries its brood, in its own frame: the height and
    /// station along the body of the band of eggs glued round it, or of the
    /// pouch on its back, and that band's or pouch's half width and half
    /// height.
    pub brood: Vec4,
    /// What the wings' frames are made of: bone (1), or iron (6) for
    /// hunters that cut with their wing edges.
    pub edge: u32,
}

/// What a parent is carrying, in `kind.z`: a clutch of eggs glued round
/// its body, young in its pouch, or a pouch with no young in it.
pub const CLUTCH: u32 = 1;
pub const FULL_POUCH: u32 = 2;
pub const EMPTY_POUCH: u32 = 3;

/// Animals in a living coat, with membrane or feathered wings.
pub const COAT: u32 = 0;

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

    /// The same look, with its membranes marked this way in this second
    /// colour; see [`Look::pattern`].
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

/// How many of the world's creatures are drawn at once, besides the
/// player and their mount: those that cover the most of the screen.
pub const HERD: usize = 24;

/// The clear envelope of a drifting gas-bag, for the world renderer.
fn any_envelope(point: Vec3, c: &CreatureInstance) -> Surface {
    if c.kind.x == Kind::Aerogigas as u32 {
        auratus::envelope(point, c)
    } else if c.kind.x == Kind::Levitans as u32 {
        taeniatus::envelope(point, c)
    } else if c.kind.x == Kind::Nubivesica as u32 {
        caerulea::envelope(point, c)
    } else {
        Surface::new(1e6, 0)
    }
}

/// The solid surface of a creature along a ray, for the world's opaque pass.
///
/// The player and their mount are drawn by their own tracers; every other
/// species by one tracer over [`any_surface`], in its family's materials: the
/// same shapes and colours as its preview.
pub fn render_opaque(ray: Ray, c: &CreatureInstance) -> Hit {
    let look = look_of(c.kind.x);
    if look.family == OWN && c.life.w <= 0.0 {
        return if c.kind.x == Kind::Nae as u32 {
            nae::render_opaque(ray, c)
        } else {
            naetu::render_opaque(ray, c)
        };
    }
    let grain = (c.position.w * 0.15).max(0.7);
    let body = |p: Vec3, c: &CreatureInstance| {
        let surface = any_surface(p, c);
        if look.family == COAT {
            with_wings(surface, p, c, look)
        } else {
            surface
        }
    };
    let mut hit = trace_material(ray, c, body, |p, id| {
        if look.family == PLANT {
            plant_material(p, id, look.pigment.truncate(), look.accent)
        } else if look.family == DRIFT {
            gel(p, look.pigment.w * 0.45, look.accent)
        } else {
            coat(p / grain, id, look)
        }
    });
    hit.transmittance = Vec3::ZERO;
    hit
}

/// The thin, see-through surfaces of a creature along a ray, such as wing
/// membranes and gas envelopes, for the transparent pass.
pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, c: &CreatureInstance, mut emit: F) {
    let look = look_of(c.kind.x);
    if look.family == OWN && c.life.w <= 0.0 {
        if c.kind.x == Kind::Nae as u32 {
            nae::visit_thin(ray, c, emit);
        } else {
            naetu::visit_thin(ray, c, emit);
        }
        return;
    }
    if look.family == DRIFT {
        let tint = look.accent;
        let mut hit = trace_material(ray, c, any_envelope, |p, _| {
            let mut skin = gel(p, look.pigment.w, tint);
            skin.transmission = (-(Vec3::ONE - tint) * 0.2).exp() * 0.9;
            skin
        });
        if hit.is_hit() {
            hit.transmittance = (-(Vec3::ONE - tint) * 0.2).exp() * 0.9;
            hit.part = 2;
            emit(hit);
        }
        return;
    }
    if c.life.w <= 0.0 {
        visit_wings(ray, c, look, emit);
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
/// It is for parts blended into one another: `ellipsoid` switches to a box
/// bound away from the surface, whose corners blending and soft shadows pick up
/// as facets. Slightly overstated near the ends of long ones, so it is scaled
/// back.
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
/// Its radius swells from `start` to its broadest, `middle`, a third of the way
/// along, then narrows to `end`; it is ringed by `rings` shallow grooves
/// `groove` deep as a share of the radius, as worms, leeches and the abdomens
/// of small flyers are. Its cross-section is squashed to `flat` of its width
/// vertically.
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

/// A sucking disc facing along `facing` from `centre`: a thick rim around a
/// shallow cup.
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

/// Small domed eyes, a pair to each of `rows` along the head, as leeches
/// and many-eyed grazers carry them.
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

/// A tetralate body, shaped by a life spent gliding in updrafts with no ground
/// to stand on.
///
/// A lifting body, flattened and widest at the wing roots, is drawn out towards
/// the tail, with gas stored in a soft ridge along the spine, clear of the
/// breathing passages. There is no neck or skull: the forebody spreads into a
/// broad, thin cowl whose rim cleaves the oncoming air and whose underside
/// funnels drifting food back to a slit mouth beneath it. Behind the cowl,
/// slanted louvres open onto the pleated breathing passages, ventilated by
/// flight itself and wringing the water back out of every breath.
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

/// The cowl ahead of a tetralate body `length` long: broad and thin, its
/// rim lifting towards the edges, a little dihedral for stability.
pub fn cowl(p: Vec3, length: f32, head: f32) -> f32 {
    let q = p - cowl_centre(length, head);
    let lift = q.x * q.x / (2.2 * head).powi(2) * 0.35 * head;
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

/// A body with slanted louvres cut into its flanks behind the cowl, over
/// the pleated breathing passages.
pub fn louvres(body: f32, p: Vec3, size: Vec3) -> f32 {
    let mut d = body;
    let flank = vec3(p.x.abs(), p.y, p.z);
    for i in 0..5 {
        let z = -size.z * 0.72 + i as f32 * size.z * 0.11;
        let slant = flank.z - z - flank.y * 0.35;
        let louvre = (slant.abs() - size.z * 0.016)
            .max(flank.y.abs() - size.y * 0.42)
            .max(size.x * 0.55 - flank.x);
        d = d.max(-louvre);
    }
    d
}

/// The centre of a tetralate's cowl, ahead of a body `length` long.
fn cowl_centre(length: f32, head: f32) -> Vec3 {
    vec3(0.0, 0.25 * head, -length - 0.9 * head)
}

/// Where a tetralate's slit mouth opens beneath its cowl, and its mouth
/// tentacles with it.
pub fn mouth(length: f32, head: f32) -> Vec3 {
    cowl_centre(length, head) + vec3(0.0, -0.55 * head, 0.25 * head)
}

pub fn tail(p: Vec3, c: &CreatureInstance) -> f32 {
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

pub fn feathers(p: Vec3, c: &CreatureInstance, width: f32) -> f32 {
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
        // A shaft along its length, thinning to the tip; a vane either side
        // of it, widest at the middle and arched across its width; and the
        // tip curling down under the air's load.
        let t = ((swept.z + 4.5) / 9.0).clamp(0.0, 1.0);
        let half = reach * 0.32 * (t * (1.0 - t) * 4.0).sqrt().max(0.08);
        let droop = swept.y + 0.35 * t * t;
        let arch = -0.12 * half * (1.0 - (swept.x / half).powi(2)).max(0.0);
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
/// A row of small eyes around the cowl's rim watches every bearing at once, for
/// in open sky a threat can come from anywhere; a larger pair beneath looks
/// down at the glowing aur, where both the light and the deadliest danger lie.
pub fn eyes(p: Vec3, length: f32, head: f32) -> f32 {
    let q = vec3(p.x.abs(), p.y, p.z);
    let centre = cowl_centre(length, head);
    let mut d = 1e6f32;
    for i in 0..3 {
        let angle = 0.55 + i as f32 * 0.38;
        let x = angle.sin() * 2.05 * head;
        // On the rim's edge, bulging out past it, so each sees level and
        // down as well as up.
        let rim = centre
            + vec3(
                x * 1.0,
                x * x / (2.2 * head).powi(2) * 0.35 * head,
                -angle.cos() * 2.0 * head,
            );
        d = d.min((q - rim).length() - 0.11 * head);
    }
    let below = centre + vec3(0.68 * head, -0.62 * head, 0.35 * head);
    d.min((q - below).length() - 0.21 * head)
}

pub fn gills(p: Vec3, length: f32, pairs: u32) -> f32 {
    let p = vec3(p.x.abs(), p.y, p.z);
    let mut d = 1e6f32;
    for i in 0..pairs {
        let f = i as f32;
        let root = vec3(1.4, 0.3 + f * 0.4, -length + f * 0.5);
        let tip = vec3(4.8 - f * 0.65, 1.0 + f * 1.25, -length + 3.0 + f * 0.6);
        d = d.min(capsule(p, root, tip, 0.18));
        for j in 1..4 {
            let branch = root.lerp(tip, j as f32 * 0.23);
            d = d.min(capsule(p, branch, branch + vec3(0.7, 0.8, 0.85), 0.12));
        }
    }
    d
}

/// Pose trailing strands by simulating them.
///
/// `count` chains hang side by side from `root`, `spacing` apart and `length`
/// long, passive but for a little stiffness, swinging under gravity in the
/// animal's own frame and streaming back in the air it flies through. Settled
/// from two starts on a fixed grid and crossfaded, so the pose moves smoothly
/// frame to frame.
pub fn rig_ribbons(c: &mut CreatureInstance, count: usize, root: Vec3, spacing: f32, length: f32) {
    const STEP: f32 = 1.0 / 120.0;
    const SETTLE: f32 = 2.0;
    let turn = Quat::from_array(c.rotation.to_array()).conjugate();
    let gravity = turn * vec3(0.0, -9.8, 0.0);
    let flow = vec3(0.0, 0.0, c.animation.w.max(2.0));
    let end = c.animation.z.max(0.0);
    let segment = length / 5.0;
    for chain in 0..count.min(5) {
        let lane = chain as f32 - (count as f32 - 1.0) * 0.5;
        let anchor = root + vec3(lane * spacing, 0.0, 0.0);
        let settle = |start: f32| {
            let start = start.max(0.0);
            let mut points = [Vec3::ZERO; 6];
            for (knot, point) in points.iter_mut().enumerate() {
                *point = anchor + vec3(0.0, -0.4, 0.9) * (segment * knot as f32);
            }
            let mut previous = points;
            let steps = ((end - start) / STEP) as usize;
            for step in 0..steps {
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
                    points[knot] += velocity + (gravity * 0.3 + drag) * STEP * STEP;
                }
                for _ in 0..2 {
                    for knot in 1..6 {
                        let delta = points[knot] - points[knot - 1];
                        let fix = delta * (1.0 - segment / delta.length().max(1e-6));
                        if knot == 1 {
                            points[knot] -= fix;
                        } else {
                            points[knot - 1] += fix * 0.5;
                            points[knot] -= fix * 0.5;
                        }
                    }
                }
                points[0] = anchor;
            }
            points
        };
        let grid = (end / SETTLE).floor();
        let weight = end / SETTLE - grid;
        let earlier = settle((grid - 1.0) * SETTLE);
        let later = settle(grid * SETTLE);
        for knot in 0..6 {
            c.ribbons[chain * 6 + knot] = earlier[knot]
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

/// Trailing strands posed by [`rig_ribbons`]: smooth, tapering, a little
/// flattened, blended through their knots.
pub fn ribbons(p: Vec3, c: &CreatureInstance, count: usize) -> f32 {
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
                let c = CreatureInstance::specimen(
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
                        let hit = render(ray, &c).front();
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
                                !render(
                                    Ray {
                                        max_distance: hit.distance() - 0.1,
                                        ..ray
                                    },
                                    &c
                                )
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
        let c = CreatureInstance::new(
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
        let hit = render(ray, &c).front();
        assert!(hit.is_hit(), "central ray missed transformed Naetu");
        assert!(
            hit.previous_position
                .distance(ray.origin + ray.direction * hit.distance())
                < 0.001,
            "stationary specimen produced motion"
        );
    }
}
