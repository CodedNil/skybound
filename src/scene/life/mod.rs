use spirv_std::glam::{FloatExt, UVec4, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[cfg(target_arch = "spirv")]
use crate::scene::surface::Hit;
use crate::scene::surface::Ray;
#[cfg(not(target_arch = "spirv"))]
use crate::scene::surface::TraceResult;

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

pub mod anatomy;
pub mod animalia;
pub mod behaviour;
pub mod biology;
pub mod development;
pub mod physiology;
pub const BIOLOGY: biology::Biology = biology::ANCESTRAL;
pub mod plantae;

pub const TAXON: Taxon = Taxon {
    name: "Vita",
    english_name: "Life",
    description: "Life above the aur ocean is sustained by rising energy, moisture and wind. \
        Organisms occupy changing layers of the sky, exchanging nutrients through grazing, decomposition, parasitism and mutual dependence; buoyancy and attachment are as important as active flight.",
    rank: Rank::Root,
    parent: None,
};

macro_rules! species {
    ($($id:ident: $($path:ident)::+;)*) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Kind { $($id),* }
        pub const KINDS: &[Kind] = &[$(Kind::$id),*];
        impl Kind {
            pub const fn render_radius(self) -> f32 {
                match self { $(Self::$id => $($path)::+::RENDER_RADIUS,)* }
            }
            pub const fn biology(self) -> biology::Biology {
                match self { $(Self::$id => $($path)::+::BIOLOGY,)* }
            }
        }
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES: &[&Taxon] = &[$(&$($path)::+::TAXON),*];
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES_MODULES: &[&str] = &[$(stringify!($($path)::+)),*];
        pub fn behave(kind: Kind, senses: &behaviour::Senses) -> behaviour::Controls {
            behaviour::organism(senses, kind.biology())
        }
        #[cfg(not(target_arch = "spirv"))]
        pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
            if c.life.w > 0.0 { return development::render_juvenile(ray, c); }
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::render(ray, c),)* _ => TraceResult::MISS }
        }
        #[cfg(not(target_arch = "spirv"))]
        pub fn body_surface(point: Vec3, c: &CreatureInstance) -> crate::scene::surface::Surface {
            if c.life.w > 0.0 { return development::juvenile_surface(point, c); }
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::surface(point, c),)* _ => crate::scene::surface::Surface::new(1e6, 0) }
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

#[cfg(target_arch = "spirv")]
pub fn render_opaque(ray: Ray, c: &CreatureInstance) -> Hit {
    if c.kind.x == Kind::Nae as u32 {
        animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::nae::render_opaque(
            ray, c,
        )
    } else {
        animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::naetu::render_opaque(
            ray, c,
        )
    }
}

pub fn ellipsoid(p: Vec3, radius: Vec3) -> f32 {
    ((p / radius).length() - 1.0) * radius.min_element()
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

pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).saturate();
    b.lerp(a, h) - k * h * (1.0 - h)
}

pub fn torso(p: Vec3, size: Vec3, head: f32) -> f32 {
    let body = ellipsoid(p, size);
    let skull = ellipsoid(
        p - vec3(0.0, 0.65 * head, -size.z),
        vec3(1.9, 1.5, 2.4) * head,
    );
    let snout = ellipsoid(
        p - vec3(0.0, 0.5 * head, -size.z - 1.7 * head),
        vec3(1.7, 0.95, 1.3) * head,
    );
    smooth_min(smooth_min(body, skull, 0.7), snout, 0.5)
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
        distance = distance.min(ellipsoid(swept, vec3(reach * 0.32, 0.12, 4.5)));
    }
    distance
}

pub fn eyes(p: Vec3, length: f32, head: f32) -> f32 {
    ellipsoid(
        vec3(p.x.abs(), p.y, p.z) - vec3(1.52 * head, 1.12 * head, -length - 1.2 * head),
        vec3(0.30, 0.32, 0.38) * head,
    )
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

pub fn ribbons(p: Vec3, c: &CreatureInstance, count: usize) -> f32 {
    let mut d = 1e6f32;
    for i in 0..count {
        for j in 0..5 {
            let t = (j + 1) as f32 / 5.0;
            d = d.min(
                capsule(
                    p * vec3(1.0, 3.0, 1.0),
                    c.ribbons[i * 6 + j].truncate(),
                    c.ribbons[i * 6 + j + 1].truncate(),
                    0.5 * (1.0 - t * 0.75),
                ) / 3.0,
            );
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use spirv_std::glam::Quat;

    use super::*;

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

#[cfg(target_arch = "spirv")]
pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, c: &CreatureInstance, emit: F) {
    if c.kind.x == Kind::Nae as u32 {
        animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::nae::visit_thin(
            ray, c, emit,
        );
    } else {
        animalia::tetralata::vivipara::ferrilaminata::naeturidae::naeturus::naetu::visit_thin(
            ray, c, emit,
        );
    }
}
