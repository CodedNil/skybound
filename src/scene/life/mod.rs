use crate::scene::surface::{Hit, Ray};
use spirv_std::glam::{FloatExt, UVec4, Vec3, Vec4, vec3};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CreatureInstance {
    pub position: Vec4,
    pub rotation: Vec4,
    pub animation: Vec4,
    pub previous_position: Vec4,
    pub previous_rotation: Vec4,
    pub kind: UVec4,
    pub tail: [Vec4; 6],
    pub ribbons: [Vec4; 30],
}

#[cfg(not(target_arch = "spirv"))]
impl CreatureInstance {
    pub fn new(position: Vec4, rotation: Vec4, animation: Vec4, kind: u32) -> Self {
        let mut c = Self {
            position,
            rotation,
            animation,
            previous_position: position,
            previous_rotation: rotation,
            kind: UVec4::new(kind, 0, 0, 0),
            ..Self::default()
        };
        prepare_instance(&mut c);
        c
    }
}

#[cfg(not(target_arch = "spirv"))]
fn prepare_pose(c: &mut CreatureInstance, length: f32, tail: f32, ribbons: usize) {
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

#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rank {
    Root,
    Phylum,
    Class,
    Order,
    Family,
    Genus,
    Species,
}

#[cfg(not(target_arch = "spirv"))]
#[derive(Clone, Copy)]
pub struct Taxon {
    pub name: &'static str,
    pub rank: Rank,
    pub specimen: &'static str,
    pub description: &'static str,
    pub diet: &'static str,
    pub species_count: u32,
    pub processes_aur: bool,
    pub radius: f32,
    pub parent: Option<&'static Self>,
}

#[cfg(not(target_arch = "spirv"))]
impl Taxon {
    pub const fn new(parent: &'static Self, name: &'static str, description: &'static str) -> Self {
        let rank = match parent.rank {
            Rank::Root => Rank::Phylum,
            Rank::Phylum => Rank::Class,
            Rank::Class => Rank::Order,
            Rank::Order => Rank::Family,
            Rank::Family => Rank::Genus,
            Rank::Genus | Rank::Species => Rank::Species,
        };
        Self {
            name,
            description,
            rank,
            parent: Some(parent),
            specimen: "",
            diet: "",
            species_count: 0,
            processes_aur: false,
            radius: 0.0,
        }
    }
    pub const fn ecology(
        mut self,
        diet: &'static str,
        species_count: u32,
        processes_aur: bool,
    ) -> Self {
        self.diet = diet;
        self.species_count = species_count;
        self.processes_aur = processes_aur;
        self
    }
    pub const fn specimen(mut self, specimen: &'static str, radius: f32) -> Self {
        self.specimen = specimen;
        self.radius = radius;
        self
    }
}

pub mod bullidorsata;
pub mod sanguivora;
pub mod tetralata;

#[cfg(not(target_arch = "spirv"))]
pub const TAXON: Taxon = Taxon {
    name: "Life",
    rank: Rank::Root,
    specimen: "",
    description: "",
    diet: "",
    species_count: 0,
    processes_aur: false,
    radius: 0.0,
    parent: None,
};

macro_rules! species {
    ($($id:ident: $($path:ident)::+;)*) => {
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Kind { $($id),* }
        #[cfg(not(target_arch = "spirv"))]
        pub const SPECIES: &[&Taxon] = &[$(&$($path)::+::TAXON),*];
        #[cfg(not(target_arch = "spirv"))]
        pub fn render(ray: Ray, c: &CreatureInstance) -> Hit {
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::render(ray, c),)* _ => Hit::MISS }
        }
        #[cfg(not(target_arch = "spirv"))]
        fn prepare_instance(c: &mut CreatureInstance) {
            match c.kind.x { $(kind if kind == Kind::$id as u32 => $($path)::+::animate(c),)* _ => {} }
        }
    };
}
species! {
    Lucentiae: tetralata::specialia::detritivora::lucentiae;
    Manducidae: tetralata::specialia::detritivora::manducidae;
    Textricinae: tetralata::specialia::rapaxina::textricinae;
    Luminiscidae: tetralata::volitilia::parvifuga::luminiscidae;
    Nebulaphoridae: tetralata::volitilia::parvifuga::nebulaphoridae;
    Ramiphagae: tetralata::volitilia::petrolata::ramiphagae;
    Terrapodae: tetralata::volitilia::petrolata::terrapodae;
    Aerogigans: bullidorsata::caeliformes::aerogigantiform::aerogigantidae::aerogigans;
    Levita: bullidorsata::caeliformes::nubidriftonidae::lentiformidae::levita;
    Nubiballona: bullidorsata::caeliformes::nubidriftonidae::leviformidae::nubiballona;
    Hematozoa: sanguivora::erratia::errantia::parasitidae::hematozoa;
    Adherans: sanguivora::erratia::errantia::tenacidae::adherans;
    Fluitans: sanguivora::erratia::errantia::tenacidae::fluitans;
    Ambigena: sanguivora::voluceria::dissimuliformes::mimicidae::ambigena;
    Aerohirudo: sanguivora::voluceria::voluceriformes::voluceridae::aerohirudo;
    Ptilohirudo: sanguivora::voluceria::voluceriformes::voluceridae::ptilohirudo;
    Purgatoridae: tetralata::specialia::detritivora::manducidae::purgatoridae;
    Sputorexidae: tetralata::specialia::rapaxina::textricinae::sputorexidae;
    Aculeorhynchus: tetralata::viviparia::ferrilaminata::aculeorhinidae::aculeorhynchus;
    Crepusculorbis: tetralata::viviparia::ferrilaminata::ferrivolvidae::crepusculorbis;
    Ferrispinae: tetralata::viviparia::ferrilaminata::ferrivolvidae::ferrispinae;
    Herbivolus: tetralata::viviparia::grandivoriformes::ruminavidae::herbivolus;
    Umbraevolidae: tetralata::viviparia::grandivoriformes::ruminavidae::umbraevolidae;
    Lucinani: sanguivora::voluceria::dissimuliformes::mimicidae::candelis::lucinani;
    Custos: tetralata::viviparia::ferrilaminata::ferrivolvidae::ferrispinae::custos;
    Naetu: tetralata::viviparia::ferrilaminata::naeturidae::naeturae::naetu;
    Antiquus: tetralata::viviparia::grandivoriformes::vocanathidae::vocanathus::antiquus;
}

#[cfg(target_arch = "spirv")]
pub fn render_player(ray: Ray, c: &CreatureInstance) -> Hit {
    tetralata::viviparia::ferrilaminata::naeturidae::naeturae::naetu::render(ray, c)
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
    let mut d = 1e6f32;
    for i in 1..6 {
        let w = width * (1.0 - i as f32 / 5.0 * 0.65);
        let knot = c.tail[i].truncate();
        d = d.min(ellipsoid(
            vec3(
                (p.x - knot.x).abs() - w * 0.55,
                p.y - knot.y,
                p.z - knot.z - 0.8,
            ),
            vec3(w, 0.16, 2.8),
        ));
    }
    d
}

pub fn legs(p: Vec3, size: Vec3, pairs: u32) -> f32 {
    let p = vec3(p.x.abs(), p.y, p.z);
    let mut d = 1e6f32;
    for i in 0..pairs {
        let root = vec3(size.x * 0.7, -size.y * 0.4, i as f32 * 1.8 - 1.8);
        let knee = root + vec3(2.0, -1.0, 1.0);
        d = d.min(capsule(p, root, knee, 0.22)).min(capsule(
            p,
            knee,
            knee + vec3(0.5, -1.5, -0.7),
            0.15,
        ));
    }
    d
}

pub fn eyes(p: Vec3, length: f32, head: f32) -> f32 {
    ellipsoid(
        vec3(p.x.abs(), p.y, p.z) - vec3(1.52 * head, 1.12 * head, -length - 1.2 * head),
        vec3(0.47, 0.48, 0.6) * head,
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
    use super::*;
    use spirv_std::glam::Quat;

    #[test]
    fn catalogue_is_connected_and_every_specimen_renders() {
        for (kind, taxon) in SPECIES.iter().enumerate() {
            let mut ancestor = *taxon;
            while let Some(parent) = ancestor.parent {
                assert!(parent.rank < ancestor.rank, "invalid taxonomy hierarchy");
                ancestor = parent;
            }
            let c = CreatureInstance::new(
                Vec3::ZERO.extend(taxon.radius),
                Vec4::W,
                Vec4::ZERO,
                kind as u32,
            );
            let origin = vec3(0.0, 0.0, -taxon.radius * 1.7);
            let mut hits = 0;
            for x in -5..=5 {
                for y in -5..=5 {
                    let direction = vec3(x as f32 * 0.07, y as f32 * 0.07, 1.0).normalize();
                    let ray = Ray {
                        origin,
                        direction,
                        light: Vec3::Y,
                        max_distance: 1e6,
                    };
                    let hit = render(ray, &c);
                    if hit.is_hit() {
                        hits += 1;
                        assert!(
                            hit.color_depth.is_finite() && hit.normal.is_finite(),
                            "invalid hit for {}",
                            taxon.name
                        );
                        assert!((hit.normal.length() - 1.0).abs() < 0.001, "invalid normal");
                        assert!(
                            hit.distance() > 0.0 && hit.distance() < taxon.radius * 3.0,
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
            assert!(hits > 0, "empty render for {}", taxon.name);
        }
    }

    #[test]
    fn transformed_species_keep_depth_and_motion_consistent() {
        let radius = SPECIES[Kind::Naetu as usize].radius;
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
        };
        let hit = render(ray, &c);
        assert!(hit.is_hit(), "central ray missed transformed Naetu");
        assert!(
            hit.previous_position
                .distance(ray.origin + ray.direction * hit.distance())
                < 0.001,
            "stationary specimen produced motion"
        );
    }
}
