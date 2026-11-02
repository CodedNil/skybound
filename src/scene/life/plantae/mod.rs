use crate::scene::life::surface::RINGED;
use core::ops::{Deref, DerefMut};
pub mod chlorophyta;
#[cfg(not(target_arch = "spirv"))]
pub mod physiology;
pub mod pneumatophyta;
pub mod vasculophyta;
pub mod young;

/// What a plant adds to an organism: how far the air bends it, and how much water it holds.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Plant {
    pub organism: CreatureInstance,
    /// How far the moving air bends it, and whether that was set at all.
    pub bend: Vec4,
    pub previous_bend: Vec4,
    /// How full of water it is, as a Pondplant's basin shows.
    pub water: Vec4,
    /// Where the silk tying it to another plant reaches, in its own frame, and how thick the silk
    /// is; nothing if it is tied to none.
    pub tether: Vec4,
}

impl Deref for Plant {
    type Target = CreatureInstance;
    fn deref(&self) -> &CreatureInstance {
        &self.organism
    }
}

impl DerefMut for Plant {
    fn deref_mut(&mut self) -> &mut CreatureInstance {
        &mut self.organism
    }
}

impl Body for Plant {
    fn organism(&self) -> &CreatureInstance {
        &self.organism
    }
}

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Kingdom,
    "Plantae",
    "Plants",
    [
        "Plants build organic tissue using dispersed aur, scattered aur light, wind deformation or storm electricity.",
        "Hygroscopic surfaces recover airborne moisture, while selective tissues retain minerals from dust and rain.",
        "Some remain aloft through low mass, gas chambers or drag surfaces, and others attach to living bodies that carry their weight.",
        "Larger vascular forms connect distant tissues through flexible channels, allowing flowers, woody frameworks and hanging roots to share resources.",
        "Spores, fragments and specialised seeds disperse through currents or on animal hosts, with no soil or fixed ground stage in their life cycles.",
    ],
);

/// One of an individual plant's inherited traits, from zero to one.
///
/// A plant has no mind or behaviour to tell it apart from its neighbours, only its growth, and
/// every spore or seed carries its parents' ways of growing: how many arms, how long a stem, which
/// way a coil turns. Plants read the whole of their kingdom's part of the genome for it, from the
/// eighth gene on, each trait at its own place, so a plant's traits vary independently of one
/// another and are passed on to its young.
pub fn gene(c: &CreatureInstance, place: f32) -> f32 {
    c.gene(Gene(8 + (place as u32) % 24))
}

/// One of `count` forms a trait can take, read from a gene.
pub fn pick(c: &CreatureInstance, salt: f32, count: u32) -> u32 {
    ((gene(c, salt) * count as f32) as u32).min(count - 1)
}

/// A whole number of parts from `fewest` to `most`, read from a gene.
pub fn count(c: &CreatureInstance, salt: f32, fewest: u32, most: u32) -> u32 {
    fewest + pick(c, salt, most - fewest + 1)
}

/// A number from zero to one fixed for each cell of a grid a plant's parts are scattered over, so
/// each tuft, thorn or bead in a repeated pattern is its own.
pub fn speckle(cell: Vec2, salt: f32) -> f32 {
    ((cell.dot(vec2(127.1, 311.7)) + salt * 17.3).sin() * 43_758.547).fract_gl()
}

/// A gripping pad facing `out`, `p` relative to its centre: a low dome with a cupped face, by which
/// a climbing or clinging plant holds on to bark, hide or another plant.
pub fn sucker_pad(p: Vec3, out: Vec3, radius: f32) -> f32 {
    let across = (p - out * p.dot(out)).length();
    let up = p.dot(out);
    let pad = (vec2(across, up) / vec2(radius, radius * 0.45)).length() - 1.0;
    pad * radius * 0.45
}

pub fn rotate(p: Vec3, angle: f32) -> Vec3 {
    let (sine, cosine) = angle.sin_cos();
    vec3(cosine * p.x - sine * p.z, p.y, sine * p.x + cosine * p.z)
}

pub fn leaf(p: Vec3, length: f32, width: f32, curl: f32) -> f32 {
    let t = (p.z / length).clamp(0.0, 1.0);
    let profile = (t * PI).sin().max(0.0).sqrt();
    let height = curl * t * t + (p.x / width * 5.0 + t * 9.0).sin() * width * 0.04 * profile;
    let lateral = p.x.abs() - width * profile;
    let axial = (-p.z).max(p.z - length);
    let thickness = (p.y - height).abs() - length * 0.003;
    lateral.max(axial).max(thickness) * 0.55
}

pub fn frond(p: Vec3, length: f32, width: f32, curl: f32, lobes: f32) -> f32 {
    let t = (p.z / length).clamp(0.0, 1.0);
    leaf(
        p,
        length,
        width * (0.77 + 0.23 * (t * lobes * TAU).cos()),
        curl,
    )
}

pub fn rosette(p: Vec3, petals: usize, length: f32, width: f32, curl: f32) -> f32 {
    let mut distance = 1e6f32;
    for i in 0..petals {
        distance = distance.min(leaf(
            rotate(p, i as f32 * TAU / petals as f32),
            length,
            width,
            curl,
        ));
    }
    distance
}

pub fn sway(mut p: Vec3, c: &CreatureInstance) -> Vec3 {
    p.y += c.life.z * p.z.abs() * 0.15;
    p.x -= (p.y * 0.4 + c.animation.z * 0.7).sin() * p.y.abs() * 0.045;
    p.z -= (p.y * 0.3 + c.animation.z * 0.5).cos() * p.y.abs() * 0.035;
    p
}

/// Where a point of the plant's unbent body lies once the air has bent it.
///
/// The bend the simulation found, as how far the top of the plant is pushed, grows from nothing at
/// its foot to all of it at its top, as a stem bends under a load at its tip.
pub fn bent(p: Vec3, c: &Plant) -> Vec3 {
    if c.bend.w <= 0.0 {
        return p;
    }
    let height = c.opaque_bounds.y.max(0.5);
    let up = ((p.y + height) / (2.0 * height)).clamp(0.0, 1.0);
    p - c.bend.truncate() * up * up
}

/// The material of the silk a Weaver ties plants together with.
pub const SILK: u32 = 63;

/// A silk cable from the middle of a plant out to `tether`, the far end, its thickness in `w`:
/// drawn three strands twisted together, sagging a little between its ends.
pub fn silk(p: Vec3, tether: Vec4) -> f32 {
    let end = tether.truncate();
    let length = end.length().max(1e-3);
    let middle = end * 0.5 - Vec3::Y * length * 0.04;
    let thick = tether.w;
    let mut cable = tapered_tail(p, Vec3::ZERO, middle, thick, thick).min(tapered_tail(
        p,
        middle,
        end,
        thick,
        thick,
    ));
    // The twist of its strands, a ridge running round it.
    let along = p.dot(end) / length;
    cable += (along / thick * 1.3 + p.x.atan2(p.y) * 3.0).sin() * thick * 0.08;
    cable
}

/// Weaver silk: pale, faintly lustrous and a little translucent.
pub const fn silk_material() -> Material {
    let mut m = Material::skin(vec3(0.82, 0.8, 0.72), 0.35);
    m.translucency = 0.35;
    m
}
/// Bark, leaf, glowing tissue, gloss and water: a plant's materials.
pub fn plant_material(p: Vec3, id: u32, pigment: Vec3, glow: Vec3) -> Material {
    let grain = (p.y * 31.0 + (p.x * 4.0).sin()).sin() * 0.04;
    let veins = (p.x * 8.0 + (p.z * 3.0).sin())
        .sin()
        .abs()
        .smoothstep(0.9, 0.98);
    let mut material = Material::skin(pigment * (0.8 + grain + veins * 0.2), 0.48);
    // Soft tissue: light passes through it, tinted by whatever colours it, and a waxy film glints
    // on it.
    material.scatter = 0.9;
    material.film = 0.35;
    match id {
        0 => {
            material.scatter = 0.1;
            material.film = 0.05;
            material.color = vec3(0.18, 0.085, 0.035) * (1.0 + grain);
            material.roughness = 0.82;
            material.texture = RINGED;
            material.relief = 0.5;
            material.grain = 0.8;
        }
        1 => {
            material.transmission = pigment.sqrt() * 0.28;
            material.ior = 1.38;
            material.translucency = 0.45;
        }
        2 => {
            material.color = glow;
            material.emission = 1.8;
            material.roughness = 0.3;
        }
        3 => {
            material.roughness = 0.22;
            material.color = pigment.lerp(glow, (p.y * 0.2 + 0.5).clamp(0.0, 1.0));
        }
        4 => {
            material.color = vec3(0.2, 0.5, 0.62);
            material.transmission = vec3(0.72, 0.88, 0.94);
            material.ior = 1.333;
            material.roughness = 0.07;
        }
        _ => {}
    }
    material
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{PLANT, Physiology};

    pub const PHYSIOLOGY: Physiology = PLANT;
}
