#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::KINDS;
use crate::scene::life::{
    CreatureInstance, PLANT,
    animalia::{
        bullidorsata::{self, young::POLYP},
        sanguinivora,
        tetralata::{
            self,
            vivipara::grandivora::{self, young::POUCH_YOUNG},
            young::{CLUTCH, DRIFTING_EGG, EGG_STRINGS, TETHERED_EGG},
        },
    },
    biology::Biology,
    look_of,
    plantae::{
        self,
        young::{BUD, SPORE, SPROUT},
    },
    surface::{Material, Ray, SMOOTH, Surface, TraceResult, coat, trace_material},
};
use core::f32::consts::TAU;
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec4;
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const DAY: f32 = 86_400.0;

pub fn grid_age(biology: Biology, panel: usize) -> f32 {
    if panel < 6 {
        biology.maturity_days * panel as f32 / 5.0
    } else {
        biology.maturity_days
            + (biology.lifespan_days - biology.maturity_days) * (panel - 5) as f32 / 3.0
    }
}

#[cfg(not(target_arch = "spirv"))]
pub fn prepare(c: &mut CreatureInstance, biology: Biology, age_days: f32, seed: u32) {
    let random = |channel: u32| {
        if seed == 0 {
            return 0.5;
        }
        let mut value = seed ^ channel.wrapping_mul(0x9e37_79b9);
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        ((value ^ (value >> 16)) >> 8) as f32 / 16_777_215.0
    };
    let age = age_days.max(0.0);
    let young = (age / biology.maturity_days).clamp(0.0, 1.0);
    let old = age.smoothstep(biology.lifespan_days * 0.72, biology.lifespan_days);
    // Each kingdom, and each lineage within it, has its own young forms.
    let kind = KINDS[c.kind.x as usize];
    let stage = kind.young_form(young) as f32;
    c.kind.y = seed;
    c.life = vec4(age, young, old, stage);
    c.identity = vec4(random(1), random(2), random(3), random(4));
    let growth = (0.25 + young * 0.75) * (1.0 - old * 0.08) * biology.size;
    let variation = (c.identity.xyz() - 0.5) * 0.35 + 1.0;
    c.proportions = (variation
        * vec3(
            1.25 - young * 0.25,
            1.15 - young * 0.15,
            0.65 + young * 0.35,
        )
        * growth
        * vec3(
            1.0 + (c.animation.z * 1.4).sin() * 0.012,
            1.0 + (c.animation.z * 1.4).sin() * 0.02,
            1.0,
        ))
    .extend(young.smoothstep(0.55, 1.0));
    c.animation.x *= 1.0 - old * 0.45;
}

pub fn scale(c: &CreatureInstance) -> Vec3 {
    if c.proportions.x > 0.0 {
        c.proportions.xyz()
    } else {
        Vec3::ONE
    }
}

pub fn body_point(mut point: Vec3, c: &CreatureInstance) -> Vec3 {
    let infant = 1.0 - c.life.y;
    let front = (-point.z / c.opaque_bounds.z.max(0.01)).smoothstep(0.25, 0.8);
    point.x *= 1.0 - infant * front * 0.15;
    point.y *= 1.0 - infant * front * 0.12;
    // Individuals differ in build as well as size: some heavier in the head, some lighter, some
    // with a gentle arch to the back, some a sag.
    let individual = c.identity - 0.5;
    let reach = point.z / c.opaque_bounds.z.max(0.01);
    let head = (-reach).smoothstep(0.2, 0.8);
    point *= 1.0 - individual.w * 0.32 * head;
    point.y -= individual.z * 0.12 * reach * reach * c.opaque_bounds.y;
    point
}

pub fn appearance(mut material: Material, point: Vec3, c: &CreatureInstance) -> Material {
    if c.proportions.x <= 0.0 {
        return material;
    }
    let individual = c.identity - 0.5;
    let pigment = material
        .color
        .lerp(material.color.yzx(), individual.x.abs() * 0.45);
    let pattern = (point.z * (2.0 + c.identity.y * 5.0)
        + (point.x * 3.0 + individual.z * TAU).sin() * 2.0)
        .sin();
    material.color = pigment * (1.0 + individual.w * 0.3 + pattern * individual.y * 0.16);
    material.color = material
        .color
        .lerp(material.color.sqrt(), (1.0 - c.life.y) * 0.4);
    material.color = material.color.lerp(vec3(0.42, 0.34, 0.27), c.life.z * 0.35);
    material.roughness =
        (material.roughness + individual.z * 0.12 + c.life.z * 0.18).clamp(0.04, 1.0);
    material.emission *= (0.3 + c.life.y * 0.7) * (1.0 - c.life.z * 0.65);
    material
}

/// The body of a young organism in one of its early forms, by
/// [`CreatureInstance::life`]'s developmental form, scaled to its bounds.
pub fn juvenile_surface(point: Vec3, c: &CreatureInstance) -> Surface {
    // Most forms are drawn to scale in their bounds; a plant's young and a string of eggs are
    // stretched to fill theirs.
    let form = c.life.w as u32;
    let bounds = c.opaque_bounds.xyz() * 0.85;
    let extent = if matches!(form, SPORE | SPROUT | BUD | EGG_STRINGS) {
        bounds
    } else {
        Vec3::splat(bounds.max_element())
    };
    let mut p = point / extent;
    let time = c.animation.z;
    let age = c.life.y;
    p.x -= (p.z * 4.0 - time * 2.0).sin() * 0.08 * p.z;
    // Each lineage draws its own young; the shader can only choose among them by number.
    let mut hit = match form {
        POLYP => bullidorsata::young::polyp(p, age, time),
        sanguinivora::young::LARVA => sanguinivora::young::leech_larva(p, age),
        tetralata::young::LARVA => tetralata::young::larva(p, age),
        TETHERED_EGG => tetralata::young::tethered_egg(p),
        DRIFTING_EGG => tetralata::young::drifting_egg(p, time),
        EGG_STRINGS => tetralata::young::egg_strings(p, time),
        CLUTCH => tetralata::young::clutch(p),
        POUCH_YOUNG => grandivora::young::pouch_young(p, age),
        SPORE => plantae::young::spore(p),
        SPROUT => plantae::young::sprout(p, age, time),
        _ => plantae::young::bud(p, age),
    };
    hit.distance *= extent.min_element() * 0.7;
    hit
}

pub fn render_juvenile(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::MISS;
    let look = look_of(c.kind.x);
    result.opaque = trace_material(ray, c, juvenile_surface, |point, id| {
        // Young animals already wear their species' colours.
        if look.family != PLANT {
            let mut material = coat(
                point / (c.position.w * 0.15).max(0.7),
                match id {
                    1 => 7,
                    2 => 1,
                    3 => 3,
                    _ => 0,
                },
                look,
            );
            material.translucency = 0.35;
            // Eggshell is smooth and leathery, and the young's skin finer than the adult's.
            material.relief *= if id == 1 { 0.1 } else { 0.5 };
            if id == 1 {
                material.texture = SMOOTH;
                material.translucency = 0.5;
            }
            return material;
        }
        let color = match id {
            1 => vec3(0.64, 0.49, 0.26),
            2 => vec3(0.40, 0.26, 0.12),
            3 => vec3(0.24, 0.59, 0.37),
            _ => vec3(0.58, 0.75, 0.65),
        };
        let mut material = Material::skin(color, if id == 1 { 0.7 } else { 0.3 });
        material.translucency = 0.35;
        material
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::life::{KINDS, Kind};
    use spirv_std::glam::Vec4;

    #[test]
    fn every_young_form_is_its_own_and_every_kind_grows_through_known_ones() {
        let forms = [
            POLYP,
            sanguinivora::young::LARVA,
            tetralata::young::LARVA,
            TETHERED_EGG,
            DRIFTING_EGG,
            EGG_STRINGS,
            CLUTCH,
            POUCH_YOUNG,
            SPORE,
            SPROUT,
            BUD,
        ];
        for (index, form) in forms.iter().enumerate() {
            assert!(*form > 0, "form {form} is taken by the grown body");
            assert!(
                !forms[index + 1..].contains(form),
                "two lineages declared young form {form}"
            );
        }
        for kind in KINDS {
            for step in 0..=10 {
                let form = kind.young_form(step as f32 / 10.0);
                assert!(
                    form == 0 || forms.contains(&form),
                    "{kind:?} grows through an undeclared form {form}"
                );
            }
        }
    }

    #[test]
    fn lifecycle_grid_includes_birth_maturity_and_end_of_life() {
        for kind in KINDS {
            let biology = kind.biology();
            assert!(grid_age(biology, 0).abs() < f32::EPSILON);
            assert!((grid_age(biology, 5) - biology.maturity_days).abs() < 0.1);
            assert!((grid_age(biology, 8) - biology.lifespan_days).abs() < 0.1);
            for panel in 1..9 {
                assert!(grid_age(biology, panel) > grid_age(biology, panel - 1));
            }
        }
    }

    #[test]
    fn identity_is_stable_and_development_changes_body_plan() {
        for kind in [
            Kind::Nae,
            Kind::Nebulaphora,
            Kind::Nubivesica,
            Kind::Polliniflora,
        ] {
            let make = |age, seed| {
                CreatureInstance::specimen(
                    Vec4::W * kind.render_radius(),
                    Vec4::W,
                    Vec4::ZERO,
                    kind as u32,
                    age,
                    seed,
                )
            };
            let a = make(0.0, 123);
            let b = make(kind.biology().maturity_days, 123);
            assert_eq!(a.identity, b.identity);
            assert_ne!(a.identity, make(0.0, 456).identity);
            assert!(a.life.y < b.life.y);
            if kind != Kind::Nae {
                assert!(a.life.w > 0.0 && b.life.w == 0.0);
            }
        }
    }
}
