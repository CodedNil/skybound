use crate::scene::life::{
    CreatureInstance, PLANT,
    animalia::tetralata::{
        self,
        young::{CLUTCH, DRIFTING_EGG, EGG_STRINGS, TETHERED_EGG},
    },
    biology::Biology,
    genome::{ARCH, BUILD, HUE, PATTERN, SHADE},
    look_of,
    plantae::{self, young::SPORE},
    surface::{Material, Ray, SMOOTH, Surface, TraceResult, coat, trace_material},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    KINDS,
    genome::{Genome, SIZE},
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
pub fn prepare(
    c: &mut CreatureInstance,
    biology: Biology,
    age_days: f32,
    seed: u32,
    genes: Genome,
) {
    let age = age_days.max(0.0);
    let young = (age / biology.maturity_days).clamp(0.0, 1.0);
    let old = age.smoothstep(biology.lifespan_days * 0.72, biology.lifespan_days);
    // Each kingdom, and each lineage within it, has its own young forms.
    let kind = KINDS[c.kind.x as usize];
    let stage = kind.young_form(young) as f32;
    c.kind.y = seed;
    c.life = vec4(age, young, old, stage);
    c.genes = genes.packed();
    c.identity = c.genes[0];
    let growth = (0.25 + young * 0.75) * (1.0 - old * 0.08) * biology.size;
    // Larger or smaller as its genes have it, broader or longer, higher or lower in the back.
    let size = (genes.get(SIZE) - 0.5) * 0.35;
    let build = genes.get(BUILD) - 0.5;
    let variation = vec3(
        1.0 + size + build * 0.2,
        1.0 + size + (genes.get(ARCH) - 0.5) * 0.12,
        1.0 + size - build * 0.15,
    );
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
    // The young are not small adults: the brain and eyes are nearly grown at birth while the rest
    // has far to go, so a newborn's head is large for its body and its hindquarters slight, and
    // both even out as it grows. The head swells about its own centre so it stays on the neck.
    let infant = 1.0 - c.life.y;
    let length = c.opaque_bounds.z.max(0.01);
    let front = (-point.z / length).smoothstep(0.25, 0.8);
    let rear = (point.z / length).smoothstep(0.1, 0.9);
    let centre = vec3(0.0, 0.0, -0.6 * length);
    point = centre + (point - centre) * (1.0 - infant * front * 0.2);
    point.x *= 1.0 + infant * rear * 0.18;
    point.y *= 1.0 + infant * rear * 0.12;
    // Individuals differ in build as well as size, as their genes have it: some heavier in the
    // head, some lighter, some with a gentle arch to the back, some a sag.
    let reach = point.z / c.opaque_bounds.z.max(0.01);
    let head = (-reach).smoothstep(0.2, 0.8);
    point *= 1.0 - (c.gene(BUILD) - 0.5) * 0.32 * head;
    point.y -= (c.gene(ARCH) - 0.5) * 0.12 * reach * reach * c.opaque_bounds.y;
    // A body carrying young swells round its brood chamber as they grow, slowly at first and most
    // in the last of its carrying, the fuller the larger the litter, and the swelling hangs a
    // little below the chamber under its weight.
    if c.belly.w > 0.0 && c.stores.y > 0.0 {
        let carried = c.stores.y * c.stores.y * (1.0 + (c.stores.z - 1.0).clamp(0.0, 4.0) * 0.2);
        let centre = c.belly.xyz() - vec3(0.0, c.belly.w * 0.5, 0.0);
        let reach = c.belly.w * 2.8;
        let offset = point - centre;
        let near = (-offset.length_squared() / (reach * reach)).exp();
        point -= offset * near * carried * 0.45;
    }
    point
}

/// A cheap, smooth wobble in about [0, 1] to break up the edges of scars.
fn noise_like(p: Vec3) -> f32 {
    ((p.x * 1.7 + (p.y * 2.3).sin()).sin() * (p.z * 1.9 + (p.x * 1.3).cos()).cos()) * 0.5 + 0.5
}

pub fn appearance(mut material: Material, point: Vec3, c: &CreatureInstance) -> Material {
    if c.proportions.x <= 0.0 {
        return material;
    }
    // Its colouring is its genes': its hue, how dark or pale, and how boldly patterned.
    let hue = c.gene(HUE) - 0.5;
    let shade = c.gene(SHADE) - 0.5;
    let bold = c.gene(PATTERN);
    let pigment = material.color.lerp(material.color.yzx(), hue.abs() * 0.45);
    let pattern = (point.z * (2.0 + bold * 5.0) + (point.x * 3.0 + hue * TAU).sin() * 2.0).sin();
    material.color = pigment * (1.0 + shade * 0.3 + pattern * (bold - 0.5) * 0.16);
    // Every wound it lived through left its scar where it landed: a pale, puckered patch, the
    // larger the worse the wound, crossed by seams where the skin knitted.
    let mut scar = 0.0f32;
    for index in 0..4 {
        let wound = c.wounds[index];
        if wound.w > 0.0 {
            let reach = 0.4 + wound.w * 1.6;
            let near = 1.0 - ((point - wound.truncate()).length() / reach).min(1.0);
            let edge = noise_like(point * 2.3 + wound.w * 17.0) * 0.5;
            scar = scar.max((near * 1.6 - edge).clamp(0.0, 1.0));
        }
    }
    if scar > 0.0 {
        let seam = |turn: f32| {
            let across = point.x * turn.cos() + point.z * turn.sin() + point.y * 0.3;
            ((across + (point.y * 2.3 + turn * 5.0).sin() * 0.3) * 5.0)
                .sin()
                .abs()
        };
        let lines = 1.0 - seam(0.7).min(seam(2.3)).smoothstep(0.0, 0.12);
        let pale = scar * (0.45 + lines * 0.55);
        material.color = material.color.lerp(vec3(0.84, 0.77, 0.71), pale * 0.75);
        material.roughness += pale * 0.3;
        material.scatter *= 1.0 - pale * 0.6;
    }
    // Its skin a little rougher or smoother than another's, as it happened to grow.
    let individual = c.chance(9) - 0.5;
    material.color = material
        .color
        .lerp(material.color.sqrt(), (1.0 - c.life.y) * 0.4);
    material.color = material.color.lerp(vec3(0.42, 0.34, 0.27), c.life.z * 0.35);
    material.roughness =
        (material.roughness + individual * 0.12 + c.life.z * 0.18).clamp(0.04, 1.0);
    material.emission *= (0.3 + c.life.y * 0.7) * (1.0 - c.life.z * 0.65);
    material
}

/// An egg or spore, by [`CreatureInstance::life`]'s developmental form, scaled to its bounds.
///
/// Everything that has hatched or taken hold is drawn by its species' own tracer instead, so it
/// grows smoothly into the adult rather than snapping from one shape to another.
pub fn juvenile_surface(point: Vec3, c: &CreatureInstance) -> Surface {
    // Most forms are drawn to scale in their bounds; a spore and a string of eggs are stretched to
    // fill theirs.
    let form = c.life.w as u32;
    let bounds = c.opaque_bounds.xyz() * 0.85;
    let extent = if matches!(form, SPORE | EGG_STRINGS) {
        bounds
    } else {
        Vec3::splat(bounds.max_element())
    };
    let mut p = point / extent;
    let time = c.animation.z;
    p.x -= (p.z * 4.0 - time * 2.0).sin() * 0.08 * p.z;
    // Each lineage draws its own young; the shader can only choose among them by number.
    let mut hit = match form {
        TETHERED_EGG => tetralata::young::tethered_egg(p),
        DRIFTING_EGG => tetralata::young::drifting_egg(p, time),
        EGG_STRINGS => tetralata::young::egg_strings(p, time),
        CLUTCH => tetralata::young::clutch(p),
        _ => plantae::young::spore(p),
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
        let forms = [TETHERED_EGG, DRIFTING_EGG, EGG_STRINGS, CLUTCH, SPORE];
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
            // Only an egg or spore has a form of its own; the grown body has none.
            assert_eq!(a.life.w, kind.young_form(0.0) as f32);
            assert_eq!(b.life.w, 0.0);
        }
    }
}
