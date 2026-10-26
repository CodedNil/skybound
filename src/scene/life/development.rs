#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::KINDS;
use crate::scene::life::{
    CreatureInstance, PLANT,
    biology::Biology,
    ellipsoid, look_of, plantae, smooth_min,
    surface::{Material, Ray, SMOOTH, Surface, TraceResult, coat, trace_material},
};
use core::f32::consts::{PI, TAU};
#[cfg(not(target_arch = "spirv"))]
use spirv_std::glam::vec4;
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4Swizzles, vec2, vec3};
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
    let growth = (0.25 + young * 0.75) * (1.0 - old * 0.08);
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
    let extent = if matches!(form, 5..=7 | 9) {
        bounds
    } else {
        Vec3::splat(bounds.max_element())
    };
    let mut p = point / extent;
    let time = c.animation.z;
    let age = c.life.y;
    p.x -= (p.z * 4.0 - time * 2.0).sin() * 0.08 * p.z;
    let mut hit = match form {
        1 => polyp(p, age, time),
        2 => leech_larva(p, age),
        3 => larva(p, age),
        4 => tethered_egg(p),
        5 => spore(p),
        6 => sprout(p, age, time),
        8 => drifting_egg(p, time),
        9 => egg_strings(p, time),
        10 => clutch(p),
        11 => pouch_young(p, age),
        _ => bud(p, age),
    };
    hit.distance *= extent.min_element() * 0.7;
    hit
}

/// A porous animal's first form: a soft dome fringed at its rim, its gas chambers swelling as
/// it grows.
fn polyp(p: Vec3, age: f32, time: f32) -> Surface {
    let mut body = ellipsoid(p, vec3(0.7, 0.4 + age * 0.3, 0.75));
    let fringe = (vec2(vec2(p.x, p.z).length() - 0.64, p.y + 0.2).length()
        - 0.08
        - (p.x.atan2(p.z) * 24.0 + time * 3.0).sin() * 0.025)
        .max(p.y);
    body = smooth_min(body, fringe, 0.12);
    for index in 0..3 {
        let chamber = ellipsoid(
            p - vec3((index as f32 - 1.0) * 0.35, 0.25, 0.0),
            vec3(0.25, age * 0.6 + 0.05, 0.35),
        );
        body = smooth_min(body, chamber, 0.15);
    }
    Surface::new(body, 0)
}

/// A young blood-feeder: a ringed body led by its sucker, its fins growing out along its flanks.
fn leech_larva(p: Vec3, age: f32) -> Surface {
    let mut body = 1e6f32;
    for index in 0..7 {
        let along = index as f32 / 6.0;
        let radius = 0.20 + (along * PI).sin() * 0.13;
        body = smooth_min(
            body,
            ellipsoid(
                p - vec3(0.0, 0.0, (along - 0.5) * 1.35),
                vec3(radius, radius * 0.85, 0.19),
            ),
            0.06,
        );
    }
    let sucker = ellipsoid(p - vec3(0.0, 0.0, -0.67), vec3(0.25, 0.22, 0.12));
    body = smooth_min(body, sucker, 0.04);
    for pair in 0..2 {
        let fin = vec3(
            p.x.abs() - 0.28,
            p.y + 0.02,
            p.z - (pair as f32 - 0.5) * 0.5,
        );
        body = body.min(ellipsoid(fin, vec3(0.12 + age * 0.3, 0.035, 0.19)));
    }
    Surface::new(body, 0)
}

/// The larva recalls the lineage's limbless ancestors: a plump, ringed grub wriggling through the
/// vegetation it hatched in, the cowl only a bud over its blunt head, the wing pairs rounded buds
/// folded along its back and the mouth tentacles short stubs, all growing towards the adult's as it
/// nears its change. Two dark eyes already look down.
fn larva(p: Vec3, age: f32) -> Surface {
    let grow = ((age - 0.3) / 0.4).clamp(0.0, 1.0);
    let mut body = 1e6f32;
    for index in 0..9 {
        let along = index as f32 / 8.0;
        let radius = 0.13 + (along * PI).sin().powf(0.6) * 0.1 - along * 0.03;
        body = smooth_min(
            body,
            ellipsoid(
                p - vec3(0.0, 0.0, (along - 0.45) * 1.4),
                vec3(radius, radius * 0.92, 0.12),
            ),
            0.035,
        );
    }
    let head = p - vec3(0.0, 0.02, -0.68);
    body = smooth_min(body, ellipsoid(head, vec3(0.17, 0.15, 0.14)), 0.05);
    let cowl = ellipsoid(
        head - vec3(0.0, 0.08, -0.04),
        vec3(0.14 + grow * 0.12, 0.04, 0.12 + grow * 0.06),
    );
    body = smooth_min(body, cowl, 0.04);
    let flank = vec3(p.x.abs(), p.y, p.z);
    for stub in 0..2 {
        let root = vec3(0.05 + stub as f32 * 0.05, -0.1, -0.76);
        body = smooth_min(
            body,
            (flank - root - vec3(0.02, -0.05, -0.06) * ((root.z - flank.z) / 0.06).clamp(0.0, 1.0))
                .length()
                - 0.014,
            0.02,
        );
    }
    let mut hit = Surface::new(body, 0);
    // Wing buds, rounded lobes lying back along the upper flanks.
    for pair in 0..2 {
        let lobe = flank - vec3(0.14 + grow * 0.04, 0.13, -0.32 + pair as f32 * 0.32);
        let (sine, cosine) = 0.5f32.sin_cos();
        let tilted = vec3(
            lobe.x * cosine + lobe.y * sine,
            lobe.y * cosine - lobe.x * sine,
            lobe.z,
        );
        hit.join(
            ellipsoid(tilted, vec3(0.05 + grow * 0.14, 0.04, 0.07 + grow * 0.05)),
            2,
        );
    }
    hit.join(
        ellipsoid(flank - vec3(0.09, -0.08, -0.79), vec3(0.04, 0.035, 0.035)),
        3,
    );
    hit
}

/// An egg, laid as most of the lineage lays them: tethered to living vegetation, safe from the long
/// fall to the aur, in a leathery, ridged case.
fn tethered_egg(p: Vec3) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.46, 0.4, 0.85)), 1);
    let tether = ((p - vec3(0.0, 0.36, -0.55))
        - vec3(0.0, 0.55, -0.4) * ((p.y - 0.36) / 0.55).clamp(0.0, 1.0))
    .length()
        - 0.025;
    let anchor = ellipsoid(p - vec3(0.0, 0.93, -0.97), vec3(0.18, 0.03, 0.18));
    hit.join(tether.min(anchor), 0);
    hit
}

/// A plant's spore, under a sail that the updraft carries.
fn spore(p: Vec3) -> Surface {
    let body = ellipsoid(p, vec3(0.3, 0.45, 0.3));
    let sail = ellipsoid(p - vec3(0.0, 0.4, 0.0), vec3(0.7, 0.06, 0.65));
    Surface::new(smooth_min(body, sail, 0.06), 2)
}

/// A plant's seedling: a bulb on a stalk, opening its first leaves.
fn sprout(p: Vec3, age: f32, time: f32) -> Surface {
    let bulb = ellipsoid(p, vec3(0.35, 0.3, 0.35));
    let stalk = ellipsoid(p - vec3(0.0, 0.3, 0.0), vec3(0.06, 0.5, 0.06));
    let mut body = smooth_min(bulb, stalk, 0.06);
    for index in 0..4 {
        let angle = index as f32 * TAU / 4.0;
        let turned = plantae::rotate(p - vec3(0.0, 0.1 + index as f32 * 0.12, 0.0), angle);
        body = body.min(plantae::leaf(
            turned,
            0.35 + age * 0.75,
            0.14 + age * 0.22,
            0.08 + (time + angle).sin() * 0.04,
        ));
    }
    Surface::new(body, 3)
}

/// A plant dividing into two.
fn bud(p: Vec3, age: f32) -> Surface {
    let body = smooth_min(
        ellipsoid(p - vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
        ellipsoid(p + vec3(age * 0.45, 0.0, 0.0), Vec3::splat(0.35)),
        0.15,
    );
    Surface::new(body, 3)
}

/// An egg cast loose on the updraft, held up by a crown of fine threads that the rising air
/// catches, as down holds up a seed.
fn drifting_egg(p: Vec3, time: f32) -> Surface {
    let rise = p - vec3(0.0, -0.3, 0.0);
    let shell = ellipsoid(rise, vec3(0.26, 0.3, 0.26))
        + (rise.y * 40.0 + rise.x.atan2(rise.z) * 3.0).sin() * 0.006;
    let mut hit = Surface::new(shell, 1);
    let sector = TAU / 11.0;
    let turn = (p.z.atan2(p.x) / sector).round();
    let (sine, cosine) = (turn * sector).sin_cos();
    let out = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let lean = ((turn * 5.31).sin() * 43_758.547).fract_gl();
    let sway = (time * 1.3 + turn).sin() * 0.04;
    // Each thread rises from the egg's crown and arcs outwards.
    let up = ((p.y + 0.02) / 0.8).clamp(0.0, 1.0);
    let line = vec2(
        out - (0.08 + up * (0.55 + lean * 0.3) - up * up * 0.15),
        across - sway * up,
    )
    .length()
        - 0.012 * (1.0 - up * 0.6);
    hit.join(line.max(-p.y - 0.02).max(p.y - 0.78) * 0.7, 0);
    hit
}

/// Strings of eggs paid out behind the parent, dozens to a string at no even spacing, each string
/// sagging and swaying its own way in the wake.
fn egg_strings(p: Vec3, time: f32) -> Surface {
    let mut hit = Surface::new(1e6, 1);
    let behind = ((p.z + 0.9) / 1.85).clamp(0.0, 1.0);
    for string in 0..3 {
        let lane = string as f32;
        let centre = vec2(
            (lane - 1.0) * 0.18 * behind
                + (time * 1.1 + p.z * 4.0 + lane * 2.0).sin() * 0.07 * behind,
            -behind * behind * (0.25 + lane * 0.1)
                + (time * 0.8 + p.z * 3.0 + lane).cos() * 0.05 * behind,
        );
        let strand = (vec2(p.x, p.y) - centre).length();
        hit.join(
            vec2(strand - 0.012, (p.z.abs() - 0.92).max(0.0)).length(),
            0,
        );
        let spacing = 0.065;
        let cell = (p.z / spacing).round();
        let seed = ((cell * 12.9898 + lane * 78.233).sin() * 43_758.547).fract_gl();
        if seed > 0.25 && (cell * spacing).abs() < 0.9 {
            let along = p.z - cell * spacing - (seed - 0.5) * spacing * 0.4;
            let egg = vec3(p.x - centre.x, p.y - centre.y + (seed - 0.6) * 0.03, along).length()
                - (0.022 + seed * 0.022);
            hit.join(egg, 1);
        }
    }
    hit
}

/// A clutch fallen from its parent's body, still glued together.
fn clutch(p: Vec3) -> Surface {
    let mut eggs = 1e6f32;
    for egg in 0..7 {
        let turn = egg as f32;
        let at = if egg == 0 {
            Vec3::ZERO
        } else {
            let angle = turn * TAU / 6.0;
            vec3(
                angle.cos() * 0.42,
                (turn * 1.7).sin() * 0.14,
                angle.sin() * 0.42,
            )
        };
        eggs = smooth_min(eggs, ellipsoid(p - at, vec3(0.24, 0.22, 0.3)), 0.03);
    }
    Surface::new(eggs, 1)
}

/// Born long before it is formed, to finish growing in the pouch: a curled, translucent body, the
/// great dark eyes not yet open, the wings and mouth tentacles only nubs.
fn pouch_young(p: Vec3, age: f32) -> Surface {
    let grow = age / 0.2;
    let mut body = 1e6f32;
    for index in 0..7 {
        let along = index as f32 / 6.0;
        let bend = along * 2.6 - 0.6;
        let at = vec3(0.0, bend.sin() * 0.45 - 0.1, -bend.cos() * 0.45 + 0.15);
        let radius = 0.24 - along * 0.12 + (1.0 - along) * grow * 0.06;
        body = smooth_min(
            body,
            ellipsoid(p - at, vec3(radius, radius, radius * 0.9)),
            0.08,
        );
    }
    let side = vec3(p.x.abs(), p.y, p.z);
    let mut nubs = 1e6f32;
    for pair in 0..2 {
        nubs = nubs.min(ellipsoid(
            side - vec3(0.22, 0.12, -0.2 + pair as f32 * 0.3),
            vec3(0.1 + grow * 0.08, 0.025, 0.08),
        ));
    }
    let mut hit = Surface::new(smooth_min(body, nubs, 0.04), 0);
    hit.join(
        ellipsoid(side - vec3(0.14, -0.02, -0.58), vec3(0.09, 0.1, 0.09)),
        3,
    );
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
