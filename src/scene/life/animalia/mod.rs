use crate::scene::life::{
    CLUTCH, CreatureInstance, FULL_POUCH, Look,
    biology::{Anatomy, Biology},
    ellipsoid, rounded, smooth_min,
    surface::{Hit, Ray, Surface, TraceResult, trace, visit_paired_sheets, wing_bones},
};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec3, Vec4, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
pub mod bullidorsata;
pub mod sanguinivora;
pub mod tetralata;

pub const BIOLOGY: Biology = Biology {
    anatomy: Anatomy::Porous,
    ..super::BIOLOGY
};

taxon!(
    Kingdom,
    "Animalia",
    "Animals",
    [
        "Animals obtain usable energy by consuming organic material made by other organisms.",
        "Porous forms absorb food across wet surfaces, while active lineages gather it into a digestive crop and distribute nutrients through pulsatile channels.",
        "Moisture-recovery folds conserve water, although sustained flight can still force animals to seek Pondplants or humid currents.",
        "Lift comes from gas chambers, wings or attachment to another living body, and every lineage must survive the weaker updraft of the dim aur phase.",
        "Direct exposure to concentrated aur remains lethal except within the carefully limited mist-skimming adaptations of the Nætu lineage.",
    ],
);

/// Render the body in its coat and its wing pairs; a pair of no span is
/// absent.
pub fn render_wings(
    ray: Ray,
    c: &CreatureInstance,
    surface: impl Fn(Vec3, &CreatureInstance) -> Surface,
    look: Look,
) -> TraceResult {
    let body = |p: Vec3, c: &CreatureInstance| with_wings(surface(p, c), p, c, look);
    let mut result = TraceResult::opaque(trace(ray, c, body, look));
    visit_wings(ray, c, look, |hit| result.add(hit));
    result
}

/// A grown body with its wing frames and whatever brood it carries.
///
/// The frames grow out of the body, blended in at the roots so each wing
/// visibly springs from the flank. The young have their own forms, buds and
/// all.
pub fn with_wings(body: Surface, p: Vec3, c: &CreatureInstance, look: Look) -> Surface {
    if c.life.w > 0.0 {
        return body;
    }
    let mut frame = 1e6f32;
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x > 0.0 {
            frame = frame.min(wing_bones(p, c, look, index));
        }
    }
    let joined = smooth_min(body.distance, frame, look.wings[0].y * 0.12);
    let mut surface = Surface::new(
        joined,
        if frame < body.distance {
            look.edge
        } else {
            body.material
        },
    );
    if c.kind.z != 0 && look.brood.w > 0.0 {
        brood(&mut surface, p, c, look.brood);
    }
    surface
}

/// A parent's brood, where its species carries it. A clutch is a band of
/// eggs glued round the body, two rows deep and packed unevenly, wrapping
/// the flanks and belly but leaving the back clear for the wings. A pouch
/// is a fold of skin on the back opening forwards and upwards, so nothing
/// can fall out of it towards the aur: flat when empty, swollen when full,
/// with a youngster's head and dark eyes looking out of its mouth.
fn brood(surface: &mut Surface, p: Vec3, c: &CreatureInstance, at: Vec4) {
    let q = p - vec3(0.0, at.x, at.y);
    if c.kind.z == CLUTCH {
        let angle = (q.y / at.w).atan2(q.x / at.z);
        let sector = TAU / 14.0;
        let mut eggs = 1e6f32;
        for row in 0..2 {
            let shift = row as f32 * 0.5;
            let turn = (angle / sector - shift).round() + shift;
            // Round the belly and flanks only.
            let around = turn * sector;
            if around.sin() > 0.45 {
                continue;
            }
            let seed = ((turn * 12.9898 + row as f32 * 4.1 + c.identity.x * 311.7).sin()
                * 43_758.547)
                .fract_gl();
            let size = at.z.min(at.w) * (0.2 + seed * 0.07);
            let centre = vec3(
                around.cos() * (at.z + size * 0.6),
                around.sin() * (at.w + size * 0.6),
                (row as f32 - 0.5) * size * 2.0 + (seed - 0.5) * size * 0.6,
            );
            eggs = eggs.min(ellipsoid(q - centre, vec3(size, size, size * 1.25)));
        }
        if eggs < surface.distance {
            surface.distance = smooth_min(surface.distance, eggs, at.z * 0.03);
            surface.material = 7;
        }
        return;
    }
    let full = if c.kind.z == FULL_POUCH { 1.0 } else { 0.0 };
    let height = at.w * (0.45 + full * 0.55);
    let pouch = rounded(
        q - vec3(0.0, -at.w * 0.3, 0.0),
        vec3(at.z, height + at.w * 0.3, at.z * 1.4),
    );
    // Its mouth, a slot across the front facing forwards and up.
    let mouth = rounded(
        q - vec3(0.0, height * 0.8, -at.z * 1.25),
        vec3(at.z * 0.55, height * 0.35 + 0.02, at.z * 0.4),
    );
    surface.distance = smooth_min(surface.distance, pouch.max(-mouth), at.z * 0.25);
    if full > 0.0 {
        let head = q - vec3(0.0, height * 0.75, -at.z * 1.05);
        let young = rounded(head, vec3(at.z * 0.4, at.z * 0.32, at.z * 0.45));
        let eye = ellipsoid(
            vec3(head.x.abs(), head.y, head.z) - vec3(at.z * 0.2, at.z * 0.08, -at.z * 0.3),
            Vec3::splat(at.z * 0.1),
        );
        surface.join(young, 0);
        surface.join(eye, 3);
    }
}

/// The wing membranes of a species' look along a ray.
pub fn visit_wings<F: FnMut(Hit)>(ray: Ray, c: &CreatureInstance, look: Look, mut emit: F) {
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x <= 0.0 {
            continue;
        }
        visit_paired_sheets(ray, c, look, index, &mut emit);
    }
}
