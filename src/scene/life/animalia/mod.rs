use crate::scene::life::{
    CreatureInstance, Look,
    biology::{Anatomy, Biology},
    smooth_min,
    surface::{Hit, Ray, Surface, TraceResult, trace, visit_paired_sheets, wing_bones},
};
use spirv_std::glam::{Vec3, Vec4, vec3};
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
    let mut result = TraceResult::opaque(trace(ray, c, body, look.pigment, look.accent));
    visit_wings(ray, c, look, |hit| result.add(hit));
    result
}

/// Where a wing pair's roots meet the body: on its midline, where the
/// body itself hides the membrane's inner edge.
fn wing_root(wing: Vec4) -> Vec3 {
    vec3(0.0, 0.0, wing.z)
}

/// A body with the frames of its wings grown out of it, blended in at the
/// roots so each wing visibly springs from the flank.
pub fn with_wings(body: Surface, p: Vec3, c: &CreatureInstance, look: Look) -> Surface {
    let mut frame = 1e6f32;
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x > 0.0 {
            frame = frame.min(wing_bones(
                p,
                c,
                wing_root(wing),
                wing.truncate().truncate(),
                wing.w > 0.9,
            ));
        }
    }
    let joined = smooth_min(body.distance, frame, look.wings[0].y * 0.12);
    Surface::new(
        joined,
        if frame < body.distance {
            1
        } else {
            body.material
        },
    )
}

/// The wing membranes of a species' look along a ray.
pub fn visit_wings<F: FnMut(Hit)>(ray: Ray, c: &CreatureInstance, look: Look, mut emit: F) {
    for index in 0..2 {
        let wing = look.wings[index];
        if wing.x <= 0.0 {
            continue;
        }
        visit_paired_sheets(
            ray,
            c,
            wing_root(wing),
            wing.truncate().truncate(),
            look.accent.extend(wing.w),
            &mut emit,
        );
    }
}
