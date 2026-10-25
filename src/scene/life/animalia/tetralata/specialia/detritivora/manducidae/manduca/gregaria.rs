pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{curved_tube, oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth,
    surface::{Ray, Surface, TraceResult},
    torso,
};
use spirv_std::glam::{Vec3, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 12.0,
    lifespan_days: 120.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 21.6;

taxon!(
    Species,
    "Manduca gregaria",
    "Nibblers",
    [
        "Nibblers are tiny four-winged scavengers whose compact bodies turn quickly around suspended organic debris.",
        "Feeding swarms gather where damaged plants, carcasses or crowded grazing routes release loose material.",
        "Each animal follows nearby motion and food cues, producing coordinated murmurations without a fixed leader.",
        "Rapid growth lets a swarm exploit a short-lived concentration before the current scatters it.",
        "By intercepting debris before it falls toward the aur ocean, they keep nutrients circulating among the Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(4.0, 4.0, 4.0).extend(0.0);
    rig_tentacles(c, mouth(1.3, 0.42), 0.35, BIOLOGY.oral_pairs);
}

/// A swarming scavenger built like a beetle grub with wings: a hard head
/// with domed eyes and curved mandibles, a humped thorax, and a soft,
/// ringed abdomen trailing behind.
/// A compact swarming scavenger: a short, deep body, a cowl broadened into
/// a scoop that sweeps drifting scraps back to the mouth, and hard
/// shearing plates at the mouth's corners for working tough carcass.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(0.75, 0.7, 1.3), 0.42), 0);
    let lip = mouth(1.3, 0.42);
    let mirrored = vec3(p.x.abs(), p.y, p.z);
    hit.join(
        curved_tube(
            mirrored,
            lip + vec3(0.32, 0.02, 0.05),
            lip + vec3(0.5, -0.12, -0.3),
            lip + vec3(0.18, -0.16, -0.5),
            0.07,
        ),
        7,
    );
    hit.join(eyes(p, 1.3, 0.42), 3);
    hit.join(oral_tentacles(p, c, lip, 0.35, BIOLOGY.oral_pairs), 8);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.47, 0.26, 0.18, 0.0),
    vec3(0.86, 0.56, 0.28),
    [vec4(3.5, 1.2, -0.3, 0.3), vec4(2.2, 1.0, 0.5, 0.35)],
);

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
