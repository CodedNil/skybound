pub use super::render_wings;
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::curved_tube,
    biology::{Biology, Brood},
    louvres, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    brood: Brood::Strings,
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
        "Nibblers are tiny four-winged scavengers, teardrop bodies on two pairs of short, round wings set close together, that beat so fast they can turn on the spot around a drifting scrap.",
        "Their mouth tentacles have stiffened into a cage of hooked fingers under the head, which closes on a fragment and holds it while the animal feeds in flight.",
        "Each animal follows nearby motion and food cues, producing coordinated murmurations without a fixed leader; the pale eyespots on every wing flash as a neighbour turns, so the turn ripples through the swarm.",
        "Rapid growth lets a swarm exploit a short-lived concentration before the current scatters it, and adults pay out their eggs on long strings that drift in the swarm's wake.",
        "By intercepting debris before it falls toward the aur ocean, they keep nutrients circulating among the Gardens.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(3.6, 3.0, 3.6).extend(0.0);
}

/// Where the feeding cage hangs, under the front of the head.
const MOUTH: Vec3 = vec3(0.0, -0.35, -1.05);

/// A swarming scavenger, small and quick.
///
/// A teardrop body, blunt in front and tapering behind, carries two pairs of
/// short, round wings set close together so it can pivot on the spot. Its mouth
/// tentacles have stiffened into a cage of five hooked fingers, curving
/// forwards and down and in again, which closes round a drifting scrap and
/// holds it while the animal feeds on the wing; two domed eyes beneath the head
/// look down into the cage and the air below.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let body = tapered_tail(
        p,
        vec3(0.0, 0.05, -0.55),
        vec3(0.0, -0.05, 1.25),
        0.62,
        0.28,
    );
    let head = rounded(p - vec3(0.0, -0.05, -0.92), vec3(0.5, 0.42, 0.42));
    let mut hit = Surface::new(
        louvres(smooth_min(body, head, 0.3), p, vec3(0.6, 0.55, 1.2)),
        0,
    );
    // Each finger of the cage its own length, and a little open or closed.
    let close = 0.15 + c.identity.z * 0.2;
    for finger in 0..5 {
        let splay = ((finger as f32 - 2.0) * 0.45).sin();
        let root = MOUTH + vec3(splay * 0.28, 0.0, 0.0);
        let reach = 0.9 + ((finger as f32 * 7.1 + c.identity.x * 31.0).sin() * 0.5 + 0.5) * 0.25;
        hit.join(
            curved_tube(
                p,
                root,
                root + vec3(splay * 0.55, -0.12, -0.6 * reach),
                root + vec3(splay * (0.35 - close), -0.6 * reach, -0.3 - close),
                0.05,
            ),
            7,
        );
    }
    let side = vec3(p.x.abs(), p.y, p.z);
    hit.join((side - vec3(0.3, -0.3, -1.05)).length() - 0.15, 3);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.47, 0.26, 0.18, 0.0),
    vec3(0.86, 0.50, 0.24),
    [vec4(2.1, 1.2, -0.45, 0.4), vec4(2.0, 1.15, 0.25, 0.4)],
)
.hinged([vec2(0.5, 0.25), vec2(0.45, 0.2)])
.styled([vec4(0.2, 1.0, 0.0, 0.0), vec4(0.2, 1.0, 0.0, 0.0)])
.framed([0.5, 0.5])
.skinned(vec4(1.0, 1.0, 2.2, 0.05))
.flying(vec4(7.00, 0.50, 0.15, 0.40))
.marked(vec4(2.00, 4.00, 0.90, 0.42), vec3(0.96, 0.92, 0.74));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
