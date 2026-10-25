use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::{Biology, Brood},
    capsule, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec3, Vec4, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    brood: Brood::Carried,
    maturity_days: 1800.0,
    lifespan_days: 60000.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 46.8;

taxon!(
    Species,
    "Lithosoma grave",
    "Grounders",
    [
        "Grounders carry heavy mineral armour over dense bodies supported by broad lifting surfaces.",
        "They move slowly through the lower Gardens, grazing Root-Stars while staying above the lethal Veiland mists.",
        "Both wing pairs have fused into one broad, upturned saucer that brakes descent against strong updrafts while the dense body hangs beneath it like a pendulum, so the animals favour reliable rising columns rather than weak or rapidly shifting air.",
        "Mineral-rich currents renew the crust, cracking it into plates and growing clusters of crystal on the saucer's upper face, but expose them to dusty haze and the danger of nearby geyser jets.",
        "Their feeding routes skirt those jets, and a mother carries her clutch glued round the hanging ballast, where the eggs ride steady until they hatch.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(15.0, 10.0, 15.0).extend(0.0);
    rig_tentacles(c, MOUTH, 3.2, BIOLOGY.oral_pairs);
}

/// A heavy, stone-backed flyer: a broad, ringed body roofed by
/// overlapping plates like a woodlouse, a blunt armoured head with deep-set
/// eyes, and thick mouth tentacles beneath.
/// Where the rasp meets the plant, and the gripping tentacles beside it.
const MOUTH: Vec3 = vec3(0.0, 2.2, -11.0);

/// A living drogue, riding level beneath a saucer of fused wings.
///
/// Both wing pairs have fused into one broad, rigid,
/// upturned saucer that the strong lower updrafts press against, braking
/// its descent where lighter animals would flap; beneath it a dense body
/// hangs like a pendulum, so the whole animal rides level through rough
/// columns. The mineral currents have crusted it into cracked plates and
/// grown clusters of crystal on its upper face. A rasp boss rises from the
/// saucer's leading edge, facing up and forward, to grind Root-Star tissue
/// from below while short, thick tentacles hold the plant against it, and
/// a ring of eyes round the underside of the rim watches the mist it must
/// never sink into.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let radial = vec2(p.x, p.z).length();
    let angle = p.z.atan2(p.x);
    // Every Grounder's saucer is its own: broader or narrower, with more or
    // fewer scallops, more or less upturned at the rim.
    let own = c.identity;
    let span = 13.0 * (0.85 + own.x * 0.3);
    let scallops = 5.0 + (own.y * 5.0).floor();
    let rim = span * (1.0 + (angle * scallops).cos() * 0.05);
    let upturn = 1.6 * (0.5 + own.z);
    let lift = (radial / span).powi(2) * upturn;
    let thickness = 1.3 * (1.0 - (radial / rim).powi(2)).max(0.0) + 0.25;
    let saucer = ((p.y - lift).abs() - thickness * 0.5).max(radial - rim) * 0.6;
    let heft = 0.8 + own.w * 0.4;
    let ballast = rounded(p - vec3(0.0, -3.6 * heft, 0.5), vec3(4.0, 3.8, 5.0) * heft);
    let stem = capsule(p, vec3(0.0, -0.8, 0.5), vec3(0.0, -2.6, 0.5), 2.6);
    let boss = rounded(p - vec3(0.0, 1.3, -9.4), vec3(2.5, 1.4, 2.3));
    let mut crust = smooth_min(smooth_min(saucer, stem, 1.6), ballast, 1.8);
    crust = smooth_min(crust, boss, 1.2);
    // Cracked into plates by the crust's own growth.
    let cracks = ((p.x * 0.8 + (p.z * 0.55).sin() * 1.3).sin()
        * (p.z * 0.8 + (p.x * 0.5).sin() * 1.3).sin())
    .abs();
    crust += (1.0 - cracks.smoothstep(0.0, 0.12)) * 0.08;
    // Grinding ridges across the rasp.
    let rasp = p - vec3(0.0, 2.5, -10.2);
    if rasp.length() < 2.4 {
        crust = crust.max(-(((rasp.x * 3.0).sin().abs() - 0.15) * 0.1 - (rasp.y - 0.2).max(0.0)));
    }
    let mut hit = Surface::new(crust, 5);
    // Crystal clusters, one to a sector of the saucer, each its own size.
    let sector = TAU / (5.0 + (own.z * 5.0).floor());
    let turn = (angle / sector).round();
    let at = turn * sector;
    let (sine, cosine) = at.sin_cos();
    let along = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let seed = ((turn * 12.9898 + c.identity.x * 311.7).sin() * 43_758.547).fract_gl();
    let out = 5.0 + seed * 4.5;
    let surface = (out / span).powi(2) * upturn + 0.6;
    // Some sectors have grown no crystal at all.
    let bare = ((turn * 7.31 + own.w * 97.1).sin() * 43_758.547).fract_gl() < 0.3;
    for crystal in 0..3 {
        if bare {
            break;
        }
        let offset = crystal as f32 - 1.0;
        let lean = offset * 0.35;
        let local = vec3(
            across - offset * 0.6,
            p.y - surface,
            along - out - (offset + 1.0) * 0.3,
        );
        let local = vec3(
            local.x * lean.cos() - local.y * lean.sin(),
            local.x * lean.sin() + local.y * lean.cos(),
            local.z,
        );
        let height = (1.0 + seed * 1.6) * (1.0 - offset.abs() * 0.4);
        let prism = (vec2(local.x, local.z).length() - 0.32)
            .max(local.y - height)
            .max(-local.y - 0.3);
        hit.join(prism * 0.8, 6);
    }
    // A ring of eyes beneath the rim, looking down at the mist.
    let eye_sector = TAU / 9.0;
    let eye_at = (angle / eye_sector).round() * eye_sector;
    let (sine, cosine) = eye_at.sin_cos();
    let eye_out = span * 0.91;
    let eye = vec3(
        cosine * eye_out,
        (eye_out / span).powi(2) * upturn - 0.45,
        sine * eye_out,
    );
    hit.join((p - eye).length() - 0.32, 3);
    hit.join(oral_tentacles(p, c, MOUTH, 3.2, BIOLOGY.oral_pairs), 8);
    hit
}

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.24, 0.24, 0.29, 0.0),
    vec3(0.46, 0.32, 0.56),
    // Its wings are fused into the saucer: there are no membranes.
    [Vec4::ZERO; 2],
)
.skinned(vec4(2.0, 1.8, 0.7, 0.20))
.brooding(vec4(-3.8, 0.5, 3.5, 3.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}
