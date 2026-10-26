use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::Biology,
    capsule,
    genome::Gene,
    louvres, prepare_pose, rounded, smooth_min,
    surface::{Ray, Surface, TraceResult},
    tapered_tail,
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    animalia::physiology::BREATHER,
    animalia::physiology::BROOD,
    animalia::physiology::CROP,
    animalia::physiology::FLOAT,
    animalia::physiology::GANGLION,
    animalia::physiology::PUMP,
    animalia::tetralata::CISTERN,
    animalia::tetralata::FLIGHT_MUSCLE,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{Vec3, vec2, vec3, vec4};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 2.7,
    maturity_days: 18000.0,
    lifespan_days: 365_000.0,
    adult_mass: 200.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 55.6;

/// How long its horns grow and how wide their bells flare, so each elder's voice is its own, and
/// how full its bladders swell.
pub const HORNS: Gene = Gene(30);
pub const SWELL: Gene = Gene(31);

#[cfg(not(target_arch = "spirv"))]
pub use super::ETHOLOGY;
#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Vocigenus antiquus",
    "Old Wise Ones",
    [
        "Old Wise Ones are rare giants whose long, pale bodies carry a chain of swelling gas bladders along the back, with long, pale mouth tentacles hanging beneath the blunt front like a flowing beard.",
        "Slow wing strokes and careful use of rising air carry their immense bodies between kelp forests and Laevensails.",
        "Three pairs of mouth tentacles gather only those soft plants, giving the lineage a narrow dependence on healthy floating forests.",
        "Rich calls, shaped by a throat sac and a double rank of hollow resonating horns along the back, pass between companions long before their pale bodies become visible through the haze.",
        "Their young develop slowly within a live-bearing social lineage whose journeys preserve access to feeding areas across many generations.",
    ],
);

/// How far the body runs fore and aft of its middle.
const LENGTH: f32 = 15.0;

/// Where the mouth opens, ahead of the chin and clear of the head, so the mouth tentacles hang
/// free beneath it.
const MOUTH: Vec3 = vec3(0.0, -3.1, -16.6);

/// The bladders along the back, from the shoulders to the hips.
const BLADDERS: u32 = 5;

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 22.0, 10.0, 0);
    c.opaque_bounds = vec3(13.0, 15.0, 30.0).extend(0.0);
    rig_tentacles(c, MOUTH, 3.4, ORAL_PAIRS);
}

/// The body's half-width and half-depth a distance `z` along it, and how high its midline runs: a
/// blunt, broad front, deepest at the chest where the flight muscles are, drawn out to the tail.
fn girth(z: f32) -> Vec3 {
    let t = ((z + LENGTH) / (2.0 * LENGTH)).clamp(0.0, 1.0);
    let front = (t / 0.16).min(1.0).sqrt();
    let width = 5.3 * front * (1.0 - 0.72 * t.powf(1.5));
    vec3(width, width * (0.86 - t * 0.12), 0.4 - t * 0.9)
}

/// The top of the back a distance `z` along it.
fn back(z: f32) -> f32 {
    let girth = girth(z);
    girth.z + girth.y
}

/// A living airship.
///
/// Its long, pale body carries a chain of gas bladders sunk into the back, each ribbed with the
/// vessels that warm it and webbed to the next, so a giant that lives for centuries can idle on
/// slow strokes between distant floating forests. Flanking them rise two ranks of hollow horns,
/// each growing from a fleshy collar and curving back to a flared bell, ringed by the growth of
/// every year; through them it shapes the calls carried across open sky long before the animal
/// itself is seen. Its blunt head overhangs a brow under which rows of small eyes sit sunk in
/// lidded sockets, looking down and ahead. Its three pairs of mouth tentacles, longer than any
/// other live-bearer's, hang like a pale beard from the mouth and throat, gathering soft plants,
/// and behind them swells the pleated throat sac it sings through. Its bladders swell, its horns
/// lengthen, its skin creases and air moss spreads over its back through its long life, so no two
/// elders are alike.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let age = c.life.z;
    let girth = girth(p.z);
    let middle = vec3(p.x, p.y - girth.z, 0.0);
    let reach = vec2(middle.x / girth.x.max(0.3), middle.y / girth.y.max(0.3)).length();
    let mut body = (reach - 1.0) * girth.y.min(girth.x).max(0.3) * 0.7;
    body = body.max(p.z - LENGTH - 2.0).max(-p.z - LENGTH);
    // The head: a broad dome over the brow, a heavy jaw beneath it.
    let head = rounded(p - vec3(0.0, 0.9, -12.2), vec3(4.6, 3.7, 4.4));
    body = smooth_min(body, head, 1.4);
    let brow = rounded(p - vec3(0.0, 1.7, -13.6), vec3(4.9, 1.1, 2.6));
    body = smooth_min(body, brow, 0.7);
    let jaw = rounded(p - vec3(0.0, -2.2, -14.6), vec3(2.6, 1.3, 2.4));
    body = smooth_min(body, jaw, 0.9);
    // The throat sac, slack between calls, its pleats opening out as it fills.
    let throat = p - vec3(0.0, -4.0, -9.0);
    let pleats = (throat.x * 3.2).cos() * 0.09 * (-throat.y - 1.0).clamp(0.0, 1.0);
    body = smooth_min(body, rounded(throat, vec3(3.0, 2.5, 3.8)) + pleats, 1.3);
    // Shoulders over the roots of each pair of wings.
    let flank = vec3(p.x.abs(), p.y, p.z);
    for pair in 0..2 {
        let (z, rise) = (-3.5 + pair as f32 * 8.0, 2.0 - pair as f32 * 0.5);
        let shoulder = rounded(flank - vec3(girth.x * 0.72, rise, z), vec3(1.9, 1.6, 2.6));
        body = smooth_min(body, shoulder, 1.2);
    }
    // A low keel along the belly, which steadies it and holds the streamers.
    let keel = capsule(p, vec3(0.0, -3.6, -5.0), vec3(0.0, -2.6, 9.0), 0.9);
    body = smooth_min(body, keel, 1.2);
    // The skin creases along its length with the years, and in rings where it bends.
    let creases = (p.y * 2.3 + (p.z * 0.4).sin()).sin() * (p.z * 0.9).cos().abs() * 0.07;
    body += creases * (0.4 + age * 0.6);
    // The bladders, sunk into the back and webbed to one another.
    let mut bladders = 1e6f32;
    if p.y > 0.5 {
        for i in 0..BLADDERS {
            let f = i as f32;
            let swell = (0.75 + age * 0.25 + c.gene(SWELL) * 0.2)
                * (1.0 + (f * 12.9898 + c.identity.x * 311.7).sin() * 0.18);
            let z = -7.5 + f * 4.6;
            let at = p - vec3(0.0, back(z) + 0.9 * swell, z);
            let size = vec3(3.4, 2.8, 2.5) * swell;
            // Vessels running over each one from crown to base.
            let around = at.x.atan2(at.z);
            let vessels = (around * 7.0).cos().abs() * 0.07 * size.y;
            bladders = smooth_min(bladders, rounded(at, size) - vessels, 0.9);
        }
        let web = capsule(
            p,
            vec3(0.0, back(-7.5) + 1.2, -7.5),
            vec3(0.0, back(10.9) + 0.8, 10.9),
            1.5,
        );
        bladders = smooth_min(bladders, web, 0.8);
    }
    body = smooth_min(body, bladders, 0.7);
    body = louvres(body, p - vec3(0.0, -0.4, -2.0), vec3(5.0, 3.0, 11.0));
    let mut hit = Surface::new(body, 0);
    // Air moss settles on the backs of the oldest, in patches that the years spread.
    let warp = vec2(p.x + (p.z * 1.7).sin() * 0.8, p.z + (p.x * 2.3).sin() * 0.6);
    let patches = (warp.x * 0.9 + warp.y * 0.5).sin() * (warp.y * 0.7 - warp.x * 1.1).sin()
        + (p.x * 3.7 + p.z * 2.9).sin() * (p.z * 4.1).sin() * 0.3;
    // As much of its back as the moss has covered, the patches spreading out from the highest.
    let mossy = (patches - 1.1 + c.scars.w * 1.3).min((p.y - 1.5) * 0.5);
    if mossy > 0.0 {
        hit.blend(body - mossy.min(0.12), 10, 0.08);
    }
    hit.socket(eyes(flank), 0.3);
    if p.y > 0.0 {
        // The horns grow out of the back through their collars, skin fading into horn.
        hit.blend(horns(flank, c), 7, 0.9);
    }
    // A broad paddle at the tail's end, which it trims its long glides with.
    let mut tail = 1e6f32;
    for i in 0..5 {
        let t = i as f32 / 5.0;
        tail = smooth_min(
            tail,
            tapered_tail(
                p,
                c.tail[i].truncate(),
                c.tail[i + 1].truncate(),
                2.0 * (1.0 - t) + 0.4,
                2.0 * (0.8 - t) + 0.4,
            ),
            0.5,
        );
    }
    // Two broad lobes at the tail's end, which it trims its long glides with.
    let end = c.tail[5].truncate();
    let lobe = vec3((p.x - end.x).abs(), p.y - end.y, p.z - end.z);
    let fluke = rounded(
        lobe - vec3(1.7, 0.0, 0.9 + lobe.x * 0.35),
        vec3(2.1, 0.22, 0.9),
    );
    hit.blend(smooth_min(tail, fluke, 0.5), 0, 0.6);
    hit.blend(oral_tentacles(p, c, MOUTH, 3.4, ORAL_PAIRS), 8, 0.3);
    hit
}

/// Two ranks of resonating horns, `p` folded onto the right flank. Each rises from a fleshy
/// collar on the back, curves up and back, and opens in a flared bell; its bore runs from the bell
/// down into the body, and every year of growth leaves a ring on it.
fn horns(p: Vec3, c: &CreatureInstance) -> f32 {
    let grown = 0.6 + c.life.z * 0.25 + c.gene(HORNS) * 0.3;
    let mut nearest = 1e6f32;
    for i in 0..4 {
        let f = i as f32;
        let z = -5.6 + f * 4.6;
        let foot = vec3(girth(z).x * 0.62, back(z) - 1.4, z);
        let length = (9.0 - f * 1.4) * grown;
        let rise = foot + vec3(0.7, length * 0.55, -0.2);
        let sweep = rise + vec3(0.4, length * 0.35, length * 0.45);
        let tip = sweep + vec3(0.1, length * 0.1, length * 0.3);
        if (p - rise).length() > length * 1.6 + 2.0 {
            continue;
        }
        let rings = |q: Vec3, a: Vec3| ((q - a).length() * 5.0).sin() * 0.04;
        let shaft = smooth_min(
            tapered_tail(p, foot, rise, 1.0, 0.7),
            smooth_min(
                tapered_tail(p, rise, sweep, 0.7, 0.5),
                tapered_tail(p, sweep, tip, 0.5, 0.42),
                0.3,
            ),
            0.3,
        ) + rings(p, foot);
        // The bell, flaring wide at the mouth.
        let axis = (tip - sweep).normalize();
        let along = (p - tip).dot(axis);
        let off = (p - tip - axis * along).length();
        let widening = ((along + 2.0) / 2.2).clamp(0.0, 1.0);
        let flare = 0.4 + widening.powi(3) * 0.9;
        let bell = ((off - flare).abs() - 0.1)
            .max(along - 0.2)
            .max(-along - 2.0)
            * 0.7;
        let bore = tapered_tail(p, foot + (rise - foot) * 0.5, tip, 0.34, 0.3);
        let horn = smooth_min(shaft, bell, 0.25).max(-bore);
        let collar = rounded(p - foot - vec3(0.0, 0.9, 0.0), vec3(1.7, 1.3, 1.9));
        nearest = nearest.min(smooth_min(horn, collar, 0.6));
    }
    nearest
}

/// Rows of small eyes along each side of the head, `p` folded onto the right, set into its
/// curve under the brow and looking down and ahead, as a tetralate's eyes do: the front ones
/// largest, watching where it flies, the rest watching the forest drift by below.
fn eyes(p: Vec3) -> f32 {
    let mut nearest = 1e6f32;
    for i in 0..6 {
        let f = i as f32;
        let z = -15.0 + f * 0.85;
        let y = 0.55 - f * 0.12 - (i % 2) as f32 * 0.55;
        let across = 1.0 - ((z + 12.2) / 4.4).powi(2) - ((y - 0.9) / 3.7).powi(2);
        let at = vec3(4.6 * across.max(0.0).sqrt(), y, z);
        nearest = nearest.min((p - at).length() - 0.3 * (1.0 - f * 0.07));
    }
    nearest
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.27, -10.98), vec3(0.92, 0.85, 2.71)),
    site(PUMP, vec3(0.00, -0.27, -5.62), vec3(1.77, 1.77, 3.09)),
    site(CROP, vec3(0.00, -0.27, 1.88), vec3(1.91, 1.70, 4.95)),
    pair(BREATHER, vec3(3.00, 0.54, -8.04), vec3(0.80, 0.80, 0.80)),
    site(FLOAT, vec3(0.00, 3.52, -3.48), vec3(0.63, 0.63, 0.63)),
    pair(
        FLIGHT_MUSCLE,
        vec3(3.00, 1.62, -3.75),
        vec3(0.81, 0.81, 0.81),
    ),
    site(CISTERN, vec3(0.00, -0.27, 7.77), vec3(1.44, 1.44, 3.09)),
    site(BROOD, vec3(0.00, -0.27, 12.05), vec3(0.99, 0.88, 1.92)),
    site(FLOAT, vec3(0.00, 3.52, -6.70), vec3(0.54, 0.54, 0.54)),
    site(FLOAT, vec3(0.00, 3.52, 6.70), vec3(0.66, 0.66, 0.66)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::coat(
    vec4(0.75, 0.73, 0.64, 0.0),
    vec3(0.94, 0.92, 0.83),
    [vec4(26.0, 8.0, -3.5, 0.98), vec4(18.0, 6.0, 4.5, 0.98)],
)
.hinged([vec2(5.0, 1.5), vec2(4.6, 1.3)])
.styled([vec4(0.4, 0.9, 6.0, 0.0), vec4(0.5, 0.8, 4.0, 0.0)])
.skinned(vec4(1.0, 1.9, 0.6, 0.10))
.flying(vec4(0.18, 0.40, 0.45, 1.30))
.marked(vec4(4.00, 4.00, 0.40, 0.42), vec3(0.98, 0.97, 0.92))
.brooding(vec4(2.8, 12.0, 2.2, 1.6));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render_wings(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::{
        Kind,
        animalia::physiology::{Light, Physiology},
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        light: Light::Sunlit,
        speed: 8.0,
        eats: |_, food| matches!(food, Kind::Aerothallus | Kind::Laevivelum),
        ..super::super::PHYSIOLOGY
    };
}
