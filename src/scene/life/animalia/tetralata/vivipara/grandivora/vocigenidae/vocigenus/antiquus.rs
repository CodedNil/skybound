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
use crate::scene::{
    life::{
        CreatureInstance, Look,
        animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
        biology::Biology,
        capsule,
        genome::Gene,
        louvres, prepare_pose, rounded, smooth_min,
        surface::{Ray, Surface, TraceResult, fbm},
        tapered_tail,
    },
    math::Power,
};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3, vec4};
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
    let flank = vec3(p.x.abs(), p.y, p.z);
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
    for pair in 0..2 {
        let (z, rise) = (-3.5 + pair as f32 * 8.0, 2.0 - pair as f32 * 0.5);
        let shoulder = rounded(flank - vec3(girth.x * 0.72, rise, z), vec3(1.9, 1.6, 2.6));
        body = smooth_min(body, shoulder, 1.2);
    }
    // A low keel along the belly, which steadies it and holds the streamers.
    let keel = capsule(p, vec3(0.0, -3.6, -5.0), vec3(0.0, -2.6, 9.0), 0.9);
    body = smooth_min(body, keel, 1.2);
    body += hide(p, flank, girth, age);
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
    moss(&mut hit, p, girth, c.scars.w);
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

/// Two ranks of resonating horns, `p` folded onto the right flank.
///
/// Each grows out of the back from a broad, fleshy base, tapering as it sweeps up and back with a
/// slight twist, and opens at its tip in a modest lip: the hollow horn its calls resonate in. Every
/// year of growth leaves a ridge around it, wound a little about it as the horn twists, crowded
/// near the base and fading towards the smooth young tip.
fn horns(p: Vec3, c: &CreatureInstance) -> f32 {
    // A calf is born with only nubs where its horns will be, and they lengthen fastest as it
    // nears maturity, when it first needs its full voice.
    let calf = c.life.y;
    let grown = (0.6 + c.life.z * 0.25 + c.gene(HORNS) * 0.3) * (0.12 + 0.88 * calf * calf);
    let stout = 0.45 + 0.55 * calf;
    let mut nearest = 1e6f32;
    for rank in 0..4 {
        let z = -5.6 + rank as f32 * 4.6;
        let foot = vec3(girth(z).x * 0.55, back(z) - 1.2, z);
        let length = (9.0 - rank as f32 * 1.4) * grown;
        let bend = foot + vec3(0.9, length * 0.62, -0.4);
        let tip = foot + vec3(1.6, length * 0.78, length * 0.62);
        if (p - bend).length() > length * 1.1 + 2.5 {
            continue;
        }
        // The nearest point of the horn's curve, found by a few steps along it.
        let linear = (bend - foot) * 2.0;
        let quadratic = tip - bend * 2.0 + foot;
        let mut along =
            ((p - foot).dot(tip - foot) / (tip - foot).length_squared()).clamp(0.0, 1.0);
        for _ in 0..4 {
            let tangent = linear + quadratic * (2.0 * along);
            let off = foot + linear * along + quadratic * along * along - p;
            let curve = tangent.length_squared() + off.dot(quadratic) * 2.0;
            along = (along - off.dot(tangent) / curve.abs().max(1e-5)).clamp(0.0, 1.0);
        }
        let centre = foot + linear * along + quadratic * along * along;
        let axis = (linear + quadratic * (2.0 * along)).normalize();
        let offset = p - centre;
        // Its bearing about the curve, for ridges wound round it.
        let side = axis.cross(Vec3::Y).normalize();
        let turn = offset.dot(axis.cross(side)).atan2(offset.dot(side));
        let radius = (1.05 * (1.0 - along * 0.68) + (along - 0.85).max(0.0) * 1.6) * stout;
        let ridges = (along * 26.0 + turn * 0.35).sin().max(0.0) * 0.06 * (1.0 - along).sqrt();
        let horn = (offset.length() - radius - ridges) * 0.7;
        // The hollow inside, open at the lip.
        let bore = (offset.length() - radius * 0.6).max(0.82 - along);
        nearest = nearest.min(horn.max(-bore));
    }
    nearest
}

/// What the years and the bending of so great a body do to its hide, as displacement: deep folds
/// where it bends behind the head, round the wing roots and at the tail; long grooves down the
/// throat, as a whale's; wrinkles fanning from the eyes and the corners of the mouth; and rough,
/// lumpy callus on the brow and snout, which every cliff of kelp it has browsed has scoured. The
/// older it is the deeper all of it runs.
fn hide(p: Vec3, flank: Vec3, girth: Vec3, age: f32) -> f32 {
    let depth = 0.6 + age * 0.6;
    // Rings of folds where the body bends.
    let fold = |at: f32, spread: f32, every: f32| {
        let near = (-(p.z - at) * (p.z - at) / spread).exp();
        let wobble = (p.x * 0.8 + p.y * 0.6).sin() * 0.4;
        -((p.z - at + wobble) * every).sin().abs() * near * 0.09
    };
    let mut skin = fold(-9.4, 3.0, 6.0) + fold(12.5, 4.0, 5.0);
    // Folds gathered round each wing root.
    for pair in 0..2 {
        let (z, rise) = (-3.5 + pair as f32 * 8.0, 2.0 - pair as f32 * 0.5);
        let root = vec3(girth.x * 0.8, rise, z);
        let round = (flank - root).length();
        let near = (-round * round / 6.0).exp();
        skin -= (round * 4.5).sin().abs() * near * 0.07;
    }
    // Grooves down the throat, from the chin back to the throat sac.
    if p.y < -1.5 && p.z < -5.0 {
        let groove = (p.x * 4.2).cos().power(8);
        let span = (1.0 - ((p.z + 10.0) / 5.0).abs()).clamp(0.0, 1.0);
        skin -= groove * span * 0.08;
    }
    // Wrinkles fanning back from the eyes and the mouth.
    let eye = flank - vec3(4.0, 0.3, -13.8);
    let fan = (eye.y.atan2(eye.z) * 9.0).sin().abs();
    skin -= (1.0 - fan).power(4) * (-eye.length() * 0.9).exp() * 0.06;
    // Callus over the brow and snout.
    let callus = (-((flank - vec3(1.6, 2.4, -14.4)).length() - 1.8).max(0.0) * 1.5).exp();
    let lumps = fbm(p * 1.8) - 0.45;
    skin += lumps.max(0.0) * callus * 0.35;
    skin * depth
}

/// Air moss over its hide, as much as has grown on it, `cover`. It takes hold where the hide faces
/// the sky, fading out down the flanks rather than stopping at any line, and grows in thick, uneven
/// mats, deepest where it began and thinning to a fuzz at its edges.
fn moss(hit: &mut Surface, p: Vec3, girth: Vec3, cover: f32) {
    if cover <= 0.01 {
        return;
    }
    // How far round the body this point faces upward, from the belly to the crest of the back.
    let facing = ((p.y - girth.z) / girth.y.max(0.3)).clamp(-1.0, 1.0);
    let open = facing.smoothstep(-0.2, 0.7);
    let patches = fbm(p * 0.35 + vec3(3.1, 0.0, 7.7));
    let grown = (patches * open + cover * 0.75 - 0.75).max(0.0) * 3.0;
    if grown <= 0.0 {
        return;
    }
    let fuzz = (fbm(p * 6.0) - 0.5) * 0.05;
    let thick = grown.min(1.0) * 0.16 + fuzz * grown.min(1.0);
    let mat = hit.distance - thick;
    hit.blend(mat, 10, 0.06);
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
        let across = 1.0 - ((z + 12.2) / 4.4).power(2) - ((y - 0.9) / 3.7).power(2);
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
.skinned(vec4(1.0, 2.6, 1.3, 0.10))
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
