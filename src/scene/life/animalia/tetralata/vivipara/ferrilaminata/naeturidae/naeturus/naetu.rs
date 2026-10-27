use super::{TailFeathers, feather, render_tail_feathers, tail_shaft, wings};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles, vivipara::ORAL_PAIRS},
    biology::{Biology, Litter},
    cowl, ellipsoid, eyes,
    genome::Gene,
    louvres, mouth, prepare_pose, rounded, slit, smooth_min,
    surface::{
        FURRED, Hit, Material, Ray, Surface, TraceResult, WRINKLED, trace_material, visit_shell,
    },
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
    animalia::tetralata::vivipara::ferrilaminata::naeturidae::AUR_SAC,
    organs::{Viscera, pair, site},
};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 7300.0,
    lifespan_days: 180_000.0,
    adult_mass: 150.0,
    // A single calf after a long carrying; twins are rare.
    litter: Litter::new(1, 1, [0.94, 0.06, 0.0, 0.0, 0.0, 0.0]),
    gestation_days: 540.0,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 70.0;

/// How full its gas chambers bulge, how large its aur sacs, how far its exchange frills fan, and
/// how deep the fibres of its back grow.
///
/// Full chambers carry more and lose less to a wound, large sacs hold more of the mists' charge,
/// long frills breathe more of the air it flies through, and a deep pelt holds the Næ's dwellings
/// fast.
pub const CHAMBERS: Gene = Gene(38);
pub const AUR_SACS: Gene = Gene(39);
pub const FRILLS: Gene = Gene(40);
pub const FIBRES: Gene = Gene(41);

#[cfg(not(target_arch = "spirv"))]
pub use nature::{ETHOLOGY, PHYSIOLOGY};

taxon!(
    Species,
    "Naetu",
    "Nætu",
    [
        "Nætu are enormous four-winged carriers with broad bodies, external exchange frills and trailing feathered tails.",
        "Three pairs of mouth tentacles gather vegetation, while redundant gas chambers and pleated breathing channels reduce the cost of slow flight.",
        "Grassy dorsal fibres anchor Næ dwellings above pale folds surrounding a large digestive crop.",
        "Carefully controlled dives skim the mists above the aur ocean to recharge isolated aur sacs, sustaining healing and a protective dome without contact with pure aur.",
        "Recognition through touch, grooming and vocal exchanges binds travelling herds to the Næ families that care for them.",
    ],
);

/// The Nætu's body: its length and breadth, and the size of its cowl.
const LENGTH: f32 = 19.0;

const HEAD: f32 = 2.4;

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 20.0, 25.0, 0);
    creature.opaque_bounds = vec3(14.0, 12.0, 52.0).extend(0.0);
    rig_tentacles(creature, mouth(LENGTH, HEAD), 3.0, ORAL_PAIRS);
}

/// A flying island: a lifting body far broader than it is deep.
///
/// Its back is flattened where the grassy fibres grow and the Næ build, and a deep keel beneath
/// houses the great crop inside its pale, pleated folds. Redundant gas chambers bulge along the
/// upper flanks, so no single wound can bring it down; the louvres of its breathing channels open
/// behind the cowl, and its aur sacs glow in a row along each side, charged on its dives through
/// the mists.
pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let along = (point.z / LENGTH).clamp(-1.0, 1.0);
    let breadth = 7.2 * (1.0 - 0.25 * along * along);
    let mut body = rounded(point - vec3(0.0, -0.3, 0.0), vec3(breadth, 5.0, LENGTH)) * 0.9;
    // The back flattened, a smooth shoulder down to the flanks.
    body = -smooth_min(-body, -(point.y - 5.15), 0.9);
    let keel = rounded(point - vec3(0.0, -3.4, -2.0), vec3(3.6, 2.6, 9.5));
    let mut folds = smooth_min(body, keel, 2.0);
    if point.y < -2.0 && point.x.abs() < 4.0 {
        folds += (1.0 - (point.z * 1.4).sin().abs()) * 0.1;
    }
    let mut d = folds;
    for i in 0..3 {
        let z = i as f32 * 10.0 - 10.0;
        let chamber = rounded(
            vec3(point.x.abs(), point.y, point.z) - vec3(5.0, 3.0, z),
            vec3(2.5, 1.7, 4.2) * (0.85 + creature.gene(CHAMBERS) * 0.3),
        );
        d = smooth_min(d, chamber, 1.4);
    }
    d = smooth_min(d, cowl(point, LENGTH, HEAD), 2.2);
    d = louvres(
        slit(d, point, LENGTH, HEAD),
        point,
        vec3(breadth, 5.0, LENGTH),
    );
    let fur = (point.x * 23.0 + (point.z * 7.0).sin() * 2.0).sin()
        * (point.z * 29.0 + (point.x * 3.0).sin() * 3.0).sin();
    if point.y > 4.4 {
        d -= fur.abs() * (0.008 + creature.gene(FIBRES) * 0.016);
    }
    let mut hit = Surface::new(d, if point.y > 4.4 { 10 } else { 0 });
    if point.z > 9.0 {
        let tail = tail_shaft(point, creature, vec3(3.0, 3.0, 1.0), 2.4);
        let distance = smooth_min(hit.distance, tail.distance, 0.7);
        hit.join(tail.distance, tail.material);
        hit.distance = distance;
    }
    if point.z < -12.0 {
        hit = head_surface(point, creature, hit);
    }
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    for i in 0..3 {
        let z = i as f32 * 5.5 - 8.5;
        hit.blend(
            rounded(
                mirrored - vec3(7.0, 1.2, z),
                vec3(0.8, 0.65, 1.4) * (0.8 + creature.gene(AUR_SACS) * 0.4),
            ),
            9,
            0.3,
        );
    }
    for i in 0..5 {
        let z = i as f32 * 3.1 - 10.0;
        hit.join(feather(mirrored, vec3(6.2, 3.1, z), 4.0, 0.55, 0.4), 1);
    }
    hit.blend(
        ellipsoid(point - vec3(0.0, 5.12, 2.0), vec3(0.9, 0.16, 0.9)),
        9,
        0.2,
    );
    hit
}

/// The head: the eyes around the cowl, the exchange frills fanning from behind it where the air it
/// flies through strikes them first, and the mouth tentacles beneath.
fn head_surface(point: Vec3, creature: &CreatureInstance, mut hit: Surface) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    hit.socket(eyes(point, LENGTH, HEAD), HEAD * 0.16);
    for i in 0..3 {
        let index = i as f32;
        let root = vec3(5.4, 0.6 + index * 0.75, -17.5 + index * 0.8);
        let tip = root
            + vec3(6.0 - index * 0.9, 1.2 + index * 1.1, 4.0) * (0.8 + creature.gene(FRILLS) * 0.4);
        let axis = (tip - root).normalize();
        let across = axis.cross(Vec3::Z).normalize();
        let normal = axis.cross(across);
        let relative = mirrored - root;
        let along = relative.dot(axis);
        let serration = 1.0 - (along * 5.0).sin().abs() * 0.22;
        let frond = vec3(
            relative.dot(across) / serration,
            relative.dot(normal),
            along - (tip - root).length() * 0.5,
        );
        let membrane = ellipsoid(frond, vec3(0.85, 0.12, (tip - root).length() * 0.55)) * 0.65;
        let stem = tapered_tail(mirrored, root, tip, 0.24, 0.035);
        hit.blend(smooth_min(membrane, stem, 0.10), 2, 0.15);
    }
    hit.blend(
        oral_tentacles(point, creature, mouth(LENGTH, HEAD), 3.0, ORAL_PAIRS),
        2,
        0.1,
    );
    hit
}

pub fn render_opaque(ray: Ray, creature: &CreatureInstance) -> Hit {
    let body = trace_material(ray, creature, surface, material);
    let feathers = render_tail_feathers(
        Ray {
            max_distance: body.distance().min(ray.max_distance),
            ..ray
        },
        creature,
        TailFeathers {
            scale: vec3(3.0, 3.0, 1.0),
            length: 7.0,
            width: 1.5,
        },
        material,
    );
    if feathers.is_hit() { feathers } else { body }
}

fn material(point: Vec3, id: u32) -> Material {
    let bands = (point.z * 0.7 + (point.x * 0.6).sin() * 2.0)
        .sin()
        .smoothstep(0.55, 0.85);
    let speckles = ((point.x * 2.3).sin() * (point.z * 3.1).sin()).abs();
    let belly = (-point.y).smoothstep(0.0, 3.4);
    let colour = vec3(0.035, 0.24, 0.28)
        .lerp(vec3(0.31, 0.61, 0.54), bands * 0.55)
        .lerp(vec3(0.81, 0.77, 0.59), belly);
    let mut m = Material::skin(colour, 0.4);
    m.texture = WRINKLED;
    m.relief = 0.3;
    m.grain = 0.45;
    match id {
        1 => {
            let vane = (point.z * 4.0 + point.x.abs() * 7.0).sin() * 0.035;
            let gold = (point.x.abs() * 0.55 + point.z * 0.3)
                .sin()
                .smoothstep(0.6, 0.9);
            m.color = vec3(0.08, 0.30, 0.36).lerp(vec3(0.63, 0.53, 0.25), gold) * (1.0 + vane);
            m.roughness = 0.65;
        }
        2 => {
            m.color = vec3(0.79, 0.29, 0.39).lerp(vec3(0.98, 0.72, 0.52), speckles);
            m.translucency = 0.4;
        }
        3 => {
            m.color = vec3(0.008, 0.025, 0.04);
            m.roughness = 0.08;
        }
        8 => {
            m.color = vec3(0.09, 0.065, 0.09);
        }
        9 => {
            m.color = vec3(0.19, 0.82, 0.86);
            m.emission = 0.7;
            m.roughness = 0.18;
        }
        10 => {
            m.color = vec3(0.07, 0.21, 0.12).lerp(vec3(0.23, 0.36, 0.16), speckles);
            m.roughness = 0.92;
            m.texture = FURRED;
            m.relief = 0.6;
            m.grain = 1.5;
        }
        _ => {}
    }
    m
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(GANGLION, vec3(0.00, 0.54, -17.18), vec3(1.11, 0.92, 1.11)),
    site(PUMP, vec3(0.00, -1.18, -8.82), vec3(1.55, 0.99, 3.00)),
    site(CROP, vec3(0.00, -0.75, 2.32), vec3(2.98, 1.91, 4.12)),
    pair(BREATHER, vec3(2.75, 1.50, -10.21), vec3(0.74, 0.74, 0.74)),
    site(FLOAT, vec3(0.00, 3.54, -0.46), vec3(0.70, 0.70, 0.70)),
    pair(
        FLIGHT_MUSCLE,
        vec3(2.75, 1.50, -4.64),
        vec3(0.92, 0.92, 0.92),
    ),
    site(CISTERN, vec3(0.00, -0.75, 10.68), vec3(1.55, 1.34, 4.30)),
    site(BROOD, vec3(0.00, -0.32, 16.25), vec3(1.55, 0.99, 1.69)),
    pair(AUR_SAC, vec3(4.75, -0.21, -0.93), vec3(1.43, 0.92, 1.49)),
]);

/// Drawn by its own tracers.
pub const LOOK: Look = Look::OWN;

pub fn render(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::opaque(render_opaque(ray, creature));
    let thin = render_thin(ray, creature);
    for i in 0..thin.count as usize {
        result.add(thin.layers[i]);
    }
    result
}

pub fn render_thin(ray: Ray, creature: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::MISS;
    visit_thin(ray, creature, |hit| result.add(hit));
    result
}

pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, creature: &CreatureInstance, mut emit: F) {
    wings(
        ray,
        creature,
        vec3(4.6, 0.7, -5.0),
        Vec2::new(34.0, 7.0),
        vec3(0.18, 0.52, 0.48),
        &mut emit,
    );
    wings(
        ray,
        creature,
        vec3(4.3, 0.2, 8.0),
        Vec2::new(22.0, 5.0),
        vec3(0.22, 0.41, 0.53),
        &mut emit,
    );
    visit_shell(
        ray,
        creature,
        vec3(0.0, 5.6, 3.0),
        vec3(10.0, 8.0, 16.0),
        (vec3(0.25, 0.68, 0.9)).extend(0.035),
        4.5,
        &mut emit,
    );
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    //! A Nætu's voice is a call so deep it is felt more than heard, rolling through the great gas
    //! chambers and carrying across a hundred kilometres of open sky. A herd strung out across the
    //! clouds keeps together by it, and the Næ who live on them ask them to call for them.

    use crate::scene::life::{
        animalia::{
            mind::{Ethology, Mind, Situation},
            physiology::Physiology,
            tetralata::{FOLLOW, vivipara::ferrilaminata::naeturidae::LOYALTY},
        },
        behaviour::Goal,
    };

    pub const PHYSIOLOGY: Physiology = Physiology {
        speed: 24.0,
        voice: 20_000.0,
        ..super::super::PHYSIOLOGY
    };

    pub const ETHOLOGY: Ethology = Ethology {
        act,
        ..super::super::ETHOLOGY
    };

    /// Making for the herd, a Nætu that has lost sight of it calls, and the herd answers.
    fn act(mind: &mut Mind, s: &Situation) -> Option<Goal> {
        let goal = (super::super::ETHOLOGY.act)(mind, s);
        // The more loyal, the sooner it calls after the herd drawing away.
        let limit = 600.0 - s.temper(LOYALTY) * 400.0;
        let apart = s.seen.kin.is_none_or(|kin| kin.length() > limit);
        if mind.intent == FOLLOW && apart {
            return Some(goal.unwrap_or_else(|| (FOLLOW.act)(mind, s)).calling());
        }
        goal
    }
}
