use super::{TailFeathers, render_tail_feathers, tail_shaft, wings};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    capsule, prepare_pose, rounded, smooth_min,
    surface::{Hit, Material, Ray, Surface, TraceResult, trace_material},
};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 5500.0,
    lifespan_days: 44000.0,
    oral_pairs: 4,
    speed: 30.0,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 3.2;

taxon!(
    Species,
    "Nae",
    "Næ",
    [
        "Næ are small, intelligent four-winged animals with pear-shaped bodies, swept crests and long gripping tails.",
        "Four pairs of sensitive mouth tentacles extend the live-bearing lineage's usual three pairs, with long outer tips holding tools while shorter tips steady objects and guide food.",
        "Moisture-recovery folds support short flights, although active travel still requires drinks from living basins.",
        "Isolated aur sacs support their magical abilities and are replenished through the protected life they share with bonded Nætu.",
        "Families groom their companions, tend sheltered homes among dorsal fibres and navigate between drifting communities by currents and the distant polar beams.",
    ],
);

/// Where the Næ's mouth opens beneath its hood, and its tentacles with it.
const MOUTH: Vec3 = vec3(0.0, 0.46, -0.13);

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 0.45, 1.5, 0);
    for point in &mut creature.tail {
        point.y -= 1.0;
        point.z -= 0.15;
    }
    creature.opaque_bounds = vec3(0.9, 1.2, 2.4).extend(0.0);
    rig_tentacles(creature, MOUTH, 0.5, BIOLOGY.oral_pairs);
}

/// A point in the frame of the Næ's hood, which leans forward and down
/// over its front.
fn hood_frame(point: Vec3) -> Vec3 {
    let q = point - vec3(0.0, 0.56, -0.08);
    let (sine, cosine) = 0.45f32.sin_cos();
    vec3(q.x, q.y * cosine - q.z * sine, q.y * sine + q.z * cosine)
}

/// A Næ: small and pear-shaped, held upright when perched with its weight
/// low in the crop. It has no face. A hood-like cowl leans forward over its
/// front, eyes set round the rim for every bearing and a larger pair
/// beneath looking down; under it hang the four pairs of mouth tentacles
/// that are its hands, the long outer pair for tools and the short inner
/// ones to steady things and feed. A swept crest runs back from the
/// hood's crown, aur sacs glow at the chest, and louvres on the flanks win
/// back moisture on short flights. It wears one thing it made: a woven
/// strap with a bronze clasp.
pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let belly = rounded(point - vec3(0.0, -0.16, 0.02), vec3(0.26, 0.3, 0.23));
    let chest = rounded(point - vec3(0.0, 0.2, 0.0), vec3(0.18, 0.22, 0.16));
    let neck = capsule(point, vec3(0.0, 0.3, 0.0), vec3(0.0, 0.5, -0.05), 0.09);
    let mut d = smooth_min(smooth_min(belly, chest, 0.12), neck, 0.07);
    let hood = hood_frame(point);
    let lift = hood.x * hood.x / 0.0676 * 0.03;
    d = smooth_min(
        d,
        rounded(hood - vec3(0.0, lift, 0.0), vec3(0.26, 0.065, 0.24)),
        0.05,
    );
    d = smooth_min(
        d,
        rounded(hood - vec3(0.0, 0.07, 0.16), vec3(0.018, 0.075, 0.22)),
        0.03,
    );
    let side = vec3(point.x.abs(), point.y, point.z);
    for i in 0..3 {
        let slant = side.y + 0.05 + i as f32 * 0.09 + side.z * 0.4;
        let louvre = (slant.abs() - 0.012)
            .max(side.z.abs() - 0.14)
            .max(0.2 - side.x);
        d = d.max(-louvre);
    }
    let mut hit = Surface::new(d, 0);
    let rim = vec3(hood.x.abs(), hood.y, hood.z);
    let mut eyes = (rim - vec3(0.08, -0.075, -0.05)).length() - 0.03;
    for i in 0..3 {
        let angle = 0.55 + i as f32 * 0.38;
        let x = angle.sin() * 0.25;
        eyes =
            eyes.min((rim - vec3(x, x * x / 0.0676 * 0.03, -angle.cos() * 0.23)).length() - 0.02);
    }
    hit.join(eyes, 3);
    hit.join((side - vec3(0.1, 0.12, -0.14)).length() - 0.045, 9);
    let strap = point - vec3(0.0, 0.08, 0.0);
    let (sine, cosine) = 0.6f32.sin_cos();
    let strap = vec3(
        strap.x * cosine - strap.y * sine,
        strap.x * sine + strap.y * cosine,
        strap.z,
    );
    hit.join(
        vec2(vec2(strap.x * 0.95, strap.z).length() - 0.215, strap.y).length() - 0.016,
        5,
    );
    hit.join(
        rounded(point - vec3(0.07, 0.15, -0.19), vec3(0.03, 0.03, 0.012)),
        7,
    );
    hit.join(
        oral_tentacles(point, creature, MOUTH, 0.5, BIOLOGY.oral_pairs),
        2,
    );
    let tail = tail_shaft(point, creature, vec3(0.20, 0.25, 1.0), 0.07);
    hit.join(tail.distance, tail.material);
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
            scale: vec3(0.20, 0.25, 1.0),
            length: 0.38,
            width: 0.085,
        },
        material,
    );
    if feathers.is_hit() { feathers } else { body }
}

fn material(point: Vec3, id: u32) -> Material {
    let mask = (point.y - 0.58).smoothstep(0.0, 0.1) * (-point.z).smoothstep(0.10, 0.22);
    let mut m = Material::skin(
        vec3(0.16, 0.38, 0.40).lerp(vec3(0.80, 0.74, 0.52), mask),
        0.56,
    );
    match id {
        2 => {
            m.color = vec3(0.34, 0.58, 0.54)
                .lerp(vec3(0.84, 0.65, 0.40), (-point.z).smoothstep(0.25, 0.7));
            m.roughness = 0.4;
            m.translucency = 0.25;
        }
        1 => {
            m.color = vec3(0.10, 0.35, 0.37).lerp(
                vec3(0.76, 0.54, 0.24),
                (point.z * 24.0).sin().smoothstep(0.0, 0.7),
            );
            m.roughness = 0.7;
        }
        3 => {
            m.color = vec3(0.006, 0.014, 0.021);
            m.roughness = 0.075;
        }
        4 => {
            m.color = vec3(0.23, 0.16, 0.12);
        }
        5 => {
            let stitch = ((point.x.abs() - 0.17).abs()).smoothstep(0.006, 0.018);
            m.color = vec3(0.74, 0.49, 0.18).lerp(vec3(0.035, 0.10, 0.16), stitch);
            m.roughness = 0.85;
        }
        7 => {
            m.color = vec3(0.84, 0.57, 0.18);
            m.metal = 0.65;
            m.roughness = 0.25;
        }
        8 => {
            m.color = vec3(0.17, 0.075, 0.055);
            m.roughness = 0.3;
        }
        9 => {
            // Aur sacs, glowing with what they hold.
            m.color = vec3(0.18, 0.72, 0.70);
            m.roughness = 0.16;
            m.emission = 0.6;
        }
        _ => {}
    }
    m
}

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
        vec3(0.18, 0.23, 0.12),
        Vec2::new(1.15, 0.42),
        vec3(0.24, 0.53, 0.48),
        &mut emit,
    );
    wings(
        ray,
        creature,
        vec3(0.16, -0.12, 0.25),
        Vec2::new(0.80, 0.30),
        vec3(0.31, 0.34, 0.54),
        &mut emit,
    );
}
