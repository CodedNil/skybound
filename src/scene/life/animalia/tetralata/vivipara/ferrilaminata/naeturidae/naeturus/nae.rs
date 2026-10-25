use super::{TailFeathers, render_tail_feathers, tail_shaft, wings};
use crate::scene::life::{
    CreatureInstance, Look,
    animalia::tetralata::{oral_tentacles, rig_tentacles},
    biology::Biology,
    eyes, mouth, prepare_pose, rounded, smooth_min,
    surface::{Hit, Material, Ray, Surface, TraceResult, WRINKLED, trace_material},
    torso,
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

/// The Næ's body: its half-length and the size of its cowl.
const LENGTH: f32 = 0.36;

const HEAD: f32 = 0.14;

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 0.63, 1.5, 0);
    creature.opaque_bounds = vec3(0.9, 1.2, 2.4).extend(0.0);
    rig_tentacles(creature, mouth(LENGTH, HEAD), 0.5, BIOLOGY.oral_pairs);
}

/// A Næ: a small tetralate laid along its flight like all its kin.
///
/// Its body is pear-shaped, heavy behind where the crop sits and narrowing to
/// the cowl, with eyes round the rim and a downward pair beneath. Under the
/// cowl hang the four pairs of mouth tentacles that are its hands, the long
/// outer pair for tools and the short inner ones to steady things and feed. A
/// swept crest runs back from the cowl, aur sacs glow on its flanks, and its
/// long gripping tail trails behind. It wears one thing it made: a woven girth
/// strap with a bronze clasp on the back.
pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let body = torso(point, vec3(0.2, 0.17, LENGTH), HEAD);
    let crop = rounded(point - vec3(0.0, -0.03, 0.14), vec3(0.27, 0.22, 0.25));
    let crest = rounded(point - vec3(0.0, 0.17, -0.2), vec3(0.016, 0.07, 0.24));
    let mut hit = Surface::new(smooth_min(smooth_min(body, crop, 0.1), crest, 0.03), 0);
    hit.join(eyes(point, LENGTH, HEAD), 3);
    let side = vec3(point.x.abs(), point.y, point.z);
    hit.join((side - vec3(0.25, 0.04, 0.06)).length() - 0.045, 9);
    let strap = vec2(
        vec2(point.x * 0.95, point.y + 0.02).length() - 0.255,
        point.z - 0.0,
    )
    .length()
        - 0.016;
    hit.join(strap, 5);
    hit.join(
        rounded(point - vec3(0.0, 0.215, 0.0), vec3(0.035, 0.012, 0.03)),
        7,
    );
    hit.join(
        oral_tentacles(
            point,
            creature,
            mouth(LENGTH, HEAD),
            0.5,
            BIOLOGY.oral_pairs,
        ),
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
    // A pale cowl over a darker belly: the lineage's countershading, turned
    // over for a world lit from below.
    let mask = (-point.z).smoothstep(0.36, 0.5) * point.y.smoothstep(-0.03, 0.06);
    let belly = (-point.y).smoothstep(0.0, 0.2);
    let mut m = Material::skin(
        vec3(0.16, 0.38, 0.40)
            .lerp(vec3(0.80, 0.74, 0.52), mask)
            .lerp(vec3(0.05, 0.13, 0.16), belly * 0.6),
        0.56,
    );
    m.texture = WRINKLED;
    m.relief = 0.3;
    m.grain = 1.6;
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
        vec3(0.18, 0.05, -0.08),
        Vec2::new(1.15, 0.42),
        vec3(0.24, 0.53, 0.48),
        &mut emit,
    );
    wings(
        ray,
        creature,
        vec3(0.17, 0.03, 0.15),
        Vec2::new(0.80, 0.30),
        vec3(0.31, 0.34, 0.54),
        &mut emit,
    );
}
