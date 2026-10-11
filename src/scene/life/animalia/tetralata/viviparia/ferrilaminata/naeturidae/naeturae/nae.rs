use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    oral_pairs: 4,
    speed: 30.0,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

use super::{feather, feathered_tail, wings};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{CreatureInstance, Rank, Taxon, ellipsoid, prepare_pose, smooth_min};
use crate::scene::surface::{Hit, Material, Ray, Surface, TraceResult, trace_material};

pub const TAXON: Taxon = Taxon {
    name: "Næ",
    english_name: "Nae",
    description: "Næ are small, intelligent, four-winged animals with pear-shaped bodies, swept feather crests and long gripping tails. \
        Eight sensitive oral tentacles emerge around the mouth and throat; the longest pair reaches outward to hold tools while shorter pairs stabilise objects and guide food into the digestive crop. \
        Fine sensory pads and independently controlled tips allow delicate craftwork. \
        Moisture-recovery folds and compact breathing channels support short flights. \
        Isolated aur sacs recharge in the Veiland mists and supply their magical abilities. \
        Families live with bonded Nætu, tending their companions and building sheltered homes on their backs.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
    radius: 3.2,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::organism(senses, BIOLOGY)
}

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 0.45, 1.5, 0);
    for point in &mut creature.tail {
        point.y -= 1.0;
        point.z -= 0.15;
    }
    creature.opaque_bounds = vec3(0.9, 1.2, 2.4).extend(0.0);
}

pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let chest = ellipsoid(point - vec3(0.0, 0.08, 0.0), vec3(0.23, 0.37, 0.17));
    let waist = ellipsoid(point - vec3(0.0, -0.24, 0.02), vec3(0.17, 0.20, 0.15));
    let head = ellipsoid(point - vec3(0.0, 0.64, -0.015), vec3(0.205, 0.26, 0.19));
    let neck = ellipsoid(point - vec3(0.0, 0.39, 0.0), vec3(0.115, 0.18, 0.12));
    let mut hit = Surface::new(
        smooth_min(
            smooth_min(chest, waist, 0.08),
            smooth_min(head, neck, 0.06),
            0.04,
        ),
        0,
    );
    hit = head_surface(point, creature, hit);
    hit.join(
        ellipsoid(point - vec3(0.0, 0.26, -0.17), vec3(0.044, 0.055, 0.026)),
        7,
    );
    let tail = feathered_tail(point, creature, vec3(0.20, 0.25, 1.0), 0.07, 0.38, 0.085);
    hit.join(tail.distance, tail.material);
    hit
}

fn head_surface(point: Vec3, creature: &CreatureInstance, mut hit: Surface) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let cheek = ellipsoid(
        mirrored - vec3(0.13, 0.59, -0.16),
        vec3(0.085, 0.065, 0.065),
    );
    hit.distance = smooth_min(hit.distance, cheek, 0.045);
    hit.join(
        ellipsoid(point - vec3(0.0, 0.54, -0.235), vec3(0.07, 0.014, 0.018)),
        8,
    );
    hit.join(
        ellipsoid(
            mirrored - vec3(0.128, 0.70, -0.18),
            vec3(0.058, 0.044, 0.032),
        ),
        3,
    );
    hit.join(
        ellipsoid(
            mirrored - vec3(0.128, 0.70, -0.213),
            vec3(0.027, 0.032, 0.009),
        ),
        9,
    );
    hit.join(
        ellipsoid(
            mirrored - vec3(0.128, 0.70, -0.222),
            vec3(0.009, 0.026, 0.004),
        ),
        3,
    );
    hit.join(
        oral_tentacles(
            point,
            creature,
            vec3(0.0, 0.56, -0.22),
            0.60,
            BIOLOGY.oral_pairs,
        ),
        2,
    );
    for i in 0..5 {
        let index = i as f32;
        hit.join(
            feather(
                vec3(point.x, -point.z, point.y),
                vec3((index - 2.0) * 0.035, -0.025, 0.84),
                0.18 - (index - 2.0).abs() * 0.028,
                0.035,
                (index - 2.0) * 0.18,
            ),
            1,
        );
    }
    hit
}

pub fn render_opaque(ray: Ray, creature: &CreatureInstance) -> Hit {
    trace_material(ray, creature, surface, |point, id| {
        let mask = (point.y - 0.58).smoothstep(0.0, 0.1) * (-point.z).smoothstep(0.10, 0.22);
        let mut m = Material::skin(
            vec3(0.16, 0.38, 0.40).lerp(vec3(0.80, 0.74, 0.52), mask),
            0.56,
        );
        let clothed = id == 0 && point.y > -0.32 && point.y < 0.30 && point.x.abs() < 0.24;
        match if clothed { 5 } else { id } {
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
                m.color = vec3(0.18, 0.72, 0.70);
                m.roughness = 0.16;
                m.emission = 0.15;
            }
            _ => {}
        }
        m
    })
}

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
