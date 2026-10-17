use crate::scene::life::animalia::tetralata::oral_tentacles;
use crate::scene::life::biology::Biology;

pub const BIOLOGY: Biology = Biology {
    maturity_days: 7300.0,
    lifespan_days: 180_000.0,
    speed: 24.0,
    adult_mass: 150.0,
    ..super::BIOLOGY
};

use super::{TailFeathers, feather, render_tail_feathers, tail_shaft, wings};
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::{
    CreatureInstance, Rank, Taxon, ellipsoid, prepare_pose, smooth_min, tapered_tail,
};
use crate::scene::surface::{
    Hit, Material, Ray, Surface, TraceResult, trace_material, visit_shell,
};

pub const RENDER_RADIUS: f32 = 70.0;

pub const TAXON: Taxon = Taxon {
    name: "Naetu",
    english_name: "Nætu",
    description: "Nætu are enormous four-winged carriers with a broad, gently tapered body, soft external exchange frills and a trailing feathered tail. \
        Six oral tentacles gather vegetation and hold food against the broad mouth. \
        Grassy dorsal fibres anchor Næ dwellings, while pale ventral folds expand around a large digestive crop. \
        Redundant gas chambers and pleated respiratory channels support slow economical flight. \
        Aur sacs isolated from the nutrient circulation recharge during careful mist-skimming dives, sustaining healing and a protective dome over travelling families. \
        Herd members recognise their Næ companions through close contact, grooming and vocal exchanges.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
};

pub fn animate(creature: &mut CreatureInstance) {
    prepare_pose(creature, 20.0, 25.0, 0);
    creature.opaque_bounds = vec3(14.0, 12.0, 52.0).extend(0.0);
}

pub fn surface(point: Vec3, creature: &CreatureInstance) -> Surface {
    let mut body = ellipsoid(point, vec3(5.8, 5.2, 19.0));
    body = smooth_min(
        body,
        ellipsoid(point - vec3(0.0, 0.8, -18.5), vec3(4.1, 3.4, 5.6)),
        1.4,
    );
    body = smooth_min(
        body,
        ellipsoid(point - vec3(0.0, 0.0, -23.0), vec3(3.3, 2.1, 3.9)),
        1.0,
    );
    body += (point.x * 7.0).sin().powi(2) * (-point.y).smoothstep(1.5, 4.0) * 0.035;
    let fur = (point.x * 23.0 + (point.z * 7.0).sin() * 2.0).sin()
        * (point.z * 29.0 + (point.x * 3.0).sin() * 3.0).sin();
    if point.y > 3.5 {
        body -= fur.abs() * 0.015;
    }
    let mut hit = Surface::new(body, if point.y > 3.5 { 10 } else { 0 });
    if point.z > 9.0 {
        let tail = tail_shaft(point, creature, vec3(3.0, 3.0, 1.0), 2.4);
        let distance = smooth_min(hit.distance, tail.distance, 0.7);
        hit.join(tail.distance, tail.material);
        hit.distance = distance;
    }
    if point.z < -12.0 {
        hit = head_surface(point, creature, hit);
    }
    for i in 0..5 {
        let z = i as f32 * 3.1 - 10.0;
        let mirrored = vec3(point.x.abs(), point.y, point.z);
        hit.join(feather(mirrored, vec3(4.5, 3.1, z), 4.0, 0.55, 0.4), 1);
    }
    hit.join(
        ellipsoid(point - vec3(0.0, 5.12, 2.0), vec3(0.9, 0.16, 0.9)),
        9,
    );
    hit
}

fn head_surface(point: Vec3, creature: &CreatureInstance, mut hit: Surface) -> Surface {
    let mirrored = vec3(point.x.abs(), point.y, point.z);
    let eye = vec3(3.25, 1.8, -22.0);
    hit.join(ellipsoid(mirrored - eye, vec3(0.58, 0.61, 0.55)), 3);
    hit.join(
        ellipsoid(
            mirrored - eye - vec3(0.10, 0.02, -0.40),
            vec3(0.25, 0.33, 0.14),
        ),
        9,
    );
    hit.join(
        ellipsoid(
            mirrored - eye - vec3(0.03, 0.52, 0.04),
            vec3(0.72, 0.13, 0.52),
        ),
        0,
    );
    for i in 0..3 {
        let index = i as f32;
        let root = vec3(3.1, 0.4 + index * 0.75, -18.0 + index * 0.8);
        let tip = root + vec3(6.0 - index * 0.9, 1.2 + index * 1.1, 4.0);
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
        hit.join(smooth_min(membrane, stem, 0.10), 2);
    }
    hit.join(
        oral_tentacles(
            point,
            creature,
            vec3(0.0, -0.65, -25.6),
            3.0,
            BIOLOGY.oral_pairs,
        ),
        2,
    );
    let lip = vec3(
        mirrored.x,
        point.y + 0.7 + mirrored.x * mirrored.x * 0.08,
        point.z + 25.5,
    );
    hit.join(ellipsoid(lip, vec3(2.65, 0.10, 0.22)), 8);
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
        }
        _ => {}
    }
    m
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
