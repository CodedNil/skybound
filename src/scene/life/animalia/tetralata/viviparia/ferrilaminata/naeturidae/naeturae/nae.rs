use spirv_std::glam::{Vec2, Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, capsule, ellipsoid, prepare_pose, smooth_min, tapered_tail,
};
use crate::scene::surface::{Hit, Ray, Surface, TraceResult, trace, visit_paired_sheets};

pub const TAXON: Taxon = Taxon {
    name: "Næ",
    english_name: "Nae",
    description: "Næ are small intelligent humanoids whose former legs have become an additional pair of arms and hands. \
        Small wings provide short-distance flight, while a long feathered tail grips surfaces and anchors the body. \
        They craft tools and can collect aur for devices such as cannons. \
        Families live with bonded Nætu, carefully grooming and decorating their companions with flowers, trophies and attached structures.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
    radius: 3.2,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::forage(senses, 30.0)
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 0.45, 1.5, 0);
    c.opaque_bounds = vec3(0.8, 1.0, 2.4).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.22, 0.42, 0.18)), 0);
    hit.distance = smooth_min(
        hit.distance,
        ellipsoid(p - vec3(0.0, 0.55, -0.02), vec3(0.25, 0.27, 0.23)),
        0.06,
    );
    let q = vec3(p.x.abs(), p.y, p.z);
    for i in 0..2 {
        let y = 0.12 - i as f32 * 0.34;
        let root = vec3(0.18, y, 0.0);
        let elbow = vec3(0.43, y - 0.16, 0.04);
        let palm = vec3(0.54, y - 0.38, -0.02);
        hit.join(
            capsule(q, root, elbow, 0.055).min(capsule(q, elbow, palm, 0.045)),
            0,
        );
        hit.join(ellipsoid(q - palm, vec3(0.065, 0.09, 0.035)), 0);
        for finger in 0..3 {
            let start = palm + vec3((finger as f32 - 1.0) * 0.04, -0.05, -0.015);
            hit.join(
                capsule(q, start, start + vec3(0.01, -0.11, -0.015), 0.012),
                0,
            );
        }
    }
    hit.join(
        ellipsoid(q - vec3(0.145, 0.59, -0.195), vec3(0.065, 0.065, 0.055)),
        3,
    );
    hit.join(
        ellipsoid(p - vec3(0.0, 0.52, -0.23), vec3(0.055, 0.07, 0.07)),
        0,
    );
    hit.join(
        capsule(
            p,
            vec3(-0.07, 0.43, -0.215),
            vec3(0.07, 0.43, -0.215),
            0.015,
        ),
        8,
    );
    hit.join(
        ellipsoid(q - vec3(0.17, 0.75, 0.02), vec3(0.09, 0.14, 0.05)),
        1,
    );
    hit.join(
        ellipsoid(p - vec3(0.0, 0.02, -0.12), vec3(0.23, 0.30, 0.11)),
        5,
    );
    hit.join(
        ellipsoid(p - vec3(0.0, 0.18, -0.235), vec3(0.045, 0.055, 0.03)),
        9,
    );
    for crest in 0..3 {
        let offset = crest as f32 * 0.08;
        hit.join(
            tapered_tail(
                p,
                vec3(0.0, 0.74, offset),
                vec3(0.0, 0.93 - offset * 0.6, 0.18 + offset),
                0.035,
                0.004,
            ),
            1,
        );
    }
    for i in 0..5 {
        let tail_start = c.tail[i].truncate() * vec3(0.20, 0.25, 1.0);
        let tail_end = c.tail[i + 1].truncate() * vec3(0.20, 0.25, 1.0);
        hit.join(tapered_tail(p, tail_start, tail_end, 0.065, 0.035), 4);
        for index in 0..2 {
            let side = if index == 0 { -1.0 } else { 1.0 };
            let feather = tail_end + vec3(side * 0.14, -0.03, 0.12);
            let feather_point = vec3(
                (p.x - feather.x) * 1.8,
                (p.y - feather.y) * 5.0,
                p.z - feather.z,
            );
            hit.join(ellipsoid(feather_point, vec3(0.20, 0.06, 0.28)) / 5.0, 1);
        }
    }
    hit
}

pub fn render_opaque(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.58, 0.43, 0.32, 0.0),
        vec3(0.24, 0.49, 0.58),
    )
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::opaque(render_opaque(ray, c));
    let thin = render_thin(ray, c);
    for i in 0..thin.count as usize {
        result.add(thin.layers[i]);
    }
    result
}

pub fn render_thin(ray: Ray, c: &CreatureInstance) -> TraceResult {
    let mut result = TraceResult::MISS;
    visit_thin(ray, c, |hit| result.add(hit));
    result
}

pub fn visit_thin<F: FnMut(Hit)>(ray: Ray, c: &CreatureInstance, mut emit: F) {
    visit_paired_sheets(
        ray,
        c,
        vec3(0.0, 0.23, 0.10),
        Vec2::new(1.25, 0.40),
        vec3(0.21, 0.47, 0.58).extend(0.98),
        &mut emit,
    );
    visit_paired_sheets(
        ray,
        c,
        vec3(0.0, -0.13, 0.34),
        Vec2::new(0.85, 0.30),
        vec3(0.45, 0.31, 0.59).extend(0.98),
        &mut emit,
    );
}
