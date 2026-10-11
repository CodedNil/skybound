use spirv_std::glam::{Vec2, Vec3, vec3, vec4};

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::{
    CreatureInstance, Rank, Taxon, ellipsoid, eyes, feathers, gills, prepare_pose, smooth_min,
    tail, torso,
};
use crate::scene::surface::{
    Hit, Ray, Surface, TraceResult, trace, visit_paired_sheets, visit_shell,
};

pub const TAXON: Taxon = Taxon {
    name: "Nætu",
    english_name: "Naetu",
    description: "Nætu are enormous flying animals combining whale-like mass, dragon-like length, gentle bird-like wings and axolotl-like features. \
        Their patterned feathered tails droop behind bodies whose grassy upper fur safely anchors Næ buildings. \
        Internal sacs collect aur during careful dives into the Veiland mists, sustaining healing and a domed protective shield. \
        The shield shelters travelling families and helps keep them attached to the back. \
        Herds include both agile smaller carriers and huge steady individuals, bound to Næ through a close mutualistic relationship.",
    rank: Rank::Species,
    parent: Some(&super::TAXON),
    radius: 70.0,
};

pub fn behave(senses: &Senses) -> Controls {
    senses.companion.map_or_else(
        || behaviour::seek_aur(senses, 6.0),
        |companion| {
            let speed = ((companion.length() - 45.0) * 0.4).clamp(0.0, 24.0);
            let mut controls = behaviour::steer(senses, Some(companion), speed);
            controls.response = 1.2;
            if speed == 0.0 {
                controls.facing = Vec3::ZERO;
            }
            controls
        },
    )
}

pub fn animate(c: &mut CreatureInstance) {
    prepare_pose(c, 20.0, 25.0, 0);
    c.opaque_bounds = vec3(12.0, 11.0, 48.0).extend(0.0);
}

pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut hit = Surface::new(torso(p, vec3(5.8, 5.2, 20.0), 2.0), 0);
    if p.z > 9.0 {
        hit.distance = smooth_min(hit.distance, tail(p / vec3(3.0, 3.0, 1.0), c), 1.0);
        hit.join(feathers(p, c, 9.0), 1);
    }
    if p.z < -11.0 {
        hit.join(gills(p / 1.5, 20.0 / 1.5, 3) * 1.5, 2);
        hit.join(eyes(p, 20.0, 2.0), 3);
        hit.join(
            ellipsoid(
                vec3(p.x.abs(), p.y, p.z) - vec3(1.0, 2.0, -25.5),
                vec3(0.30, 0.14, 0.10),
            ),
            3,
        );
    }
    if hit.material == 0 && p.y > 3.5 {
        hit.material = 10;
    }
    hit.join(ellipsoid(p - vec3(0.0, 5.12, 2.0), vec3(0.9, 0.16, 0.9)), 9);
    hit
}

pub fn render_opaque(ray: Ray, c: &CreatureInstance) -> Hit {
    trace(
        ray,
        c,
        surface,
        vec4(0.035, 0.23, 0.29, 0.0),
        vec3(0.16, 0.39, 0.37),
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
        vec3(0.0, 0.5, -4.0),
        Vec2::new(35.0, 8.0),
        (vec3(0.16, 0.39, 0.37)).extend(0.98),
        &mut emit,
    );
    visit_paired_sheets(
        ray,
        c,
        vec3(0.0, 0.5, 10.0),
        Vec2::new(23.0, 6.0),
        (vec3(0.16, 0.39, 0.37)).extend(0.98),
        &mut emit,
    );
    visit_shell(
        ray,
        c,
        vec3(0.0, 7.0, 3.0),
        vec3(10.0, 8.0, 16.0),
        (vec3(0.25, 0.68, 0.9)).extend(0.035),
        4.8,
        &mut emit,
    );
}
