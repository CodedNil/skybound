use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::life::behaviour::{self, Controls, Senses};
use crate::scene::life::plantae::{rosette, rotate};
use crate::scene::life::{CreatureInstance, Rank, Taxon, capsule};
use crate::scene::surface::{Ray, Surface, TraceResult};

pub const TAXON: Taxon = Taxon {
    name: "Rubilion",
    english_name: "Brambles",
    description: "Brambles form dense spheres or chains of thorny woody vines that drift independently on strong currents. \
        Large lotus-shaped leaves catch the updraft and help support the framework. \
        Thorns carry a mild auric charge that deters large grazers. \
        Small, resilient animals shelter inside the tangles, while specialised Driftmunchers carefully strip the outer bark and spines.",
    rank: Rank::Genus,
    parent: Some(&super::TAXON),
    radius: 13.0,
};

pub fn behave(senses: &Senses) -> Controls {
    behaviour::seek_aur(senses, 0.15)
}

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(7.0, 7.0, 7.0).extend(0.0);
}

pub fn surface(p: Vec3, _: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(1e6, 0);
    for i in 0..6 {
        let local = rotate(p, i as f32 * 0.52);
        let radial = vec3(local.x, local.y * (0.9 + i as f32 * 0.03), 0.0).length();
        shape.join(vec3(radial - 3.5, 0.0, local.z).length() - 0.2, 0);
        for j in 0..6 {
            let angle = j as f32 * 1.047 + i as f32;
            let start = vec3(angle.cos() * 3.5, angle.sin() * 3.5, 0.0);
            let tip = start * 1.32;
            shape.join(capsule(local, start, tip, 0.09), 2);
        }
    }
    shape.join(rosette(p - vec3(0.0, 2.9, 0.0), 7, 5.0, 1.4, 0.8), 1);
    shape
}

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, vec3(0.16, 0.35, 0.1), vec3(0.4, 0.6, 0.14))
}
