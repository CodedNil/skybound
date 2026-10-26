use crate::scene::life::{
    CreatureInstance, Look,
    biology::Biology,
    ellipsoid,
    plantae::{count, gene, leaf, rotate, sucker_pad},
    surface::{Ray, Surface, TraceResult},
};
#[cfg(not(target_arch = "spirv"))]
use crate::scene::life::{
    organs::{Viscera, site},
    plantae::physiology::{BLADDER, FROND, ROOT, STEM},
};
use core::f32::consts::TAU;
use spirv_std::glam::{Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const BIOLOGY: Biology = Biology {
    size: 0.31,
    maturity_days: 30.0,
    lifespan_days: 4000.0,
    adult_mass: 0.3,
    ..super::BIOLOGY
};

pub const RENDER_RADIUS: f32 = 12.0;

#[cfg(not(target_arch = "spirv"))]
pub use nature::PHYSIOLOGY;

taxon!(
    Species,
    "Flexilora spiralis",
    "Cicaia",
    [
        "Cicaia forms long spiralling tendrils that attach to trees or animals while loose coils move in the passing air.",
        "Piezoelectric fibres convert stretching, bending and twisting into chemical energy.",
        "Several flexible contact points spread the load, letting the plant exploit a host's motion without making every gust a tearing force.",
        "Strongly sheared currents near Rivers provide abundant movement but also increase the risk of breakage.",
        "Surviving fragments can attach elsewhere and grow, turning that mechanical risk into a means of reproduction.",
    ],
);

pub fn animate(c: &mut CreatureInstance) {
    c.opaque_bounds = vec3(6.0, 8.0, 6.0).extend(0.0);
}

/// A cicaia.
///
/// From a small knot spring long tendrils, each coiling round its own axis: the coils stretch,
/// bend and twist in the passing air, and the fibres in them turn that strain into the plant's
/// food. Pads along the coils and at their ends hold on to a host at several points. Each plant
/// coils its own way: one tendril or several, tight coils or loose, wound left or right, long or
/// short, bare or set with small leaves, gripping at the tips alone or all along.
pub fn surface(p: Vec3, c: &CreatureInstance) -> Surface {
    let mut shape = Surface::new(ellipsoid(p, vec3(0.45, 0.42, 0.45)), 0);
    let tendrils = count(c, 1.0, 1, 4);
    let thick = 0.06 + gene(c, 2.0) * 0.06;
    let leafy = gene(c, 3.0) > 0.5;
    let pads = count(c, 4.0, 1, 4);
    for i in 0..tendrils {
        let lane = i as f32;
        let hand = if gene(c, 5.0 + lane) > 0.5 { 1.0 } else { -1.0 };
        let turns = 2.0 + gene(c, 6.0 + lane) * 7.0;
        let coil = 0.25 + gene(c, 7.0 + lane) * 0.9;
        let long = 4.0 + gene(c, 8.0 + lane) * 3.5;
        // Most hang beneath the knot; some rise or reach out sideways along the host.
        let bearing = lane * TAU / tendrils as f32 + gene(c, 9.0) * TAU;
        let droop = -1.0 + 1.3 * (i % 2) as f32 * gene(c, 10.0);
        let axis = vec3(bearing.cos() * 0.5, droop, bearing.sin() * 0.5).normalize();
        // A frame about the axis; the shader cannot use `normalize_or_zero`, whose check for an
        // infinite length it cannot compile.
        let side = axis.cross(Vec3::Z);
        let side = if side.length_squared() < 0.04 {
            Vec3::X
        } else {
            side.normalize()
        };
        let other = axis.cross(side);
        let stretch = 1.0 + (c.animation.z * 0.9 + lane).sin() * 0.06;
        let at = |t: f32| {
            let angle = t * turns * TAU * hand;
            let radius = coil * (t * 6.0).min(1.0);
            axis * t * long * stretch + (side * angle.cos() + other * angle.sin()) * radius
        };
        // Skip a tendril that cannot be near the point.
        let reach = (p - axis * long * 0.5).length() - long * 0.6 - coil - 0.6;
        if reach > shape.distance {
            continue;
        }
        // The nearest turn of the coil: where along it the coil comes round to the point's
        // bearing about the axis, tried on the turns either side of where the point lies.
        let bearing = p.dot(other).atan2(p.dot(side)) * hand / TAU;
        let along = (p.dot(axis) / (long * stretch)).clamp(0.0, 1.0);
        let turn = (along * turns - bearing).round();
        for step in -1..=1 {
            let t = ((turn + step as f32 + bearing) / turns).clamp(0.0, 1.0);
            let width = thick * (1.0 - t * 0.6);
            shape.join(((p - at(t)).length() - width) * 0.8, 1);
        }
        // Leaves and gripping pads at fixed places along the coil.
        let out_of = |t: f32| {
            let out = at(t) - axis * t * long * stretch;
            out / out.length().max(1e-4)
        };
        if leafy {
            let t = ((along * 10.0).round() / 10.0).clamp(0.05, 1.0);
            let out = out_of(t);
            let local = rotate(p - at(t), out.z.atan2(out.x) - TAU / 4.0);
            shape.join(leaf(local, 0.5, 0.14, 0.1), 1);
        }
        let t = ((along * pads as f32).round() / pads as f32).max(1.0 / pads as f32);
        shape.join(sucker_pad(p - at(t), out_of(t), thick * 2.2), 3);
    }
    shape
}

/// Where its organs lie in its body; see [`Viscera`].
#[cfg(not(target_arch = "spirv"))]
pub const ORGANS: Viscera = Viscera::new(&[
    site(STEM, vec3(0.00, 0.00, 0.00), vec3(0.18, 0.18, 0.18)),
    site(BLADDER, vec3(0.00, 0.25, 0.00), vec3(0.08, 0.08, 0.08)),
    site(FROND, vec3(0.25, 0.00, 0.00), vec3(0.07, 0.07, 0.07)),
    site(ROOT, vec3(0.00, -0.25, 0.00), vec3(0.10, 0.08, 0.10)),
]);

/// Colouring and wings, shared by the preview and the world renderer.
pub const LOOK: Look = Look::plant(vec3(0.24, 0.44, 0.18), vec3(0.6, 0.7, 0.2));

pub fn render(ray: Ray, c: &CreatureInstance) -> TraceResult {
    super::render(ray, c, surface, LOOK)
}

#[cfg(not(target_arch = "spirv"))]
mod nature {
    use crate::scene::life::plantae::physiology::{Physiology, Propagation, epiphyte};

    pub const PHYSIOLOGY: Physiology = Physiology {
        powered: |climate| climate.wind.length_squared() * 0.03,
        propagation: Propagation::Fragmentation,
        attaches: true,
        partner: epiphyte,
        floats: false,
        ..super::super::PHYSIOLOGY
    };
}
