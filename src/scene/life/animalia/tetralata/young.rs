//! The young of the four-winged animals: eggs, kept as each lineage keeps them, then a grub-like
//! larva that recalls the lineage's limbless ancestors.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use core::f32::consts::{PI, TAU};
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// A grub-like larva.
pub const LARVA: u32 = 3;

/// An egg tethered to living vegetation.
pub const TETHERED_EGG: u32 = 4;

/// An egg cast loose under a crown of threads.
pub const DRIFTING_EGG: u32 = 8;

/// Strings of eggs trailing behind the parent.
pub const EGG_STRINGS: u32 = 9;

/// A clutch glued together, fallen from the parent's body.
pub const CLUTCH: u32 = 10;

/// The larva, which recalls the lineage's limbless ancestors.
///
/// A plump, ringed grub wriggling through the vegetation it hatched in, the cowl only a bud over
/// its blunt head, the wing pairs rounded buds folded along its back and the mouth tentacles short
/// stubs, all growing towards the adult's as it nears its change. Two dark eyes already look down.
pub fn larva(p: Vec3, age: f32) -> Surface {
    let grow = ((age - 0.3) / 0.4).clamp(0.0, 1.0);
    let mut body = 1e6f32;
    for index in 0..9 {
        let along = index as f32 / 8.0;
        let radius = 0.13 + (along * PI).sin().powf(0.6) * 0.1 - along * 0.03;
        body = smooth_min(
            body,
            ellipsoid(
                p - vec3(0.0, 0.0, (along - 0.45) * 1.4),
                vec3(radius, radius * 0.92, 0.12),
            ),
            0.035,
        );
    }
    let head = p - vec3(0.0, 0.02, -0.68);
    body = smooth_min(body, ellipsoid(head, vec3(0.17, 0.15, 0.14)), 0.05);
    let cowl = ellipsoid(
        head - vec3(0.0, 0.08, -0.04),
        vec3(0.14 + grow * 0.12, 0.04, 0.12 + grow * 0.06),
    );
    body = smooth_min(body, cowl, 0.04);
    let flank = vec3(p.x.abs(), p.y, p.z);
    for stub in 0..2 {
        let root = vec3(0.05 + stub as f32 * 0.05, -0.1, -0.76);
        body = smooth_min(
            body,
            (flank - root - vec3(0.02, -0.05, -0.06) * ((root.z - flank.z) / 0.06).clamp(0.0, 1.0))
                .length()
                - 0.014,
            0.02,
        );
    }
    let mut hit = Surface::new(body, 0);
    // Wing buds, rounded lobes lying back along the upper flanks.
    for pair in 0..2 {
        let lobe = flank - vec3(0.14 + grow * 0.04, 0.13, -0.32 + pair as f32 * 0.32);
        let (sine, cosine) = 0.5f32.sin_cos();
        let tilted = vec3(
            lobe.x * cosine + lobe.y * sine,
            lobe.y * cosine - lobe.x * sine,
            lobe.z,
        );
        hit.join(
            ellipsoid(tilted, vec3(0.05 + grow * 0.14, 0.04, 0.07 + grow * 0.05)),
            2,
        );
    }
    hit.join(
        ellipsoid(flank - vec3(0.09, -0.08, -0.79), vec3(0.04, 0.035, 0.035)),
        3,
    );
    hit
}

/// An egg, laid as most of the lineage lays them: tethered to living vegetation, safe from the long
/// fall to the aur, in a leathery, ridged case.
pub fn tethered_egg(p: Vec3) -> Surface {
    let mut hit = Surface::new(ellipsoid(p, vec3(0.46, 0.4, 0.85)), 1);
    let tether = ((p - vec3(0.0, 0.36, -0.55))
        - vec3(0.0, 0.55, -0.4) * ((p.y - 0.36) / 0.55).clamp(0.0, 1.0))
    .length()
        - 0.025;
    let anchor = ellipsoid(p - vec3(0.0, 0.93, -0.97), vec3(0.18, 0.03, 0.18));
    hit.join(tether.min(anchor), 0);
    hit
}

/// An egg cast loose on the updraft, held up by a crown of fine threads that the rising air
/// catches, as down holds up a seed.
pub fn drifting_egg(p: Vec3, time: f32) -> Surface {
    let rise = p - vec3(0.0, -0.3, 0.0);
    let shell = ellipsoid(rise, vec3(0.26, 0.3, 0.26))
        + (rise.y * 40.0 + rise.x.atan2(rise.z) * 3.0).sin() * 0.006;
    let mut hit = Surface::new(shell, 1);
    let sector = TAU / 11.0;
    let turn = (p.z.atan2(p.x) / sector).round();
    let (sine, cosine) = (turn * sector).sin_cos();
    let out = p.x * cosine + p.z * sine;
    let across = p.z * cosine - p.x * sine;
    let lean = ((turn * 5.31).sin() * 43_758.547).fract_gl();
    let sway = (time * 1.3 + turn).sin() * 0.04;
    // Each thread rises from the egg's crown and arcs outwards.
    let up = ((p.y + 0.02) / 0.8).clamp(0.0, 1.0);
    let line = vec2(
        out - (0.08 + up * (0.55 + lean * 0.3) - up * up * 0.15),
        across - sway * up,
    )
    .length()
        - 0.012 * (1.0 - up * 0.6);
    hit.join(line.max(-p.y - 0.02).max(p.y - 0.78) * 0.7, 0);
    hit
}

/// Strings of eggs paid out behind the parent, dozens to a string at no even spacing, each string
/// sagging and swaying its own way in the wake.
pub fn egg_strings(p: Vec3, time: f32) -> Surface {
    let mut hit = Surface::new(1e6, 1);
    let behind = ((p.z + 0.9) / 1.85).clamp(0.0, 1.0);
    for string in 0..3 {
        let lane = string as f32;
        let centre = vec2(
            (lane - 1.0) * 0.18 * behind
                + (time * 1.1 + p.z * 4.0 + lane * 2.0).sin() * 0.07 * behind,
            -behind * behind * (0.25 + lane * 0.1)
                + (time * 0.8 + p.z * 3.0 + lane).cos() * 0.05 * behind,
        );
        let strand = (vec2(p.x, p.y) - centre).length();
        hit.join(
            vec2(strand - 0.012, (p.z.abs() - 0.92).max(0.0)).length(),
            0,
        );
        let spacing = 0.065;
        let cell = (p.z / spacing).round();
        let seed = ((cell * 12.9898 + lane * 78.233).sin() * 43_758.547).fract_gl();
        if seed > 0.25 && (cell * spacing).abs() < 0.9 {
            let along = p.z - cell * spacing - (seed - 0.5) * spacing * 0.4;
            let egg = vec3(p.x - centre.x, p.y - centre.y + (seed - 0.6) * 0.03, along).length()
                - (0.022 + seed * 0.022);
            hit.join(egg, 1);
        }
    }
    hit
}

/// A clutch fallen from its parent's body, still glued together.
pub fn clutch(p: Vec3) -> Surface {
    let mut eggs = 1e6f32;
    for egg in 0..7 {
        let turn = egg as f32;
        let at = if egg == 0 {
            Vec3::ZERO
        } else {
            let angle = turn * TAU / 6.0;
            vec3(
                angle.cos() * 0.42,
                (turn * 1.7).sin() * 0.14,
                angle.sin() * 0.42,
            )
        };
        eggs = smooth_min(eggs, ellipsoid(p - at, vec3(0.24, 0.22, 0.3)), 0.03);
    }
    Surface::new(eggs, 1)
}
