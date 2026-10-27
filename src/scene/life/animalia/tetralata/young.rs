//! The eggs of the four-winged animals, kept as each lineage keeps them. What hatches is a small
//! body of its parents' shape, drawn by the grown body's own tracer.

use crate::scene::life::{ellipsoid, smooth_min, surface::Surface};
use core::f32::consts::TAU;
use spirv_std::glam::{FloatExt, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

/// An egg tethered to living vegetation.
pub const TETHERED_EGG: u32 = 4;

/// An egg cast loose under a crown of threads.
pub const DRIFTING_EGG: u32 = 8;

/// Strings of eggs trailing behind the parent.
pub const EGG_STRINGS: u32 = 9;

/// A clutch glued together, fallen from the parent's body.
pub const CLUTCH: u32 = 10;

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
