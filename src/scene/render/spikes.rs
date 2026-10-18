//! Crystal spires: sphere traced through their altitude band and lit by the
//! glowing ocean they rise from.
use spirv_std::glam::{FloatExt, Vec3, Vec4Swizzles, vec2, vec3};
use spirv_std::num_traits::Float;

use crate::scene::aur::{SPIRE_CELL, energy_color};
use crate::scene::life::surface::Hit;
use crate::scene::planet::{light_position, surface_coordinates, surface_frame};
use crate::scene::render::ocean::caustics;
use crate::scene::{PLANET_RADIUS, ViewUniform, aur};

/// Beyond this a spire is narrower than a pixel and the fog hides it anyway.
const RANGE: f32 = 400_000.0;

/// Surface units stretch longitude towards the poles; scale them back to a
/// world-space bound.
fn field(point: Vec3) -> f32 {
    let metric = point.truncate().length() / point.length().max(1.0);
    aur::spire_field(surface_coordinates(point)) * metric.max(0.05)
}

pub fn trace(origin: Vec3, direction: Vec3, view: &ViewUniform, limit: f32, angle: f32) -> Hit {
    let ro = view.to_planet(origin);
    let rd = view.planet_rotation.conjugate() * direction;
    let height = ro.length();
    let band = aur::spire_band(
        height - PLANET_RADIUS,
        rd.dot(ro / height),
        limit.min(RANGE),
    );
    let mut t = band.x;
    let mut relaxed = 1.6;
    let mut stride = 0.0;
    let mut previous = 0.0;
    for _ in 0..160 {
        if t >= band.y {
            return Hit::MISS;
        }
        let point = ro + rd * t;
        let surface = surface_coordinates(point);
        let cell = (surface.truncate() / SPIRE_CELL).floor();
        let epsilon = (t * angle * 0.5).max(0.5);
        if aur::spire_count(cell) == 0 {
            // Cross open ocean one cell at a time: step straight to the
            // cell's far edge using the ray's velocity in surface units.
            let frame = surface_frame(point);
            let metric = point.truncate().length() / point.length().max(1.0);
            let velocity = vec2(
                frame.x_axis.dot(rd) / metric.max(0.05),
                -frame.y_axis.dot(rd),
            ) * (PLANET_RADIUS / point.length());
            let local = surface.truncate() - cell * SPIRE_CELL;
            let exit = |position: f32, speed: f32| {
                if speed > 1.0e-6 {
                    (SPIRE_CELL - position) / speed
                } else if speed < -1.0e-6 {
                    -position / speed
                } else {
                    f32::MAX
                }
            };
            let across = exit(local.x, velocity.x).min(exit(local.y, velocity.y));
            t += (across + 20.0).max(epsilon);
            relaxed = 1.6;
            stride = 0.0;
            previous = 0.0;
            continue;
        }
        // Over-relaxed sphere tracing: stride further while consecutive
        // spheres overlap, and fall back to the plain step when they do not.
        let distance = field(point);
        if relaxed > 1.0 && distance + previous < stride {
            t += stride * (1.0 - relaxed) / relaxed;
            relaxed = 1.0;
            previous = 0.0;
            continue;
        }
        if distance < epsilon {
            return shade(point, origin + direction * t, rd, t, view, epsilon);
        }
        stride = distance.max(epsilon) * relaxed;
        previous = distance;
        t += stride;
    }
    Hit::MISS
}

fn shade(point: Vec3, world: Vec3, rd: Vec3, t: f32, view: &ViewUniform, epsilon: f32) -> Hit {
    let e = epsilon.max(2.0);
    let a = vec3(1.0, -1.0, -1.0);
    let b = vec3(-1.0, -1.0, 1.0);
    let c = vec3(-1.0, 1.0, -1.0);
    let d = Vec3::ONE;
    let normal = (a * field(point + a * e)
        + b * field(point + b * e)
        + c * field(point + c * e)
        + d * field(point + d * e))
    .normalize();
    let up = point.normalize();
    let surface = surface_coordinates(point);
    let altitude = surface.z;
    let time = view.time();
    let light = (light_position(point) - point).normalize();
    let pole = view.pole_color.xyz().min(Vec3::splat(1.5));
    let ambient = view.ambient_color.xyz().min(Vec3::splat(0.8));

    // Obsidian-dark glass: little diffuse, a sharp sheen and a sky fresnel.
    // A thin-film tint drifts from violet to teal with the viewing angle.
    let film = (1.0 - (-rd).dot(normal).abs()).powi(2);
    let albedo = vec3(0.035, 0.03, 0.05).lerp(vec3(0.02, 0.06, 0.07), film);
    let reflected = rd - normal * (2.0 * rd.dot(normal));
    let fresnel = 0.04 + 0.96 * (1.0 - (-rd).dot(normal).max(0.0)).powi(5);
    let diffuse = albedo * (pole * normal.dot(light).max(0.0) + ambient);
    let sheen = pole * reflected.dot(light).max(0.0).powf(48.0) * 0.6;
    let sky = ambient
        * vec3(0.8, 0.9, 1.2).lerp(vec3(1.2, 0.8, 1.1), film)
        * (0.4 + 0.6 * reflected.dot(up).max(0.0))
        * fresnel;

    // The ocean below is the brightest thing in the sky. Faces turned down
    // and outward catch it, strongest near the waterline.
    let waterline = (-altitude.max(0.0) / 1800.0).exp();
    let facing_sea = 0.35 + 0.65 * (0.5 - 0.5 * normal.dot(up));
    let net = caustics(
        surface.truncate() * 0.0016 + vec2(altitude * 0.0004, 0.0),
        time * 0.6,
    );
    let sea = vec3(0.42, 0.36, 0.62)
        * view.sky_horizon.y
        * waterline
        * facing_sea
        * (0.55 + net * 1.8 * waterline);
    let rim = vec3(0.4, 0.62, 0.6) * view.sky_horizon.y * (-altitude.max(0.0) / 160.0).exp() * 1.5;

    // Veins of aur follow the twisted flutes and pulse upwards.
    let cell = (surface.truncate() / SPIRE_CELL).floor();
    let local = surface.truncate() - (cell + 0.5) * SPIRE_CELL;
    let mut vein = 0.0;
    let mut nearest = f32::MAX;
    for index in 0..aur::spire_count(cell) {
        let spire = aur::spire(cell, index);
        let gap = aur::spire_distance(&spire, local, altitude);
        if gap < nearest {
            nearest = gap;
            let (centre, u) = aur::spire_axis(&spire, altitude);
            let q = local - centre;
            let ridge = (spire.flutes * q.y.atan2(q.x) + spire.twist * u).cos();
            let pulse = 0.5 + 0.5 * (u * 38.0 - time * 1.7 + spire.twist).sin();
            vein = ridge.smoothstep(0.86, 0.99) * (0.25 + pulse * pulse) * (1.0 - u);
        }
    }
    let charge = energy_color(0.45 + 0.3 * (time * 0.3).sin(), 0.8);
    let emission = charge * vein * view.sky_horizon.y * 2.2 * (0.3 + waterline);

    let color = diffuse + sheen + sky + (sea + rim) * (albedo * 6.0 + fresnel) + emission;
    Hit {
        color_depth: color.extend(t),
        normal: view.planet_rotation * normal,
        previous_position: world,
        transmittance: Vec3::ZERO,
        part: 0,
    }
}
