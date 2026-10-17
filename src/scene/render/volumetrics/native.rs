//! Native opacity and emission; the view grid supplies cloud illumination only.
use super::super::utils::{AtmosphereData, Textures, intersect_sphere, ray_shell_intersect};
use super::{RaymarchResult, aur_ocean, clouds, poles};
use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    cloud::{Cloud, RayEvent, ray_event},
    froxel,
};
use spirv_std::Image;
use spirv_std::glam::{Vec2, Vec3, Vec4, Vec4Swizzles};
use spirv_std::num_traits::Float;

const VEIL_STEP: f32 = 112.0;

fn cloud_density(
    event: &RayEvent,
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
    records: &[Cloud],
    grid: &[u32],
) -> f32 {
    if event.overflow {
        return clouds::extinction(point, view, footprint, textures, records, grid);
    }
    let mut sigma = 0.0;
    for index in 0..event.count {
        sigma += clouds::sample_cloud(
            records[event.ids[index as usize] as usize],
            point,
            view,
            footprint,
            textures,
        );
    }
    sigma
}

fn guidance(
    uv: Vec2,
    distance: f32,
    textures: &Textures,
    field: &Image!(3D, type=f32, sampled=true),
) -> Vec4 {
    let margin = Vec2::new(0.5 / froxel::WIDTH as f32, 0.5 / froxel::HEIGHT as f32);
    let coordinates = uv.clamp(margin, Vec2::ONE - margin).extend(
        froxel::depth(distance).clamp(0.5 / froxel::DEPTH as f32, 1.0 - 0.5 / froxel::DEPTH as f32),
    );
    field.sample_by_lod(*textures.sampler, coordinates, 0.0)
}

fn boundary(range: Vec2, t: f32, end: f32) -> f32 {
    if range.y <= range.x {
        return end;
    }
    if range.x > t + 0.25 {
        range.x.min(end)
    } else if range.y > t + 0.25 {
        range.y.min(end)
    } else {
        end
    }
}

pub fn integrate(
    uv: Vec2,
    origin: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    limit: f32,
    angle: f32,
    atmosphere: &AtmosphereData,
    textures: &Textures,
    field: &Image!(3D, type=f32, sampled=true),
    records: &[Cloud],
    grid: &[u32],
) -> RaymarchResult {
    let ro = view.to_planet(origin);
    let rd = view.planet_rotation.conjugate() * direction;
    let profile = view.camera_position.w as u32;
    let absent = Vec2::new(limit, 0.0);
    let shell = if profile & 2 == 0 {
        ray_shell_intersect(
            ro,
            rd,
            Vec3::ZERO,
            clouds::CLOUD_BOTTOM_HEIGHT,
            clouds::CLOUD_TOP_HEIGHT,
        )
    } else {
        Vec4::new(limit, 0.0, limit, 0.0)
    };
    let ocean = if profile & 4 == 0 {
        intersect_sphere(ro, rd, PLANET_RADIUS + aur_ocean::OCEAN_TOP_HEIGHT)
    } else {
        absent
    };
    let polar = poles::poles_raymarch_entry(ro, rd, limit);
    let mut result = RaymarchResult::empty(limit);
    let mut t = 0.0;
    let mut event = RayEvent {
        ids: [0; 8],
        count: 0,
        end: 0.0,
        step: 4000.0,
        overflow: false,
    };
    for _ in 0..512 {
        if t >= limit || result.color.w < 0.005 {
            break;
        }
        let point = ro + rd * t;
        let footprint = (t * angle).max(2.0);
        let inside_clouds =
            (t + 0.25 >= shell.x && t < shell.y) || (t + 0.25 >= shell.z && t < shell.w);
        let inside_ocean = t + 0.25 >= ocean.x && t < ocean.y;
        let inside_poles = t + 0.25 >= polar.x && t < polar.y;
        let mut end = boundary(
            shell.xy(),
            t,
            boundary(shell.zw(), t, boundary(ocean, t, boundary(polar, t, limit))),
        );
        if inside_clouds {
            if t + 0.25 >= event.end {
                event = ray_event(ro, rd, t, end, view.cloud_time.x, records, grid);
            }
            end = end.min(event.end);
        }
        let mut step = end - t;
        let mut ocean_height = 1.0e10;
        let mut ocean_shape = Vec4::ZERO;
        let mut ocean_palette = Vec4::ZERO;
        let mut curve = Vec3::new(1.0e10, 0.0, 0.0);
        if inside_clouds && event.count > 0 {
            step = step.min(event.step.max(footprint));
        }
        if inside_ocean {
            let altitude = point.length() - PLANET_RADIUS;
            let clearance = guidance(uv, t, textures, field).w;
            if clearance < 64.0 {
                curve = textures.ejection_field(
                    crate::scene::planet::surface_coordinates(point + rd * (VEIL_STEP * 0.5)),
                    crate::scene::aur::EJECTION_WIDTH * 6.0 + 4000.0,
                );
            } else {
                step = step.min(clearance);
            }
            // Bound the spatial change of the wave, then resolve the thin veil
            // deterministically instead of jittering long coarse depth slabs.
            let body_step = if altitude < 3700.0 {
                (ocean_shape, ocean_palette) =
                    super::super::cache::sample(point, view, footprint, textures);
                let wave = if ocean_palette.w > 0.0 {
                    ocean_shape.x * 4096.0
                } else {
                    aur_ocean::wave_height(point, view.time(), footprint, textures)
                };
                let height = altitude - wave;
                ocean_height = height;
                let radial_slope = rd.dot(point.normalize()).abs();
                let clearance = (height - 320.0).max(0.0) * 0.7 / (radial_slope + 0.35);
                clearance.max(VEIL_STEP + (-height).max(0.0) * 0.49 + footprint * 0.5)
            } else {
                (altitude - 3700.0).max(80.0)
            };
            step = step.min(body_step.min(4000.0));
            if curve.z > 0.0 && profile & 128 == 0 {
                step = step.min(
                    ((curve.x - crate::scene::aur::EJECTION_WIDTH * 2.0) * 0.45)
                        .max(100.0 + footprint),
                );
            }
        }
        if inside_poles {
            let radial = point.truncate().length();
            let width = 1200.0 + (point.z.abs() - PLANET_RADIUS).max(0.0).sqrt() * 5.0;
            step = step.min((radial * 0.2).max(width * 0.5).max(footprint));
            if radial < crate::scene::polar::REACTION_SPREAD * 1.6 {
                let shell_distance = (point.length() - crate::scene::polar::REACTION_RADIUS).abs();
                step = step.min(((shell_distance - 45_000.0) * 0.5).max(400.0 + footprint));
            }
        }
        step = step.max(0.5).min(end - t);
        if step <= 0.0 {
            t += 0.5;
            continue;
        }
        let distance = t + step * 0.5;
        let sample_point = ro + rd * distance;
        let mut sigma = 0.0;
        let mut scattering = Vec3::ZERO;
        if inside_clouds && event.count > 0 {
            // Two-point Gauss quadrature integrates smooth billows without
            // animated noise and resolves the entry/exit fade within each bound.
            let a = cloud_density(
                &event,
                ro + rd * (t + step * 0.211_325),
                view,
                footprint,
                textures,
                records,
                grid,
            );
            let b = cloud_density(
                &event,
                ro + rd * (t + step * 0.788_675),
                view,
                footprint,
                textures,
                records,
                grid,
            );
            let density = (a + b) * 0.5;
            sigma += density;
            if density > 0.0000001 {
                let light =
                    (crate::scene::polar::light_position(sample_point) - sample_point).normalize();
                let near = cloud_density(
                    &event,
                    sample_point + light * 1200.0,
                    view,
                    footprint,
                    textures,
                    records,
                    grid,
                );
                let shadow = 0.4 + 0.6 * (-near * 1800.0).exp();
                scattering += guidance(uv, distance, textures, field).xyz() * density * shadow;
            }
        }
        if inside_ocean {
            // The near-field arc query is centred on the short ocean step;
            // reuse it for emission as well as stepping instead of traversing
            // the same planetary candidates twice.
            let jet = if profile & 128 == 0 {
                aur_ocean::ejections(sample_point, view, footprint, curve, textures)
            } else {
                super::VolumeSample::default()
            };
            let density = jet.density * 0.018;
            sigma += density;
            scattering += jet.emission * density;
            if sample_point.length() - PLANET_RADIUS <= 3700.0 {
                let wave = if ocean_palette.w > 0.0 {
                    // The cache already provides a height gradient. Reuse it
                    // across this short segment instead of fetching the same
                    // two cache textures again at its midpoint.
                    let up = point.normalize();
                    let normal = ocean_shape.yzw().normalize();
                    let gradient = (up - normal / normal.dot(up).max(0.01)) * 0.1;
                    ocean_shape.x * 4096.0 + gradient.dot(rd) * step * 0.5
                } else {
                    aur_ocean::wave_height(sample_point, view.time(), footprint, textures)
                };
                let span = if ocean_height < 1.0e9 {
                    (sample_point.length() - PLANET_RADIUS - wave - ocean_height).abs()
                } else {
                    0.0
                };
                let body = aur_ocean::sample_body(
                    sample_point,
                    view,
                    footprint.max(step * 0.5),
                    if profile & 128 == 0 {
                        curve
                    } else {
                        Vec3::new(1.0e10, 0.0, 0.0)
                    },
                    wave,
                    span,
                    if profile & 256 == 0 {
                        ocean_palette
                    } else {
                        Vec4::ZERO
                    },
                    textures,
                );
                let density = body.density * 0.018;
                sigma += density;
                scattering += body.emission * density;
            }
        }
        if inside_poles {
            let beam = poles::sample_poles(sample_point, view, footprint, textures);
            let density = beam.density * 0.018;
            sigma += density;
            scattering += (beam.emission * view.sky_horizon.y
                + beam.color * view.ambient_color.xyz() * 0.2)
                * density;
        }
        if sigma > 0.00000001 {
            result.absorb(scattering / sigma, sigma * step, distance);
        }
        t += step;
    }
    let transmission = textures.air_transmittance(ro, rd, limit);
    let front = textures.air_transmittance(ro, rd, result.depth.min(limit));
    let air = (1.0 - front) + result.color.w * (front - transmission).max(0.0);
    result.color =
        (result.color.xyz() * front + atmosphere.sky * air).extend(result.color.w * transmission);
    result
}
