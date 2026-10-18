//! Native opacity and emission; the view grid supplies cloud illumination only.
use spirv_std::Image;
use spirv_std::glam::{Vec2, Vec3, Vec4, Vec4Swizzles, vec3};
use spirv_std::num_traits::Float;

use crate::scene::aur::EJECTION_WIDTH;
use crate::scene::climate::{CLOUD_BOTTOM, CLOUD_TOP, Cloud, RayEvent, ray_event};
use crate::scene::math::{finish_opaque, intersect_sphere, ray_shell_intersect, square};
use crate::scene::planet::{REACTION_RADIUS, REACTION_SPREAD, light_position, surface_coordinates};
use crate::scene::render::ocean::sample;
use crate::scene::render::textures::{AtmosphereData, Textures};
use crate::scene::render::{clouds, ocean, poles};
use crate::scene::{PLANET_RADIUS, ViewUniform, froxel};

const VEIL_STEP: f32 = 112.0;

#[derive(Default)]
pub(super) struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

#[derive(Copy, Clone)]
pub struct RaymarchResult {
    pub color: Vec4,
    pub depth: f32,
}

impl RaymarchResult {
    pub(super) fn empty(limit: f32) -> Self {
        Self {
            color: Vec4::W,
            depth: limit,
        }
    }
    pub(super) fn absorb(&mut self, radiance: Vec3, optical_depth: f32, distance: f32) {
        let transmission = (-optical_depth.max(0.0)).exp();
        let weight = self.color.w * (1.0 - transmission);
        let opacity = 1.0 - self.color.w;
        if opacity + weight > 0.00001 {
            self.depth = (self.depth * opacity + distance * weight) / (opacity + weight);
        }
        self.color = (self.color.xyz() + radiance * weight).extend(self.color.w * transmission);
        self.color = finish_opaque(self.color, radiance);
    }
}

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
        ray_shell_intersect(ro, rd, Vec3::ZERO, CLOUD_BOTTOM, CLOUD_TOP)
    } else {
        Vec4::new(limit, 0.0, limit, 0.0)
    };
    let ocean = if profile & 4 == 0 {
        intersect_sphere(ro, rd, PLANET_RADIUS + ocean::OCEAN_TOP_HEIGHT)
    } else {
        absent
    };
    // The body is bounded far below the arcs. Entering its shell exactly
    // replaces a geometric crawl down towards it from the arc ceiling.
    let body = if profile & 4 == 0 {
        intersect_sphere(ro, rd, PLANET_RADIUS + ocean::BODY_TOP_HEIGHT)
    } else {
        absent
    };
    let polar = poles::poles_raymarch_entry(ro, rd, limit);
    let mut column = ocean::Column {
        flash: Vec3::ZERO,
        sun: Vec3::ZERO,
        climate: Vec4::ZERO,
    };
    let mut column_point = Vec3::splat(1.0e10);
    let mut result = RaymarchResult::empty(limit);
    let mut t = 0.0;
    let mut event = RayEvent {
        ids: [0; 8],
        count: 0,
        end: 0.0,
        step: 4000.0,
        overflow: false,
    };
    let mut steps = 0u32;
    for _ in 0..512 {
        if t >= limit || result.color.w < 0.005 {
            break;
        }
        steps += 1;
        let point = ro + rd * t;
        let footprint = (t * angle).max(2.0);
        let inside_clouds =
            (t + 0.25 >= shell.x && t < shell.y) || (t + 0.25 >= shell.z && t < shell.w);
        let inside_ocean = t + 0.25 >= ocean.x && t < ocean.y;
        let inside_body = t + 0.25 >= body.x && t < body.y;
        let inside_poles = t + 0.25 >= polar.x && t < polar.y;
        let mut end = boundary(
            shell.xy(),
            t,
            boundary(
                shell.zw(),
                t,
                boundary(ocean, t, boundary(body, t, boundary(polar, t, limit))),
            ),
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
        let mut curve = Vec4::new(1.0e10, 0.0, 0.0, 0.0);
        let mut arcs = false;
        if inside_clouds && event.count > 0 {
            step = step.min(event.step.max(footprint));
        }
        if inside_ocean && profile & 128 == 0 {
            let clearance = guidance(uv, t, textures, field).w;
            if clearance < 64.0 {
                // The arc field is an exact distance in surface coordinates.
                // Probe ahead by half of a step through an arc, so the same
                // query also serves the sample at the middle of that step.
                let reach = EJECTION_WIDTH * 3.0;
                let inner = (EJECTION_WIDTH * 0.12).max(footprint * 0.75);
                let probe = inner * 0.5;
                let ahead = point + rd * probe;
                curve = textures.ejection_field(surface_coordinates(ahead));
                // Longitude stretches towards the poles, so surface distances
                // overstate world distances by up to this factor.
                let metric = ahead.truncate().length() / ahead.length().max(1.0);
                let free = (curve.x - reach) * metric - probe;
                if free > inner {
                    step = step.min(free);
                } else {
                    arcs = curve.z > 0.0;
                    step = step.min(inner);
                }
            } else {
                step = step.min(clearance);
            }
        }
        if inside_body {
            let altitude = point.length() - PLANET_RADIUS;
            let up = point / (altitude + PLANET_RADIUS);
            (ocean_shape, _) = sample(point, view, footprint, textures);
            let wave = if ocean_shape.yzw().length_squared() > 0.25 {
                ocean_shape.x * 4096.0
            } else {
                ocean::wave_height(point, view.time(), footprint, textures)
            };
            let mut height = altitude - wave;
            ocean_height = height;
            if height < 1500.0 {
                // The veil climbs any spire it meets; step against that.
                height -= ocean::wrap(point, view.time(), footprint).lift;
            }
            // Above the veil, a cone bound on the wave slope gives a safe
            // step. Inside it, steps grow with depth as the medium saturates
            // and with distance as the footprint widens.
            let body_step = if height > 450.0 {
                (height - 320.0) * 0.8 / ((-rd.dot(up)).max(0.0) + 0.35)
            } else {
                VEIL_STEP + (-height).max(0.0) * 0.49
            };
            step = step.min(body_step.max(footprint * 0.75));
            // A cone step lies wholly above the veil by construction; its far
            // end is too distant to extrapolate the wave to, so skip it.
            if height > 450.0 && step >= body_step {
                ocean_height = 1.0e10;
            }
        }
        if inside_poles {
            let radial = point.truncate().length();
            let width = 1200.0 + (point.z.abs() - PLANET_RADIUS).max(0.0).sqrt() * 5.0;
            step = step.min((radial * 0.2).max(width * 0.5).max(footprint));
            if radial < REACTION_SPREAD * 1.6 {
                let shell_distance = (point.length() - REACTION_RADIUS).abs();
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
                let light = (light_position(sample_point) - sample_point).normalize();
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
                // The ocean's light reaches cloud bases as a drifting caustic
                // net; too fine to resolve far away, where it averages out.
                let altitude = sample_point.length() - PLANET_RADIUS;
                let resolve = (-footprint / 1500.0).exp() * (-altitude / 9000.0).exp();
                if resolve > 0.01 {
                    let surface = surface_coordinates(sample_point);
                    let net = ocean::caustics(surface.truncate() * 0.0003, view.time() * 0.35);
                    scattering += vec3(0.40, 0.34, 0.52)
                        * view.sky_horizon.y
                        * ((net - 0.25) * resolve * 0.6)
                        * density;
                }
            }
        }
        if arcs {
            let jet = ocean::ejections(sample_point, view, footprint, curve, textures);
            let density = jet.density * 0.018;
            sigma += density;
            scattering += jet.emission * density;
        }
        if ocean_height < 1.0e9 {
            let altitude = sample_point.length() - PLANET_RADIUS;
            // The cached height gradient extrapolates the wave across this
            // short segment instead of fetching it again at the midpoint.
            let up = point.normalize();
            let normal = if ocean_shape.yzw().length_squared() > 0.25 {
                ocean_shape.yzw().normalize()
            } else {
                up
            };
            let gradient = (up - normal / normal.dot(up).max(0.01)) * 0.1;
            let mut wave =
                point.length() - PLANET_RADIUS - ocean_height + gradient.dot(rd) * step * 0.5;
            let span = (altitude - wave - ocean_height).abs();
            let wrap = if altitude - wave - span < 1300.0 {
                ocean::wrap(sample_point, view.time(), footprint)
            } else {
                ocean::Wrap::NONE
            };
            // The veil also heaves up around the feet of a loop.
            wave += wrap.lift + 450.0 * (-square(curve.x / (EJECTION_WIDTH * 2.2))).exp() * curve.z;
            let height = altitude - wave;
            if altitude <= ocean::BODY_TOP_HEIGHT && height - span < 450.0 {
                let reuse = (footprint * 6.0).max(1500.0);
                if sample_point.distance_squared(column_point) > reuse * reuse {
                    column_point = sample_point;
                    column = ocean::column(sample_point, view, textures);
                }
                let body = ocean::sample_body(
                    sample_point,
                    view,
                    footprint.max(step * 0.5),
                    if profile & 128 == 0 {
                        curve
                    } else {
                        Vec4::new(1.0e10, 0.0, 0.0, 0.0)
                    },
                    wave,
                    span,
                    normal,
                    column,
                    wrap,
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
    if profile & 512 != 0 {
        // Bands: <16 blue, <32 cyan, <64 green, <128 yellow, <256 red, else white.
        let band = if steps < 16 {
            Vec4::new(0.0, 0.0, 1.0, 0.0)
        } else if steps < 32 {
            Vec4::new(0.0, 1.0, 1.0, 0.0)
        } else if steps < 64 {
            Vec4::new(0.0, 1.0, 0.0, 0.0)
        } else if steps < 128 {
            Vec4::new(1.0, 1.0, 0.0, 0.0)
        } else if steps < 256 {
            Vec4::new(1.0, 0.0, 0.0, 0.0)
        } else {
            Vec4::new(1.0, 1.0, 1.0, 0.0)
        };
        result.color = band * 0.5;
        return result;
    }
    let transmission = textures.air_transmittance(ro, rd, limit);
    let front = textures.air_transmittance(ro, rd, result.depth.min(limit));
    let air = (1.0 - front) + result.color.w * (front - transmission).max(0.0);
    result.color =
        (result.color.xyz() * front + atmosphere.haze * air).extend(result.color.w * transmission);
    result
}
