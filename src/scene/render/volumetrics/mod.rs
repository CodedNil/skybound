mod aur_ocean;
mod clouds;
mod poles;
mod storms;

use aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds};
use poles::{poles_raymarch_entry, sample_poles};
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4, Vec4Swizzles, vec3, vec4};
use spirv_std::num_traits::Float;

use crate::scene::atmosphere::air;
use crate::scene::aur::{EJECTION_HEIGHT, EJECTION_WIDTH};
use crate::scene::math::square;
use crate::scene::render::utils::{
    AtmosphereData, Textures, intersect_sphere, ray_shell_intersect,
};
use crate::scene::{PLANET_RADIUS, ViewUniform};

const DENSITY: f32 = 0.018;
const CLOUD_DENSITY: f32 = 2.0;
const AUR_LIGHT_COLOR_A: Vec3 = vec3(0.40, 0.34, 0.52);
const AUR_LIGHT_COLOR_B: Vec3 = vec3(0.28, 0.48, 0.44);

const SCATTERING_ALBEDO: f32 = 0.92;

#[derive(Default)]
pub(super) struct VolumeSample {
    pub density: f32,
    pub cloud_density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

impl VolumeSample {
    fn add(&mut self, other: Self) {
        self.color += other.color * other.density;
        self.emission += other.emission * other.density;
        self.density += other.density;
        self.cloud_density += other.cloud_density;
    }
}

fn sample_volume(
    planet_point: Vec3,
    view: &ViewUniform,
    clouds: u32,
    ocean: bool,
    poles: bool,
    footprint: f32,
    textures: &Textures,
) -> VolumeSample {
    let mut sample = VolumeSample::default();
    if clouds != 0 {
        let density = clouds::sample_selected(planet_point, view, footprint, clouds, textures);
        sample.add(VolumeSample {
            density: density * CLOUD_DENSITY,
            cloud_density: density,
            color: Vec3::ONE,
            emission: Vec3::ZERO,
        });
    }
    if ocean {
        sample.add(sample_ocean(
            crate::scene::planet::surface_coordinates(planet_point),
            planet_point,
            view,
            footprint,
            textures,
        ));
    }
    if poles {
        let mut polar = sample_poles(planet_point, view, footprint, textures);
        polar.emission *= view.sky_horizon.y;
        sample.add(polar);
    }
    if sample.density > 0.0 {
        sample.color /= sample.density;
        sample.emission /= sample.density;
    }
    sample
}

fn cloud_lighting(
    position: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    density: f32,
    footprint: f32,
    textures: &Textures,
) -> Vec3 {
    if view.camera_position.w as u32 & 1 != 0 {
        return Vec3::ONE;
    }
    let source = crate::scene::polar::light_position(position);
    let light = (source - position).normalize();
    let sunlight = textures.pole_light(position, source) * view.sky_horizon.y;
    let mut optical_depth = 0.0;
    let mut start = 0.0;
    let mut width = 160.0;
    for _ in 0..3 {
        let distance = start + width * 0.5;
        let shadow = sample_clouds(
            position + light * distance,
            view,
            footprint.max(width * 0.08),
            textures,
        );
        optical_depth += shadow * width * (DENSITY * CLOUD_DENSITY);
        if optical_depth > 12.0 {
            break;
        }
        start += width;
        width *= 3.0;
    }
    let cosine = direction.dot(light);
    let phase = |g: f32| {
        let denominator = (1.0 + g * g - 2.0 * g * cosine).max(0.001);
        (1.0 - g * g) / (denominator * denominator.sqrt())
    };
    let scattering = (-optical_depth).exp() * (phase(0.65) * 0.8 + phase(-0.25) * 0.2) * 0.25
        + (-optical_depth * 0.35).exp() * 0.12
        + (-optical_depth * 0.12).exp() * 0.06;
    let altitude = position.length() - PLANET_RADIUS;
    let mut aur_depth = density * 160.0 * (DENSITY * CLOUD_DENSITY);
    if altitude < 25_000.0 {
        aur_depth += sample_clouds(
            position - position.normalize() * 400.0,
            view,
            footprint,
            textures,
        ) * 640.0
            * (DENSITY * CLOUD_DENSITY);
    }
    let upward = aur_lighting(position, view.time(), textures)
        * view.sky_horizon.y
        * ((-aur_depth).exp() * 0.7 + (-aur_depth * 0.2).exp() * 0.3);
    ((sunlight * 0.025 + vec3(0.03, 0.045, 0.075)) + sunlight * scattering + upward)
        * SCATTERING_ALBEDO
}

fn aur_lighting(pos: Vec3, time: f32, textures: &Textures) -> Vec3 {
    let altitude = pos.length() - PLANET_RADIUS;
    let fade = (-altitude.max(0.0) / 16_000.0).exp();
    if altitude > EJECTION_HEIGHT + EJECTION_WIDTH * 8.0 {
        return Vec3::ZERO;
    }
    let surface = pos.normalize() * PLANET_RADIUS - crate::scene::aur::CURRENT * time;
    let noise = textures.billows(surface * 0.000014 + Vec3::Z * time * 0.004, 0.0);
    let caustic = noise.w.smoothstep(0.35, 0.7);
    let curve = textures.ejection_field(crate::scene::planet::surface_coordinates(pos), f32::MAX);
    let arc = (-curve.x / (EJECTION_WIDTH * 6.0)).exp() * curve.z;
    AUR_LIGHT_COLOR_A.lerp(AUR_LIGHT_COLOR_B, noise.y) * (1.5 + caustic * 2.5) * fade
        + vec3(1.5, 0.85, 0.4) * arc * 3.0
}

#[derive(Copy, Clone)]
pub struct RaymarchResult {
    pub color: Vec4,
    pub depth: f32,
}

fn next_boundary(boundaries: Vec4, t: f32, end: f32) -> f32 {
    Vec4::select(
        boundaries.cmpgt(Vec4::splat(t)),
        boundaries,
        Vec4::splat(end),
    )
    .min_element()
    .min(end)
}

pub fn raymarch_volumetrics(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    t_max: f32,
    dither: f32,
    pixel_angle: f32,
    textures: &Textures,
) -> RaymarchResult {
    let t_max = if t_max >= 999_999.0 {
        10_000_000.0
    } else {
        t_max
    };
    let ro = view.to_planet(ro);
    let rd = view.planet_rotation.conjugate() * rd;
    let planet_center = Vec3::ZERO;
    let clouds_entry_exit = if view.camera_position.w as u32 & 2 != 0
        || (view.weather.x >= 0.0 && view.weather.max_element() <= 0.0)
    {
        vec4(t_max, 0.0, t_max, 0.0)
    } else {
        ray_shell_intersect(ro, rd, planet_center, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT)
    };
    let clouds_entry_exit1 = clouds_entry_exit.xy();
    let clouds_entry_exit2 = clouds_entry_exit.zw();
    let ocean_entry_exit = if view.camera_position.w as u32 & 4 != 0 {
        spirv_std::glam::Vec2::new(t_max, 0.0)
    } else {
        intersect_sphere(ro - planet_center, rd, PLANET_RADIUS + OCEAN_TOP_HEIGHT)
    };
    let poles_entry_exit = poles_raymarch_entry(ro, rd, t_max);

    let starts = vec4(
        clouds_entry_exit1.x,
        clouds_entry_exit2.x,
        ocean_entry_exit.x,
        poles_entry_exit.x,
    );
    let ends = vec4(
        clouds_entry_exit1.y,
        clouds_entry_exit2.y,
        ocean_entry_exit.y,
        poles_entry_exit.y,
    );
    let valid = ends.cmpgt(Vec4::ZERO) & starts.cmplt(ends);
    let t_start = Vec4::select(valid, starts, Vec4::splat(t_max))
        .min_element()
        .max(0.0);
    let t_end = Vec4::select(valid, ends, Vec4::ZERO)
        .max_element()
        .min(t_max);

    if t_start >= t_end {
        let transmission = textures.air_transmittance(ro, rd, t_max);
        return RaymarchResult {
            color: (atmosphere.sky * (1.0 - transmission)).extend(transmission),
            depth: t_max,
        };
    }

    let mut acc_color = Vec3::ZERO;
    let mut volume_empty = true;
    let mut t = t_start.max(0.0);
    let mut accumulated_weighted_depth = 0.0;
    let mut accumulated_density = 0.0;
    let mut threshold_depth: f32 = t_max;
    let mut cloud_light = Vec3::ZERO;
    let mut lit_distance = -1.0e10;
    let mut lit_density = -1.0f32;
    let active_clouds = clouds::active_kinds(view);
    let mut cloud_event = Vec2::ZERO;
    let mut cloud_kinds = 0;
    let mut cloud_layers = clouds::LayerTraversal::default();

    let mut transmittance = if t_start > 0.0 {
        let initial_fog_transmittance = textures.air_transmittance(ro, rd, t_start);
        acc_color = atmosphere.sky * (1.0 - initial_fog_transmittance);
        initial_fog_transmittance
    } else {
        1.0
    };
    let pole_axis = Vec3::Z;
    let fog_color =
        atmosphere.sky + aur_lighting(ro, view.time(), textures) * view.sky_horizon.y * 0.1;

    let mut sequence = dither;
    while t < t_end && transmittance >= 0.01 {
        let jitter = sequence.fract() * 0.9 + 0.05;
        sequence = (sequence + 0.754_877).fract();

        let inside_cloud_shell = (t >= clouds_entry_exit1.x && t < clouds_entry_exit1.y)
            || (t >= clouds_entry_exit2.x && t < clouds_entry_exit2.y);
        if inside_cloud_shell && t >= cloud_event.y {
            let event =
                clouds::ray_event(ro, rd, t, view, active_clouds, &mut cloud_layers, textures);
            cloud_event = Vec2::splat(t) + event.bounds;
            cloud_kinds = event.kinds;
        }
        let inside_clouds = inside_cloud_shell && t + 8.0 >= cloud_event.x;
        let cloud_boundary = if inside_cloud_shell {
            cloud_event.y.max(t + 16.0)
        } else {
            t_end
        };
        let inside_ocean = t >= ocean_entry_exit.x && t < ocean_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t < poles_entry_exit.y;

        if !inside_clouds && !inside_ocean && !inside_poles {
            let next_t = next_boundary(starts, t, t_end).min(cloud_boundary);

            let segment_length = next_t - t;
            if segment_length > 0.0 {
                let segment_fog_transmittance =
                    textures.air_transmittance(ro + rd * t, rd, segment_length);
                let light = if inside_cloud_shell {
                    fog_color
                } else {
                    atmosphere.sky
                };
                acc_color += light * (1.0 - segment_fog_transmittance) * transmittance;
                transmittance *= segment_fog_transmittance;
            }
            t = next_t.max(t + (t * 0.000_000_2).max(0.125)).min(t_end);
            continue;
        }

        let segment_end =
            next_boundary(ends, t, next_boundary(starts, t, t_end)).min(cloud_boundary);
        let depth = (PLANET_RADIUS - (ro + rd * t).distance(planet_center)).max(0.0);
        let mut step = if inside_clouds || (inside_cloud_shell && !volume_empty) {
            (32.0 + t * 0.002).max(t * pixel_angle * 0.5).min(1500.0)
        } else if inside_cloud_shell {
            (120.0 + t * 0.02).min(2000.0)
        } else if volume_empty {
            (120.0 + t * 0.025).min(4000.0)
        } else if inside_ocean {
            (24.0 + t * 0.008 + depth * 0.04).min(1500.0)
        } else {
            (48.0 + t * 0.012).min(2500.0)
        }
        .min(segment_end - t);
        if inside_clouds && cloud_kinds & crate::scene::climate::clouds::HIGH_KINDS != 0 {
            let up = (ro + rd * t).normalize();
            let profile_step = 90.0 / rd.dot(up).abs().max(0.1);
            step = step.min(profile_step.max(t * pixel_angle * 0.5));
        }
        if inside_poles {
            let relative = ro + rd * t - planet_center;
            let along = relative.dot(pole_axis);
            let radial = (relative - pole_axis * along).length();
            let width = 1200.0 + (along.abs() - PLANET_RADIUS).max(0.0).sqrt() * 5.0;
            let bound = (radial * 0.2).max(width * 0.5).max(t * pixel_angle * 0.5);
            step = if inside_cloud_shell || inside_ocean {
                step.min(bound)
            } else {
                bound.min(segment_end - t)
            };
            if radial < crate::scene::polar::REACTION_SPREAD * 1.6 {
                let shell = (relative.length() - crate::scene::polar::REACTION_RADIUS).abs();
                let detail_step = (t * pixel_angle * 0.5).max(200.0);
                step = step.min(((shell - 45_000.0) * 0.5).max(detail_step));
            }
        }
        if inside_ocean && volume_empty {
            let point = ro + rd * t;
            let altitude = point.distance(planet_center) - PLANET_RADIUS;
            let world = crate::scene::planet::surface_coordinates(point);
            let mut clearance = ((textures.ejection_field(world, EJECTION_WIDTH + 4000.0).x
                - EJECTION_WIDTH)
                * 0.5)
                .clamp(0.0, 2000.0);
            if altitude > 3700.0 {
                clearance = clearance.min(altitude - 3700.0);
            }
            step = step.min(clearance.max(120.0));
        }
        step = step.min(segment_end - t);
        let sample_distance = t + jitter * step;
        let pos_raw = ro + rd * sample_distance;
        let planet_point = pos_raw;
        let altitude = pos_raw.distance(planet_center) - PLANET_RADIUS;
        let sample = sample_volume(
            planet_point,
            view,
            if inside_clouds { cloud_kinds } else { 0 },
            inside_ocean,
            inside_poles,
            (t * pixel_angle).max(4.0),
            textures,
        );
        let step_density = sample.density;
        volume_empty = step_density <= 0.0001;

        let volume_transmittance = (-(DENSITY * step_density) * step).exp();

        let fog_transmittance_step = (-air::density(altitude) * step).exp();
        let alpha_step = 1.0 - volume_transmittance;

        acc_color += fog_color * (1.0 - fog_transmittance_step) * transmittance;

        if step_density > 0.0 {
            let in_scattering = if sample.cloud_density > 0.0001 {
                let point = planet_point;
                let footprint = (t * pixel_angle).max(4.0);
                if sample_distance - lit_distance > 240.0 + footprint
                    || (sample.cloud_density - lit_density).abs() > 0.08
                {
                    cloud_light =
                        cloud_lighting(point, rd, view, sample.cloud_density, footprint, textures);
                    lit_distance = sample_distance;
                    lit_density = sample.cloud_density;
                }
                cloud_light
            } else {
                atmosphere.ambient * 0.2
            };

            let emission = sample.emission;
            let volume_color = in_scattering * sample.color + emission;
            acc_color += volume_color * transmittance * alpha_step;

            let contribution = alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;
        }

        transmittance *= volume_transmittance * fog_transmittance_step;

        if threshold_depth == t_max && transmittance < 0.5 {
            threshold_depth = t;
        }

        t = (t + step).max(t + (t * 0.000_000_2).max(0.125)).min(t_end);
    }

    // Fog beyond the sampled volumes.
    let post_dist = (t_max - t.min(t_max)).max(0.0);
    if post_dist > 0.0 && transmittance > 0.0 {
        let fg = textures.air_transmittance(ro + rd * t, rd, post_dist);
        acc_color += atmosphere.sky * (1.0 - fg) * transmittance;
        transmittance *= fg;
    }

    let avg_depth = accumulated_weighted_depth / accumulated_density.max(0.0001);
    let final_depth = if threshold_depth < t_max {
        threshold_depth
    } else if accumulated_density > 0.0001 {
        avg_depth
    } else {
        t_max
    };

    RaymarchResult {
        color: Vec4::from((acc_color.max(Vec3::ZERO), transmittance.clamp(0.0, 1.0))),
        depth: final_depth,
    }
}

pub fn short_volume(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    length: f32,
    _dither: f32,
    textures: &Textures,
) -> RaymarchResult {
    let mut color = Vec3::ZERO;
    let mut transmission = 1.0;
    let step = length.max(0.0) / 8.0;
    let center = view.planet_center();
    let planet_direction = view.planet_rotation.conjugate() * rd;
    let mut illumination = atmosphere.ambient * 0.2;
    let mut lit = false;
    for index in 0..8 {
        let point = ro + rd * ((index as f32 + 0.5) * step);
        let altitude = point.distance(center) - PLANET_RADIUS;
        let sample = sample_volume(
            view.to_planet(point),
            view,
            if altitude > CLOUD_BOTTOM_HEIGHT && altitude < CLOUD_TOP_HEIGHT {
                crate::scene::climate::clouds::ALL_KINDS
            } else {
                0
            },
            altitude < OCEAN_TOP_HEIGHT,
            false,
            step,
            textures,
        );
        let extinction = DENSITY * sample.density + air::density(altitude);
        let absorbed = 1.0 - (-extinction * step).exp();
        if !lit && sample.cloud_density > 0.0001 {
            illumination = cloud_lighting(
                view.to_planet(point),
                planet_direction,
                view,
                sample.cloud_density,
                step,
                textures,
            );
            lit = true;
        }
        color += (sample.color * illumination + sample.emission + atmosphere.sky * 0.001)
            * absorbed
            * transmission;
        transmission *= 1.0 - absorbed;
    }
    RaymarchResult {
        color: color.extend(transmission),
        depth: length,
    }
}
