mod aur_ocean;
mod clouds;
mod poles;

use aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds};
use poles::{poles_raymarch_entry, sample_poles};
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3, vec4};
use spirv_std::num_traits::Float;

use crate::scene::aur::ejection_clearance;
use crate::scene::render::utils::{
    AtmosphereData, Textures, intersect_sphere, ray_shell_intersect,
};
use crate::scene::{PLANET_RADIUS, ViewUniform};

const DENSITY: f32 = 0.018;
const MAX_STEPS: i32 = 512;
const AUR_LIGHT_COLOR_A: Vec3 = vec3(0.36, 0.18, 0.48);
const AUR_LIGHT_COLOR_B: Vec3 = vec3(0.18, 0.09, 0.36);

const SCATTERING_ALBEDO: f32 = 0.65;
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000_000_8;

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
    pos: Vec3,
    planet_point: Vec3,
    view: &ViewUniform,
    clouds: bool,
    ocean: bool,
    poles: bool,
    footprint: f32,
    textures: &Textures,
) -> VolumeSample {
    let mut sample = VolumeSample::default();
    if clouds {
        let density = sample_clouds(pos, view, false, footprint, textures);
        sample.add(VolumeSample {
            density,
            cloud_density: density,
            color: Vec3::ONE,
            emission: Vec3::ZERO,
        });
    }
    if ocean {
        sample.add(sample_ocean(pos, view, false, footprint, textures));
    }
    if poles {
        let mut polar = sample_poles(planet_point, view, footprint, textures);
        polar.emission *= view.sky_horizon.y;
        sample.add(polar);
    }
    if sample.density > 0.0001 {
        sample.color /= sample.density;
        sample.emission /= sample.density;
    }
    sample
}

fn cloud_lighting(
    position: Vec3,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    density: f32,
    footprint: f32,
    textures: &Textures,
) -> Vec3 {
    let light = view.pole_direction();
    let near = sample_clouds(
        position + light * 250.0,
        view,
        true,
        footprint.max(80.0),
        textures,
    );
    let far = sample_clouds(
        position + light * 1500.0,
        view,
        true,
        footprint.max(240.0),
        textures,
    );
    let optical_depth = (density * 100.0 + near * 500.0 + far * 1400.0) * 0.009;
    let direct = (-optical_depth).exp() * 0.6
        + (-optical_depth * 0.5).exp() * 0.25
        + (-optical_depth * 0.25).exp() * 0.15;
    let diffuse = (-optical_depth * 0.25).exp();
    let camera = (view.world_position.xy() + view.camera_offset().truncate())
        .extend(view.ro_relative().z - PLANET_RADIUS);
    let facing = (position - camera).normalize().dot(light).max(0.0).powi(4);
    let upward =
        vec3(0.04, 0.006, 0.09) * (-position.z.max(0.0) / 6000.0).exp() * view.sky_horizon.y;
    (atmosphere.ambient * (0.20 + diffuse * 0.30)
        + view.pole_color.xyz() * direct * (0.15 + facing * 0.85)
        + upward * diffuse)
        * SCATTERING_ALBEDO
}

fn aur_lighting(pos: Vec3, time: f32, textures: &Textures) -> Vec3 {
    let fade = (1.0 - pos.z.max(0.0) / 12000.0).saturate().powi(2);
    if fade <= 0.0 {
        return Vec3::ZERO;
    }
    let noise = textures.weather(pos.xy() * 0.00008 + time * 0.01);
    let caustic = (1.0 - (noise.x * 2.0 - 1.0).abs()).powi(6);
    AUR_LIGHT_COLOR_A.lerp(AUR_LIGHT_COLOR_B, noise.y) * (0.8 + caustic * 12.0 * fade) * fade
}

#[derive(Copy, Clone)]
pub struct RaymarchResult {
    pub color: Vec4,
    pub depth: f32,
}

pub fn raymarch_volumetrics(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    t_max: f32,
    dither: f32,
    textures: &Textures,
) -> RaymarchResult {
    let t_max = if t_max >= 999_999.0 {
        10_000_000.0
    } else {
        t_max
    };
    let planet_center = view.planet_center();
    let clouds_entry_exit = if view.weather.x >= 0.0 && view.weather.max_element() <= 0.0 {
        vec4(t_max, 0.0, t_max, 0.0)
    } else {
        ray_shell_intersect(ro, rd, planet_center, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT)
    };
    let clouds_entry_exit1 = clouds_entry_exit.xy();
    let clouds_entry_exit2 = clouds_entry_exit.zw();
    let ocean_entry_exit =
        intersect_sphere(ro - planet_center, rd, PLANET_RADIUS + OCEAN_TOP_HEIGHT);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    let segments = [
        clouds_entry_exit1,
        clouds_entry_exit2,
        ocean_entry_exit,
        poles_entry_exit,
    ];
    let mut t_start = t_max;
    let mut t_end: f32 = 0.0;
    for i in 0..4 {
        let interval = segments[i];
        if interval.y > 0.0 && interval.x < interval.y {
            t_start = t_start.min(interval.x);
            t_end = t_end.max(interval.y);
        }
    }
    t_start = t_start.max(0.0);
    t_end = t_end.min(t_max);

    if t_start >= t_end {
        return RaymarchResult {
            color: vec4(0.0, 0.0, 0.0, 1.0),
            depth: t_max,
        };
    }

    let mut acc_color = Vec3::ZERO;
    let mut volume_empty = true;
    let mut t = t_start.max(0.0);
    let mut accumulated_weighted_depth = 0.0;
    let mut accumulated_density = 0.0;
    let mut threshold_depth: f32 = t_max;

    let camera_alt = (ro - planet_center).length() - PLANET_RADIUS;
    let init_fog = ATMOSPHERIC_FOG_DENSITY * (-camera_alt.max(0.0) / 20_000.0).exp();
    let mut transmittance = if t_start > 0.0 {
        let initial_fog_transmittance = (-init_fog * t_start).exp();
        acc_color = atmosphere.ambient * (1.0 - initial_fog_transmittance);
        initial_fog_transmittance
    } else {
        1.0
    };
    let camera_off = view.camera_offset();
    let pole_axis = view.planet_rotation.mul_vec3(Vec3::Z).normalize();
    let aur = aur_lighting(ro, view.time(), textures) * view.sky_horizon.y;
    let fog_color = atmosphere.sky + aur * 0.1;

    for _ in 0..MAX_STEPS {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        let inside_clouds = (t >= clouds_entry_exit1.x && t <= clouds_entry_exit1.y)
            || (t >= clouds_entry_exit2.x && t <= clouds_entry_exit2.y);
        let inside_ocean = t >= ocean_entry_exit.x && t <= ocean_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        if !inside_clouds && !inside_ocean && !inside_poles {
            let mut next_t = t_end;
            for i in 0..4 {
                if segments[i].x > t {
                    next_t = next_t.min(segments[i].x);
                }
            }

            let segment_length = next_t - t;
            if segment_length > 0.0 {
                let mid_pos = ro + rd * (t + segment_length * 0.5);
                let mid_alt = (mid_pos - planet_center).length() - PLANET_RADIUS;
                let seg_fog = ATMOSPHERIC_FOG_DENSITY * (-mid_alt.max(0.0) / 20_000.0).exp();
                let segment_fog_transmittance = (-seg_fog * segment_length).exp();
                acc_color += atmosphere.ambient * (1.0 - segment_fog_transmittance) * transmittance;
                transmittance *= segment_fog_transmittance;
            }
            t = next_t;
            continue;
        }

        let mut segment_end = t_end;
        for i in 0..4 {
            if segments[i].x > t {
                segment_end = segment_end.min(segments[i].x);
            }
            if segments[i].y > t {
                segment_end = segment_end.min(segments[i].y);
            }
        }
        // Probe the un-eroded hull first; enter fine mode before integrating a cloudy interval.
        if inside_clouds && !inside_ocean && !inside_poles && volume_empty {
            let coarse = (300.0 + t * 0.012).min(1400.0).min(segment_end - t);
            let probe = ro + rd * (t + coarse);
            let probe_alt = probe.distance(planet_center) - PLANET_RADIUS;
            let probe_world = (probe.xy() + camera_off.xy()).extend(probe_alt);
            let hull = sample_clouds(probe_world, view, true, 12.0 + t * 0.004, textures);
            if hull > 0.001 {
                volume_empty = false;
                continue;
            }
            let fog = ATMOSPHERIC_FOG_DENSITY * (-probe_alt.max(0.0) / 20_000.0).exp();
            let transmission = (-fog * coarse).exp();
            acc_color += fog_color * (1.0 - transmission) * transmittance;
            transmittance *= transmission;
            t += coarse;
            continue;
        }
        let depth = (PLANET_RADIUS - (ro + rd * t).distance(planet_center)).max(0.0);
        let mut step = if inside_clouds && !volume_empty {
            (48.0 + t * 0.004).min(600.0)
        } else if inside_clouds {
            (120.0 + t * 0.02).min(2000.0)
        } else if volume_empty {
            (120.0 + t * 0.025).min(4000.0)
        } else if inside_ocean {
            (24.0 + t * 0.008 + depth * 0.04).min(1500.0)
        } else {
            (48.0 + t * 0.012).min(2500.0)
        }
        .min(segment_end - t);
        if inside_poles {
            let relative = ro + rd * t - planet_center;
            let along = relative.dot(pole_axis);
            let radial = (relative - pole_axis * along).length();
            let width = 1200.0 + (along.abs() - PLANET_RADIUS).max(0.0).sqrt() * 5.0;
            let bound = (radial * 0.2).max(width * 0.25);
            step = if inside_clouds || inside_ocean {
                step.min(bound)
            } else {
                bound.min(segment_end - t)
            };
        }
        if inside_ocean && volume_empty {
            let point = ro + rd * t;
            let altitude = point.distance(planet_center) - PLANET_RADIUS;
            let world = (point.xy() + camera_off.xy()).extend(altitude);
            let mut clearance = ejection_clearance(world, view.time());
            if altitude > 1900.0 {
                clearance = clearance.min(altitude - 1900.0);
            }
            step = step.min(clearance.max(120.0));
        }
        let jitter = dither * 0.9 + 0.05;
        let pos_raw = ro + rd * (t + jitter * step);
        let altitude = pos_raw.distance(planet_center) - PLANET_RADIUS;
        let world_pos = (pos_raw.xy() + camera_off.xy()).extend(altitude);
        let sample = sample_volume(
            world_pos,
            pos_raw - planet_center,
            view,
            inside_clouds,
            inside_ocean,
            inside_poles,
            12.0 + t * 0.004,
            textures,
        );
        let step_density = sample.density;
        volume_empty = step_density <= 0.0001;

        let volume_transmittance = (-(DENSITY * step_density) * step).exp();

        let alt_fog = ATMOSPHERIC_FOG_DENSITY * (-altitude.max(0.0) / 20_000.0).exp();
        let fog_transmittance_step = (-alt_fog * step).exp();
        let alpha_step = 1.0 - volume_transmittance;

        acc_color += fog_color * (1.0 - fog_transmittance_step) * transmittance;

        if step_density > 0.0 {
            let in_scattering = if sample.cloud_density > 0.0001 {
                cloud_lighting(
                    world_pos,
                    view,
                    atmosphere,
                    sample.cloud_density,
                    12.0 + t * 0.004,
                    textures,
                )
            } else {
                atmosphere.ambient * 0.2
            };

            let emission = sample.emission;
            let cloud_aur_boost = if sample.cloud_density > 0.0001 {
                aur * (-sample.cloud_density * 24.0).exp() * (sample.cloud_density / step_density)
            } else {
                Vec3::ZERO
            };

            let volume_color = in_scattering * sample.color + emission + cloud_aur_boost;
            acc_color += volume_color * transmittance * alpha_step;

            let contribution = alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;
        }

        transmittance *= volume_transmittance * fog_transmittance_step;

        if threshold_depth == t_max && transmittance < 0.5 {
            threshold_depth = t;
        }

        t += step;
    }

    // Fog beyond the sampled volumes.
    let post_dist = (t_max - t.min(t_max)).max(0.0);
    if post_dist > 0.0 && transmittance > 0.0 {
        let mid_t = t + post_dist * 0.5;
        let post_mid = ro + rd * mid_t;
        let post_alt = (post_mid - planet_center).length() - PLANET_RADIUS;
        let post_fog = ATMOSPHERIC_FOG_DENSITY * (-post_alt.max(0.0) / 20_000.0).exp();
        let fg = (-post_fog * post_dist).exp();
        acc_color += atmosphere.ambient * (1.0 - fg) * transmittance;
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
    let offset = view.camera_offset();
    let mut illumination = atmosphere.ambient * 0.2;
    let mut lit = false;
    for index in 0..8 {
        let point = ro + rd * ((index as f32 + 0.5) * step);
        let altitude = point.distance(center) - PLANET_RADIUS;
        let world = (point.xy() + offset.xy()).extend(altitude);
        let sample = sample_volume(
            world,
            point - center,
            view,
            altitude > CLOUD_BOTTOM_HEIGHT && altitude < CLOUD_TOP_HEIGHT,
            altitude < OCEAN_TOP_HEIGHT,
            false,
            step,
            textures,
        );
        let extinction = DENSITY * sample.density + ATMOSPHERIC_FOG_DENSITY;
        let absorbed = 1.0 - (-extinction * step).exp();
        if !lit && sample.cloud_density > 0.0001 {
            illumination = cloud_lighting(
                world,
                view,
                atmosphere,
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
