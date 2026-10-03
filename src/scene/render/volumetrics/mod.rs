mod aur_ocean;
mod clouds;
mod poles;

use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    lighting::henyey_greenstein,
    render::utils::{AtmosphereData, Textures, intersect_sphere, ray_shell_intersect},
};
use aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds};
use poles::{poles_raymarch_entry, sample_poles};
use spirv_std::glam::{FloatExt, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec3, vec4};
use spirv_std::num_traits::Float;

const DENSITY: f32 = 0.25;
const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 120.0;
const STEP_SIZE_OUTSIDE: f32 = 240.0;
const SCALING_END: f32 = 200_000.0;
const SCALING_MAX: f32 = 6.0;
const SCALING_MAX_VERTICAL: f32 = 2.0;
const SCALING_MAX_OCEAN: f32 = 2.0;
const CLOSE_THRESHOLD: f32 = 2000.0;

const LIGHT_STEPS: u32 = 4;
const LIGHT_STEP_SIZE: f32 = 100.0;
const AUR_LIGHT_COLOR_A: Vec3 = vec3(0.36, 0.18, 0.48);
const AUR_LIGHT_COLOR_B: Vec3 = vec3(0.18, 0.09, 0.36);

const EXTINCTION: f32 = 0.07;
const SCATTERING_ALBEDO: f32 = 0.65;
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000_004;

#[derive(Default)]
pub(super) struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

impl VolumeSample {
    fn add(&mut self, other: Self) {
        self.color += other.color * other.density;
        self.emission += other.emission * other.density;
        self.density += other.density;
    }
}

fn sample_volume(
    pos: Vec3,
    view: &ViewUniform,
    clouds: bool,
    ocean: bool,
    poles: bool,
    textures: &Textures,
) -> VolumeSample {
    let mut sample = VolumeSample::default();
    if clouds {
        let density = sample_clouds(pos, view, false, textures);
        sample.add(VolumeSample {
            density,
            color: Vec3::ONE,
            emission: Vec3::ZERO,
        });
    }
    if ocean {
        sample.add(sample_ocean(pos, view.time(), false, textures));
    }
    if poles {
        sample.add(sample_poles(pos));
    }
    if sample.density > 0.0001 {
        sample.color /= sample.density;
        sample.emission *= sample.density.saturate() / sample.density;
    }
    sample
}

fn sample_shadowing(
    world_pos: Vec3,
    view: &ViewUniform,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    step_density: f32,
    sun_dir: Vec3,
    textures: &Textures,
) -> Vec3 {
    let mut optical_depth = (step_density * 1.5).saturate() * EXTINCTION * LIGHT_STEP_SIZE;

    let dynamic_light_step = 350.0;
    for j in 0..LIGHT_STEPS {
        if optical_depth > 8.0 {
            break;
        }
        let step_index = j as f32 + 1.0;
        let distance_along = step_index * dynamic_light_step;
        let lightmarch_pos = world_pos + sun_dir * distance_along;
        let clouds = sample_clouds(lightmarch_pos, view, true, textures);
        let ocean = sample_ocean(lightmarch_pos, view.time(), true, textures).density;

        optical_depth +=
            (clouds + ocean).max(0.0).powf(1.1) * (EXTINCTION * 0.6) * dynamic_light_step;
    }

    let sun_transmittance = (-optical_depth).exp();
    let powder_law = 1.0 - (-optical_depth * 2.5).exp();
    let transmittance = sun_transmittance * 1.0.lerp(powder_law, 0.6);

    let cos_theta = rd.dot(sun_dir);
    let altitude_factor = (world_pos.z / 30000.0).saturate();

    let g_forward = 0.88.lerp(0.92, altitude_factor);
    let g_back = -0.4;
    let phase_forward = henyey_greenstein(cos_theta, g_forward);
    let phase_back = henyey_greenstein(cos_theta, g_back);
    let phase = phase_forward.lerp(
        phase_back * (2.0 + altitude_factor * 4.0),
        0.15 + altitude_factor * 0.15,
    );

    let sun_brightness =
        (5.0 / (1.0 + 8.0 * (1.0 + cos_theta)).max(0.1)).min(3.0 + altitude_factor * 5.0);
    let single_scattering = transmittance * atmosphere.sun * phase * sun_brightness;

    let ambient_occlusion = (1.0 - (step_density * 2.0).saturate()).powf(1.2);
    let multiple_scattering = atmosphere.ambient * sun_transmittance * ambient_occlusion * 2.5;

    let shadow_boost = 1.0 - sun_transmittance;
    let ambient_base = 0.35 + 0.5 * (step_density * 4.0).saturate();
    let ambient_floor_vec = atmosphere.ambient * ambient_base * (0.6 + 0.4 * shadow_boost.max(0.0))
        + atmosphere.sky * (0.5 + altitude_factor * 0.4) * ambient_occlusion;

    (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO
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
    pub opaque_distance: f32,
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
    let planet_center = view.planet_center();
    let clouds_entry_exit =
        ray_shell_intersect(ro, rd, planet_center, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT);
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
            opaque_distance: t_max,
        };
    }

    let mut acc_color = Vec3::ZERO;
    let mut step = STEP_SIZE_OUTSIDE;
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
    let sun_world_pos = atmosphere.sun_pos + camera_off;

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
        step = step.min(segment_end - t);
        let pos_raw = ro + rd * (t + (0.2 + dither * 0.6) * step);
        let altitude = pos_raw.distance(planet_center) - PLANET_RADIUS;
        let world_pos = (pos_raw.xy() + camera_off.xy()).extend(altitude);
        let sample = sample_volume(
            world_pos,
            view,
            inside_clouds,
            inside_ocean,
            inside_poles,
            textures,
        );
        let step_density = sample.density;

        let distance_scale = (t / SCALING_END).saturate();
        let directional_max_scale = SCALING_MAX.lerp(SCALING_MAX_VERTICAL, rd.z.abs());
        let max_scale = if inside_ocean {
            SCALING_MAX_OCEAN
        } else {
            directional_max_scale
        };
        let mut step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        let distance_to_surface = t_max - t;
        let proximity_factor = 1.0 - (distance_to_surface / CLOSE_THRESHOLD).saturate();
        step_scaler = step_scaler.lerp(0.1, proximity_factor);

        let base_step = STEP_SIZE_OUTSIDE.lerp(STEP_SIZE_INSIDE, (step_density * 10.0).saturate());
        let next_step = base_step * step_scaler;

        let volume_transmittance = (-(DENSITY * step_density) * step).exp();

        let aur = aur_lighting(world_pos, view.time(), textures);
        let fog_color = atmosphere.sky + aur * 0.1;

        let alt_fog = ATMOSPHERIC_FOG_DENSITY * (-altitude.max(0.0) / 20_000.0).exp();
        let fog_transmittance_step = (-alt_fog * step).exp();
        let alpha_step = 1.0 - volume_transmittance;

        acc_color += fog_color * (1.0 - fog_transmittance_step) * transmittance;

        if step_density > 0.0 {
            let sun_dir = (sun_world_pos - world_pos).normalize();
            let in_scattering = sample_shadowing(
                world_pos,
                view,
                rd,
                atmosphere,
                step_density,
                sun_dir,
                textures,
            );

            let emission = sample.emission * 1000.0;
            let cloud_aur_boost = if inside_clouds {
                aur * (-step_density * 24.0).exp()
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
        step = next_step;
    }

    let opaque_distance = if transmittance < 0.01 { t } else { t_max };

    // Fog from the end of all volumes to t_max
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
        color: Vec4::from((acc_color, transmittance)).saturate(),
        depth: final_depth,
        opaque_distance,
    }
}
