mod aur_ocean;
mod clouds;
mod poles;

use crate::lighting::henyey_greenstein;
use crate::utils::{AtmosphereData, Textures, intersect_sphere, ray_shell_intersect};
use aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds};
use poles::{poles_raymarch_entry, sample_poles};
use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3, vec4};
use spirv_std::num_traits::Float;

// --- Constants ---
const DENSITY: f32 = 0.25;

// --- Raymarching Constants ---
const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 120.0;
const STEP_SIZE_OUTSIDE: f32 = 240.0;

const SCALING_END: f32 = 200_000.0;
const SCALING_MAX: f32 = 6.0;
const SCALING_MAX_VERTICAL: f32 = 2.0;
const SCALING_MAX_OCEAN: f32 = 2.0;
const CLOSE_THRESHOLD: f32 = 2000.0;

// --- Lighting Constants ---
const LIGHT_STEPS: u32 = 6;
const LIGHT_STEP_SIZE: f32 = 100.0;
const SUN_CONE_ANGLE: f32 = 0.003;
const AUR_LIGHT_DIR: Vec3 = vec3(0.0, 0.0, -1.0);
const AUR_LIGHT_COLOR_A: Vec3 = vec3(0.36, 0.18, 0.48);
const AUR_LIGHT_COLOR_B: Vec3 = vec3(0.18, 0.09, 0.36);

// --- Material Properties ---
const EXTINCTION: f32 = 0.07;
const SCATTERING_ALBEDO: f32 = 0.65;
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000_004;

const DISK_SAMPLES: [Vec2; 16] = [
    vec2(0.00, 0.00),
    vec2(0.71, 0.00),
    vec2(-0.71, 0.00),
    vec2(0.00, 0.71),
    vec2(0.00, -0.71),
    vec2(0.50, 0.50),
    vec2(-0.50, 0.50),
    vec2(0.50, -0.50),
    vec2(-0.50, -0.50),
    vec2(0.93, 0.38),
    vec2(-0.93, 0.38),
    vec2(0.38, 0.93),
    vec2(-0.38, 0.93),
    vec2(0.38, -0.93),
    vec2(-0.38, -0.93),
    vec2(0.00, 0.00),
];

struct VolumeSample {
    density: f32,
    color: Vec3,
    emission: Vec3,
}

fn sample_volume(
    pos: Vec3,
    view: &ViewUniform,
    clouds: bool,
    ocean: bool,
    poles: bool,
    textures: &Textures,
) -> VolumeSample {
    let mut sample = VolumeSample {
        density: 0.0,
        color: Vec3::ONE,
        emission: Vec3::ZERO,
    };

    let mut blended_color = Vec3::ZERO;
    if clouds {
        let cloud_sample = sample_clouds(pos, view, false, textures);
        if cloud_sample > 0.0 {
            blended_color += Vec3::ONE * cloud_sample;
            sample.density += cloud_sample;
        }
    }
    if ocean {
        let ocean_sample = sample_ocean(pos, view.time(), false, textures);
        if ocean_sample.density > 0.0 {
            blended_color += ocean_sample.color * ocean_sample.density;
            sample.density += ocean_sample.density;
            sample.emission += ocean_sample.emission * ocean_sample.density;
        }
    }
    if poles {
        let poles_sample = sample_poles(pos);
        if poles_sample.density > 0.0 {
            blended_color += poles_sample.color * poles_sample.density;
            sample.density += poles_sample.density;
            sample.emission += poles_sample.emission * poles_sample.density;
        }
    }

    if sample.density > 0.0001 {
        let mix_factor = sample.density.saturate();
        sample.color = blended_color / sample.density;
        sample.emission = (sample.emission / sample.density) * mix_factor;
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
    dither: f32,
    textures: &Textures,
) -> Vec3 {
    let mut optical_depth = (step_density * 1.5).saturate() * EXTINCTION * LIGHT_STEP_SIZE;

    let up = if sun_dir.z.abs() > 0.999 {
        vec3(1.0, 0.0, 0.0)
    } else {
        vec3(0.0, 0.0, 1.0)
    };
    let tangent1 = sun_dir.cross(up).normalize();
    let tangent2 = sun_dir.cross(tangent1);

    let dynamic_light_step = 350.0;
    for j in 0..LIGHT_STEPS {
        let step_index = j as f32 + 1.0;
        let distance_along = step_index * dynamic_light_step;
        let disk_radius = SUN_CONE_ANGLE.tan() * distance_along;

        let idxf = (dither + step_index * 0.618_034).fract_gl();
        let idx = (idxf * 16.0).floor() as i32;
        let sample_offset = DISK_SAMPLES[idx as usize];
        let disk_offset =
            tangent1 * (sample_offset.x * disk_radius) + tangent2 * (sample_offset.y * disk_radius);

        let lightmarch_pos = world_pos + sun_dir * distance_along + disk_offset;
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
    let shadow_boost_clamped = shadow_boost.max(0.0);
    let ambient_floor_vec = atmosphere.ambient * ambient_base * (0.6 + 0.4 * shadow_boost_clamped)
        + atmosphere.sky * (0.5 + altitude_factor * 0.4) * ambient_occlusion;

    (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO
}

fn compute_aur_turbulence(pos: Vec2, time: f32) -> f32 {
    let mut p = pos * 0.0022;
    let mut noise = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    for _ in 0..6 {
        let pattern = (p.x * freq + time * 0.7).sin() * (p.y * freq + time * 0.4).cos();
        noise += pattern * amp;
        let p_rotated = vec2(p.x * 0.809 - p.y * 0.588, p.x * 0.588 + p.y * 0.809);
        p = p_rotated * 1.8;
        amp *= 0.45;
        freq *= 1.618;
    }
    (1.0 - (noise * 0.35).abs()).saturate().powf(6.0)
}

fn sample_aur_lighting(world_pos: Vec3, view: &ViewUniform, textures: &Textures) -> Vec3 {
    let altitude_fade = (1.0 - (world_pos.z / 12000.0).saturate()).powf(2.0);
    if altitude_fade <= 0.0 {
        return Vec3::ZERO;
    }

    let p_raw = world_pos.xy() * 0.002;
    // Domain warp
    let warp_uv = world_pos.xy() * 0.00008 + view.time() * 0.01;
    let warp_sample: Vec4 = textures.weather(warp_uv);
    let warp = vec2(warp_sample.x - 0.5, warp_sample.y - 0.5) * 6.0;
    let p = p_raw + warp;

    let t_scale = view.time() * 0.5;
    let turb1 = (p.x + t_scale).sin() * (p.y + t_scale).cos();
    let turb2 = (p.x * 1.8 - t_scale * 0.4).cos() * (p.y * 1.5 + t_scale * 0.7).sin();

    let turb = (1.0 - (turb1 * 0.35 + turb2 * 0.2).abs())
        .saturate()
        .powf(6.0);
    // Add color variance using a second noise-like pattern for lerping between two purples
    let color_mix = (p.x * 0.3 + view.time() * 0.1).sin() * 0.5 + 0.5;
    let base_color = Vec3::lerp(AUR_LIGHT_COLOR_A, AUR_LIGHT_COLOR_B, color_mix);

    let aur_color = base_color * (0.8 + turb * 12.0 * altitude_fade);

    let mut aur_optical_depth = 0.0;

    let short_steps: u32 = 2;
    let short_step_size: f32 = 40.0;
    for j in 0..short_steps {
        let march_pos = world_pos + AUR_LIGHT_DIR * ((j as f32 + 1.0) * short_step_size);
        let sample = sample_clouds(march_pos, view, true, textures);
        aur_optical_depth += sample * 0.3 * short_step_size;
    }

    aur_color * (-aur_optical_depth.max(0.0)).exp() * altitude_fade
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
    let clouds_entry_exit =
        ray_shell_intersect(ro, rd, view, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT);
    let clouds_entry_exit1 = clouds_entry_exit.xy();
    let clouds_entry_exit2 = clouds_entry_exit.zw();
    let ocean_entry_exit = intersect_sphere(
        ro - view.planet_center(),
        rd,
        PLANET_RADIUS + OCEAN_TOP_HEIGHT,
    );
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    let mut t_start = t_max;
    let mut t_end = 0.0;
    if clouds_entry_exit1.y > 0.0 && clouds_entry_exit1.x < clouds_entry_exit1.y {
        t_start = t_start.min(clouds_entry_exit1.x);
        t_end = t_end.max(clouds_entry_exit1.y);
    }
    if clouds_entry_exit2.y > 0.0 && clouds_entry_exit2.x < clouds_entry_exit2.y {
        t_start = t_start.min(clouds_entry_exit2.x);
        t_end = t_end.max(clouds_entry_exit2.y);
    }
    if ocean_entry_exit.y > 0.0 && ocean_entry_exit.x < ocean_entry_exit.y {
        t_start = t_start.min(ocean_entry_exit.x);
        t_end = t_end.max(ocean_entry_exit.y);
    }
    if poles_entry_exit.y > 0.0 && poles_entry_exit.x < poles_entry_exit.y {
        t_start = t_start.min(poles_entry_exit.x);
        t_end = t_end.max(poles_entry_exit.y);
    }
    t_start = t_start.max(0.0) + dither * STEP_SIZE_OUTSIDE;
    t_end = t_end.min(t_max);

    if t_start >= t_end {
        return RaymarchResult {
            color: vec4(0.0, 0.0, 0.0, 1.0),
            depth: t_max,
        };
    }

    let mut acc_color = Vec3::ZERO;
    let mut step = STEP_SIZE_OUTSIDE;
    let mut t = t_start.max(0.0);
    let mut accumulated_weighted_depth = 0.0;
    let mut accumulated_density = 0.0;
    let mut threshold_depth: f32 = t_max;
    let mut threshold_captured: bool = false;

    // Scale fog density by altitude so it thins toward space.
    let camera_alt = (ro - view.planet_center()).length() - PLANET_RADIUS;
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
            if clouds_entry_exit1.x > t {
                next_t = next_t.min(clouds_entry_exit1.x);
            }
            if clouds_entry_exit2.x > t {
                next_t = next_t.min(clouds_entry_exit2.x);
            }
            if ocean_entry_exit.x > t {
                next_t = next_t.min(ocean_entry_exit.x);
            }
            if poles_entry_exit.x > t {
                next_t = next_t.min(poles_entry_exit.x);
            }

            let segment_length = next_t - t;
            if segment_length > 0.0 {
                let mid_pos = ro + rd * (t + segment_length * 0.5);
                let mid_alt = (mid_pos - view.planet_center()).length() - PLANET_RADIUS;
                let seg_fog = ATMOSPHERIC_FOG_DENSITY * (-mid_alt.max(0.0) / 20_000.0).exp();
                let segment_fog_transmittance = (-seg_fog * segment_length).exp();
                acc_color += atmosphere.ambient * (1.0 - segment_fog_transmittance) * transmittance;
                transmittance *= segment_fog_transmittance;
            }
            t = next_t;
            continue;
        }

        let pos_raw = ro + rd * (t + dither * step);
        let altitude = pos_raw.distance(view.planet_center()) - PLANET_RADIUS;
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
        step = base_step * step_scaler;

        let volume_transmittance = (-(DENSITY * step_density) * step).exp();

        let fog_turb = compute_aur_turbulence(world_pos.xy(), view.time() * 0.5);
        let fog_caustic =
            1.0 + fog_turb * 3.0 * (1.0 - (world_pos.z / 20000.0).saturate()).powf(1.5);

        let color_mix_fog = (world_pos.x * 0.001 + view.time() * 0.1).sin() * 0.5 + 0.5;
        let aur_fog_color = Vec3::lerp(AUR_LIGHT_COLOR_A, AUR_LIGHT_COLOR_B, color_mix_fog);

        let fog_color = atmosphere.sky.lerp(
            aur_fog_color * fog_caustic,
            0.5 * (1.0 - (world_pos.z / 30000.0).saturate()),
        );

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
                dither,
                textures,
            );

            let emission = sample.emission * 1000.0;
            let cloud_aur_boost = if inside_clouds {
                sample_aur_lighting(world_pos, view, textures)
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

        if !threshold_captured && transmittance < 0.5 {
            threshold_captured = true;
            threshold_depth = t;
        }

        t += step;
    }

    // Fog from the end of all volumes to t_max (solid surface or sky)
    let post_dist = (t_max - t.min(t_max)).max(0.0);
    if post_dist > 0.0 && transmittance > 0.0 {
        let mid_t = t + post_dist * 0.5;
        let post_mid = ro + rd * mid_t;
        let post_alt = (post_mid - view.planet_center()).length() - PLANET_RADIUS;
        let post_fog = ATMOSPHERIC_FOG_DENSITY * (-post_alt.max(0.0) / 20_000.0).exp();
        let fg = (-post_fog * post_dist).exp();
        acc_color += atmosphere.ambient * (1.0 - fg) * transmittance;
        transmittance *= fg;
    }

    let avg_depth = accumulated_weighted_depth / accumulated_density.max(0.0001);
    let final_depth = if threshold_captured {
        threshold_depth
    } else if accumulated_density > 0.0001 {
        avg_depth
    } else {
        t_max
    };

    RaymarchResult {
        color: Vec4::from((acc_color, transmittance)).saturate(),
        depth: final_depth,
    }
}
