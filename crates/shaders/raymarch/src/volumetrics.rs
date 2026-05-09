use crate::aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use crate::clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, get_cloud_layer_above, sample_clouds};
use crate::poles::{poles_raymarch_entry, sample_poles};
use crate::utils::{AtmosphereData, intersect_plane, intersect_sphere, ray_shell_intersect};
use skybound_shared::{PLANET_RADIUS, ViewUniform};
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3, vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler};

// --- Constants ---
const DENSITY: f32 = 0.2; // Base density for lighting

// --- Raymarching Constants ---
const MAX_STEPS: i32 = 512; // Maximum number of steps to take in transmittance march
const STEP_SIZE_INSIDE: f32 = 120.0;
const STEP_SIZE_OUTSIDE: f32 = 240.0;

const SCALING_END: f32 = 200_000.0; // Distance from camera to use max step size
const SCALING_MAX: f32 = 6.0; // Maximum scaling factor to increase by
const SCALING_MAX_VERTICAL: f32 = 2.0; // Scale less if the ray is vertical
const SCALING_MAX_OCEAN: f32 = 2.0; // Scale less if the ray is through ocean
const CLOSE_THRESHOLD: f32 = 2000.0; // Distance from solid objects to begin more precise raymarching

// --- Lighting Constants ---
const LIGHT_STEPS: u32 = 4; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 90.0; // Step size for lightmarching steps
const SUN_CONE_ANGLE: f32 = 0.005; // Angular radius of the sun / area light (radians). Increase for softer shadows
const AUR_LIGHT_DIR: Vec3 = vec3(0.0, 0.0, -1.0); // Direction the aur light comes from (straight below)
const AUR_LIGHT_DISTANCE: f32 = 60000.0; // How high before the aur light becomes negligable
const AUR_LIGHT_COLOR: Vec3 = vec3(0.36, 0.18, 0.48); // Color of the aur light from below
const SHADOW_FADE_END: f32 = 80000.0; // Distance at which shadows from layers above are fully faded

// --- Material Properties ---
const EXTINCTION: f32 = 0.05; // Overall density/darkness of the cloud material
const AUR_EXTINCTION: f32 = 0.05; // Lower extinction for aur light to penetrate more
const SCATTERING_ALBEDO: f32 = 0.65; // Scattering albedo (0..1)
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000_004; // Density of the atmospheric fog

// Precomputed disk samples (unit-disk offsets). These are cheap to index
// and paired with per-pixel `dither` produce low-cost stochastic sampling.
const DISK_SAMPLE_COUNT: u32 = 16;
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

// Samples the density, color, and emission from the various volumes
pub struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

pub fn sample_volume(
    pos: Vec3,
    view: &ViewUniform,
    time: f32,
    clouds: bool,
    ocean: bool,
    poles: bool,
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> VolumeSample {
    let mut sample = VolumeSample {
        density: 0.0,
        color: Vec3::ONE,
        emission: Vec3::ZERO,
    };

    let mut blended_color = Vec3::ZERO;
    if clouds {
        let cloud_sample = sample_clouds(
            pos,
            view,
            time,
            false,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );
        if cloud_sample > 0.0 {
            blended_color += Vec3::ONE * cloud_sample;
            sample.density += cloud_sample;
        }
    }
    if ocean {
        let ocean_sample = sample_ocean(pos, time, false, details_texture, sampler);
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

// Calculates the incoming light (in-scattering) at a given point within the volume
pub fn sample_shadowing(
    world_pos: Vec3,
    view: &ViewUniform,
    atmosphere: &AtmosphereData,
    step_density: f32,
    time: f32,
    sun_dir: Vec3,
    dither: f32,
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> Vec3 {
    // Start with a conservative local optical depth term
    let mut optical_depth = step_density * EXTINCTION * LIGHT_STEP_SIZE * 0.5;

    // Build an orthonormal basis around the sun direction for disk sampling
    let up = if sun_dir.z.abs() > 0.999 {
        vec3(1.0, 0.0, 0.0)
    } else {
        vec3(0.0, 0.0, 1.0)
    };
    let tangent1 = sun_dir.cross(up).normalize();
    let tangent2 = sun_dir.cross(tangent1);

    // Stochastic cone sampling: for each step march a short distance along the sun,
    // then sample within a disk whose radius grows with distance to approximate the
    // solid-angle integration of an extended light source (softens shadows).
    for j in 0..LIGHT_STEPS {
        let step_index = j as f32 + 1.0;
        let distance_along = step_index * LIGHT_STEP_SIZE;

        // Disk radius proportional to distance (cone aperture)
        let disk_radius = distance_along * distance_along.tan(); // Assuming SUN_CONE_ANGLE is baked or use .tan()

        // Select a precomputed disk sample indexed by per-pixel dither + step index.
        let nf = DISK_SAMPLE_COUNT as f32;
        let idxf = (dither + step_index * 0.618_034).fract();
        let idx = (idxf * nf).floor() as i32;
        let sample_offset = DISK_SAMPLES[idx as usize];
        let disk_offset =
            tangent1 * (sample_offset.x * disk_radius) + tangent2 * (sample_offset.y * disk_radius);

        let lightmarch_pos = world_pos + sun_dir * distance_along + disk_offset;

        let clouds = sample_clouds(
            lightmarch_pos,
            view,
            time,
            false,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );
        let ocean = sample_ocean(lightmarch_pos, time, true, details_texture, sampler).density;
        let light_sample_density = clouds + ocean;

        optical_depth += light_sample_density.max(0.0) * EXTINCTION * LIGHT_STEP_SIZE;
    }

    // Optimised sampling of cloud layers above for shadows
    if step_density > 0.0 {
        for i in 1..=16 {
            let next_layer = get_cloud_layer_above(world_pos.z, i as f32);
            if next_layer <= 0.0 {
                break;
            }

            // Find intersection with the layer midpoint plane
            let intersection_t = intersect_plane(world_pos, sun_dir, next_layer);
            // If the intersection is beyond our fade distance, we can stop checking further layers
            if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END {
                continue;
            }

            // Calculate a falloff factor based on the distance to the shadow-casting layer
            let shadow_falloff = 1.0 - (intersection_t / SHADOW_FADE_END);

            // Small jitter within the cone for layer sampling using the disk lookup table
            let layer_disk_radius = intersection_t * SUN_CONE_ANGLE.tan();
            let nf = DISK_SAMPLE_COUNT as f32;
            let idxf = (dither + i as f32 * 0.618_034).fract();
            let idx = (idxf * nf).floor() as i32;
            let sample_offset = DISK_SAMPLES[idx as usize];
            let disk_offset = tangent1 * (sample_offset.x * layer_disk_radius)
                + tangent2 * (sample_offset.y * layer_disk_radius);

            let layer_sample_pos = world_pos + sun_dir * intersection_t + disk_offset;
            let light_sample_density = sample_clouds(
                layer_sample_pos,
                view,
                time,
                true,
                base_texture,
                details_texture,
                weather_texture,
                sampler,
            );

            // Apply the falloff to the optical depth contribution.
            optical_depth +=
                light_sample_density.max(0.0) * EXTINCTION * LIGHT_STEP_SIZE * shadow_falloff;
        }
    }

    let sun_transmittance = (-optical_depth).exp();

    // Calculate Single Scattering (Direct Light)
    let single_scattering = sun_transmittance * atmosphere.sun;

    // Calculate Multiple Scattering (Ambient Light)
    let multiple_scattering = atmosphere.ambient * sun_transmittance;

    // Compute a dynamic ambient floor
    let shadow_boost = 1.0 - sun_transmittance;
    let ambient_base = 0.12 + 0.6 * (step_density * 3.0).saturate();
    let ambient_floor_vec =
        atmosphere.ambient * ambient_base * (0.6 + 0.4 * shadow_boost) + atmosphere.sky * 0.03;

    (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO
}

// Calculates the aur light contribution from below.
pub fn sample_aur_lighting(
    world_pos: Vec3,
    view: &ViewUniform,
    time: f32,
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> Vec3 {
    // Fade out the light intensity with altitude
    let altitude_fade = 1.0 - ((world_pos.z / AUR_LIGHT_DISTANCE).powf(0.2)).saturate();
    if altitude_fade <= 0.0 {
        return Vec3::ZERO;
    }

    // Perform a single light step upwards to check for density that would shadow the point
    let lightmarch_pos = world_pos + AUR_LIGHT_DIR * LIGHT_STEP_SIZE;
    let light_sample_density = sample_clouds(
        lightmarch_pos,
        view,
        time,
        false,
        base_texture,
        details_texture,
        weather_texture,
        sampler,
    );

    // Calculate optical depth from this single sample
    let optical_depth = light_sample_density.max(0.0) * AUR_EXTINCTION * LIGHT_STEP_SIZE;
    let transmittance = (-optical_depth).exp();

    // The final value is the colored light, attenuated by occlusion and altitude
    AUR_LIGHT_COLOR * transmittance * altitude_fade
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
    base_texture: Image!(3D, type=f32, sampled=true),
    details_texture: Image!(3D, type=f32, sampled=true),
    weather_texture: Image!(2D, type=f32, sampled=true),
    sampler: Sampler,
) -> RaymarchResult {
    // Get entry exit points for each volume
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

    // Get initial start and end of the volumes
    let mut t_start = t_max;
    let mut t_end = 0.0;
    let time = view.time();
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

    // Accumulation variables
    let mut acc_color = Vec3::ZERO;
    let mut step = STEP_SIZE_OUTSIDE;
    let mut t = t_start.max(0.0);
    let mut accumulated_weighted_depth = 0.0;
    let mut accumulated_density = 0.0;
    // Depth at 50% opacity: stable front-face, unaffected by distant clouds behind.
    let mut threshold_depth: f32 = t_max;
    let mut threshold_captured: bool = false;

    // Pre-calculate fog for the initial empty space from camera (t=0) to t_start.
    let mut transmittance = if t_start > 0.0 {
        let initial_fog_transmittance = (-ATMOSPHERIC_FOG_DENSITY * t_start).exp();
        acc_color = atmosphere.sky * (1.0 - initial_fog_transmittance);
        initial_fog_transmittance
    } else {
        1.0
    };
    let sun_world_pos = atmosphere.sun_pos + view.camera_offset().extend(0.0);

    for _ in 0..MAX_STEPS {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        // Check if we are inside any volume
        let inside_clouds = (t >= clouds_entry_exit1.x && t <= clouds_entry_exit1.y)
            || (t >= clouds_entry_exit2.x && t <= clouds_entry_exit2.y);
        let inside_ocean = t >= ocean_entry_exit.x && t <= ocean_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        // If not inside any volume, skip to the next entry point
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

            // Account for atmospheric fog across the empty, skipped space
            let segment_length = next_t - t;
            if segment_length > 0.0 {
                let segment_fog_transmittance = (-ATMOSPHERIC_FOG_DENSITY * segment_length).exp();
                acc_color += atmosphere.sky * (1.0 - segment_fog_transmittance) * transmittance;
                transmittance *= segment_fog_transmittance;
            }

            t = next_t;
            continue;
        }

        // Sample the density
        let pos_raw = ro + rd * (t + dither * step);
        let altitude = pos_raw.distance(view.planet_center()) - PLANET_RADIUS;
        let world_pos = (pos_raw.xy() + view.camera_offset()).extend(altitude);
        let sample = sample_volume(
            world_pos,
            view,
            time,
            inside_clouds,
            inside_ocean,
            inside_poles,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );
        let step_density = sample.density;

        // Scale step size based on distance from camera
        let distance_scale = (t / SCALING_END).saturate();
        let directional_max_scale = SCALING_MAX.lerp(SCALING_MAX_VERTICAL, rd.z.abs());
        let max_scale = if inside_ocean {
            SCALING_MAX_OCEAN
        } else {
            directional_max_scale
        };
        let mut step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        // Reduce scaling when close to surfaces
        let distance_to_surface = t_max - t;
        let proximity_factor = 1.0 - (distance_to_surface / CLOSE_THRESHOLD).saturate();
        step_scaler = step_scaler.lerp(0.1, proximity_factor);

        // Base step size is small in dense areas, large in sparse ones.
        let base_step = STEP_SIZE_OUTSIDE.lerp(STEP_SIZE_INSIDE, (step_density * 10.0).saturate());
        step = base_step * step_scaler;

        // Determine the total density for this step by combining cloud and atmospheric fog
        let volume_transmittance = (-(DENSITY * step_density) * step).exp();
        let fog_transmittance_step = (-ATMOSPHERIC_FOG_DENSITY * step).exp();
        let alpha_step = 1.0 - volume_transmittance;

        // Add the in-scattered light from the atmospheric fog in this step
        acc_color += atmosphere.sky * (1.0 - fog_transmittance_step) * transmittance;

        if step_density > 0.0 {
            // Get the incoming light (in-scattering)
            let sun_dir = (sun_world_pos - world_pos).normalize();
            let in_scattering = sample_shadowing(
                world_pos,
                view,
                atmosphere,
                step_density,
                time,
                sun_dir,
                dither,
                base_texture,
                details_texture,
                weather_texture,
                sampler,
            );

            // Add the incoming aur light from below to the emission
            let emission = sample.emission * 1000.0
                + sample_aur_lighting(
                    world_pos,
                    view,
                    time,
                    base_texture,
                    details_texture,
                    weather_texture,
                    sampler,
                );

            // Calculate the color of the volume at this point
            let volume_color = in_scattering * sample.color + emission;

            // Accumulate the color, weighted by its contribution
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

    // Prefer threshold depth; fall back to weighted average for thin volumes; t_max for sky.
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
