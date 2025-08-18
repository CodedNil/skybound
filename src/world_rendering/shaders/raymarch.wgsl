#define_import_path skybound::raymarch
#import skybound::utils::{AtmosphereData, View, intersect_plane, ray_shell_intersect, intersect_sphere}
#import skybound::clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds, get_cloud_layer_above}
#import skybound::aur_fog::{FOG_TOP_HEIGHT, sample_fog}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var weather_texture: texture_2d<f32>;

// --- Constants ---
const DENSITY: f32 = 0.2;               // Base density for lighting

// --- Raymarching Constants ---
const MAX_STEPS: i32 = 2048;            // Maximum number of steps to take in transmittance march
const STEP_SIZE_INSIDE: f32 = 32;
const STEP_SIZE_OUTSIDE: f32 = 64;
const MAX_SAMPLES: i32 = 16;            // Maximum number of samples for decoupled raymarching

const SCALING_END: f32 = 100000.0;      // Distance from camera to use max step size
const SCALING_MAX: f32 = 8.0;           // Maximum scaling factor to increase by
const SCALING_MAX_VERTICAL: f32 = 2.0;  // Scale less if the ray is vertical
const SCALING_MAX_FOG: f32 = 2.0;       // Scale less if the ray is through fog
const CLOSE_THRESHOLD: f32 = 200.0;     // Distance from solid objects to begin more precise raymarching

// --- Lighting Constants ---
const LIGHT_STEPS: u32 = 2;             // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = 90.0;

// --- Material Properties ---
const EXTINCTION: f32 = 0.4;                   // Overall density/darkness of the cloud material
const SCATTERING_ALBEDO: f32 = 0.6;            // Color of the cloud. 0.9 is white, lower values are darker grey
const AMBIENT_OCCLUSION_STRENGTH: f32 = 0.03;  // How much shadows affect ambient light. Lower = brighter shadows
const AMBIENT_FLOOR: f32 = 0.02;               // Minimum ambient light to prevent pitch-black shadows
const SHADOW_FADE_END: f32 = 20000.0;          // Distance at which shadows from layers above are fully faded
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000003;             // Density of the atmospheric fog

// Samples the density, color, and emission from the various volumes
struct VolumeSample {
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume_color(pos: vec3<f32>, time: f32, clouds: bool, fog: bool, poles: bool, linear_sampler: sampler) -> VolumeSample {
    var sample: VolumeSample;

    if clouds {
        sample.color = vec3<f32>(1.0);
    }

    if fog {
        let fog_sample = sample_fog(pos, time, false, details_texture, linear_sampler);
        sample.color = fog_sample.color;
        sample.emission = fog_sample.emission;
    }

    if poles {
        let poles_sample = sample_poles(pos, time, linear_sampler);
        if poles_sample.density > 0.0 {
            sample.color += poles_sample.color;
            sample.emission += poles_sample.emission;
        }
    }

    return sample;
}

fn sample_volume_density(pos: vec3<f32>, time: f32, clouds: bool, fog: bool, poles: bool, linear_sampler: sampler) -> f32 {
    var sample: f32;

    if clouds {
        let cloud_sample = sample_clouds(pos, time, false, base_texture, details_texture, weather_texture, linear_sampler);
        sample = cloud_sample;
    }

    if fog {
        let fog_sample = sample_fog(pos, time, false, details_texture, linear_sampler);
        sample = fog_sample.density;
    }

    if poles {
        let poles_sample = sample_poles(pos, time, linear_sampler);
        if poles_sample.density > 0.0 {
            sample += poles_sample.density;
        }
    }

    sample = min(sample, 1.0);
    return sample;
}

// Calculates the incoming light (in-scattering) at a given point within the volume
fn sample_shadowing(world_pos: vec3<f32>, atmosphere: AtmosphereData, step_density: f32, time: f32, view: View, linear_sampler: sampler) -> vec3<f32> {
    var optical_depth = step_density * EXTINCTION * LIGHT_STEP_SIZE * 0.5;
    var lightmarch_pos = world_pos;
    for (var j: u32 = 0; j < LIGHT_STEPS; j++) {
        lightmarch_pos += view.sun_direction * LIGHT_STEP_SIZE;

        let clouds = sample_clouds(lightmarch_pos, time, false, base_texture, details_texture, weather_texture, linear_sampler);
        let fog = sample_fog(lightmarch_pos, time, true, details_texture, linear_sampler).density;
        let light_sample_density = clouds + fog;

        optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE;
    }

    // Optimised sampling of cloud layers above for shadows
    if step_density > 0.0 && view.sun_direction.z > 0.01 {
        for (var i: u32 = 1; i <= 4; i++) {
            let next_layer = get_cloud_layer_above(world_pos.z, f32(i));
            if next_layer <= 0.0 { break; }

            // Find intersection with the layer midpoint plane
            let intersection_t = intersect_plane(world_pos, view.sun_direction, next_layer);
            // If the intersection is beyond our fade distance, we can stop checking further layers
            if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END { continue; }

            // Calculate a falloff factor based on the distance to the shadow-casting layer
            let shadow_falloff = 1.0 - (intersection_t / SHADOW_FADE_END);

            // Sample at the layer midpoint
            let lightmarch_pos = world_pos + view.sun_direction * intersection_t;
            let light_sample_density = sample_clouds(lightmarch_pos, time, true, base_texture, details_texture, weather_texture, linear_sampler);

            // Apply the falloff to the optical depth contribution.
            optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE * shadow_falloff;
        }
    }
    let sun_transmittance = exp(-optical_depth);

    // Calculate Single Scattering (Direct Light)
    let single_scattering = sun_transmittance * atmosphere.sun;

    // Calculate Multiple Scattering (Ambient Light)
    let multiple_scattering = atmosphere.ambient * mix(1.0, sun_transmittance, AMBIENT_OCCLUSION_STRENGTH);

    let in_scattering = (single_scattering + multiple_scattering + AMBIENT_FLOOR) * SCATTERING_ALBEDO;
    return in_scattering;
}

// Main raymarching entry point
struct RaymarchResult {
    color: vec3<f32>,
    depth: f32
}
struct PackedSample {
    dist: f32,
    density: f32,
    contribution: f32,
};
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, view: View, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> RaymarchResult {
    // Get entry exit points for each volume
    let clouds_entry_exit = ray_shell_intersect(ro, rd, view, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT);
    let fog_entry_exit = intersect_sphere(ro - view.planet_center, rd, view.planet_radius + FOG_TOP_HEIGHT);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    // Get initial start and end of the volumes
    var t_start = t_max;
    var t_end = 0.0;
    if clouds_entry_exit.y > 0.0 && clouds_entry_exit.x < clouds_entry_exit.y {
        t_start = min(clouds_entry_exit.x, t_start);
        t_end = max(clouds_entry_exit.y, t_end);
    }
    if fog_entry_exit.y > 0.0 && fog_entry_exit.x < fog_entry_exit.y {
        t_start = min(fog_entry_exit.x, t_start);
        t_end = max(fog_entry_exit.y, t_end);
    }
    if poles_entry_exit.y > 0.0 && poles_entry_exit.x < poles_entry_exit.y {
        t_start = min(poles_entry_exit.x, t_start);
        t_end = max(poles_entry_exit.y, t_end);
    }
    t_end = min(t_end, t_max);

    if t_start >= t_end {
        return RaymarchResult(atmosphere.sky, t_max);
    }

    // Pass 1: March through the volume to gather important samples using a finer step
    var samples: array<PackedSample, MAX_SAMPLES>;
    var sample_count = 0;
    var step = STEP_SIZE_OUTSIDE;
    var t = max(t_start, 0.0);
    var transmittance = 1.0;
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_end || transmittance < 0.01 || sample_count >= MAX_SAMPLES {
            break;
        }

        // Check if we are inside any volume
        let inside_clouds = t >= clouds_entry_exit.x && t <= clouds_entry_exit.y;
        let inside_fog = t >= fog_entry_exit.x && t <= fog_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        // If not inside any volume, skip to the next entry point
        if !inside_clouds && !inside_fog && !inside_poles {
            var next_t = t_end;
            if clouds_entry_exit.x > t {
                next_t = min(next_t, clouds_entry_exit.x);
            }
            if fog_entry_exit.x > t {
                next_t = min(next_t, fog_entry_exit.x);
            }
            if poles_entry_exit.x > t {
                next_t = min(next_t, poles_entry_exit.x);
            }
            t = next_t;
            continue;
        }

        // Sample the density
        let pos_raw = ro + rd * (t + dither * step);
        let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
        let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);
        let step_density = sample_volume_density(world_pos, time, inside_clouds, inside_fog, inside_poles, linear_sampler);

        // Scale step size based on distance from camera
        let distance_scale = saturate(t / SCALING_END);
        let directional_max_scale = mix(SCALING_MAX, SCALING_MAX_VERTICAL, abs(rd.z));
        let max_scale = select(directional_max_scale, SCALING_MAX_FOG, inside_fog);
        var step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        // Reduce scaling when close to surfaces
        let distance_left = t_max - t;
        let proximity_factor = 1.0 - saturate((CLOSE_THRESHOLD - distance_left) / CLOSE_THRESHOLD);
        step_scaler = mix(0.5, step_scaler, proximity_factor);

        // Base step size is small in dense areas, large in sparse ones.
        let base_step = mix(STEP_SIZE_OUTSIDE, STEP_SIZE_INSIDE, saturate(step_density * 10.0));
        step = base_step * step_scaler;

        // Determine the total density for this step by combining cloud and atmospheric fog
        let total_step_density = DENSITY * step_density ;
        let step_transmittance = exp(-total_step_density * step);
        let alpha_step = 1.0 - step_transmittance;

        if step_density > 0.01 {
            samples[sample_count].dist = t;
            samples[sample_count].density = step_density;
            samples[sample_count].contribution = transmittance * alpha_step;
            sample_count++;

            // The contribution of this step towards depth average
            let contribution = step_density * alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;
        }

        transmittance *= step_transmittance;
        t += step;
    }

    // Pass 2: Integrate lighting over the collected samples
    var acc_color = vec3(0.0);
    if sample_count > 0 {
        for (var i: i32 = 0; i < sample_count; i++) {
            let sample = samples[i];

            // Recompute the exact world position for this sample
            let pos_raw = ro + rd * sample.dist;
            let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
            let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);

            // Check which volumes we are inside at this sample's position
            let inside_clouds = sample.dist >= clouds_entry_exit.x && sample.dist <= clouds_entry_exit.y;
            let inside_fog = sample.dist >= fog_entry_exit.x && sample.dist <= fog_entry_exit.y;
            let inside_poles = sample.dist >= poles_entry_exit.x && sample.dist <= poles_entry_exit.y;

            // Get the color and emission properties of the volume at this point
            let volume_info = sample_volume_color(world_pos, time, inside_clouds, inside_fog, inside_poles, linear_sampler);

            // Get the incoming light (in-scattering)
            let in_scattering = sample_shadowing(world_pos, atmosphere, sample.density, time, view, linear_sampler);

            // Calculate the final color for this step
            let total_step_density = DENSITY * sample.density;
            let cloud_density_ratio = select(0.0, (DENSITY * sample.density) / total_step_density, total_step_density > 0.0);
            let blended_color = mix(atmosphere.sky, in_scattering * volume_info.color + volume_info.emission, cloud_density_ratio);

            // Apply atmospheric fog based on distance
            let fog_transmittance = exp(-sample.dist * ATMOSPHERIC_FOG_DENSITY);
            let fogged_color = mix(atmosphere.sky, blended_color, fog_transmittance);

            // Accumulate the color, weighted by its contribution
            acc_color += fogged_color * sample.contribution;
        }
    }

    let final_rgb = acc_color + atmosphere.sky * transmittance;

    // Calculate weighted average depth if a volume was hit, otherwise default to t_max.
    let avg_depth = accumulated_weighted_depth / max(accumulated_density, 0.0001);
    let final_depth = select(t_max, avg_depth, accumulated_density > 0.0001);

    var output: RaymarchResult;
    output.color = clamp(final_rgb, vec3(0.0), vec3(1.0));
    output.depth = final_depth;
    return output;
}
