#define_import_path skybound::volumetrics
#import skybound::utils::{AtmosphereData, View, intersect_plane, ray_shell_intersect, intersect_sphere}
#import skybound::clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds, get_cloud_layer_above}
#import skybound::aur_fog::{FOG_TOP_HEIGHT, sample_fog}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var weather_texture: texture_2d<f32>;

// --- Constants ---
const DENSITY: f32 = 0.2;             // Base density for lighting

// --- Raymarching Constants ---
const MAX_STEPS: i32 = 512;           // Maximum number of steps to take in transmittance march
const STEP_SIZE_INSIDE: f32 = 120;
const STEP_SIZE_OUTSIDE: f32 = 240;

const SCALING_END: f32 = 200000;      // Distance from camera to use max step size
const SCALING_MAX: f32 = 6;           // Maximum scaling factor to increase by
const SCALING_MAX_VERTICAL: f32 = 2;  // Scale less if the ray is vertical
const SCALING_MAX_FOG: f32 = 2;       // Scale less if the ray is through fog
const CLOSE_THRESHOLD: f32 = 2000;    // Distance from solid objects to begin more precise raymarching

// --- Lighting Constants ---
const LIGHT_STEPS: u32 = 4;                         // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 90.0;                  // Step size for lightmarching steps
const SUN_CONE_ANGLE: f32 = 0.005;                  // Angular radius of the sun / area light (radians). Increase for softer shadows
const AUR_LIGHT_DIR = vec3<f32>(0, 0, -1);          // Direction the aur light comes from (straight below)
const AUR_LIGHT_DISTANCE = 60000;                   // How high before the aur light becomes negligable
const AUR_LIGHT_COLOR = vec3(0.6, 0.3, 0.8) * 0.6;  // Color of the aur light from below
const SHADOW_FADE_END: f32 = 80000;                 // Distance at which shadows from layers above are fully faded

// --- Material Properties ---
const EXTINCTION: f32 = 0.05;                    // Overall density/darkness of the cloud material
const AUR_EXTINCTION: f32 = 0.05;                // Lower extinction for aur light to penetrate more
const SCATTERING_ALBEDO: f32 = 0.65;            // Scattering albedo (0..1)
const ATMOSPHERIC_FOG_DENSITY: f32 = 0.000004;  // Density of the atmospheric fog

// Precomputed disk samples (unit-disk offsets). These are cheap to index
// and paired with per-pixel `dither` produce low-cost stochastic sampling.
const DISK_SAMPLE_COUNT: u32 = 16u;
const DISK_SAMPLES: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(0.000000, 0.000000),
    vec2<f32>(0.707107, 0.000000),
    vec2<f32>(-0.707107, 0.000000),
    vec2<f32>(0.000000, 0.707107),
    vec2<f32>(0.000000, -0.707107),
    vec2<f32>(0.500000, 0.500000),
    vec2<f32>(-0.500000, 0.500000),
    vec2<f32>(0.500000, -0.500000),
    vec2<f32>(-0.500000, -0.500000),
    vec2<f32>(0.923880, 0.382683),
    vec2<f32>(-0.923880, 0.382683),
    vec2<f32>(0.382683, 0.923880),
    vec2<f32>(-0.382683, 0.923880),
    vec2<f32>(0.382683, -0.923880),
    vec2<f32>(-0.382683, -0.923880),
    vec2<f32>(0.000000, 0.000000)
);

// Samples the density, color, and emission from the various volumes
struct VolumeSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume(pos: vec3<f32>, time: f32, clouds: bool, fog: bool, poles: bool, linear_sampler: sampler) -> VolumeSample {
    var sample: VolumeSample;
    sample.color = vec3<f32>(1.0);

    var blended_color: vec3<f32> = vec3(0.0);
    if clouds {
        let cloud_sample = sample_clouds(pos, time, false, base_texture, details_texture, weather_texture, linear_sampler);
        if cloud_sample > 0.0 {
            blended_color += vec3<f32>(1.0) * cloud_sample;
            sample.density += cloud_sample;
        }
    }
    if fog {
        let fog_sample = sample_fog(pos, time, false, details_texture, linear_sampler);
        if fog_sample.density > 0.0 {
            blended_color += fog_sample.color * fog_sample.density;
            sample.density += fog_sample.density;
            sample.emission += fog_sample.emission * fog_sample.density;
        }
    }
    if poles {
        let poles_sample = sample_poles(pos, time, linear_sampler);
        if poles_sample.density > 0.0 {
            blended_color += poles_sample.color * poles_sample.density;
            sample.density += poles_sample.density;
            sample.emission += poles_sample.emission * poles_sample.density;
        }
    }

    if sample.density > 0.0001 {
        let mix_factor = saturate(sample.density);
        sample.color = blended_color / sample.density;
        sample.emission = (sample.emission / sample.density) * mix_factor;
    }
    return sample;
}

// Calculates the incoming light (in-scattering) at a given point within the volume
fn sample_shadowing(world_pos: vec3<f32>, atmosphere: AtmosphereData, step_density: f32, time: f32, sun_dir: vec3<f32>, linear_sampler: sampler, dither: f32) -> vec3<f32> {
    // Start with a conservative local optical depth term
    var optical_depth = step_density * EXTINCTION * LIGHT_STEP_SIZE * 0.5;

    // Build an orthonormal basis around the sun direction for disk sampling
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(sun_dir.z) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent1 = normalize(cross(sun_dir, up));
    let tangent2 = cross(sun_dir, tangent1);

    // Stochastic cone sampling: for each step march a short distance along the sun,
    // then sample within a disk whose radius grows with distance to approximate the
    // solid-angle integration of an extended light source (softens shadows).
    for (var j: u32 = 0; j < LIGHT_STEPS; j++) {
        let step_index = f32(j) + 1.0;
        let distance_along = step_index * LIGHT_STEP_SIZE;

        // Disk radius proportional to distance (cone aperture)
        let disk_radius = tan(SUN_CONE_ANGLE) * distance_along;

        // Select a precomputed disk sample indexed by per-pixel dither + step index.
        let nf = f32(DISK_SAMPLE_COUNT);
        let idxf = fract(dither + step_index * 0.618034);
        let idx = i32(floor(idxf * nf));
        let sample_offset = DISK_SAMPLES[idx];
        let disk_offset = tangent1 * (sample_offset.x * disk_radius) + tangent2 * (sample_offset.y * disk_radius);

        let lightmarch_pos = world_pos + sun_dir * distance_along + disk_offset;

        let clouds = sample_clouds(lightmarch_pos, time, false, base_texture, details_texture, weather_texture, linear_sampler);
        let fog = sample_fog(lightmarch_pos, time, true, details_texture, linear_sampler).density;
        let light_sample_density = clouds + fog;

        optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE;
    }

    // Optimised sampling of cloud layers above for shadows
    if step_density > 0.0 {
        for (var i: u32 = 1; i <= 16; i++) {
            let next_layer = get_cloud_layer_above(world_pos.z, f32(i));
            if next_layer <= 0.0 { break; }

            // Find intersection with the layer midpoint plane
            let intersection_t = intersect_plane(world_pos, sun_dir, next_layer);
            // If the intersection is beyond our fade distance, we can stop checking further layers
            if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END { continue; }

            // Calculate a falloff factor based on the distance to the shadow-casting layer
            let shadow_falloff = 1.0 - (intersection_t / SHADOW_FADE_END);

            // Small jitter within the cone for layer sampling using the disk lookup table
            let layer_disk_radius = tan(SUN_CONE_ANGLE) * intersection_t;
            let nf = f32(DISK_SAMPLE_COUNT);
            let idxf = fract(dither + f32(i) * 0.618034);
            let idx = i32(floor(idxf * nf));
            let sample_offset = DISK_SAMPLES[idx];
            let disk_offset = tangent1 * (sample_offset.x * layer_disk_radius) + tangent2 * (sample_offset.y * layer_disk_radius);

            let layer_sample_pos = world_pos + sun_dir * intersection_t + disk_offset;
            let light_sample_density = sample_clouds(layer_sample_pos, time, true, base_texture, details_texture, weather_texture, linear_sampler);

            // Apply the falloff to the optical depth contribution.
            optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE * shadow_falloff;
        }
    }

    let sun_transmittance = exp(-optical_depth);

    // Calculate Single Scattering (Direct Light)
    let single_scattering = sun_transmittance * atmosphere.sun;

    // Calculate Multiple Scattering (Ambient Light)
    let multiple_scattering = atmosphere.ambient * sun_transmittance;

    // Compute a dynamic ambient floor
    let shadow_boost = (1.0 - sun_transmittance);
    let ambient_base = 0.12 + 0.6 * saturate(step_density * 3.0);
    let ambient_floor_vec = atmosphere.ambient * ambient_base * (0.6 + 0.4 * shadow_boost) + atmosphere.sky * 0.03;

    let in_scattering = (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO;
    return in_scattering;
}

// Calculates the aur light contribution from below.
fn sample_aur_lighting(world_pos: vec3<f32>, time: f32, linear_sampler: sampler) -> vec3<f32> {
    // Fade out the light intensity with altitude
    let altitude_fade = 1.0 - saturate(pow(world_pos.z / AUR_LIGHT_DISTANCE, 0.2));
    if (altitude_fade <= 0.0) {
        return vec3(0.0);
    }

    // Perform a single light step upwards to check for density that would shadow the point
    let lightmarch_pos = world_pos + AUR_LIGHT_DIR * LIGHT_STEP_SIZE;
    let light_sample_density = sample_clouds(lightmarch_pos, time, false, base_texture, details_texture, weather_texture, linear_sampler);

    // Calculate optical depth from this single sample
    let optical_depth = max(0.0, light_sample_density) * AUR_EXTINCTION * LIGHT_STEP_SIZE;
    let transmittance = exp(-optical_depth);

    // The final value is the colored light, attenuated by occlusion and altitude
    return AUR_LIGHT_COLOR * transmittance * altitude_fade;
}

// Main raymarching entry point
struct RaymarchResult {
    color: vec4<f32>,
    depth: f32
}
fn raymarch_volumetrics(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, view: View, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> RaymarchResult {
    // Get entry exit points for each volume
    let clouds_entry_exit = ray_shell_intersect(ro, rd, view, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT);
    let clouds_entry_exit1 = vec2<f32>(clouds_entry_exit.x, clouds_entry_exit.y);
    let clouds_entry_exit2 = vec2<f32>(clouds_entry_exit.z, clouds_entry_exit.w);
    let fog_entry_exit = intersect_sphere(ro - view.planet_center, rd, view.planet_radius + FOG_TOP_HEIGHT);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    // Get initial start and end of the volumes
    var t_start = t_max;
    var t_end = 0.0;
    if clouds_entry_exit1.y > 0.0 && clouds_entry_exit1.x < clouds_entry_exit1.y {
        t_start = min(clouds_entry_exit1.x, t_start);
        t_end = max(clouds_entry_exit1.y, t_end);
    }
    if clouds_entry_exit2.y > 0.0 && clouds_entry_exit2.x < clouds_entry_exit2.y {
        t_start = min(clouds_entry_exit2.x, t_start);
        t_end = max(clouds_entry_exit2.y, t_end);
    }
    if fog_entry_exit.y > 0.0 && fog_entry_exit.x < fog_entry_exit.y {
        t_start = min(fog_entry_exit.x, t_start);
        t_end = max(fog_entry_exit.y, t_end);
    }
    if poles_entry_exit.y > 0.0 && poles_entry_exit.x < poles_entry_exit.y {
        t_start = min(poles_entry_exit.x, t_start);
        t_end = max(poles_entry_exit.y, t_end);
    }
    t_start = max(t_start, 0.0) + dither * STEP_SIZE_OUTSIDE;
    t_end = min(t_end, t_max);

    if t_start >= t_end {
        return RaymarchResult(vec4<f32>(0.0, 0.0, 0.0, 1.0), t_max);
    }

    // Accumulation variables
    var acc_color = vec3(0.0);
    var step = STEP_SIZE_OUTSIDE;
    var t = max(t_start, 0.0);
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    // Pre-calculate fog for the initial empty space from camera (t=0) to t_start.
    var transmittance = 1.0;
    if t_start > 0.0 {
        let initial_fog_transmittance = exp(-ATMOSPHERIC_FOG_DENSITY * t_start);
        acc_color = atmosphere.sky * (1.0 - initial_fog_transmittance);
        transmittance = initial_fog_transmittance;
    }
    let sun_altitude = distance(atmosphere.sun_pos, view.planet_center) - view.planet_radius;
    let sun_world_pos = vec3<f32>(atmosphere.sun_pos.xy + view.camera_offset, sun_altitude);

    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        // Check if we are inside any volume
        let inside_clouds = t >= clouds_entry_exit1.x && t <= clouds_entry_exit1.y || t >= clouds_entry_exit2.x && t <= clouds_entry_exit2.y;
        let inside_fog = t >= fog_entry_exit.x && t <= fog_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        // If not inside any volume, skip to the next entry point
        if !inside_clouds && !inside_fog && !inside_poles {
            var next_t = t_end;
            if clouds_entry_exit1.x > t {
                next_t = min(next_t, clouds_entry_exit1.x);
            }
            if clouds_entry_exit2.x > t {
                next_t = min(next_t, clouds_entry_exit2.x);
            }
            if fog_entry_exit.x > t {
                next_t = min(next_t, fog_entry_exit.x);
            }
            if poles_entry_exit.x > t {
                next_t = min(next_t, poles_entry_exit.x);
            }

            // Account for atmospheric fog across the empty, skipped space
            let segment_length = next_t - t;
            if segment_length > 0.0 {
                let segment_fog_transmittance = exp(-ATMOSPHERIC_FOG_DENSITY * segment_length);
                acc_color += atmosphere.sky * (1.0 - segment_fog_transmittance) * transmittance;
                transmittance *= segment_fog_transmittance;
            }

            t = next_t;
            continue;
        }

        // Sample the density
        let pos_raw = ro + rd * (t + dither * step);
        let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
        let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);
        let sample = sample_volume(world_pos, time, inside_clouds, inside_fog, inside_poles, linear_sampler);
        let step_density = sample.density;

        // Scale step size based on distance from camera
        let distance_scale = saturate(t / SCALING_END);
        let directional_max_scale = mix(SCALING_MAX, SCALING_MAX_VERTICAL, abs(rd.z));
        let max_scale = select(directional_max_scale, SCALING_MAX_FOG, inside_fog);
        var step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        // Reduce scaling when close to surfaces
        let distance_to_surface = t_max - t;
        let proximity_factor = 1.0 - saturate(distance_to_surface / CLOSE_THRESHOLD);
        step_scaler = mix(step_scaler, 0.1, proximity_factor);

        // Base step size is small in dense areas, large in sparse ones.
        let base_step = mix(STEP_SIZE_OUTSIDE, STEP_SIZE_INSIDE, saturate(step_density * 10.0));
        step = base_step * step_scaler;

        // Determine the total density for this step by combining cloud and atmospheric fog
        let volume_transmittance = exp(-(DENSITY * step_density) * step);
        let fog_transmittance_step = exp(-ATMOSPHERIC_FOG_DENSITY * step);
        let alpha_step = 1.0 - volume_transmittance;

        // Add the in-scattered light from the atmospheric fog in this step
        acc_color += atmosphere.sky * (1.0 - fog_transmittance_step) * transmittance;

        if step_density > 0.0 {
            // Get the incoming light (in-scattering)
            let sun_dir = normalize(sun_world_pos - world_pos);
            let in_scattering = sample_shadowing(world_pos, atmosphere, step_density, time, sun_dir, linear_sampler, dither);

            // Add the incoming aur light from below to the emission
            let emission = sample.emission * 1000.0 + sample_aur_lighting(world_pos, time, linear_sampler);

            // Calculate the color of the volume at this point
            let volume_color = in_scattering * sample.color + emission;

            // Accumulate the color, weighted by its contribution
            acc_color += volume_color * transmittance * alpha_step;

            // The contribution of this step towards depth average
            let contribution = step_density * alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;
        }

        transmittance *= volume_transmittance * fog_transmittance_step;
        t += step;
    }

    // Calculate weighted average depth if a volume was hit, otherwise default to t_max.
    let avg_depth = accumulated_weighted_depth / max(accumulated_density, 0.0001);
    let final_depth = select(t_max, avg_depth, accumulated_density > 0.0001);

    var output: RaymarchResult;
    output.color = clamp(vec4<f32>(acc_color, transmittance), vec4(0.0), vec4(1.0));
    output.depth = final_depth;
    return output;
}
