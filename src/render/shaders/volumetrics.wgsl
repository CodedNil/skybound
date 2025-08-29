#define_import_path skybound::volumetrics
#import skybound::utils::{AtmosphereData, View, intersect_plane, ray_shell_intersect, intersect_sphere}
#import skybound::clouds::{sample_cloud, Cloud, cloud_intersect}
#import skybound::aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var<storage, read> clouds_buffer: CloudsBuffer;
@group(0) @binding(3) var base_texture: texture_3d<f32>;
@group(0) @binding(4) var details_texture: texture_3d<f32>;

struct CloudsBuffer {
    clouds: array<Cloud, 1024>,
    total: u32,
}

// --- Constants ---
const DENSITY: f32 = 0.2;             // Base density for lighting

// --- Raymarching Constants ---
const MAX_STEPS: i32 = 512;           // Maximum number of steps to take in transmittance march
const STEP_SIZE_INSIDE: f32 = 120;
const STEP_SIZE_OUTSIDE: f32 = 240;

const SCALING_END: f32 = 200000;      // Distance from camera to use max step size
const SCALING_MAX: f32 = 6;           // Maximum scaling factor to increase by
const SCALING_MAX_VERTICAL: f32 = 2;  // Scale less if the ray is vertical
const SCALING_MAX_OCEAN: f32 = 2;       // Scale less if the ray is through ocean
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
const EXTINCTION: f32 = 0.05;                   // Overall density/darkness of the cloud material
const AUR_EXTINCTION: f32 = 0.05;               // Lower extinction for aur light to penetrate more
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

// Calculates the incoming light (in-scattering) at a given point within the volume
fn sample_shadowing(world_pos: vec3<f32>, view: View, atmosphere: AtmosphereData, sample_density: f32, time: f32, sun_dir: vec3<f32>, linear_sampler: sampler, dither: f32, cloud_queue_list: array<CloudIntersect, MAX_QUEUED>, cloud_queue_count: u32) -> vec3<f32> {
    // Start with a conservative local optical depth term
    var optical_depth = sample_density * EXTINCTION * LIGHT_STEP_SIZE * 0.5;

    // Build an orthonormal basis around the sun direction for disk sampling
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if abs(sun_dir.z) > 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent1 = normalize(cross(sun_dir, up));
    let tangent2 = cross(sun_dir, tangent1);

    for (var j: u32 = 0u; j < LIGHT_STEPS; j = j + 1u) {
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

        // Sum only queued clouds (cheap, bounded by MAX_QUEUED)
        var clouds_sum: f32 = 0.0;
        for (var k: u32 = 0u; k < cloud_queue_count; k = k + 1u) {
            let ci = cloud_queue_list[k];
            clouds_sum = clouds_sum + sample_cloud(clouds_buffer.clouds[ci.idx], lightmarch_pos, view, time, false, base_texture, details_texture, linear_sampler);
        }
        let ocean = sample_ocean(lightmarch_pos, time, true, details_texture, linear_sampler).density;
        let light_sample_density = clouds_sum + ocean;

        optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE;
    }

    let sun_transmittance = exp(-optical_depth);

    // Calculate Single Scattering (Direct Light)
    let single_scattering = sun_transmittance * atmosphere.sun;

    // Calculate Multiple Scattering (Ambient Light)
    let multiple_scattering = atmosphere.ambient * sun_transmittance;

    // Compute a dynamic ambient floor
    let shadow_boost = (1.0 - sun_transmittance);
    let ambient_base = 0.12 + 0.6 * saturate(sample_density * 3.0);
    let ambient_floor_vec = atmosphere.ambient * ambient_base * (0.6 + 0.4 * shadow_boost) + atmosphere.sky * 0.03;

    let in_scattering = (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO;
    return in_scattering;
}

// Calculates the aur light contribution from below.
fn sample_aur_lighting(world_pos: vec3<f32>, view: View, time: f32, linear_sampler: sampler, cloud_queue_list: array<CloudIntersect, MAX_QUEUED>, cloud_queue_count: u32) -> vec3<f32> {
    // Fade out the light intensity with altitude
    let altitude_fade = 1.0 - saturate(pow(world_pos.z / AUR_LIGHT_DISTANCE, 0.2));
    if altitude_fade <= 0.0 {
        return vec3(0.0);
    }

    // Perform a single light step upwards to check for density that would shadow the point
    let lightmarch_pos = world_pos + AUR_LIGHT_DIR * LIGHT_STEP_SIZE;

    var clouds_sum: f32 = 0.0;
    for (var k: u32 = 0u; k < cloud_queue_count; k = k + 1u) {
        let ci = cloud_queue_list[k];
        clouds_sum = clouds_sum + sample_cloud(clouds_buffer.clouds[ci.idx], lightmarch_pos, view, time, false, base_texture, details_texture, linear_sampler);
    }
    let light_sample_density = clouds_sum;

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

const MAX_QUEUED: u32 = 6u; // Total number of clouds to consider ahead at a time
struct CloudIntersect {
    idx: u32,
    enter: f32,
    exit: f32,
};

fn raymarch_volumetrics(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, view: View, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> RaymarchResult {
    // Get entry exit points for each volume
    let ocean_entry_exit = intersect_sphere(ro - view.planet_center, rd, view.planet_radius + OCEAN_TOP_HEIGHT);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    // Arrays to track entry and exit points for each cloud
    var next_cloud_index: u32 = 0u; // Next cloud index
    var cloud_queue_count: u32 = 0u; // Number of queued clouds
    var cloud_queue_list: array<CloudIntersect, MAX_QUEUED>;

    // Accumulation variables
    var acc_color = vec3(0.0);
    var step = STEP_SIZE_OUTSIDE;
    var t = 0.0;
    var transmittance = 1.0;
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    // Pre-calculated variables
    let sun_world_pos = atmosphere.sun_pos + vec3(view.camera_offset, 0.0);

    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_max || transmittance < 0.01 {
            break;
        }

        // Remove any expired clouds (exit ≤ t)
        var dst: u32 = 0u;
        for (var i: u32 = 0u; i < cloud_queue_count; i = i + 1u) {
            if cloud_queue_list[i].exit > t {
                cloud_queue_list[dst] = cloud_queue_list[i];
                dst++;
            }
        }
        cloud_queue_count = dst;

        // Pull in new clouds to fill the cloud_queue_list
        while cloud_queue_count < MAX_QUEUED && next_cloud_index < clouds_buffer.total {
            let cloud = clouds_buffer.clouds[next_cloud_index];
            let ts = cloud_intersect(ro, rd, cloud);
            let entry = max(ts.x, 0.0);
            let exit = min(ts.y, t_max); // Limit exit by scene depth
            if entry < exit {
                cloud_queue_list[cloud_queue_count] = CloudIntersect(next_cloud_index, entry, exit);
                cloud_queue_count++;
            }
            next_cloud_index++;
        }

        // Find the next boundary > t across clouds, ocean, and poles
        var active_count: u32 = 0u;
        var in_ocean: bool = false;
        var in_poles: bool = false;
        var next_event: f32 = t_max;

        // Consider queued clouds
        for (var i: u32 = 0u; i < cloud_queue_count; i = i + 1u) {
            let entry = cloud_queue_list[i].enter;
            let exit = cloud_queue_list[i].exit;

            if entry <= t && t <= exit {
                active_count++;
            }
            // Next event must be > t
            if entry > t && entry < next_event {
                next_event = entry;
            }
            if exit > t && exit < next_event {
                next_event = exit;
            }
        }

        // Consider ocean sphere entry/exit
        let ocean_entry = ocean_entry_exit.x;
        let ocean_exit = ocean_entry_exit.y;
        if ocean_entry < ocean_exit {
            if ocean_entry <= t && t <= ocean_exit {
                in_ocean = true;
            }
            if ocean_entry > t && ocean_entry < next_event {
                next_event = ocean_entry;
            }
            if ocean_exit > t && ocean_exit < next_event {
                next_event = ocean_exit;
            }
        }

        // Consider poles entry/exit
        let poles_entry = poles_entry_exit.x;
        let poles_exit = poles_entry_exit.y;
        if poles_entry < poles_exit {
            if poles_entry <= t && t <= poles_exit {
                in_poles = true;
            }
            if poles_entry > t && poles_entry < next_event {
                next_event = poles_entry;
            }
            if poles_exit > t && poles_exit < next_event {
                next_event = poles_exit;
            }
        }

        // If no active volumes, fast-forward t to the next_event and account for atmospheric fog across the skipped empty space.
        if active_count == 0u && !in_ocean && !in_poles {
            if next_event < t_max {
                let skip_dist = next_event - t;
                if skip_dist > 0.0 {
                    // Accumulate fog in the skipped segment and update transmittance
                    let fog_trans_skip = exp(-ATMOSPHERIC_FOG_DENSITY * skip_dist);
                    acc_color += atmosphere.sky * (1.0 - fog_trans_skip) * transmittance;
                    transmittance = transmittance * fog_trans_skip;
                }
                // Advance to the next event with a small dither offset to reduce banding
                t = next_event + (dither * STEP_SIZE_INSIDE);
                continue;
            } else {
                break; // No more volumes to march
            }
        }

        // Raymarch until next_event
        while t < next_event && transmittance > 0.01 {
            // Check if we are inside any volume (may be inside multiple simultaneously)
            let inside_clouds = active_count > 0u;

            // Scale step size based on distance from camera
            let distance_scale = saturate(t / SCALING_END);
            let directional_max_scale = mix(SCALING_MAX, SCALING_MAX_VERTICAL, abs(rd.z));
            let max_scale = select(directional_max_scale, SCALING_MAX_OCEAN, in_ocean);
            var step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

            // Reduce scaling when close to solid surfaces
            let distance_to_surface = t_max - t;
            let proximity_factor = 1.0 - saturate(distance_to_surface / CLOSE_THRESHOLD);
            step_scaler = mix(step_scaler, 0.1, proximity_factor);

            // Sample the density
            let pos_raw = ro + rd * (t + dither * step);
            let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
            let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);

            var sample_density = 0.0;
            var sample_emission = vec3<f32>(0.0);
            var blended_color: vec3<f32> = vec3<f32>(0.0);
            if inside_clouds {
                var cloud_sample: f32 = 0.0;
                for (var k: u32 = 0u; k < cloud_queue_count; k = k + 1u) {
                    let cloud = clouds_buffer.clouds[cloud_queue_list[k].idx];
                    cloud_sample = cloud_sample + sample_cloud(cloud, world_pos, view, time, false, base_texture, details_texture, linear_sampler);
                }
                if cloud_sample > 0.0 {
                    blended_color += vec3<f32>(1.0) * cloud_sample;
                    sample_density += cloud_sample;
                }
            }
            if in_ocean {
                let ocean_s = sample_ocean(world_pos, time, false, details_texture, linear_sampler);
                if ocean_s.density > 0.0 {
                    blended_color += ocean_s.color * ocean_s.density;
                    sample_density += ocean_s.density;
                    sample_emission += ocean_s.emission * ocean_s.density;
                }
            }
            if in_poles {
                let poles_s = sample_poles(world_pos, time, linear_sampler);
                if poles_s.density > 0.0 {
                    blended_color += poles_s.color * poles_s.density;
                    sample_density += poles_s.density;
                    sample_emission += poles_s.emission * poles_s.density;
                }
            }

            var sample_color = vec3<f32>(1.0);
            if sample_density > 0.0 {
                let mix_factor = saturate(sample_density);
                sample_color = blended_color / sample_density;
                sample_emission = (sample_emission / sample_density) * mix_factor;
            }

            // Base step size is small in dense areas, large in sparse ones.
            let base_step = mix(STEP_SIZE_OUTSIDE, STEP_SIZE_INSIDE, saturate(sample_density * 5.0));
            step = base_step * step_scaler;

            // Determine the total density for this step by combining cloud and atmospheric fog
            let volume_transmittance = exp(-(DENSITY * sample_density) * step);
            let fog_transmittance_step = exp(-ATMOSPHERIC_FOG_DENSITY * step);
            let alpha_step = 1.0 - volume_transmittance;

            // Add the in-scattered light from the atmospheric fog in this step
            acc_color += atmosphere.sky * (1.0 - fog_transmittance_step) * transmittance;

            if sample_density > 0.0 {
                // Get the incoming light (in-scattering)
                let sun_dir = normalize(sun_world_pos - world_pos);
                let in_scattering = sample_shadowing(world_pos, view, atmosphere, sample_density, time, sun_dir, linear_sampler, dither, cloud_queue_list, cloud_queue_count);

                // Add the incoming aur light from below to the emission
                let emission = sample_emission * 1000.0 + sample_aur_lighting(world_pos, view, time, linear_sampler, cloud_queue_list, cloud_queue_count);

                // Calculate the color of the volume at this point
                let volume_color = in_scattering * sample_color + emission;

                // Accumulate the color, weighted by its contribution
                acc_color += volume_color * transmittance * alpha_step;

                // The contribution of this step towards depth average
                let contribution = sample_density * alpha_step * transmittance;
                accumulated_weighted_depth += t * contribution;
                accumulated_density += contribution;
            }

            transmittance *= volume_transmittance * fog_transmittance_step;
            t += step;
        }
    }

    // Calculate weighted average depth if a volume was hit, otherwise default to t_max.
    let avg_depth = accumulated_weighted_depth / max(accumulated_density, 0.0001);
    let final_depth = select(t_max, avg_depth, accumulated_density > 0.0001);

    var output: RaymarchResult;
    output.color = clamp(vec4<f32>(acc_color, transmittance), vec4(0.0), vec4(1.0));
    output.depth = final_depth;
    return output;
}
