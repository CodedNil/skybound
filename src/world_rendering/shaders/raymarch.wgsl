#define_import_path skybound::raymarch
#import skybound::utils::{AtmosphereData, View, intersect_plane}
#import skybound::clouds::{clouds_raymarch_entry, sample_clouds, get_cloud_layer_above}
#import skybound::aur_fog::{fog_raymarch_entry, sample_fog}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var weather_texture: texture_2d<f32>;

const DENSITY: f32 = 0.2;            // Base density for lighting

const MAX_STEPS: i32 = 4096;
const STEP_SIZE_INSIDE: f32 = 32.0;
const STEP_SIZE_OUTSIDE: f32 = 64.0;

const SCALING_START: f32 = 1000.0;   // Distance from camera to start scaling step size
const SCALING_END: f32 = 500000.0;   // Distance from camera to use max step size
const SCALING_MAX: f32 = 16.0;       // Maximum scaling factor to increase by
const SCALING_MAX_FOG: f32 = 2.0;    // Maximum scaling factor to increase by while in fog
const CLOSE_THRESHOLD: f32 = 200.0;  // Distance from solid objects to begin more precise raymarching

const LIGHT_STEPS: u32 = 4;          // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = 90.0;

const EXTINCTION: f32 = 0.4;                   // Overall density/darkness of the cloud material
const SCATTERING_ALBEDO: f32 = 0.6;            // Color of the cloud. 0.9 is white, lower values are darker grey
const AMBIENT_OCCLUSION_STRENGTH: f32 = 0.03;  // How much shadows affect ambient light. Lower = brighter shadows
const AMBIENT_FLOOR: f32 = 0.02;               // Minimum ambient light to prevent pitch-black shadows

const FOG_DENSITY: f32 = 0.000003;      // Density of the atmospheric fog, higher values create thicker fog
const SHADOW_FADE_END: f32 = 100000.0;  // Distance at which shadows from layers above are fully faded

/// Sample from the volumes
struct VolumeSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume(pos: vec3<f32>, time: f32, volumes_inside: VolumesInside, linear_sampler: sampler) -> VolumeSample {
    var sample: VolumeSample;

    if volumes_inside.clouds {
        let cloud_sample = sample_clouds(pos, time, base_texture, details_texture, weather_texture, linear_sampler);
        if cloud_sample > 0.0 {
            sample.density += cloud_sample;
            sample.color = vec3<f32>(1.0);
        }
    }

    if volumes_inside.fog {
        let fog_sample = sample_fog(pos, time, false, details_texture, linear_sampler);
        if fog_sample.density > 0.0 {
            sample.density = fog_sample.density;
            sample.color = fog_sample.color;
            sample.emission = fog_sample.emission;
        }
    }

    if volumes_inside.poles {
        let poles_sample = sample_poles(pos, time, linear_sampler);
        if poles_sample.density > 0.0 {
            sample.density += poles_sample.density;
            sample.color += poles_sample.color;
            sample.emission += poles_sample.emission;
        }
    }

    sample.density = min(sample.density, 1.0);
    return sample;
}

fn sample_volume_light(pos: vec3<f32>, time: f32, linear_sampler: sampler) -> f32 {
    let clouds = sample_clouds(pos, time, base_texture, details_texture, weather_texture, linear_sampler);
    let fog = sample_fog(pos, time, true, details_texture, linear_sampler).density;
    return clouds + fog;
}

struct RaymarchResult {
    color: vec3<f32>,
    depth: f32
}

struct VolumesInside {
    clouds: bool,
    fog: bool,
    poles: bool,
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, view: View, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> RaymarchResult {
    // Get entry exit points for each volume
    let clouds_entry_exit = clouds_raymarch_entry(ro, rd, view, t_max);
    let fog_entry_exit = fog_raymarch_entry(ro, rd, view, t_max);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    // Get initial start and end of the volumes
    let t_start = min(min(clouds_entry_exit.x, fog_entry_exit.x), poles_entry_exit.x) + dither * STEP_SIZE_INSIDE;
    let t_end = min(max(max(clouds_entry_exit.y, fog_entry_exit.y), poles_entry_exit.y), t_max);

    // Track which volumes we are inside
    var volumes_inside: VolumesInside;

    // Accumulation variables
    var t = t_start;
    var acc_color = vec3(0.0);
    var transmittance = 1.0;
    var steps_outside = 0;

    // For calculating depth
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    // Start raymarching
    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        // Update volume membership
        volumes_inside.clouds = t >= clouds_entry_exit.x && t <= clouds_entry_exit.y;
        volumes_inside.fog = t >= fog_entry_exit.x && t <= fog_entry_exit.y;
        volumes_inside.poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        // If not inside any volume, skip to the next entry point
        if !volumes_inside.clouds && !volumes_inside.fog && !volumes_inside.poles {
            var next_entry = t_end;
            if (clouds_entry_exit.x > t && clouds_entry_exit.x < next_entry) {
                next_entry = clouds_entry_exit.x;
            }
            if (fog_entry_exit.x > t && fog_entry_exit.x < next_entry) {
                next_entry = fog_entry_exit.x;
            }
            if (poles_entry_exit.x > t && poles_entry_exit.x < next_entry) {
                next_entry = poles_entry_exit.x;
            }
            t = next_entry;
            steps_outside = 0;
            continue;
        }

        // Scale step size based on distance from camera
        let distance_scale = min(t / SCALING_END, 1.0);
        let max_scale = select(SCALING_MAX, SCALING_MAX_FOG, volumes_inside.fog);
        var step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        // Reduce scaling when close to surfaces
        let distance_left = t_max - t;
        if distance_left < CLOSE_THRESHOLD {
            let norm = clamp(distance_left / CLOSE_THRESHOLD, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }

        // Sample the density
        let pos_raw = ro + rd * (t + dither * (STEP_SIZE_OUTSIDE * step_scaler));
        let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
        let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);
        let main_sample = sample_volume(world_pos, time, volumes_inside, linear_sampler);
        let step_density = main_sample.density;

        // Get step size based on density
        let step = select(STEP_SIZE_OUTSIDE, STEP_SIZE_INSIDE, step_density > 0.0) * step_scaler;

        // Determine the total density for this step by combining cloud and atmospheric fog
        let total_step_density = DENSITY * step_density + FOG_DENSITY;
        let step_transmittance = exp(-total_step_density * step);
        let alpha_step = 1.0 - step_transmittance;

        if alpha_step > 0.0 {
            // Adjust t to effectively backtrack and take smaller steps when entering density
            if steps_outside != 0 {
                steps_outside = 0;
                t = max(t + (-STEP_SIZE_OUTSIDE + STEP_SIZE_INSIDE) * step_scaler, 0.0);
                continue;
            }

            // Lightmarching for self-shadowing
            var optical_depth: f32 = 0.0;
            var lightmarch_pos = world_pos;
            for (var j: u32 = 0; j < LIGHT_STEPS; j++) {
                lightmarch_pos += view.sun_direction * LIGHT_STEP_SIZE;
                let light_sample_density = sample_volume_light(lightmarch_pos, time, linear_sampler);
                optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE;
            }

            // Optimised sampling of cloud layers above
            if step_density > 0.0 && view.sun_direction.z > 0.01 {
                for (var i: u32 = 1; i <= 8; i++) {
                    let next_layer = get_cloud_layer_above(altitude, f32(i));
                    if next_layer <= 0.0 { break; }

                    // Find intersection with the layer midpoint plane
                    let intersection_t = intersect_plane(world_pos, view.sun_direction, next_layer);
                    // If the intersection is beyond our fade distance, we can stop checking further layers
                    if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END { continue; }

                    for (var j: f32 = 0.0; j <= 4.0; j += 1.0) {
                        let dist = intersection_t + j * 300.0;

                        // Calculate a falloff factor based on the distance to the shadow-casting layer
                        let shadow_falloff = 1.0 - (dist / SHADOW_FADE_END);

                        // Sample at the layer midpoint
                        let lightmarch_pos = world_pos + view.sun_direction * dist;
                        let light_sample_density = sample_volume_light(lightmarch_pos, time, linear_sampler);

                        // Apply the falloff to the optical depth contribution.
                        optical_depth += max(0.0, light_sample_density) * EXTINCTION * LIGHT_STEP_SIZE * shadow_falloff;
                    }
                }
            }
            let sun_transmittance = exp(-optical_depth);

            // Calculate Single Scattering (Direct Light)
            // This is the light from the sun that scatters exactly once towards the camera, it's responsible for the crisp details and the silver lining.
            let single_scattering = sun_transmittance * atmosphere.sun * atmosphere.phase;

            // The ambient light is occluded by the cloud itself.
            let ambient_occlusion = exp(-optical_depth * AMBIENT_OCCLUSION_STRENGTH);
            let multiple_scattering = atmosphere.ambient * ambient_occlusion;

            // The final color is the sum of single and multiple scattering, tinted by the cloud's albedo.
            let in_scattering = (single_scattering + multiple_scattering + AMBIENT_FLOOR) * SCATTERING_ALBEDO;

            // Blend the cloud scattering color with the sky color based on their relative densities
            let cloud_density_ratio = (DENSITY * step_density) / total_step_density;
            let blended_color = mix(atmosphere.sky, in_scattering * main_sample.color, cloud_density_ratio);

            // The contribution of this step towards depth average
            let contribution = step_density * alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;

            // Accumulate color and update transmittance
            acc_color += blended_color * transmittance * alpha_step;
            transmittance *= step_transmittance;
        } else {
            steps_outside += 1;
        }

        t += step;
    }

    // Calculate the final depth
    var final_depth: f32 = 0.0;
    if accumulated_density > 0.0001 {
        final_depth = accumulated_weighted_depth / accumulated_density;
    }

    // Blend the accumulated volume color with the sky color behind it.
    let final_rgb = acc_color + atmosphere.sky * transmittance;

    var output: RaymarchResult;
    output.color = clamp(final_rgb, vec3(0.0), vec3(1.0));
    output.depth = final_depth;
    return output;
}
