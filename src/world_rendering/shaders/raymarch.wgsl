#define_import_path skybound::raymarch
#import skybound::utils::{AtmosphereData, View, intersect_plane, ray_shell_intersect}
#import skybound::clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, sample_clouds, get_cloud_layer_above}
#import skybound::aur_fog::{FOG_BOTTOM_HEIGHT, FOG_TOP_HEIGHT, sample_fog}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var weather_texture: texture_2d<f32>;

const DENSITY: f32 = 0.2;            // Base density for lighting

const MAX_STEPS: i32 = 2048;
const STEP_SIZE_INSIDE: f32 = 32.0;
const STEP_SIZE_OUTSIDE: f32 = 64.0;

const SCALING_END: f32 = 100000.0;   // Distance from camera to use max step size
const SCALING_MAX: f32 = 8.0;       // Maximum scaling factor to increase by
const SCALING_MAX_FOG: f32 = 2.0;    // Maximum scaling factor to increase by while in fog
const CLOSE_THRESHOLD: f32 = 200.0;  // Distance from solid objects to begin more precise raymarching

const LIGHT_STEPS: u32 = 4;          // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = 90.0;

const EXTINCTION: f32 = 0.4;                   // Overall density/darkness of the cloud material
const SCATTERING_ALBEDO: f32 = 0.6;            // Color of the cloud. 0.9 is white, lower values are darker grey
const AMBIENT_OCCLUSION_STRENGTH: f32 = 0.03;  // How much shadows affect ambient light. Lower = brighter shadows
const AMBIENT_FLOOR: f32 = 0.02;               // Minimum ambient light to prevent pitch-black shadows

const FOG_DENSITY: f32 = 0.000003;     // Density of the atmospheric fog, higher values create thicker fog
const SHADOW_FADE_END: f32 = 30000.0;  // Distance at which shadows from layers above are fully faded

/// Sample from the volumes
struct VolumeSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume(pos: vec3<f32>, time: f32, clouds: bool, fog: bool, poles: bool, linear_sampler: sampler) -> VolumeSample {
    var sample: VolumeSample;

    if clouds {
        let cloud_sample = sample_clouds(pos, time, base_texture, details_texture, weather_texture, linear_sampler);
        sample.density = cloud_sample;
        sample.color = vec3<f32>(1.0);
    }

    if fog {
        let fog_sample = sample_fog(pos, time, false, details_texture, linear_sampler);
        sample.density = fog_sample.density;
        sample.color = fog_sample.color;
        sample.emission = fog_sample.emission;
    }

    if poles {
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

// Logic to gather intersections per workgroup
// 0 is none, first 4 bits are 16 cloud layers, 5th bit is fog, 6th bit is poles
// var<workgroup> intersections: array<Intersection, 18>;
// struct Intersection {
//     idx: u32,
//     distance: f32,
//     is_entry: bool,
// }

// const CLOUD_BOTTOM_HEIGHT: f32 = 1000;
// const CLOUD_TOP_HEIGHT: f32 = 48000;
// const CLOUD_LAYER_HEIGHT: f32 = 3000;
// struct CloudLayer {
//     index: u32,
//     bottom: f32,
//     height: f32,
// }
// fn get_cloud_layer(pos: vec3<f32>) -> CloudLayer {
//     let index: u32 = u32((pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_HEIGHT);
//     let bottom: f32 = CLOUD_BOTTOM_HEIGHT + f32(index) * CLOUD_LAYER_HEIGHT;

//     // Branchless validity check, height becomes 0 if invalid
//     let is_valid_layer: bool = (pos.z > CLOUD_BOTTOM_HEIGHT) && (index < CLOUD_TOTAL_LAYERS);
//     let height: f32 = select(0.0, CLOUD_LAYER_HEIGHTS[index], is_valid_layer);
//     let is_within_thickness: f32 = f32(pos.z <= bottom + height);

//     return CloudLayer(index, bottom, height * is_within_thickness);
// }

// fn gather_intersections(ro: vec3<f32>, rd: vec3<f32>, view: View) {
//     let poles_entry_exit = poles_raymarch_entry(ro, rd, view, 0.0);
//     let up = rd.z > 0.0;
//     // Possible paths, consider poles at every point
//     // Already in fog, going down has no events, just fog forever, going up has fog exit then every cloud layer
//     // Above fog below clouds, could go down into fog, entry point only one to consider, if going up then consider clouds layers
//     // Already in a cloud layer, start with volume for that layer, then exit point for it and the layers above or below entry then exit
//     // Above cloud layer, camera going up no paths, camera going down all clouds then fog

//     if up {

//     }
// }



// Raymarch
struct RaymarchResult {
    color: vec3<f32>,
    depth: f32
}
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, atmosphere: AtmosphereData, view: View, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> RaymarchResult {
    // Get entry exit points for each volume
    let clouds_entry_exit = ray_shell_intersect(ro, rd, view, CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT);
    let fog_entry_exit = ray_shell_intersect(ro, rd, view, FOG_BOTTOM_HEIGHT, FOG_TOP_HEIGHT);
    let poles_entry_exit = poles_raymarch_entry(ro, rd, view, t_max);

    // Get initial start and end of the volumes
    var t_start = t_max;
    if clouds_entry_exit.x < clouds_entry_exit.y {
        t_start = min(clouds_entry_exit.x, t_start);
    }
    if fog_entry_exit.x < fog_entry_exit.y {
        t_start = min(fog_entry_exit.x, t_start);
    }
    if poles_entry_exit.x < poles_entry_exit.y {
        t_start = min(poles_entry_exit.x, t_start);
    }
    let t_end = min(max(max(clouds_entry_exit.y, fog_entry_exit.y), poles_entry_exit.y), t_max);

    // Accumulation variables
    var t = max(t_start, 0.0) + dither * STEP_SIZE_INSIDE;
    var acc_color = vec3(0.0);
    var transmittance = 1.0;
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    // Start raymarching
    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        // Check if we are inside any volume
        let inside_clouds = t >= clouds_entry_exit.x && t <= clouds_entry_exit.y;
        let inside_fog = t >= fog_entry_exit.x && t <= fog_entry_exit.y;
        let inside_poles = t >= poles_entry_exit.x && t <= poles_entry_exit.y;

        // If not inside any volume, skip to the next entry point
        if !inside_clouds && !inside_fog && !inside_poles {
            var next_t = t_end;
            next_t = select(next_t, clouds_entry_exit.x, clouds_entry_exit.x > t);
            next_t = select(next_t, fog_entry_exit.x, fog_entry_exit.x > t);
            next_t = select(next_t, poles_entry_exit.x, poles_entry_exit.x > t);
            t = next_t;
            continue;
        }

        // Sample the density
        let pos_raw = ro + rd * t;
        let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
        let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);
        let main_sample = sample_volume(world_pos, time, inside_clouds, inside_fog, inside_poles, linear_sampler);
        let step_density = main_sample.density;

        // Scale step size based on distance from camera
        let distance_scale = saturate(t / SCALING_END);
        let max_scale = select(SCALING_MAX, SCALING_MAX_FOG, inside_fog);
        var step_scaler = 1.0 + distance_scale * distance_scale * max_scale;

        // Reduce scaling when close to surfaces
        let distance_left = t_max - t;
        let proximity_factor = 1.0 - saturate((CLOSE_THRESHOLD - distance_left) / CLOSE_THRESHOLD);
        step_scaler = mix(0.5, step_scaler, proximity_factor);

        // Base step size is small in dense areas, large in sparse ones.
        let base_step = mix(STEP_SIZE_OUTSIDE, STEP_SIZE_INSIDE, saturate(step_density * 10.0));
        let step = base_step * step_scaler;

        // Determine the total density for this step by combining cloud and atmospheric fog
        let total_step_density = DENSITY * step_density + FOG_DENSITY;
        let step_transmittance = exp(-total_step_density * step);
        let alpha_step = 1.0 - step_transmittance;

        if alpha_step > 0.0 {
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
                for (var i: u32 = 1; i <= 6; i++) {
                    let next_layer = get_cloud_layer_above(altitude, f32(i));
                    if next_layer <= 0.0 { break; }

                    // Find intersection with the layer midpoint plane
                    let intersection_t = intersect_plane(world_pos, view.sun_direction, next_layer);
                    // If the intersection is beyond our fade distance, we can stop checking further layers
                    if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END { continue; }

                    for (var j: f32 = 0.0; j <= 2.0; j += 1.0) {
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
            let single_scattering = sun_transmittance * atmosphere.sun;

            // The ambient light is occluded by the cloud itself.
            let multiple_scattering = atmosphere.ambient * mix(1.0, sun_transmittance, AMBIENT_OCCLUSION_STRENGTH);

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
        }

        t += step;
    }

    // Calculate the final depth
    let final_depth = accumulated_weighted_depth / max(accumulated_density, 0.0001);
    // Use step() to zero out depth if no density was accumulated.
    let depth_condition = step(0.0001, accumulated_density);

    // Blend the accumulated volume color with the sky color behind it.
    let final_rgb = acc_color + atmosphere.sky * transmittance;
    // let final_rgb = vec3(t_start * 0.0001);

    var output: RaymarchResult;
    output.color = clamp(final_rgb, vec3(0.0), vec3(1.0));
    output.depth = final_depth * depth_condition;
    return output;
}
