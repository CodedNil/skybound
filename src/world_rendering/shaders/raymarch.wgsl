#define_import_path skybound::raymarch
#import skybound::utils::{AtmosphereData, View, intersect_plane}
#import skybound::clouds::{clouds_raymarch_entry, sample_clouds, get_cloud_layer_above}
#import skybound::aur_fog::{fog_raymarch_entry, sample_fog}
#import skybound::poles::{poles_raymarch_entry, sample_poles}

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var motion_texture: texture_2d<f32>;
@group(0) @binding(5) var weather_texture: texture_2d<f32>;

const ALPHA_THRESHOLD: f32 = 0.99; // Max alpha to reach before stopping
const DENSITY: f32 = 0.05; // Base density for lighting

const MAX_STEPS: i32 = 4096;
const STEP_SIZE_INSIDE: f32 = 24.0;
const STEP_SIZE_OUTSIDE: f32 = 48.0;

const SCALING_START: f32 = 1000.0; // Distance from camera to start scaling step size
const SCALING_END: f32 = 500000.0; // Distance from camera to use max step size
const SCALING_MAX: f32 = 8.0; // Maximum scaling factor to increase by
const SCALING_MAX_FOG: f32 = 2.0; // Maximum scaling factor to increase by while in fog
const CLOSE_THRESHOLD: f32 = 200.0; // Distance from solid objects to begin more precise raymarching

const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = 40.0;
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));

/// Sample from the volumes
struct VolumeSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_volume(pos: vec3<f32>, dist: f32, time: f32, volumes_inside: VolumesInside, linear_sampler: sampler) -> VolumeSample {
    var sample: VolumeSample;

    if volumes_inside.clouds {
        let cloud_sample = sample_clouds(pos, dist, time, base_texture, details_texture, motion_texture, weather_texture, linear_sampler);
        if cloud_sample > 0.0 {
            sample.density += cloud_sample;
            sample.color = vec3<f32>(1.0);
        }
    }

    if volumes_inside.fog {
        let fog_sample = sample_fog(pos, dist, time, false, details_texture, linear_sampler);
        if fog_sample.density > 0.0 {
            sample.density = fog_sample.density;
            sample.color = fog_sample.color;
            sample.emission = fog_sample.emission;
        }
    }

    if volumes_inside.poles {
        let poles_sample = sample_poles(pos, dist, time, linear_sampler);
        if poles_sample.density > 0.0 {
            sample.density += poles_sample.density;
            sample.color += poles_sample.color;
            sample.emission += poles_sample.emission;
        }
    }

    sample.density = min(sample.density, 1.0);
    return sample;
}

fn sample_volume_light(pos: vec3<f32>, dist: f32, time: f32, linear_sampler: sampler) -> f32 {
    let clouds = sample_clouds(pos, dist, time, base_texture, details_texture, motion_texture, weather_texture, linear_sampler);
    let fog = sample_fog(pos, dist, time, true, details_texture, linear_sampler).density;
    return clouds + fog;
}

struct RaymarchResult {
    color: vec4<f32>,
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
    var acc_alpha = 0.0;
    var transmittance = 1.0;
    var steps_outside = 0;

    // For calculating depth
    var accumulated_weighted_depth = 0.0;
    var accumulated_density = 0.0;

    // Start raymarching
    for (var i = 0; i < MAX_STEPS; i++) {
        if t >= t_end || acc_alpha >= ALPHA_THRESHOLD {
            break;
        }

        // Track which volumes we are inside
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
            continue;
        }

        // Scale step size based on distance from camera
        var step_scaler = 1.0;
        let distance_scale = pow(smoothstep(SCALING_START, SCALING_END, t), 0.5);
        if t > SCALING_START && volumes_inside.fog {
            step_scaler = 1.0 + distance_scale * SCALING_MAX_FOG;
        } else if t > SCALING_START {
            step_scaler = 1.0 + distance_scale * SCALING_MAX;
        }
        // Reduce scaling when close to surfaces
        let distance_left = t_max - t;
        if distance_left < CLOSE_THRESHOLD {
            let norm = clamp(distance_left / CLOSE_THRESHOLD, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }
        var step = STEP_SIZE_OUTSIDE * step_scaler;

        // Sample the volumes
        let pos_raw = ro + rd * (t + dither * step);
        let altitude = distance(pos_raw, view.planet_center) - view.planet_radius;
        let world_pos = vec3<f32>(pos_raw.xy + view.camera_offset, altitude);
        let main_sample = sample_volume(world_pos, t, time, volumes_inside, linear_sampler);
        let step_density = main_sample.density;

        if step_density > 0.0 {
            // Adjust t to effectively backtrack and take smaller steps when entering density
            if steps_outside != 0 {
                steps_outside = 0;
                t = max(t + (-STEP_SIZE_OUTSIDE + STEP_SIZE_INSIDE) * step_scaler, 0.0);
                continue;
            }

            step = STEP_SIZE_INSIDE * step_scaler;

            let step_transmittance = max(0.0, 1.0 - DENSITY * step_density * step);
            let alpha_step = 1.0 - step_transmittance;

            // The contribution of this step towards depth average
            let contribution = step_density * alpha_step * transmittance;
            accumulated_weighted_depth += t * contribution;
            accumulated_density += contribution;

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var lightmarch_pos = world_pos;
            for (var j: u32 = 0; j < LIGHT_STEPS; j++) {
                lightmarch_pos += (view.sun_direction + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE;
                density_sunwards += sample_volume_light(lightmarch_pos, t, time, linear_sampler);
            }

            // // Optimised sampling of cloud layers above
            // if step_density > 0.0 && view.sun_direction.z > 0.01 {
            //     for (var i: u32 = 1; i <= 3; i++) {
            //         let next_layer = get_cloud_layer_above(altitude, f32(i));
            //         if next_layer <= 0.0 { continue; }

            //         // Find intersection with the layer midpoint plane
            //         let intersection_t = intersect_plane(world_pos, view.sun_direction, next_layer);
            //         if intersection_t <= 0.0 { continue; }

            //         // Sample at the layer midpoint with higher weight since we're skipping intermediate samples
            //         for (var j: f32 = -1.0; j <= 1.0; j += 1.0) {
            //             let lightmarch_pos = world_pos + view.sun_direction * (intersection_t + j * 40.0);
            //             density_sunwards += sample_volume_light(lightmarch_pos, t, time, linear_sampler) * 4.0;
            //         }
            //     }
            // }

            // Captures the direct lighting from the sun
            let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE);
            let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE * 0.25) * 0.7;
            let beers_total = max(beers, beers2);

			// Compute in-scattering
            let aur_intensity = smoothstep(6000.0, 0.0, altitude);
            let aur_ambient = mix(vec3(1.0), atmosphere.ground, aur_intensity);
            let ambient = aur_ambient * DENSITY * mix(atmosphere.ambient, vec3(1.0), 0.4) * (view.sun_direction.z);
            let in_scattering = ambient + beers_total * atmosphere.sun * atmosphere.phase;

            acc_alpha += alpha_step * (1.0 - acc_alpha);
            acc_color += in_scattering * transmittance * alpha_step * main_sample.color + main_sample.emission;

            transmittance *= step_transmittance;
        } else {
            steps_outside += 1;
        }

        t += step;
    }


    // Calculate the final stable depth
    var final_depth: f32 = 0.0;
    if accumulated_density > 0.0001 {
        final_depth = accumulated_weighted_depth / accumulated_density;
    }

    // Scale alpha so ALPHA_THRESHOLD becomes 1.0
    acc_alpha = min(min(acc_alpha, 1.0) * (1.0 / ALPHA_THRESHOLD), 1.0);

    var output: RaymarchResult;
    output.color = clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
    output.depth = final_depth;
    return output;
}
