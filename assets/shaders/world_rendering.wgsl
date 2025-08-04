#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::functions::blue_noise
#import skybound::clouds::sample_clouds
#import skybound::aur_fog::sample_fog
#import skybound::sky::render_sky
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
struct View {
    world_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    planet_rotation: vec4<f32>,
    latitude: f32,
    longitude: f32,
    latitude_meters: f32,
    longitude_meters: f32,
    altitude: f32,
};
@group(0) @binding(1) var<uniform> globals: Globals;
struct Globals {
    time: f32,
    planet_radius: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
}
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

@group(0) @binding(4) var perlinworley_texture: texture_3d<f32>;
@group(0) @binding(5) var perlinworley_sampler: sampler;

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 8.0;
const STEP_SIZE_OUTSIDE: f32 = 30.0;

const STEP_DISTANCE_SCALING_START: f32 = 500.0; // Distance from camera to start scaling step size
const STEP_DISTANCE_SCALING_FACTOR: f32 = 0.0003; // How much to scale step size by distance

const LIGHT_STEPS: u32 = 4; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: array<f32, 6> = array<f32, 6>(6.0, 10.0, 16.0, 24.0, 36.0, 48.0);

// Lighting Parameters
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);
const MIN_SUN_DOT: f32 = sin(radians(-8.0)); // How far below the horizon before the switching to aur light
const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const AMBIENT_COLOR: vec3<f32> = vec3(0.7, 0.8, 1.0) * 0.5;
const AMBIENT_AUR_COLOR: vec3<f32> = vec3(0.3, 0.2, 0.8) * 0.05;

const FOG_START_DISTANCE: f32 = 1000.0;
const FOG_END_DISTANCE: f32 = 100000.0;

const SHADOW_EXTINCTION: f32 = 2.0; // Higher = deeper core shadows

// Cloud Material Parameters
const BACK_SCATTERING: f32 = 1.0; // Backscattering
const BACK_SCATTERING_FALLOFF: f32 = 30.0; // Backscattering falloff
const OMNI_SCATTERING: f32 = 0.8; // Omnidirectional Scattering
const TRANSMISSION_SCATTERING: f32 = 1.0; // Transmission Scattering
const TRANSMISSION_FALLOFF: f32 = 2.0; // Transmission falloff
const BASE_TRANSMISSION: f32 = 0.1; // Light that doesn't get scattered at all

// Cloud density and colour
struct CloudSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_cloud(pos: vec3<f32>, dist: f32) -> CloudSample {
    var sample: CloudSample;

    // Thick aur fog below 0m
    let fog_sample = sample_fog(pos, dist, globals.time);
    sample.emission = fog_sample.emission;

    // Sample our clouds
    let clouds_sample = sample_clouds(pos, dist, globals.time);

    sample.density = fog_sample.contribution + clouds_sample;
    if sample.density > 0.0 {
        sample.color = (fog_sample.color * fog_sample.contribution + clouds_sample) / sample.density;
    } else {
        sample.color = vec3(1.0);
    }

    return sample;
}

// Lighting Functions
fn beer(material_amount: f32) -> f32 {
    return exp(-material_amount);
}

fn transmission(light: vec3<f32>, material_amount: f32) -> vec3<f32> {
    return beer(material_amount * (1.0 - BASE_TRANSMISSION)) * light;
}

fn light_scattering(light: vec3<f32>, angle: f32) -> vec3<f32> {
    var a = (angle + 1.0) * 0.5; // Angle between 0 and 1

    var ratio = 0.0;
    ratio += BACK_SCATTERING * pow(1.0 - a, BACK_SCATTERING_FALLOFF);
    ratio += TRANSMISSION_SCATTERING * pow(a, TRANSMISSION_FALLOFF);
    ratio = ratio * (1.0 - OMNI_SCATTERING) + OMNI_SCATTERING;

    return light * ratio * (1.0 - BASE_TRANSMISSION);
}

// Raymarch through all the clouds
@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // let uv = in.uv;
    // let pix = in.position.xy;

    // // Load depth and unproject to clip space
    // let depth = textureSample(depth_texture, linear_sampler, uv);
    // let ndc = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);

    // // Reconstruct world‑space pos
    // let world_pos4 = view.world_from_clip * ndc;
    // let world_pos3 = world_pos4.xyz / world_pos4.w;

    // // Ray origin & dir
    // let ro = view.world_position; //vec3(view.longitude_meters, view.altitude, -view.latitude_meters);
    // let rd_vec = world_pos3 - ro;
    // let t_max = length(rd_vec);
    // let rd = rd_vec / t_max;

    // let dither = fract(blue_noise(pix));

    // var accumulation = vec4(0.0);
    // var t = dither * STEP_SIZE_INSIDE;
    // var steps_outside_cloud = 0;

    // // Get sun direction and intensity, mix between aur light (straight up) and sun
    // let sun_dot = globals.sun_direction.y;
    // let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    // let sun_dir = normalize(mix(AUR_DIR, globals.sun_direction, sun_t));

    // // Compute sky color for the current view direction
    // let sky_col = render_sky(rd, globals.sun_direction);
    // let sky_col_inv = render_sky(-rd, globals.sun_direction);

    // // Compute scattering angle (dot product between view direction and light direction)
    // let scattering_angle = dot(rd, sun_dir);

    // // Calculate cylinders to render for the poles, behind all clouds
    // let pole_color = render_poles(ro, rd, view.planet_rotation, globals.planet_radius);

    // for (var i = 0; i < MAX_STEPS; i += 1) {
    //     if t >= t_max || accumulation.a >= ALPHA_THRESHOLD || t >= FOG_END_DISTANCE {
    //         break;
    //     }

    //     let pos = ro + rd * t;
    //     let cloud_sample = sample_cloud(pos, t);
    //     let step_density = cloud_sample.density;

    //     // Scale step size based on distance from camera
    //     var step_scaler = 1.0;
    //     if t > STEP_DISTANCE_SCALING_START {
    //         step_scaler = 1.0 + (t - STEP_DISTANCE_SCALING_START) * STEP_DISTANCE_SCALING_FACTOR;
    //     }
    //     // Reduce scaling when close to surfaces
    //     let close_threshold = STEP_SIZE_OUTSIDE * step_scaler;
    //     let distance_left = t_max - t;
    //     if distance_left < close_threshold {
    //         let norm = clamp(distance_left / close_threshold, 0.0, 1.0);
    //         step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
    //     }

    //     // Adjust t to effectively "backtrack" and take smaller steps when entering a cloud
    //     if step_density > 0.0 {
    //         if steps_outside_cloud != 0 {
    //             // First step into the cloud;
    //             steps_outside_cloud = 0;
    //             t = max(t + (-STEP_SIZE_OUTSIDE + STEP_SIZE_INSIDE) * step_scaler, 0.0);
    //             continue;
    //         }
    //     } else {
    //         steps_outside_cloud += 1;
    //     }

    //     var step = STEP_SIZE_OUTSIDE * step_scaler;
    //     if step_density > 0.0 {
    //         step = STEP_SIZE_INSIDE * step_scaler;

    //         let step_color = cloud_sample.color;

    //         // Lightmarching for self-shadowing
    //         var density_sunwards = max(step_density, 0.0);
    //         var lightmarch_distance = 0.0;
    //         for (var j: u32 = 1; j <= LIGHT_STEPS; j++) {
    //             var light_step_size = LIGHT_STEP_SIZE[j];
    //             lightmarch_distance += light_step_size;

    //             let light_offset = pos + sun_dir * lightmarch_distance;
    //             density_sunwards += sample_cloud(light_offset, t).density * lightmarch_distance;
    //             if density_sunwards >= 0.95 {
    //                 break;
    //             }
    //         }

    //         // Calcuate self shadowing
    //         let tau = clamp(density_sunwards, 0.0, 1.0) * SHADOW_EXTINCTION;

    //         // Height factor for reducing aur light
    //         let height_factor = max(smoothstep(15000.0, 1000.0, pos.y), 0.15);

    //         // Blend between aur and sun, modulated by sky color
    //         let sun_color = SUN_COLOR * globals.sun_intensity * 10.0;

    //         // Blend between sky based ambient and aur based on sun height, always using a little aur light
    //         let ambient_color = (AMBIENT_AUR_COLOR * height_factor * 50.0) + sky_col_inv;

    //         // Apply transmission to sun and ambient light
    //         let transmitted_sun = transmission(sun_color, tau);
    //         let transmitted_ambient = transmission(ambient_color, tau * 0.5); // Ambient attenuated less

    //         // Apply light scattering to sun light based on angle
    //         let scattered_sun = light_scattering(sun_color, scattering_angle);

    //         // Combine transmitted and scattered light, weighted by density
    //         var lit_color = ((transmitted_sun + scattered_sun) * step_density + transmitted_ambient) * step_color + cloud_sample.emission;

    //         let step_alpha = clamp(step_density * 0.4 * step, 0.0, 1.0);
    //         accumulation += vec4(lit_color * step_alpha * (1.0 - accumulation.a), step_alpha * (1.0 - accumulation.a));
    //     }

    //     t += step;
    // }

    // accumulation.a = min(accumulation.a * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    // if depth <= 0.00001 {
    //     // Blend in poles behind clouds
    //     accumulation = vec4(
    //         accumulation.rgb + pole_color.rgb * pole_color.a * (1.0 - accumulation.a),
    //         accumulation.a + pole_color.a * (1.0 - accumulation.a)
    //     );

    //     // Add our sky in the background
    //     accumulation = vec4(
    //         accumulation.rgb + sky_col * (1.0 - accumulation.a),
    //         1.0
    //     );
    // }

    // return clamp(accumulation, vec4(0.0), vec4(1.0));

    let uv = in.uv - globals.time * 0.02;

    let texture = textureSample(perlinworley_texture, perlinworley_sampler, vec3<f32>(uv.x, uv.y, 0.0));

    var col = vec3(0.0);
    if in.uv.x < 0.25 {
        col = vec3(texture.r);
    } else if in.uv.x < 0.5 {
        col = vec3(texture.g);
    } else if in.uv.x < 0.75 {
        col = vec3(texture.b);
    } else {
        col = vec3(texture.a);
    }

    return vec4(col, 1.0);
}
