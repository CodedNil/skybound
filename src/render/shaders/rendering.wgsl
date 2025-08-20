#import skybound::utils::{View, AtmosphereData, blue_noise, get_sun_position}
#import skybound::volumetrics::raymarch_volumetrics
#import skybound::sky::{render_sky, get_sun_light_color}
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var linear_sampler: sampler;

@group(0) @binding(2) var base_texture: texture_3d<f32>;
@group(0) @binding(3) var details_texture: texture_3d<f32>;
@group(0) @binding(4) var weather_texture: texture_2d<f32>;

@group(0) @binding(5) var output_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var output_motion: texture_storage_2d<rg16float, write>;
@group(0) @binding(7) var output_depth: texture_storage_2d<r32float, write>;

const INV_4_PI: f32 = 0.07957747154;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return INV_4_PI * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let texture_size = textureDimensions(output_color);

    // Boundary check: Stop execution if the thread is outside the texture's dimensions
    if (id.x >= texture_size.x || id.y >= texture_size.y) {
        return;
    }

    // Calculate the pixel coordinate and UV for the current thread
    let pix = vec2<f32>(id.xy);
    let uv = (pix + 0.5) / vec2<f32>(texture_size);
    let dither = fract(blue_noise(pix));

    // Reconstruct world-space position for the ray
    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(vec3(ndc, 0.01), view.world_from_clip);

    // Ray origin & dir
    let ro = view.world_position;
    let rd = normalize(world_pos_far - ro);
    let t_max = 10000000.0;
    let sun_pos = get_sun_position(view);
    let sun_dir = normalize(sun_pos - ro);

    // Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.9) * 0.01;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = max(hg_forward, max(hg_silver, hg_back));

	// Precalculate sun, sky and ambient colors
    var atmosphere: AtmosphereData;
    atmosphere.sun_pos = sun_pos;
    atmosphere.sky = render_sky(rd, view, sun_dir);
    atmosphere.sun = get_sun_light_color(ro, view, sun_dir) * 0.5 * phase;
    atmosphere.ambient = render_sky(normalize(vec3<f32>(1.0, 0.0, 1.0)), view, sun_dir);

    // Sample the volumes
    let raymarch_result = raymarch_volumetrics(ro, rd, atmosphere, view, t_max, dither, view.time, linear_sampler);
    let volumetrics_depth: f32 = raymarch_result.depth;
    var acc_color: vec3<f32> = raymarch_result.color;

    // Motion Vectors
    var motion_vector = vec2(0.0);
    if volumetrics_depth < t_max {
        // Find the world position of the point we rendered, project it into the previous frames screen space
        let world_pos_current = ro + rd * volumetrics_depth;

        // Project to previous frame's clip space
        let clip_pos_prev = view.prev_clip_from_world * vec4<f32>(world_pos_current, 1.0);
        // Perform perspective divide to get Normalized Device Coordinates (NDC)
        let ndc_prev = clip_pos_prev.xyz / clip_pos_prev.w;
        // Convert NDC [-1, 1] to UV [0, 1]
        let uv_prev = ndc_prev.xy * vec2<f32>(0.5, -0.5) + 0.5;

        motion_vector = uv - uv_prev;
    }

    // Normalize depth to a [0, 1] range.
    var final_volumetric_depth: f32 = select(0.0, volumetrics_depth / t_max, volumetrics_depth > 0.0);

    // Write the final results to the output storage textures
    textureStore(output_color, id.xy, clamp(vec4<f32>(acc_color, 1.0), vec4(0.0), vec4(1.0)));
    textureStore(output_motion, id.xy, vec4<f32>(motion_vector, 0.0, 0.0));
    textureStore(output_depth, id.xy, vec4<f32>(final_volumetric_depth, 0.0, 0.0, 0.0));
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_clip * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}
