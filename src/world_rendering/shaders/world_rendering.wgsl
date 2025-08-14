#import skybound::utils::{View, AtmosphereData, blue_noise}
#import skybound::raymarch::raymarch
#import skybound::sky::render_sky
#import skybound::poles::render_poles

struct FullscreenVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) motion_vector: vec2<f32>,
    @location(2) volumetric_depth: f32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var linear_sampler: sampler;

const INV_4_PI: f32 = 0.07957747154;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return INV_4_PI * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;
    let pix = in.position.xy;
    let dither = fract(blue_noise(pix));

    // Reconstruct world‑space pos
    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(vec3(ndc, 0.01), view.world_from_clip);

    // Ray origin & dir
    let ro = view.world_position;
    let rd = normalize(world_pos_far - ro);
    let t_max = 10000000.0;

	// Precalculate sun, sky and ambient colors
    var atmosphere: AtmosphereData;
    atmosphere.sky = render_sky(rd, view.sun_direction, ro.z);
    atmosphere.sun = render_sky(view.sun_direction, view.sun_direction, ro.z) * 0.5;
    atmosphere.ambient = render_sky(normalize(vec3<f32>(1.0, 0.0, 1.0)), view.sun_direction, ro.z);

	// Phase functions for silver and back scattering
    let cos_theta = dot(view.sun_direction, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.9) * 0.01;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    atmosphere.phase = max(hg_forward, max(hg_silver, hg_back));

    // Sample the volumes
    let raymarch_result = raymarch(ro, rd, atmosphere, view, t_max, dither, view.time, linear_sampler);
    let volumetrics_depth: f32 = raymarch_result.depth;
    var acc_color: vec3<f32> = raymarch_result.color.rgb;
    var acc_alpha: f32 = raymarch_result.color.a;

    // Blend the volumetrics with the sky color behind them
    acc_color = acc_color + atmosphere.sky * (1.0 - acc_alpha);

    // Calculate motion vectors using volumetric depth for better accuracy
    var motion_vector = vec2(0.0);

    if volumetrics_depth > 0.0 {
        // Use the actual volumetric sample position instead of geometry depth
        let volumetric_world_pos = ro + rd * volumetrics_depth;

        // Current screen position using volumetric depth
        let volumetric_clip_pos = view.clip_from_world * vec4(volumetric_world_pos, 1.0);
        let volumetric_ndc = volumetric_clip_pos.xyz / volumetric_clip_pos.w;
        let current_uv = volumetric_ndc.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);

        // Transform volumetric world position to previous frame's clip space
        let prev_clip_pos = view.prev_clip_from_world * vec4(volumetric_world_pos, 1.0);
        let prev_ndc = prev_clip_pos.xyz / prev_clip_pos.w;
        let prev_uv = prev_ndc.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);

        // Motion vector is the difference between current and previous UV positions
        motion_vector = current_uv - prev_uv;
    }

    var output: FragmentOutput;
    output.color = clamp(vec4(acc_color, 1.0), vec4(0.0), vec4(1.0));
    output.motion_vector = motion_vector;

    // Output volumetric depth, or far plane if no volumetric contribution
    var final_volumetric_depth: f32 = 0.0;
    if volumetrics_depth > 0.0 {
        final_volumetric_depth = volumetrics_depth / t_max;
    }
    output.volumetric_depth = final_volumetric_depth;

    return output;
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
