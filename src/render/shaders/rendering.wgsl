#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::utils::{View, AtmosphereData, blue_noise, get_sun_position}
#import skybound::volumetrics::raymarch_volumetrics
#import skybound::raymarch::raymarch_solids
#import skybound::sky::{render_sky, get_sun_light_color}
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var linear_sampler: sampler;

const INV_4_PI: f32 = 0.07957747154;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return INV_4_PI * (1.0 - g2) / pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @builtin(frag_depth) frag_depth: f32,
};

@fragment
fn main(in: FullscreenVertexOutput) -> FragmentOutput {
    let uv = in.uv;

    // Dither based on UV to approximate per-pixel blue noise
    let dither = fract(blue_noise(uv * 1024));

    // Reconstruct world-space position for a ray through the far plane
    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(vec3(ndc, 0.01), view.world_from_clip);

    // Ray origin & dir
    let ro = view.world_position;
    let rd = normalize(world_pos_far - ro);
    var t_max = 1000000.0;
    let sun_pos = get_sun_position(view);
    let sun_dir = normalize(sun_pos - ro);

    // Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.95) * 0.003;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = hg_forward + hg_back * 0.15 + hg_silver * 0.2;

	// Precalculate sun, sky and ambient colors
    var atmosphere: AtmosphereData;
    atmosphere.sun_pos = sun_pos;
    atmosphere.sky = render_sky(rd, view, sun_dir);
    atmosphere.sun = (get_sun_light_color(ro, view, sun_dir) * 0.45 + atmosphere.sky * 0.18) * phase;
    atmosphere.ambient = atmosphere.sky * 0.8 + render_sky(normalize(vec3<f32>(1.0, 0.0, 1.0)), view, sun_dir) * 0.2;

    // Run solids raymarch (solids are independent of volumetrics)
    let solids = raymarch_solids(ro, rd, view, t_max, view.time);
    var rendered_color = select(atmosphere.sky, solids.color, solids.depth < t_max);
    t_max = solids.depth;

    // Sample the volumetrics
    let volumetrics_result = raymarch_volumetrics(ro, rd, atmosphere, view, t_max, dither, view.time, linear_sampler);
    rendered_color = volumetrics_result.color.rgb + rendered_color * volumetrics_result.color.a;

    // Motion Vectors
    var motion_vector = vec2(0.0);
    let depth: f32 = min(solids.depth, volumetrics_result.depth);
    if depth < t_max {
        // Find the world position of the point we rendered, project it into the previous frames screen space
        let world_pos_current = ro + rd * depth;

        // Project to previous frame's clip space
        let clip_pos_prev = view.prev_clip_from_world * vec4<f32>(world_pos_current, 1.0);
        // Perform perspective divide to get Normalized Device Coordinates (NDC)
        let ndc_prev = clip_pos_prev.xyz / clip_pos_prev.w;
        // Convert NDC [-1, 1] to UV [0, 1]
        let uv_prev = ndc_prev.xy * vec2<f32>(0.5, -0.5) + 0.5;

        motion_vector = uv - uv_prev;
    }

    // Normalize depth to a [0, 1] range.
    var final_volumetric_depth: f32 = select(0.0, depth / t_max, depth > 0.0);

    var out: FragmentOutput;
    out.color = clamp(vec4<f32>(rendered_color, 1.0), vec4(0.0), vec4(1.0));
    out.motion = vec4<f32>(motion_vector, 0.0, 0.0);
    out.frag_depth = final_volumetric_depth;
    return out;
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
