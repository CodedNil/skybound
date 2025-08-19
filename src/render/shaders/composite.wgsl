#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1; // Default blend rate to use when no confidence in history
const MIN_HISTORY_BLEND_RATE: f32 = 0.015; // Minimum blend rate allowed, to ensure at least some of the current sample is used

@group(0) @binding(0) var view_target: texture_2d<f32>; // low-res cloud color
@group(0) @binding(1) var history: texture_2d<f32>; // full-res history (RGBA: rgb=color, a=confidence)
@group(0) @binding(2) var motion_vectors: texture_2d<f32>; // low-res motion vectors (uv_delta in normalized UV space)
@group(0) @binding(3) var depth: texture_2d<f32>; // low-res depth (normalized 0..1)
@group(0) @binding(4) var nearest_sampler: sampler;
@group(0) @binding(5) var linear_sampler: sampler;

struct Output {
    @location(0) view_target: vec4<f32>,
    @location(1) history: vec4<f32>,
}

// ----- Utility helpers -----
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(x: vec3<f32>) -> f32 { return max(x.r, max(x.g, x.b)); }

// The following 3 functions are from Playdead (MIT-licensed)
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return saturate(vec3(r, g, b));
}

fn clip_towards_aabb_center(history_color: vec3<f32>, current_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;
    let v_clip = history_color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max3(a_unit);
    if ma_unit > 1.0 {
        return p_clip + (v_clip / ma_unit);
    } else {
        return history_color;
    }
}

// ----- Sampling helpers -----
fn sample_history(u: f32, v: f32) -> vec3<f32> {
    return textureSample(history, linear_sampler, vec2(u, v)).rgb;
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    return textureSample(depth, nearest_sampler, uv).r;
}

fn sample_view_target(uv: vec2<f32>) -> vec3<f32> {
    // view_target is low-res; return YCoCg of the sampled color
    let sample = textureSample(view_target, nearest_sampler, uv).rgb;
    return RGB_to_YCoCg(sample);
}

// ----- Inline bilateral upsample (3x3) -----
// - Samples the low-res color and depth to produce an edge-aware upsampled color for this full-res pixel.
// - Uses low-res texel offsets (1 / low_res_size) sampled around the normalized uv.
fn bilateral_upsample(uv: vec2<f32>, low_res_size: vec2<f32>) -> vec3<f32> {
    let low_texel = 1.0 / low_res_size;

    // center sample values
    let center_depth = textureSample(depth, nearest_sampler, uv).r;
    let center_color = textureSample(view_target, linear_sampler, uv).rgb;

    var color_acc: vec3<f32> = vec3(0.0);
    var weight_acc: f32 = 0.0;

    // kernel radius 1 (3x3)
    // spatial sigma chosen heuristically; tune to taste: larger sigma => smoother result
    let spatial_sigma: f32 = 1.0; // in texel units
    let spatial_coeff = -0.5 / (spatial_sigma * spatial_sigma);

    // depth weight scale controls edge preservation: higher => stronger edge guarding
    // depth values are normalized 0..1; tune this to your scene's depth distribution
    let depth_sigma: f32 = 0.02; // in normalized depth units (tune)
    let depth_coeff = -0.5 / (depth_sigma * depth_sigma);

    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * low_texel;
            let sample_uv = uv + offset;
            let sample_color = textureSample(view_target, linear_sampler, sample_uv).rgb;
            let sample_depth = textureSample(depth, nearest_sampler, sample_uv).r;

            // spatial weight (Gaussian in texel space)
            let dist2 = dot(vec2<f32>(f32(x), f32(y)), vec2<f32>(f32(x), f32(y)));
            let w_spatial = exp(dist2 * spatial_coeff);

            // depth weight (edge-aware)
            let depth_diff = sample_depth - center_depth;
            let w_depth = exp(depth_diff * depth_diff * depth_coeff);

            let w = w_spatial * w_depth;
            color_acc = color_acc + sample_color * w;
            weight_acc = weight_acc + w;
        }
    }

    // fallback to center if numerically unstable
    if weight_acc <= 1e-6 {
        return center_color;
    }
    return color_acc / weight_acc;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> Output {
    let uv = in.uv;

    // low-resolution input texture size (cloud render targets)
    let low_res_size = vec2<f32>(textureDimensions(view_target));
    // low-res texel size in normalized UV space
    let low_texel_size = 1.0 / low_res_size;

    // Estimate full-res texel size using derivatives of UV (dpdx/dpdy)
    // dpdx/dpdy produce change in UV per pixel; invert to get approximate full-res size in pixels.
    var full_res_texel_uv = abs(dpdx(uv));
    let fy = abs(dpdy(uv));
    full_res_texel_uv = vec2<f32>(max(full_res_texel_uv.x, fy.x), max(full_res_texel_uv.y, fy.y));
    full_res_texel_uv = max(full_res_texel_uv, vec2<f32>(1e-6, 1e-6));
    let full_res_size = vec2<f32>(1.0) / full_res_texel_uv;

    // ----- Spatial upsample (inline) -----
    // Produce an upsampled current color from the low-res cloud texture using edge-aware bilateral filtering
    var upsampled_color = bilateral_upsample(uv, low_res_size);
    var current_color = upsampled_color;

    // Pick the closest motion_vector from 5 samples (reduces aliasing on the edges of moving entities)
    // Use low-res texel offsets for sampling motion/depth (they are authored at low-res)
    let offset = low_texel_size * 2.0;
    let d_uv_tl = uv + vec2(-offset.x, offset.y);
    let d_uv_tr = uv + vec2(offset.x, offset.y);
    let d_uv_bl = uv + vec2(-offset.x, -offset.y);
    let d_uv_br = uv + vec2(offset.x, -offset.y);
    var closest_uv = uv;
    let d_tl = sample_depth(d_uv_tl);
    let d_tr = sample_depth(d_uv_tr);
    var closest_depth = sample_depth(uv);
    let d_bl = sample_depth(d_uv_bl);
    let d_br = sample_depth(d_uv_br);
    if d_tl > closest_depth {
        closest_uv = d_uv_tl;
        closest_depth = d_tl;
    }
    if d_tr > closest_depth {
        closest_uv = d_uv_tr;
        closest_depth = d_tr;
    }
    if d_bl > closest_depth {
        closest_uv = d_uv_bl;
        closest_depth = d_bl;
    }
    if d_br > closest_depth {
        closest_uv = d_uv_br;
    }

    // Sample motion vector at the chosen location (low-res texture)
    // The motion vectors generated by the raymarch compute shader are uv - uv_prev
    // (normalized UV deltas), so they can be used directly in normalized UV space for reprojection.
    var closest_motion_vector = textureSample(motion_vectors, nearest_sampler, closest_uv).rg;

    // Reproject to find the equivalent sample from the past (history is full-res)
    // Uses 5-sample Catmull-Rom filtering (reduces blurriness)
    let history_uv = uv - closest_motion_vector;
    let sample_position = history_uv * full_res_size;
    let texel_center = floor(sample_position - 0.5) + 0.5;
    let f = sample_position - texel_center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let texel_position_0 = (texel_center - 1.0) * (1.0 / full_res_size);
    let texel_position_3 = (texel_center + 2.0) * (1.0 / full_res_size);
    let texel_position_12 = (texel_center + (w2 / w12)) * (1.0 / full_res_size);

    var history_color = sample_history(texel_position_12.x, texel_position_0.y) * w12.x * w0.y;
    history_color += sample_history(texel_position_0.x, texel_position_12.y) * w0.x * w12.y;
    history_color += sample_history(texel_position_12.x, texel_position_12.y) * w12.x * w12.y;
    history_color += sample_history(texel_position_3.x, texel_position_12.y) * w3.x * w12.y;
    history_color += sample_history(texel_position_12.x, texel_position_3.y) * w12.x * w3.y;

    // Constrain past sample with 3x3 YCoCg variance clipping (reduces ghosting)
    let s_tl = sample_view_target(uv + vec2(-low_texel_size.x,  low_texel_size.y));
    let s_tm = sample_view_target(uv + vec2( 0.0,               low_texel_size.y));
    let s_tr = sample_view_target(uv + vec2( low_texel_size.x,  low_texel_size.y));
    let s_ml = sample_view_target(uv + vec2(-low_texel_size.x,  0.0));
    let s_mm = RGB_to_YCoCg(current_color);
    let s_mr = sample_view_target(uv + vec2( low_texel_size.x,  0.0));
    let s_bl = sample_view_target(uv + vec2(-low_texel_size.x, -low_texel_size.y));
    let s_bm = sample_view_target(uv + vec2( 0.0,              -low_texel_size.y));
    let s_br = sample_view_target(uv + vec2( low_texel_size.x, -low_texel_size.y));
    let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    let moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) + (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) + (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
    let mean = moment_1 / 9.0;
    let variance = (moment_2 / 9.0) - (mean * mean);
    let std_deviation = sqrt(max(variance, vec3(0.0)));
    history_color = RGB_to_YCoCg(history_color);
    history_color = clip_towards_aabb_center(history_color, s_mm, mean - std_deviation, mean + std_deviation);
    history_color = YCoCg_to_RGB(history_color);

    // How confident we are that the history is representative of the current frame
    var history_confidence = textureSample(history, nearest_sampler, uv).a;
    // pixel_motion_vector in full-res pixel units
    let pixel_motion_vector = abs(closest_motion_vector) * full_res_size;
    if pixel_motion_vector.x < 0.01 && pixel_motion_vector.y < 0.01 {
        // Increment when pixels are not moving
        history_confidence += 10.0;
    } else {
        // Else reset
        history_confidence = 1.0;
    }

    // Blend current and past sample
    // Use more of the history if we're confident in it (reduces noise when there is no motion)
    var current_color_factor = clamp(1.0 / history_confidence, MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE);

    // Reject history when motion vectors point off screen (disocclusion)
    if any(saturate(history_uv) != history_uv) {
        current_color_factor = 1.0;
        history_confidence = 1.0;
    }

    current_color = mix(history_color, current_color, current_color_factor);

    // Write output to history and view target
    var out: Output;
    out.history = vec4(current_color, history_confidence);

    // Preserve alpha from original low-res sample if present
    let original_alpha = textureSample(view_target, nearest_sampler, uv).a;
    out.view_target = vec4(current_color, original_alpha);
    return out;
}
