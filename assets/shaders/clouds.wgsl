#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> clouds: VolumetricClouds;
@group(0) @binding(1) var<uniform> view: View;
struct VolumetricClouds {
    time: f32,
}
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;


const SUNDIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const MAX_DISTANCE: f32 = 2000.0;


// Simple noise function for white noise
fn hash1(p: f32) -> f32 {
    var p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}
fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash3i(p: vec3<i32>) -> f32 {
    var n: i32 = p.x * 3 + p.y * 113 + p.z * 311;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * f32(n & 0x0fffffff) / f32(0x0fffffff);
}

// Procedural blue noise approximation
fn blue_noise(uv: vec2<f32>) -> f32 {
    let v = hash2(uv + vec2(-1, 0)) + hash2(uv + vec2(1, 0)) + hash2(uv + vec2(0, 1)) + hash2(uv + vec2(0, -1));
    return hash2(uv) - v / 4.0 + 0.5;
}

// Simple 3D noise function
fn noise3(x: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(x));
    let w: vec3<f32> = fract(x);

    // Smoothstep weights
    let u: vec3<f32> = w * w * (3.0 - 2.0 * w);

    // Hash values at cube corners, interpolate along x
    let lerp_x0 = mix(hash3i(i + vec3<i32>(0, 0, 0)), hash3i(i + vec3<i32>(1, 0, 0)), u.x);
    let lerp_x1 = mix(hash3i(i + vec3<i32>(0, 1, 0)), hash3i(i + vec3<i32>(1, 1, 0)), u.x);
    let lerp_x2 = mix(hash3i(i + vec3<i32>(0, 0, 1)), hash3i(i + vec3<i32>(1, 0, 1)), u.x);
    let lerp_x3 = mix(hash3i(i + vec3<i32>(0, 1, 1)), hash3i(i + vec3<i32>(1, 1, 1)), u.x);

    // Interpolate along y
    let lerp_y0 = mix(lerp_x0, lerp_x1, u.y);
    let lerp_y1 = mix(lerp_x2, lerp_x3, u.y);

    // Interpolate along z and return
    return mix(lerp_y0, lerp_y1, u.z);
}

// FBM
const m3: mat3x3f = mat3x3f(
    vec3(0.8, 0.6, 0.0),
    vec3(-0.6, 0.8, 0.0),
    vec3(0.0, 0.0, 1.0)
) * 2.0;

fn fbm(p_original: vec3<f32>) -> f32 {
    var p = p_original;
    var f: f32 = 0.0;

    f = f + 0.5000 * noise3(p);
    p = m3 * p;

    f = f + 0.2500 * noise3(p);
    p = m3 * p;

    f = f + 0.1250 * noise3(p);
    p = m3 * p;

    f = f + 0.0625 * noise3(p);
    p = m3 * p;

    f = f + 0.03125 * noise3(p);

    return f / 0.96875;
}

// Density map
fn density(pos: vec3<f32>) -> f32 {
    var density = 0.0;

    if pos.y < 10.0 {
        // Thick fog
        let fbm_value = fbm(pos - vec3(0.0, 0.1, 1.0) * clouds.time);
        density = clamp(1.5 - pos.y - 2.0 + 1.75 * fbm_value, 0.0, 1.0);
    } else if pos.y < 60.0 {
        // Normal clouds
        let cloud_scale = 30.0;
        let fbm_value = fbm(pos / cloud_scale - vec3(0.0, 0.1, 1.0) * clouds.time / cloud_scale);
        density = smoothstep(0.2, 0.4, fbm_value);
    } else {
        density = 0.0; // Clear sky
    }

    return density;
}

// Raymarch function
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, t_scene: f32, offset: f32) -> vec4<f32> {
    var sumCol = vec4(0.0);
    var t = offset; // Dithering offset
    let min_step = 0.2;
    let k = 0.01; // The fall-off of step size with distance
    let epsilon = 0.5; // How far to move through less dense areas

    for (var i = 0; i < 500; i = i + 1) {
        if sumCol.a > 0.99 || t > min(t_scene, MAX_DISTANCE) { break; }

        let pos = ro + rd * t;
        let den = density(pos);

        if pos.y > 100.0 && rd.y > 0.0 { break; }

        if den > 0.01 {
            // Simple self‐shadowing
            let dif = clamp((den - density(pos + SUNDIR * 0.3)) / 0.6, 0.0, 1.0);
            let lin = vec3(1.0, 0.6, 0.3) * dif + vec3(0.91, 0.98, 1.05);

            // Base cloud color lerp
            var col = mix(vec3(1.0, 0.95, 0.8), vec3(0.25, 0.30, 0.35), den);

            // Apply lighting & background dusting
            col = col * lin;

            // Accumulate in alpha‐weighted front‐to‐back
            let a = den * 0.4;
            col *= a;
            sumCol = vec4(
                sumCol.xyz + col * (1.0 - sumCol.a),
                sumCol.a + a * (1.0 - sumCol.a)
            );
        }

        let step_size = max(min_step, k * t) / (den + epsilon);
        t += step_size;
    }

    return clamp(sumCol, vec4(0.0), vec4(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // G-buffer loads
    let screen_color: vec3f = textureLoad(screen_texture, vec2<i32>(in.position.xy), 0).xyz;
    let depth: f32 = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);

    // Unproject to world
    let ndc = vec3(in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
    let world_pos_raw = view.world_from_clip * vec4(ndc, 1.0);
    let world_pos = world_pos_raw.xyz / world_pos_raw.w;

    // Form the ray
    let origin = view.world_position;
    let ray_vec = world_pos - origin;
    let t_scene = length(ray_vec);
    let ray_dir = ray_vec / t_scene;

    // Procedural blue noise dithering
    let offset = fract(blue_noise(in.position.xy));

    // Ray-march clouds
    let clouds = raymarch(origin, ray_dir, t_scene, offset);

    // Composite over background
    let col = mix(screen_color, clouds.xyz, clouds.a);

    return vec4(col, 1.0);
}
