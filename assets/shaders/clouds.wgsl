#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> clouds: VolumetricClouds;
@group(0) @binding(1) var<uniform> view: View;
struct VolumetricClouds {
    time: f32,
}
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;

@group(0) @binding(4) var noise_texture: texture_3d<f32>;
@group(0) @binding(5) var noise_sampler: sampler;


const SUNDIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const MAX_DISTANCE: f32 = 400.0;


// Simple noise function for white noise
fn rand11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}

// Procedural blue noise approximation
fn blue_noise(uv: vec2<f32>) -> f32 {
    // Generate white noise
    let n = rand11(uv.x + uv.y * 57.0 + clouds.time);
    // Simple high-pass filter: subtract average of neighboring samples
    var sum = 0.0;
    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, -1.0),
        vec2<f32>(0.0, 1.0)
    );
    for (var i = 0; i < 4; i = i + 1) {
        let offset = offsets[i] / 1024.0; // Small offset for neighboring pixels
        sum += rand11((uv + offset).x + (uv + offset).y * 57.0 + clouds.time);
    }
    let avg = sum / 4.0;
    let high_pass = n - avg;
    // Re-normalize to [0,1]
    return clamp((high_pass + 1.0) / 2.0, 0.0, 1.0);
}

// Function to get noise value from a 3D texture
fn get_texture_noise(uvw: vec3<f32>) -> f32 {
    let sampled_value = textureSample(noise_texture, noise_sampler, uvw * 0.02).r;
    return sampled_value * 2.0 - 1.0; // Remap [0, 1] to [-1, 1]
}

// 2D noise function using the 3D noise texture at fixed y
fn fbm2d(p: vec2<f32>) -> f32 {
    var f = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    let scale = 0.001; // Adjust for desired spatial variation
    var p_scaled = p * scale;

    for (var i = 0; i < 5; i = i + 1) {
        let uvw = vec3(p_scaled.x * frequency, 0.0, p_scaled.y * frequency);
        f = f + amplitude * noise3(uvw);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return f;
}

// Simple 3D noise function
fn noise3(p: vec3<f32>) -> f32 {
    let p_floor = floor(p);
    let p_fract = fract(p);
    let f = p_fract * p_fract * (3.0 - 2.0 * p_fract);
    let n = p_floor.x + p_floor.y * 57.0 + p_floor.z * 113.0;
    return mix(
        mix(
            mix(rand11(n + 0.0), rand11(n + 1.0), f.x),
            mix(rand11(n + 57.0), rand11(n + 58.0), f.x),
            f.y
        ),
        mix(
            mix(rand11(n + 113.0), rand11(n + 114.0), f.x),
            mix(rand11(n + 170.0), rand11(n + 171.0), f.x),
            f.y
        ),
        f.z
    ) * 2.0 - 1.0;
}

// FBM
const m3: mat3x3f = mat3x3f(
    vec3(0.8, 0.6, 0.0),
    vec3(-0.6, 0.8, 0.0),
    vec3(0.0, 0.0, 1.0)
) * 2.0;

fn fbm_cloud(p: vec3<f32>, octaves: i32, lacunarity: f32, gain: f32) -> f32 {
    var f = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var p_scaled = p;

    for (var i = 0; i < octaves; i = i + 1) {
        f = f + amplitude * noise3(p_scaled * frequency);
        if i == octaves - 1 {
            break;
        }
        p_scaled = m3 * p_scaled;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    return f / (1.0 - pow(gain, f32(octaves))) * (1.0 - gain);
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
        let cloud_scale = 10.0;
        let fbm_value = fbm(pos / cloud_scale - vec3(0.0, 0.1, 1.0) * clouds.time / cloud_scale);
        density = smoothstep(0.1, 0.2, fbm_value);
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
