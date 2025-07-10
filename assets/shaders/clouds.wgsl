#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> clouds: VolumetricClouds;
@group(0) @binding(1) var<uniform> view: View;
struct VolumetricClouds {
    time: f32,
}
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;


const SUNDIR: vec3<f32> = vec3(0.577350269, 0.0, -0.577350269);

// Simple noise function
fn rand11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}
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
);
const m3_1 = m3 * 2.02;
const m3_2 = m3 * 2.03;
const m3_3 = m3 * 2.01;
fn fbm(p_original: vec3<f32>) -> f32 {
    var p = p_original;
    var f: f32 = 0.0;

    f = f + 0.5000 * noise3(p);
    p = m3_1 * p;

    f = f + 0.2500 * noise3(p);
    p = m3_2 * p;

    f = f + 0.1250 * noise3(p);
    p = m3_3 * p;

    f = f + 0.0625 * noise3(p);

    return f / 0.9375;
}

// Density map
fn density(pos: vec3<f32>) -> f32 {
    // advect clouds with time
    let q = pos - vec3(0.0, 0.1, 1.0) * clouds.time;
    let d = clamp(1.5 - pos.y - 2.0 + 1.75 * fbm(q), 0.0, 1.0);
    return d;
}

// Raymarch function
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, t_scene: f32) -> vec4<f32> {
    var sumCol = vec4(0.0);
    var t = 0.0;
    let step_size = 0.2;

    for (var i = 0; i < 120; i = i + 1) {
        if sumCol.a > 0.99 || t > t_scene { break; }
        let pos = ro + rd * t;

        let den = density(pos);
        if den > 0.01 {
            // simple self‐shadowing
            let dif = clamp((den - density(pos + SUNDIR * 0.3)) / 0.6, 0.0, 1.0);
            let lin = vec3(1.0, 0.6, 0.3) * dif + vec3(0.91, 0.98, 1.05);

            // base cloud color lerp
            var col = mix(vec3(1.0, 0.95, 0.8), vec3(0.25, 0.30, 0.35), den);

            // apply lighting & background dusting
            col = col * lin;

            // accumulate in alpha‐weighted front‐to‐back
            let a = den * 0.4;
            col *= a;
            sumCol = vec4(
                sumCol.xyz + col * (1.0 - sumCol.a),
                sumCol.a + a * (1.0 - sumCol.a)
            );
        }
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

    // Ray-march clouds
    let clouds = raymarch(origin, ray_dir, t_scene);

    // Composite over background
    let col = mix(screen_color, clouds.xyz, clouds.a);

    return vec4(col, 1.0);
}
