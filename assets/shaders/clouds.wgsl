#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> clouds: VolumetricClouds;
@group(0) @binding(1) var<uniform> view: View;
struct VolumetricClouds {
    time: f32,
}
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var texture_sampler: sampler;


const SUNDIR: vec3<f32> = vec3<f32>(0.577350269, 0.0, -0.577350269);

// Simple noise function
fn rand11(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453123);
}
fn noise3(p: vec3f) -> f32 {
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
    vec3f(0.8, 0.6, 0.0),
    vec3f(-0.6, 0.8, 0.0),
    vec3f(0.0, 0.0, 1.0)
);
fn fbm(p_original: vec3f) -> f32 {
    var p = p_original;
    var f: f32 = 0.0;

    f = f + 0.5000 * noise3(p);
    p = m3 * p * 2.02;

    f = f + 0.2500 * noise3(p);
    p = m3 * p * 2.03;

    f = f + 0.1250 * noise3(p);
    p = m3 * p * 2.01;

    f = f + 0.0625 * noise3(p);

    return f / 0.9375;
}

// Density map
fn density(pos: vec3<f32>) -> f32 {
    // advect clouds with time
    let q = pos - vec3<f32>(0.0, 0.1, 1.0) * clouds.time;
    let d = clamp(1.5 - pos.y - 2.0 + 1.75 * fbm(q), 0.0, 1.0);
    return d;
}

// Raymarch function
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, bg: vec3<f32>, px: vec2<i32>) -> vec4<f32> {
    var sumCol = vec4<f32>(0.0);
    // random jitter per pixel
    var t = 0.05 * fract(sin(dot(vec2<f32>(f32(px.x & 1023), f32(px.y & 1023)), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    for (var i = 0; i < 240; i = i + 1) {
        if sumCol.a > 0.99 { break; }
        let pos = ro + rd * t;
        if pos.y < -3.0 || pos.y > 2.0 {
            t += max(0.06, 0.05 * t);
            continue;
        }

        let den = density(pos);
        if den > 0.01 {
            // simple self‐shadowing
            let dif = clamp((den - density(pos + SUNDIR * 0.3)) / 0.6, 0.0, 1.0);
            let lin = vec3<f32>(1.0, 0.6, 0.3) * dif + vec3<f32>(0.91, 0.98, 1.05);

            // base cloud color lerp
            var col = mix(vec3<f32>(1.0, 0.95, 0.8), vec3<f32>(0.25, 0.30, 0.35), den);

            // apply lighting & background dusting
            col = col * lin;
            col = mix(col, bg, exp(-0.003 * t));

            // accumulate in alpha‐weighted front‐to‐back
            let a = den * 0.4;        // per‐step opacity
            col *= a;
            sumCol = vec4<f32>(
                sumCol.xyz + col * (1.0 - sumCol.a),
                sumCol.a + a * (1.0 - sumCol.a)
            );
        }
        t += max(0.06, 0.05 * t);
    }

    return clamp(sumCol, vec4<f32>(0.0), vec4<f32>(1.0));
}

// Render function
fn render(ro: vec3<f32>, rd: vec3<f32>, bg: vec3<f32>, px: vec2<i32>) -> vec4<f32> {
    // Clouds
    let clouds: vec4<f32> = raymarch(ro, rd, bg, px);

    // Composite over background
    let col = bg * (1.0 - clouds.a) + clouds.xyz;

    return vec4<f32>(col, 1.0);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = view.world_position;

    let background_color = textureSample(screen_texture, texture_sampler, in.uv).xyz;

    // Convert UV to NDC: [0, 1] -> [-1, 1]
    let ndc = in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    // Transform UV to ray direction
    let view_position_homogeneous = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let view_ray_direction = view_position_homogeneous.xyz / view_position_homogeneous.w;
    let ray_direction = normalize((view.world_from_view * vec4(view_ray_direction, 0.0)).xyz);

    let resolution: vec2<f32> = view.viewport.zw;
    let px: vec2<i32> = vec2<i32>(in.uv * resolution - 0.5);

    // Perform raymarching
    return render(ray_origin, ray_direction, background_color, px);
}
