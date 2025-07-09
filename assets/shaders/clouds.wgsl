#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var<uniform> view: View;
struct VolumetricClouds {
    time: f32,
    intensity: f32,
}
@group(0) @binding(3) var<uniform> settings: VolumetricClouds;


const sundir: vec3<f32> = vec3<f32>(0.577350269, 0.0, -0.577350269);

// Simple hash function for noise
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// Simple 2D hash function for pseudo-random noise
fn hash2(p: vec2<i32>) -> f32 {
    let p_float: vec2<f32> = vec2<f32>(f32(p.x), f32(p.y));
    let n: f32 = dot(p_float, vec2<f32>(127.1, 311.7));
    return fract(sin(n) * 43758.5453123);
}

// Simple noise function
fn noise(p: vec3<f32>) -> f32 {
    let p_floor = floor(p);
    let p_fract = fract(p);
    let f = p_fract * p_fract * (3.0 - 2.0 * p_fract);
    let n = p_floor.x + p_floor.y * 57.0 + p_floor.z * 113.0;
    return mix(
        mix(
            mix(hash(n + 0.0), hash(n + 1.0), f.x),
            mix(hash(n + 57.0), hash(n + 58.0), f.x),
            f.y
        ),
        mix(
            mix(hash(n + 113.0), hash(n + 114.0), f.x),
            mix(hash(n + 170.0), hash(n + 171.0), f.x),
            f.y
        ),
        f.z
    ) * 2.0 - 1.0;
}

// Map functions
fn map5(p: vec3<f32>) -> f32 {
    let q = p - vec3<f32>(0.0, 0.1, 1.0) * settings.time;
    var f: f32 = 0.0;
    var scale: f32 = 1.0;
    var q_temp = q;

    f += 0.50000 * noise(q_temp);
    q_temp = q_temp * 2.02;
    f += 0.25000 * noise(q_temp);
    q_temp = q_temp * 2.03;
    f += 0.12500 * noise(q_temp);
    q_temp = q_temp * 2.01;
    f += 0.06250 * noise(q_temp);
    q_temp = q_temp * 2.02;
    f += 0.03125 * noise(q_temp);

    return clamp(1.5 - p.y - 2.0 + 1.75 * f, 0.0, 1.0);
}
fn map4(p: vec3<f32>) -> f32 {
    let q = p - vec3<f32>(0.0, 0.1, 1.0) * settings.time;
    var f: f32 = 0.0;
    var scale: f32 = 1.0;
    var q_temp = q;

    f += 0.50000 * noise(q_temp);
    q_temp = q_temp * 2.02;
    f += 0.25000 * noise(q_temp);
    q_temp = q_temp * 2.03;
    f += 0.12500 * noise(q_temp);
    q_temp = q_temp * 2.01;
    f += 0.06250 * noise(q_temp);

    return clamp(1.5 - p.y - 2.0 + 1.75 * f, 0.0, 1.0);
}
fn map3(p: vec3<f32>) -> f32 {
    let q = p - vec3<f32>(0.0, 0.1, 1.0) * settings.time;
    var f: f32 = 0.0;
    var scale: f32 = 1.0;
    var q_temp = q;

    f += 0.50000 * noise(q_temp);
    q_temp = q_temp * 2.02;
    f += 0.25000 * noise(q_temp);
    q_temp = q_temp * 2.03;
    f += 0.12500 * noise(q_temp);

    return clamp(1.5 - p.y - 2.0 + 1.75 * f, 0.0, 1.0);
}
fn map2(p: vec3<f32>) -> f32 {
    let q = p - vec3<f32>(0.0, 0.1, 1.0) * settings.time;
    var f: f32 = 0.0;
    var q_temp = q;

    f += 0.50000 * noise(q_temp);
    q_temp = q_temp * 2.02;
    f += 0.25000 * noise(q_temp);

    return clamp(1.5 - p.y - 2.0 + 1.75 * f, 0.0, 1.0);
}

// Raymarch function
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, bgcol: vec3<f32>, px: vec2<i32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    var t = 0.05 * hash2(vec2<i32>(px.x & 1023, px.y & 1023));
    let sundir = vec3<f32>(-0.7071, 0.0, -0.7071);

    for (var i = 0; i < 40; i++) {
        let pos = ro + t * rd;
        if pos.y < -3.0 || pos.y > 2.0 || sum.a > 0.99 { break; }
        let den = map5(pos);
        if den > 0.01 {
            let dif = clamp((den - map5(pos + 0.3 * sundir)) / 0.6, 0.0, 1.0);
            let lin = vec3<f32>(1.0, 0.6, 0.3) * dif + vec3<f32>(0.91, 0.98, 1.05);
            var col = vec4<f32>(
                mix(vec3<f32>(1.0, 0.95, 0.8), vec3<f32>(0.25, 0.3, 0.35), den),
                den
            );
            col = vec4<f32>(col.xyz * lin, col.w * 0.4);
            col = vec4<f32>(mix(col.xyz, bgcol, 1.0 - exp(-0.003 * t * t)), col.w);
            col = vec4<f32>(col.rgb * col.a, col.a);
            sum += col * (1.0 - sum.a);
        }
        t += max(0.06, 0.05 * t);
    }

    for (var i = 0; i < 40; i++) {
        let pos = ro + t * rd;
        if pos.y < -3.0 || pos.y > 2.0 || sum.a > 0.99 { break; }
        let den = map4(pos);
        if den > 0.01 {
            let dif = clamp((den - map4(pos + 0.3 * sundir)) / 0.6, 0.0, 1.0);
            let lin = vec3<f32>(1.0, 0.6, 0.3) * dif + vec3<f32>(0.91, 0.98, 1.05);
            var col = vec4<f32>(
                mix(vec3<f32>(1.0, 0.95, 0.8), vec3<f32>(0.25, 0.3, 0.35), den),
                den
            );
            col = vec4<f32>(col.xyz * lin, col.w * 0.4);
            col = vec4<f32>(mix(col.xyz, bgcol, 1.0 - exp(-0.003 * t * t)), col.w);
            col = vec4<f32>(col.rgb * col.a, col.a);
            sum += col * (1.0 - sum.a);
        }
        t += max(0.06, 0.05 * t);
    }

    for (var i = 0; i < 30; i++) {
        let pos = ro + t * rd;
        if pos.y < -3.0 || pos.y > 2.0 || sum.a > 0.99 { break; }
        let den = map3(pos);
        if den > 0.01 {
            let dif = clamp((den - map3(pos + 0.3 * sundir)) / 0.6, 0.0, 1.0);
            let lin = vec3<f32>(1.0, 0.6, 0.3) * dif + vec3<f32>(0.91, 0.98, 1.05);
            var col = vec4<f32>(
                mix(vec3<f32>(1.0, 0.95, 0.8), vec3<f32>(0.25, 0.3, 0.35), den),
                den
            );
            col = vec4<f32>(col.xyz * lin, col.w * 0.4);
            col = vec4<f32>(mix(col.xyz, bgcol, 1.0 - exp(-0.003 * t * t)), col.w);
            col = vec4<f32>(col.rgb * col.a, col.a);
            sum += col * (1.0 - sum.a);
        }
        t += max(0.06, 0.05 * t);
    }

    for (var i = 0; i < 30; i++) {
        let pos = ro + t * rd;
        if pos.y < -3.0 || pos.y > 2.0 || sum.a > 0.99 { break; }
        let den = map2(pos);
        if den > 0.01 {
            let dif = clamp((den - map2(pos + 0.3 * sundir)) / 0.6, 0.0, 1.0);
            let lin = vec3<f32>(1.0, 0.6, 0.3) * dif + vec3<f32>(0.91, 0.98, 1.05);
            var col = vec4<f32>(
                mix(vec3<f32>(1.0, 0.95, 0.8), vec3<f32>(0.25, 0.3, 0.35), den),
                den
            );
            col = vec4<f32>(col.xyz * lin, col.w * 0.4);
            col = vec4<f32>(mix(col.xyz, bgcol, 1.0 - exp(-0.003 * t * t)), col.w);
            col = vec4<f32>(col.rgb * col.a, col.a);
            sum += col * (1.0 - sum.a);
        }
        t += max(0.06, 0.05 * t);
    }

    return clamp(sum, vec4<f32>(0.0), vec4<f32>(1.0));
}

// Render function
fn render(ro: vec3<f32>, rd: vec3<f32>, background_color: vec3<f32>, px: vec2<i32>) -> vec4<f32> {
    let sun: f32 = clamp(dot(sundir, rd), 0.0, 1.0);

    var col = background_color;

    // Clouds
    let res: vec4<f32> = raymarch(ro, rd, col, px);
    col = col * (1.0 - res.w) + res.xyz;

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
