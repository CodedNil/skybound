#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(4) var<storage, read> clouds_buffer: CloudsBuffer;
struct CloudsBuffer {
    num_clouds: u32,
    clouds: array<Cloud, 2048>,
}
struct Cloud {
    // 2× Vec4
    transform: vec4<f32>, // xyz=centre, w used for rotation yaw
    scale: vec4<f32>,     // xyz=scale, w is dynamically used for squared radius

    // 4 floats → 16 bytes
    seed: f32,    // Unique identifier for noise
    density: f32, // Overall fill (0=almost empty mist, 1=solid cloud mass)
    detail: f32,  // Fractal/noise detail power (0=smooth blob, 1=lots of little puffs)
    form: f32,    // 0 = linear streaks like cirrus, 0.5 = solid like cumulus, 1 = anvil like cumulonimbus
}


const SUNDIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const MAX_DISTANCE: f32 = 2000.0;


// Simple noise function for white noise
fn hash1(po: f32) -> f32 {
    var p = fract(po * .1031);
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

fn fbm(po: vec3<f32>) -> f32 {
    var p = po;
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

fn get_cloud_lod(distance: f32) -> f32 {
    // Reduce quality for distant clouds
    return clamp(distance / 500.0, 0.25, 1.0);
}

// Check density against clouds
fn density_at_cloud(pos: vec3<f32>, c: Cloud) -> f32 {
    let d = pos - c.transform.xyz;
    let dist2 = dot(d, d);
    let density = 1.0 - dist2 / c.scale.w;
    return density;
    // let noise = fbm((pos - c.transform.xyz) * 0.6 + vec3(globals.time * 0.8, globals.time * -0.2, globals.time * 0.6));
    // return max(density - 0.2, 0.0) + density * noise;
}

// Raymarch function
const MIN_STEP = 0.2;
const K_STEP = 0.01; // The fall-off of step size with distance

// Lighting
const extinction = 0.04;
const g = 0.05; // anisotropy
const light_color = vec3(1.0, 0.98, 0.95); // Very white sunlight
const ambient = vec3(1.0, 1.0, 1.0) * 0.25; // Bright ambient

fn raymarch(ro: vec3<f32>, rd: vec3<f32>, tmax: f32, dither: f32) -> vec4<f32> {
    var sumCol = vec4(0.0);

    // Henyey-Greenstein phase function
    let hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * dot(SUNDIR, rd), 1.5);

    // Loop over all clouds which are sorted by camera distance
    for (var i: u32 = 0u; i < clouds_buffer.num_clouds; i = i + 1u) {
        let c = clouds_buffer.clouds[i];
        let oc = ro - c.transform.xyz;
        let b = dot(oc, rd);
        let disc = b * b - (dot(oc, oc) - c.scale.w);
        if disc <= 0.0 {
            continue;  // no intersection
        }

        let sq = sqrt(disc);
        let t0 = max(-b - sq, dither);
        let t1 = min(-b + sq, tmax);
        if t0 >= t1 {
            continue;  // missed or degenerate
        }

        // March this sphere segment
        var t = t0;
        while t < t1 && sumCol.a < 0.99 {
            let pos = ro + rd * t;
            let density = density_at_cloud(pos, c);
            if density > 0.01 {
                // Traditional Beer's Law with Beer-Powder approximation
                let beer = 1.0 / (1.0 + density * extinction);
                let powder = 1.0 / (1.0 + density * 0.8);
                let beers_powder = mix(beer, beer * powder, powder);
                var col = (light_color * beers_powder * hg + ambient) * density;

                let a = density * 0.4;
                col *= a;
                sumCol = sumCol + vec4(col * (1.0 - sumCol.a), a * (1.0 - sumCol.a));
                t += max(MIN_STEP, K_STEP * t);
            } else {
                t += max(MIN_STEP * 2.0, K_STEP * t); // Larger step in empty space
            }
        }

        // Once we saturate, stop testing any more distant clouds
        if sumCol.a >= 0.99 {
            break;
        }
    }

    return clamp(sumCol, vec4(0.0), vec4(1.0));
}


@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let current_pos = in.position.xy;

    // G-buffer loads
    let screen_color: vec3f = textureLoad(screen_texture, vec2<i32>(current_pos), 0).xyz;
    var depth: f32 = textureLoad(depth_texture, vec2<i32>(current_pos), 0);

    // Unproject to world
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
    let world_pos_raw = view.world_from_clip * vec4(ndc, 1.0);
    let world_pos = world_pos_raw.xyz / world_pos_raw.w;

    // Form the ray
    let origin = view.world_position;
    let ray_vec = world_pos - origin;
    let tmax = length(ray_vec);
    let ray_dir = ray_vec / tmax;

    // Procedural blue noise dithering
    let dither = fract(blue_noise(in.position.xy));

    // Ray-march clouds
    let clouds = raymarch(origin, ray_dir, tmax, dither);

    // Composite over background
    let col = mix(screen_color, clouds.xyz, clouds.a);

    return vec4(col, 1.0);
}
