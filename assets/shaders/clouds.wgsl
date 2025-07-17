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
    // 16 bytes
    pos: vec3<f32>, // Position of the cloud
    seed: f32, // Unique identifier for noise

    // 16 bytes
    scale: vec3<f32>, // x=width, y=height, z=length, w=radius^2
    density: f32, // Overall fill (0=almost empty mist, 1=solid cloud mass)

    // 16 bytes
    detail: f32, // Fractal/noise detail power (0=smooth blob, 1=lots of little puffs)
    color: f32, // 0 = white, 1 = black
    _padding0: f32,
    _padding1: f32,
}


// Lighting variables
const EXTINCTION: f32 = 0.04;

const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0); // Direction of the aur light from below
const AUR_COLOR = vec3(0.5, 0.7, 1.0); // Light blue

const SUN_DIR: vec3<f32> = vec3(0.70710678, -0.70710678, 0.0);
const SUN_COLOR = vec3(1.0, 0.98, 0.95); // Very white sunlight

const AMBIENT_COLOR = vec3(1.0, 1.0, 1.0) * 0.25; // Bright ambient


// Simple noise function for white noise
fn hash1(pos: f32) -> f32 {
    var p = fract(pos * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}
fn hash2(pos: vec2<f32>) -> f32 {
    var p3 = fract(vec3(pos.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash3i(pos: vec3<i32>) -> f32 {
    var n: i32 = pos.x * 3 + pos.y * 113 + pos.z * 311;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731 + 789221) + 1376312589;
    return -1.0 + 2.0 * f32(n & 0x0fffffff) / f32(0x0fffffff);
}

// Procedural blue noise approximation
fn blue_noise(uv: vec2<f32>) -> f32 {
    let v = hash2(uv + vec2(-1, 0)) + hash2(uv + vec2(1, 0)) + hash2(uv + vec2(0, 1)) + hash2(uv + vec2(0, -1));
    return hash2(uv) - v * 0.25 + 0.5;
}

// Simple 3D noise function
fn noise3(pos: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(pos));
    let f: vec3<f32> = fract(pos);
    let u: vec3<f32> = f * f * (3.0 - 2.0 * f); // Smoothstep weights

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
const MAX_OCT: u32 = 5u;
const WEIGHTS = array<f32,MAX_OCT>(0.5, 0.25, 0.125, 0.0625, 0.03125);
const NORMS = array<f32,MAX_OCT>(1.0, 0.75, 0.875, 0.9375, 0.96875);
fn fbm_lod(pos: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var freqp = pos;
    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        sum += WEIGHTS[i] * noise3(freqp);
        freqp = freqp * m3;
    }
    return sum / NORMS[octaves - 1u];
}

// Check density against clouds
const COARSE_FREQ = 0.01;
const COARSE_OCT = 2u;
const WARP_AMP = 0.2;
const DETAIL_FREQ = 0.1;
fn density_at_cloud(pos: vec3<f32>, cloud: Cloud, dist: f32, timeOffsetA: vec3<f32>, timeOffsetB: vec3<f32>) -> f32 {
    // Ellipsoid local space
    let dpos = pos - cloud.pos;
    let invRadius = 2.0 / cloud.scale;
    let normDir = dpos * invRadius;
    let edgeDistance = length(normDir);

    // Compute a very low frequency warp
    let seed = vec3(cloud.seed);
    let coarse = fbm_lod(dpos * COARSE_FREQ + timeOffsetA + seed * 0.3, COARSE_OCT);

    // Warp the sample position
    let dpos_warped = dpos + WARP_AMP * cloud.scale * (coarse - 0.5);

    // Compute the ellipsoid shape on the warped position
    let invS = 3.0 / cloud.scale;
    let d2 = dot(dpos_warped * invS, dpos_warped * invS);
    var shape = 1.0 - d2;
    if shape <= 0.0 {
        return 0.0;
    }

    // Level of detail fading for both the puff and octaves
    let lodf1 = clamp(1.0 - dist * 0.002, 0.0, 1.0);  // 0..1 over 500 units
    let lodf2 = clamp(1.0 - dist * 0.00025, 0.0, 1.0);  // 0..1 over 4000 units

    // Sample puff noise
    let octaves = u32(mix(2.0, f32(MAX_OCT), lodf1));
    let detail_noise = fbm_lod(dpos * DETAIL_FREQ + timeOffsetB + seed * 0.7, octaves);

    // Build a little “flat core” and then add noisy puffs scaled by detail & fade
    let core = max(shape - 0.2, 0.0);
    let puff = shape * detail_noise * cloud.detail * lodf2;

    return (core + puff) * cloud.density;
}

// Returns (t_near, t_far), the intersection points
fn ellipsoid_intersect(ro: vec3<f32>, rd: vec3<f32>, cloud: Cloud) -> vec2<f32> {
    // Transform ray into the unit‐sphere space of our ellipsoid:
    let invRadius = 2.0 / cloud.scale;    // = 1.0/(scale*0.5)
    let originLocal = (ro - cloud.pos) * invRadius;
    let dirLocal = rd * invRadius;

    // Build the quadratic:
    let a = dot(dirLocal, dirLocal);
    let b = dot(originLocal, dirLocal);
    let c = dot(originLocal, originLocal) - 1.0;

    // If the ray origin is outside the sphere (c>0) and the ray is pointing away from it (b>0), there is no intersection.
    if c > 0.0 && b > 0.0 {
        return vec2<f32>(1.0, 0.0); // No intersection
    }

    // Compute the discriminant:
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return vec2<f32>(1.0, 0.0); // No real roots → no intersection.
    }

    // Solve for the two roots
    let sqrtDisc = sqrt(disc);
    let invA = 1.0 / a;
    let tNear = (-b - sqrtDisc) * invA;
    let tFar = (-b + sqrtDisc) * invA;
    return vec2<f32>(tNear, tFar);
}

// Raymarch through all the clouds, first gathering the intersects
const MIN_STEP = 0.8;
const K_STEP = 0.005; // The fall-off of step size with distance
const ALPHA_TARGET = 0.99; // Max alpha to reach before stopping

const MAX_QUEUED = 12u; // Total number of clouds to consider ahead at a time
struct CloudIntersect {
    idx: u32,
    enterT: f32,
    exitT: f32,
};
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, tMax: f32, dither: f32) -> vec4<f32> {
    var sumCol = vec4<f32>(0.0); // Accumulated color
    var t = 0.0; // Current ray distance

    // Precomputed constants
    let timeOffsetB = vec3(globals.time * 0.8, globals.time * -0.2, globals.time * 0.6);
    let timeOffsetA = timeOffsetB * 0.05;

    var nextCloudIdx = 0u; // Next cloud index
    var queuedCount = 0u; // Number of queued clouds
    var queuedList: array<CloudIntersect, MAX_QUEUED>; // Queued cloud list

    while t < tMax && sumCol.a < ALPHA_TARGET {
        // Remove any expired clouds (exitT ≤ t)
        var dst = 0u;
        for (var i = 0u; i < queuedCount; i = i + 1u) {
            if queuedList[i].exitT > t {
                queuedList[dst] = queuedList[i];
                dst++;
            }
        }
        queuedCount = dst;

        // Pull in new clouds to fill the queuedList
        while queuedCount < MAX_QUEUED && nextCloudIdx < clouds_buffer.num_clouds {
            let cloud = clouds_buffer.clouds[nextCloudIdx];
            let ts = ellipsoid_intersect(ro, rd, cloud);
            let entry = max(ts.x, 0.0);
            let exit = min(ts.y, tMax);

            if entry < exit {
                queuedList[queuedCount] = CloudIntersect(nextCloudIdx, entry, exit);
                queuedCount++;
            }
            nextCloudIdx++;
        }

        // Find the next boundary > t
        var activeCount: u32 = 0u;
        var nextEvent: f32 = tMax;
        for (var i = 0u; i < queuedCount; i = i + 1u) {
            let entry = queuedList[i].enterT;
            let exit = queuedList[i].exitT;
            if entry <= t && t <= exit {
                activeCount++;
            }
            // Set next event
            if entry > t && entry < nextEvent {
                nextEvent = entry;
            }
            if exit < nextEvent {
                nextEvent = queuedList[i].exitT;
            }
        }
        // If no active clouds, fast‐forward t to the nextEvent
        if activeCount == 0u {
            if nextEvent < tMax {
                t = nextEvent + dither + dither * (nextEvent * 0.005); // Add dither based on distance so it reduces banding
                continue;
            } else {
                break; // No more clouds to ever march
            }
        }

        // Raymarch until nextEvent
        while t < nextEvent && t < tMax && sumCol.a < ALPHA_TARGET {
            // March one step through the active clouds
            let step = max(MIN_STEP, K_STEP * t);
            let pos = ro + rd * t;

            for (var i = 0u; i < queuedCount; i = i + 1u) {
                if queuedList[i].enterT <= t && t <= queuedList[i].exitT {
                    let cloud = clouds_buffer.clouds[queuedList[i].idx];
                    // let density = density_at_cloud(pos, cloud, t, timeOffsetA, timeOffsetB);
                    let density = 1.0;
                    if density > 0.01 {
                        // Single‐pass Beer approximation
                        let beer = 1.0 / (1.0 + density * EXTINCTION);
                        var col = (SUN_COLOR * beer + AMBIENT_COLOR * (1.0 - beer)) * density * cloud.color;

                        let a = density * 0.4;
                        col *= a;
                        sumCol = sumCol + vec4(col * (1.0 - sumCol.a), a * (1.0 - sumCol.a));
                    }
                }
            }

            // Advance the ray
            t += step;
        }
    }

    return clamp(sumCol, vec4(0.0), vec4(1.0));
}


@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;

    // G-buffer loads
    let sceneCol: vec3f = textureLoad(screen_texture, vec2<i32>(pix), 0).xyz;
    var depth: f32 = textureLoad(depth_texture, vec2<i32>(pix), 0);

    // Unproject to world
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
    let wpos = view.world_from_clip * vec4(ndc, 1.0);
    let worldPos = wpos.xyz / wpos.w;

    // Form the ray
    let ro = view.world_position;
    let rd_vec = worldPos - ro;
    let tMax = length(rd_vec);
    let rd = rd_vec / tMax;

    // Ray-march clouds
    let dither = fract(blue_noise(in.position.xy));
    let clouds = raymarch(ro, rd, tMax, dither);

    // Composite over background
    let col = mix(sceneCol, clouds.xyz, clouds.a);

    return vec4(col, 1.0);
}
