#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var depthTexture: texture_depth_multisampled_2d;
@group(0) @binding(3) var<storage, read> cloudsBuffer: CloudsBuffer;

struct CloudsBuffer {
    numClouds: u32,
    clouds: array<Cloud, 2048>,
}

struct Cloud {
    // 16 bytes
    pos: vec3<f32>, // Position of the cloud, 12 bytes
    data: u32, // 4 bytes

    // 16 bytes
    scale: vec3<f32>, // x=width, y=height, z=length
    _padding0: u32,
}

// Lighting variables
const SHADOW_EXTINCTION: f32 = 5.0; // Higher = deeper core shadows

const SUN_DIR: vec3<f32> = vec3(0.5, 1.0, 0.5);
const SUN_COLOR: vec3<f32> = vec3(1.0, 0.98, 0.95); // Very white sunlight
const AMBIENT_COLOR: vec3<f32> = vec3(0.7, 0.8, 1.0) * 0.25; // Bright ambient

// Functions to unpack data
const FORM_MASK: u32 = 0x00000003u;      // Bits 0-1
const FORM_SHIFT: u32 = 0u;
fn getForm(data: u32) -> u32 {
    return (data & FORM_MASK) >> FORM_SHIFT;
}

const DENSITY_MASK: u32 = 0x0000003Cu;    // Bits 2-5
const DENSITY_SHIFT: u32 = 2u;
fn getDensity(data: u32) -> f32 {
    let raw = (data & DENSITY_MASK) >> DENSITY_SHIFT;
    return f32(raw) / 15.0;
}

const DETAIL_MASK: u32 = 0x000003C0u;     // Bits 6-9
const DETAIL_SHIFT: u32 = 6u;
fn getDetail(data: u32) -> f32 {
    let raw = (data & DETAIL_MASK) >> DETAIL_SHIFT;
    return f32(raw) / 15.0;
}

const BRIGHTNESS_MASK: u32 = 0x00003C00u; // Bits 10-13
const BRIGHTNESS_SHIFT: u32 = 10u;
fn getBrightness(data: u32) -> f32 {
    let raw = (data & BRIGHTNESS_MASK) >> BRIGHTNESS_SHIFT;
    return f32(raw) / 15.0;
}

const COLOR_MASK: u32 = 0x0003C000u;      // Bits 14-17
const COLOR_SHIFT: u32 = 14u;
fn getColor(data: u32) -> u32 {
    return (data & COLOR_MASK) >> COLOR_SHIFT;
}

const SEED_MASK: u32 = 0x00FC0000u;       // Bits 18-23
const SEED_SHIFT: u32 = 18u;
fn getSeed(data: u32) -> u32 {
    return (data & SEED_MASK) >> SEED_SHIFT;
}

// Simple noise function for white noise
fn hash2(pos: vec2<f32>) -> f32 {
    var p3 = fract(vec3(pos.x, pos.y, pos.x) * 0.1031);
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
fn blueNoise(uv: vec2<f32>) -> f32 {
    let v = hash2(uv + vec2<f32>(-1.0, 0.0)) + hash2(uv + vec2<f32>(1.0, 0.0)) + hash2(uv + vec2<f32>(0.0, 1.0)) + hash2(uv + vec2<f32>(0.0, -1.0));
    return hash2(uv) - v * 0.25 + 0.5;
}

// Simple noise functions
fn noise3(pos: vec3<f32>) -> f32 {
    let i: vec3<i32> = vec3<i32>(floor(pos));
    let f: vec3<f32> = fract(pos);
    let u: vec3<f32> = f * f * (3.0 - 2.0 * f); // Smoothstep weights

    // Hash values at cube corners, interpolate along x
    let lerpX0 = mix(hash3i(i + vec3<i32>(0, 0, 0)), hash3i(i + vec3<i32>(1, 0, 0)), u.x);
    let lerpX1 = mix(hash3i(i + vec3<i32>(0, 1, 0)), hash3i(i + vec3<i32>(1, 1, 0)), u.x);
    let lerpX2 = mix(hash3i(i + vec3<i32>(0, 0, 1)), hash3i(i + vec3<i32>(1, 0, 1)), u.x);
    let lerpX3 = mix(hash3i(i + vec3<i32>(0, 1, 1)), hash3i(i + vec3<i32>(1, 1, 1)), u.x);

    // Interpolate along y
    let lerpY0 = mix(lerpX0, lerpX1, u.y);
    let lerpY1 = mix(lerpX2, lerpX3, u.y);

    // Interpolate along z and return
    return mix(lerpY0, lerpY1, u.z);
}

// Fractional Brownian Motion (FBM)
const M3: mat3x3<f32> = mat3x3<f32>(
    vec3(0.8, 0.6, 0.0),
    vec3(-0.6, 0.8, 0.0),
    vec3(0.0, 0.0, 1.0)
) * 2.0;

const MAX_OCT: u32 = 5u;
const WEIGHTS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(0.5, 0.25, 0.125, 0.0625, 0.03125);
const NORMS: array<f32, MAX_OCT> = array<f32, MAX_OCT>(1.0, 0.75, 0.875, 0.9375, 0.96875);

fn fbm3(pos: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var freqPos = pos;
    for (var i: u32 = 0u; i < octaves; i = i + 1u) {
        sum += WEIGHTS[i] * noise3(freqPos);
        freqPos = freqPos * M3;
    }
    return sum / NORMS[octaves - 1u];
}

// Check density against clouds
const COARSE_FREQ: f32 = 0.005;
const WARP_AMP: f32 = 0.3;
const DETAIL_FREQ: f32 = 0.04;
const DETAIL_AMP: f32 = 0.8;

fn densityAtCloud(pos: vec3<f32>, cloud: Cloud, dist: f32, timeOffsetA: vec3<f32>, timeOffsetB: vec3<f32>) -> f32 {
    let form = getForm(cloud.data);
    let seed = vec3(f32(getSeed(cloud.data)));

    // Ellipsoid local space
    let dPos = pos - cloud.pos;
    let invScale = 2.0 / cloud.scale;
    let ellipsoidDist = length(dPos * invScale);
    let baseDensity = clamp(1.0 - ellipsoidDist, 0.0, 1.0);

    if baseDensity <= 0.0 {
        return 0.0;
    }

    if form == 0u { // Cumulus
        // Compute a very low frequency warp
        let coarse = fbm3(dPos * COARSE_FREQ + timeOffsetA + seed * 0.3, 3u);
        let dPosWarped = dPos + (coarse - 0.5) * WARP_AMP * cloud.scale;

        // Compute the ellipsoid shape on the warped position
        let invS = 2.5 / cloud.scale;
        let d2 = dot(dPosWarped * invS, dPosWarped * invS);
        var shape = 1.0 - d2;

        // Bottom cutoff
        let flatCut = smoothstep(-0.1 * cloud.scale.y, 0.05 * cloud.scale.y, dPos.y);
        shape *= flatCut;

        if shape <= 0.0 {
            return 0.0;
        }

        // Sample puff noise
        let detailNoise = fbm3(dPos * DETAIL_FREQ + timeOffsetB + seed * 0.7, u32(round(mix(2.0, 4.0, smoothstep(800.0, 200.0, dist))))) * DETAIL_AMP;

        // Build a little “flat core” and then add noisy puffs scaled by detail & fade
        let core = max(shape - 0.2, 0.0);
        let puff = shape * detailNoise * getDetail(cloud.data) * smoothstep(12000.0, 1000.0, dist);
        return clamp(core + puff, 0.0, 1.0) * baseDensity * getDensity(cloud.data);
    } else if form == 1u { // Stratus
        // Compute a very low frequency warp
        let coarse = fbm3(dPos * COARSE_FREQ + timeOffsetA + seed * 0.3, 3u);
        // Warp the sample position
        let dPosWarped = dPos + (coarse - 0.5) * WARP_AMP * cloud.scale;

        // Compute the ellipsoid shape on the warped position
        let invS = 3.0 / cloud.scale;
        let d2 = dot(dPosWarped * invS, dPosWarped * invS);
        var shape = (1.0 - d2) * baseDensity;

        if shape <= 0.0 {
            return 0.0;
        }

        return shape * baseDensity * getDensity(cloud.data);
    } else { // Cirrus
        // Compute a very low frequency warp
        let coarse = fbm3(dPos * vec3(0.1, 1.0, 1.0) * COARSE_FREQ + timeOffsetA + seed * 0.3, 3u);
        // Warp the sample position
        let dPosWarped = dPos + (coarse - 0.5) * WARP_AMP * cloud.scale;

        // Compute the ellipsoid shape on the warped position
        let invS = 2.0 / cloud.scale;
        let d2 = dot(dPosWarped * invS, dPosWarped * invS);
        var shape = (1.0 - d2) * baseDensity;

        if shape <= 0.0 {
            return 0.0;
        }

        return shape * baseDensity * getDensity(cloud.data);
    }
}

// Returns (tNear, tFar), the intersection points
fn ellipsoidIntersect(ro: vec3<f32>, rd: vec3<f32>, cloud: Cloud) -> vec2<f32> {
    // Transform ray into the unit‐sphere space of our ellipsoid:
    let invRadius = 2.0 / cloud.scale; // = 1.0/(scale*0.5)
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
const MIN_STEP: f32 = 10.0;
const MAX_STEP: f32 = 24.0;
const K_STEP: f32 = 0.001; // The fall-off of step size with distance
const ALPHA_TARGET: f32 = 0.9; // Max alpha to reach before stopping
const MAX_QUEUED: u32 = 12u; // Total number of clouds to consider ahead at a time

const LIGHT_STEPS: i32 = 2; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 16.0; // Adjust based on scene scale

struct CloudIntersect {
    idx: u32,
    enterT: f32,
    exitT: f32,
};

fn raymarch(ro: vec3<f32>, rd: vec3<f32>, tMax: f32, dither: f32) -> vec4<f32> {
    var sumCol = vec4<f32>(0.0); // Accumulated color
    var t = dither * MIN_STEP; // Current ray distance

    // Precomputed constants
    let timeOffsetB = vec3(globals.time * 0.8, globals.time * -0.2, globals.time * 0.6) * 0.5;
    let timeOffsetA = timeOffsetB * 0.02;

    let sunDirection = normalize(SUN_DIR);

    var nextCloudIdx: u32 = 0u; // Next cloud index
    var queuedCount: u32 = 0u; // Number of queued clouds
    var queuedList: array<CloudIntersect, MAX_QUEUED>;

    while t < tMax && sumCol.a < ALPHA_TARGET {
        // Remove any expired clouds (exitT ≤ t)
        var dst: u32 = 0u;
        for (var i: u32 = 0u; i < queuedCount; i = i + 1u) {
            if queuedList[i].exitT > t {
                queuedList[dst] = queuedList[i];
                dst++;
            }
        }
        queuedCount = dst;

        // Pull in new clouds to fill the queuedList
        while queuedCount < MAX_QUEUED && nextCloudIdx < cloudsBuffer.numClouds {
            let cloud = cloudsBuffer.clouds[nextCloudIdx];
            let ts = ellipsoidIntersect(ro, rd, cloud);
            let entry = max(ts.x, 0.0);
            let exit = min(ts.y, tMax); // Limit exit by scene depth
            if entry < exit {
                queuedList[queuedCount] = CloudIntersect(nextCloudIdx, entry, exit);
                queuedCount++;
            }
            nextCloudIdx++;
        }

        // Find the next boundary > t
        var activeCount: u32 = 0u;
        var nextEvent: f32 = tMax;

        for (var i: u32 = 0u; i < queuedCount; i = i + 1u) {
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
                nextEvent = exit;
            }
        }

        // If no active clouds, fast‐forward t to the nextEvent
        if activeCount == 0u {
            if nextEvent < tMax {
                t = nextEvent + (dither * MIN_STEP); // Add dither so it reduces banding
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

            var stepDensity: f32 = 0.0;
            var stepColor: vec3<f32> = vec3(0.0);
            var shadowSum: f32 = 0.0;

            // Accumulate density, color, and shadow for all active clouds
            for (var i: u32 = 0u; i < queuedCount; i = i + 1u) {
                if queuedList[i].enterT <= t && t <= queuedList[i].exitT {
                    let cloud = cloudsBuffer.clouds[queuedList[i].idx];
                    let density = densityAtCloud(pos, cloud, t, timeOffsetA, timeOffsetB);

                    if density > 0.01 {
                        stepDensity += density;

                        let cloudColor = vec3(1.0) * getBrightness(cloud.data); // Placeholder: map getColor(cloud.data) to RGB
                        stepColor += cloudColor * density;

                        // Lightmarching for self-shadowing
                        var lightDensity: f32 = 0.0;
                        for (var j: i32 = 1; j <= LIGHT_STEPS; j = j + 1) {
                            let lightOffset = pos + sunDirection * f32(j) * LIGHT_STEP_SIZE;
                            let shadowDensity = densityAtCloud(lightOffset, cloud, t, timeOffsetA, timeOffsetB);
                            lightDensity += shadowDensity;
                        }
                        shadowSum += lightDensity * LIGHT_STEP_SIZE;
                    }
                }
            }

            if stepDensity > 0.0 {
                stepColor /= stepDensity; // Normalize to get the albedo for this point.

                let tau = clamp(shadowSum, 0.0, 1.0);
                let selfShadow = exp(-SHADOW_EXTINCTION * tau); // Inner shadow darkening, with Beer function

                // Final color with self-shadowing
                let litColor = mix(AMBIENT_COLOR, SUN_COLOR, selfShadow) * stepColor;

                let stepAlpha = clamp(stepDensity * 0.4 * step, 0.0, 1.0);
                sumCol = sumCol + vec4<f32>(litColor * stepAlpha * (1.0 - sumCol.a), stepAlpha * (1.0 - sumCol.a));
            }

            // Adjust step size based on density
            let step_scale = mix(MAX_STEP, MIN_STEP, clamp(stepDensity * 2.0, 0.0, 1.0));
            t += min(step_scale, step);
        }
    }

    sumCol.a = sumCol.a * (1.0 / ALPHA_TARGET); // Scale the sumCol so that alpha of ALPHA_TARGET becomes 1.0
    return clamp(sumCol, vec4<f32>(0.0), vec4<f32>(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // Load depth from the depthTexture
    let depth_texture_size = textureDimensions(depthTexture);
    let clamped_coords = clamp(
        vec2<i32>(
            i32(uv.x * f32(depth_texture_size.x)),
            i32(uv.y * f32(depth_texture_size.y))
        ),
        vec2<i32>(0, 0),
        vec2<i32>(i32(depth_texture_size.x - 1u), i32(depth_texture_size.y - 1u))
    );
    let depth = textureLoad(depthTexture, clamped_coords, 0);

    // Unproject to world using the depth
    let ndc = vec3(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), depth);
    let wpos = view.world_from_clip * vec4<f32>(ndc, 1.0);
    let worldPos = wpos.xyz / wpos.w;

    // Form the ray, using the scene depth as tMax
    let ro = view.world_position;
    let rdVec = worldPos - ro;
    let tMax = length(rdVec);
    let rd = rdVec / tMax;

    // Ray-march clouds
    let dither = fract(blueNoise(in.position.xy));
    let clouds = raymarch(ro, rd, tMax, dither);

    // Return the clouds with alpha
    return clouds;
}
