#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var depthTexture: texture_depth_2d;

// Raymarcher Parameters
const MAX_STEPS: i32 = 256;
const ALPHA_THRESHOLD: f32 = 0.9; // Max alpha to reach before stopping
const MIN_STEP: f32 = 6.0;
const MAX_STEP: f32 = 12.0;
const K_STEP: f32 = 0.0001; // The fall-off of step size with distance
const LIGHT_STEPS: i32 = 2; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 16.0;

// Lighting variables
const LIGHT_DIRECTION: vec3<f32> = vec3<f32>(0.0, 0.89, 0.45);
const SUN_COLOR: vec3<f32> = vec3<f32>(0.99, 0.97, 0.96);
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.52, 0.80, 0.92);
const SHADOW_EXTINCTION: f32 = 5.0; // Higher = deeper core shadows

// Cloud Material Parameters
const BACK_SCATTERING: f32 = 1.0; // Backscattering
const BACK_SCATTERING_FALLOFF: f32 = 30.0; // Backscattering falloff
const OMNI_SCATTERING: f32 = 0.8; // Omnidirectional Scattering
const TRANSMISSION_SCATTERING: f32 = 1.0; // Transmission Scattering
const TRANSMISSION_FALLOFF: f32 = 2.0; // Transmission falloff
const BASE_TRANSMISSION: f32 = 0.1; // Light that doesn't get scattered at all

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

fn hash4(pos: vec4<f32>) -> f32 {
    var p4 = fract(pos * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.x + p4.y) * (p4.z + p4.w));
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

// Sky shading
fn renderSky(rd: vec3<f32>, sunDirection: f32) -> vec3<f32> {
    let elevation = 1.0 - dot(rd, vec3<f32>(0.0, 1.0, 0.0));
    let centered = 1.0 - abs(1.0 - elevation);

    let atmosphere_color = mix(AMBIENT_COLOR, SUN_COLOR, sunDirection * 0.5);
    let base = mix(pow(AMBIENT_COLOR, vec3<f32>(4.0)), atmosphere_color, pow(clamp(elevation, 0.0, 1.0), 0.5));
    let haze = pow(centered + 0.02, 4.0) * (sunDirection * 0.2 + 0.8);

    let sky = mix(base, SUN_COLOR, clamp(haze, 0.0, 1.0));
    let sun = pow(max((sunDirection - 29.0 / 30.0) * 30.0 - 0.05, 0.0), 6.0);

    return sky + vec3<f32>(sun);
}

// Cloud density
fn densityAtCloud(pos: vec3<f32>) -> f32 {
    let clouds_type: f32 = 0.0;
    let clouds_coverage: f32 = 1.0;

    let altitude = pos.y;
    var density = 0.0;

    // Thick aur fog below 0m
    let fog_density = smoothstep(0.0, -10.0, pos.y);
    if fog_density > 0.01 {
        let fbm_value = fbm3(pos / 5.0 - vec3(0.0, 0.1, 1.0) * globals.time, 4);
        density = fbm_value * fog_density + fog_density * 0.5;
    }

    let low_gradient = smoothstep(60.0, 200.0, pos.y) * smoothstep(800.0, 700.0, pos.y);
    let mid_gradient = smoothstep(700.0, 800.0, pos.y) * smoothstep(1800.0, 1700.0, pos.y);
    let high_gradient = smoothstep(1700.0, 1800.0, pos.y) * smoothstep(2700.0, 2500.0, pos.y);

    // Cloud type blending
    let low_type = mix(
        // Stratus: Flat, layered profile
        smoothstep(0.0, 200.0, altitude) * smoothstep(1500.0, 1000.0, altitude),
        // Cumulus: Puffy, vertical development
        smoothstep(200.0, 500.0, altitude) * smoothstep(3000.0, 2000.0, altitude) * (1.0 + 0.5 * sin(altitude * 0.002)),
        saturate(clouds_type * 2.0)
    );
    // Cirrus: Thin, wispy profile
    let high_type = smoothstep(5000.0, 6000.0, altitude) * smoothstep(12000.0, 10000.0, altitude) * 0.3;

    // Generate base cloud shapes
    if low_gradient > 0.01 {
        let base_noise = fbm3(pos * 0.005 + vec3(0.0, globals.time * 0.02, 0.0), 5);
        let shaped_noise = base_noise * low_type;
        density += shaped_noise * low_gradient * clouds_coverage;
    }
    // if mid_gradient > 0.01 {
    //     let base_noise = fbm3(pos * 0.004 + vec3(0.0, globals.time * 0.02, 0.0), 4);
    //     let shaped_noise = base_noise * mix(low_type, high_type, 0.5);
    //     density += shaped_noise * mid_gradient * clouds_coverage;
    // }
    // if high_gradient > 0.01 {
    //     let base_noise = fbm3(pos * 0.003 + vec3(0.0, globals.time * 0.02, 0.0), 4);
    //     let shaped_noise = base_noise * high_type;
    //     density += shaped_noise * high_gradient * clouds_coverage * 0.7; // Thinner high clouds
    // }

    return clamp(density, 0.0, 1.0);
}

// Lighting Functions
fn computeDensityTowardsSun(pos: vec3<f32>, densityHere: f32) -> f32 {
    var densitySunwards = max(densityHere, 0.0);
    for (var j: i32 = 1; j <= LIGHT_STEPS; j = j + 1) {
        let lightOffset = pos + LIGHT_DIRECTION * f32(j) * LIGHT_STEP_SIZE;
        densitySunwards += densityAtCloud(lightOffset) * LIGHT_STEP_SIZE;
    }

    return densitySunwards;
}

fn beer(materialAmount: f32) -> f32 {
    return exp(-materialAmount);
}

fn transmission(light: vec3<f32>, materialAmount: f32) -> vec3<f32> {
    return beer(materialAmount * (1.0 - BASE_TRANSMISSION)) * light;
}

fn lightScattering(light: vec3<f32>, angle: f32) -> vec3<f32> {
    var a = (angle + 1.0) * 0.5; // Angle between 0 and 1

    var ratio = 0.0;
    ratio += BACK_SCATTERING * pow(1.0 - a, BACK_SCATTERING_FALLOFF);
    ratio += TRANSMISSION_SCATTERING * pow(a, TRANSMISSION_FALLOFF);
    ratio = ratio * (1.0 - OMNI_SCATTERING) + OMNI_SCATTERING;

    return light * ratio * (1.0 - BASE_TRANSMISSION);
}

// Raymarch through all the clouds
fn raymarch(ro: vec3<f32>, rd: vec3<f32>, tMax: f32, dither: f32) -> vec4<f32> {
    var accumulation = vec4<f32>(0.0);
    var t = dither * MIN_STEP;

    let sunDirection = dot(rd, LIGHT_DIRECTION);
    let sky = renderSky(rd, sunDirection);

    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= tMax || accumulation.a >= ALPHA_THRESHOLD {
            break;
        }
        let step = max(MIN_STEP, t * K_STEP);

        let pos = ro + rd * t;
        let stepDensity = densityAtCloud(pos);

        if stepDensity > 0.0 {
            let materialHere = stepDensity * step;
            let materialTowardsSun = computeDensityTowardsSun(pos, stepDensity);
            let lightAtParticle = transmission(SUN_COLOR, materialTowardsSun);

            let lightScatteringTowardsCamera = lightScattering(lightAtParticle * materialHere, sunDirection);
            let lightReachingCamera = transmission(lightScatteringTowardsCamera, accumulation.a + materialHere);
            accumulation += vec4(lightReachingCamera, materialHere);
        }

        // Adjust step size based on density
        let stepScale = mix(MAX_STEP, MIN_STEP, clamp(stepDensity * 2.0, 0.0, 1.0));
        t += min(stepScale, step);
    }

    accumulation.a = min(accumulation.a * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0
    accumulation += vec4(beer(accumulation.a * (1.0 - BASE_TRANSMISSION)) * sky, 1.0); // Add sky

    return clamp(accumulation, vec4<f32>(0.0), vec4<f32>(1.0));
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
