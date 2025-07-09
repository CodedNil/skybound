#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var<uniform> view: View;
struct VolumetricClouds {
    intensity: f32,
}
@group(0) @binding(3) var<uniform> settings: VolumetricClouds;

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

// Simple hash function for noise
fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// Signed distance function for a sphere at (0, 0, 0), with slight noise
fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    let noise_scale = 0.5;
    let noise_strength = 2.0;
    return length(p) - radius + noise(p * noise_scale) * noise_strength;
}

// Basic raymarching function
fn raymarch(uv: vec2<f32>, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var dist = 0.0;
    let sphere_radius = settings.intensity;

    // Raymarch parameters
    let max_steps = 100;
    let max_dist = 100.0;
    let hit_threshold = 0.01;

    for (var i = 0; i < max_steps; i = i + 1) {
        let p = ro + rd * dist;
        let d = sdf_sphere(p, sphere_radius);

        if d < hit_threshold {
            // Hit the sphere, return red color
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }

        dist += d;
        if dist > max_dist {
            break;
        }
    }

    // No hit, return background (screen texture)
    return textureSample(screen_texture, texture_sampler, uv);
}


@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = view.world_position;

    // Convert UV to NDC: [0, 1] -> [-1, 1]
    let ndc = in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    // Transform UV to ray direction
    let view_position_homogeneous = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let view_ray_direction = view_position_homogeneous.xyz / view_position_homogeneous.w;
    let ray_direction = normalize((view.world_from_view * vec4(view_ray_direction, 0.0)).xyz);

    // Perform raymarching
    return raymarch(in.uv, ray_origin, ray_direction);
}
