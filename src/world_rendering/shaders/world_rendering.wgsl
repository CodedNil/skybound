#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::utils::{View, AtmosphereData, blue_noise}
#import skybound::raymarch::raymarch
#import skybound::sky::render_sky
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var depth_texture: texture_depth_2d;

const ATMOSPHERE_HEIGHT: f32 = 100000.0;

// Lighting Parameters
const MIN_SUN_DOT: f32 = sin(radians(-8.0)); // How far below the horizon before the switching to aur light
const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0);
const AMBIENT_AUR_COLOR: vec3<f32> = vec3(0.4, 0.1, 0.6);
const SILVER_SPREAD: f32 = 0.1;
const SILVER_INTENSITY: f32 = 0.01;


const K: f32 = 0.0795774715459;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    return K * (1.0 - g * g) / (pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;
    let dither = fract(blue_noise(pix));

    // Load depth and unproject to clip space
    let depth = textureSample(depth_texture, linear_sampler, uv);
    let ndc = vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth, 1.0);

    // Reconstruct world‑space pos
    let world_pos4 = view.world_from_clip * ndc;
    let world_pos3 = world_pos4.xyz / world_pos4.w;

    // Ray origin & dir
    let ro = view.world_position + view.camera_offset;
    let rd_vec = world_pos3 - ro;
    let t_max = length(rd_vec);
    let rd = normalize(rd_vec);

    // Get sun direction and intensity, mix between aur light (straight up) and sun
    let sun_dot = view.sun_direction.y;
    let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    let sun_dir = normalize(mix(AUR_DIR, view.sun_direction, sun_t));

	// Precalculate sun, sky and ambient colors
    var atmosphere: AtmosphereData;
    atmosphere.sky = render_sky(rd, sun_dir, ro.y);
    atmosphere.sun = render_sky(sun_dir, sun_dir, ro.y) * 0.1;
    atmosphere.ambient = render_sky(normalize(vec3<f32>(1.0, 1.0, 0.0)), sun_dir, ro.y);
    atmosphere.ground = AMBIENT_AUR_COLOR * 100.0;

    atmosphere.planet_rotation = view.planet_rotation;
    atmosphere.planet_center = vec3<f32>(ro.x, -view.planet_radius, ro.z);
    atmosphere.planet_radius = view.planet_radius;
    atmosphere.sun_dir = sun_dir;

	// Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.99 - SILVER_SPREAD) * SILVER_INTENSITY;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    atmosphere.phase = max(hg_forward, max(hg_silver, hg_back)) + 0.1;

    // Sample the volumes
    let volumes_color: vec4<f32> = raymarch(ro, rd, atmosphere, view, t_max, dither, view.time, linear_sampler);
    var acc_color: vec3<f32> = volumes_color.rgb;
    var acc_alpha: f32 = volumes_color.a;

    if depth <= 0.00001 {
        // Add our sky in the background
        acc_color += vec3(atmosphere.sky * (1.0 - acc_alpha));
        acc_alpha = 1.0;
    }

    return clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
}
