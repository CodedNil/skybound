#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import skybound::functions::{ blue_noise}
#import skybound::clouds::{render_clouds}
#import skybound::aur_fog::render_fog
#import skybound::sky::{AtmosphereColors, render_sky}
#import skybound::poles::render_poles

@group(0) @binding(0) var<uniform> view: View;
struct View {
    world_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    planet_rotation: vec4<f32>,
    latitude: f32,
    longitude: f32,
    latitude_meters: f32,
    longitude_meters: f32,
    altitude: f32,
};
@group(0) @binding(1) var<uniform> globals: Globals;
struct Globals {
    time: f32,
    planet_radius: f32,
    sun_direction: vec3<f32>,
    sun_intensity: f32,
}
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_2d;

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
    let ro = view.world_position; //vec3(view.longitude_meters, view.altitude, -view.latitude_meters);
    let rd_vec = world_pos3 - ro;
    let t_max = length(rd_vec);
    let rd = normalize(rd_vec);

    // Get sun direction and intensity, mix between aur light (straight up) and sun
    let sun_dot = globals.sun_direction.y;
    let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    let sun_dir = normalize(mix(AUR_DIR, globals.sun_direction, sun_t));

	// Precalculate sun, sky and ambient colors
	var atmosphere_colors: AtmosphereColors;
	atmosphere_colors.sky = render_sky(rd, sun_dir, ro.y);
	atmosphere_colors.sun = render_sky(sun_dir, sun_dir, ro.y) * 0.1;
	atmosphere_colors.ambient = render_sky(normalize(vec3<f32>(1.0, 1.0, 0.0)), sun_dir, ro.y);
	atmosphere_colors.ground = AMBIENT_AUR_COLOR * 100.0;

	// Phase functions for silver and back scattering
    let cos_theta = dot(sun_dir, rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.99 - SILVER_SPREAD) * SILVER_INTENSITY;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    atmosphere_colors.phase = max(hg_forward, max(hg_silver, hg_back)) + 0.1;

    // Render out the world poles
    let pole_color = render_poles(ro, rd, view.planet_rotation, globals.planet_radius);

    // Start accumulation volumetric color
    var acc_color: vec3<f32> = vec3<f32>(0.0);
    var acc_alpha: f32 = 0.0;

    if ro.y > 1000.0 {
        // Sample the clouds
        let cloud_color: vec4<f32> = render_clouds(ro, rd, atmosphere_colors, sun_dir, t_max, dither, globals.time, linear_sampler);
        acc_color = cloud_color.rgb;
        acc_alpha = cloud_color.a;

        if acc_alpha < 1.0 {
            // Blend in the fog
            let fog_color: vec4<f32> = render_fog(ro, rd, atmosphere_colors, sun_dir, t_max, dither, globals.time, linear_sampler);
            if fog_color.a > 0.0 {
                acc_color += fog_color.rgb * (1.0 - acc_alpha);
                acc_alpha += fog_color.a * (1.0 - acc_alpha);
            }
        }
    } else {
        // Sample the fog first
        let fog_color: vec4<f32> = render_fog(ro, rd, atmosphere_colors, sun_dir, t_max, dither, globals.time, linear_sampler);
        acc_color = fog_color.rgb;
        acc_alpha = fog_color.a;

        if acc_alpha < 1.0 {
            // Blend in the clouds
            let cloud_color: vec4<f32> = render_clouds(ro, rd, atmosphere_colors, sun_dir, t_max, dither, globals.time, linear_sampler);
            if cloud_color.a > 0.0 {
                acc_color += cloud_color.rgb * (1.0 - acc_alpha);
                acc_alpha += cloud_color.a * (1.0 - acc_alpha);
            }
        }
    }

    if depth <= 0.00001 {
        // Blend in poles behind clouds
        acc_color += pole_color.rgb * pole_color.a * (1.0 - acc_alpha);
        acc_alpha += pole_color.a * (1.0 - acc_alpha);

        // Add our sky in the background
        acc_color += vec3(atmosphere_colors.sky * (1.0 - acc_alpha));
        acc_alpha = 1.0;
    }

    return clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
}
