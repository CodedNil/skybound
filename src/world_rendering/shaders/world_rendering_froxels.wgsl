#import skybound::utils::View
#import skybound::raymarch::VolumesInside
#import skybound::clouds::sample_clouds
#import skybound::aur_fog::sample_fog
#import skybound::poles::sample_poles

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var linear_sampler: sampler;

@group(0) @binding(3) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(4) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_motion_texture: texture_2d<f32>;
@group(0) @binding(6) var cloud_weather_texture: texture_2d<f32>;

@group(0) @binding(7) var fog_noise_texture: texture_3d<f32>;

const RESOLUTION = vec3<u32>(128, 80, 512);

const DENSITY: f32 = 0.05; // Base density for lighting
const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = array<f32, 6>(30.0, 50.0, 80.0, 160.0, 300.0, 500.0);
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));

fn sample_volume(pos: vec3<f32>, dist: f32, time: f32, volumes_inside: VolumesInside) -> f32 {
    var sample: f32;

    if volumes_inside.fog {
        let fog_sample = sample_fog(pos, dist, time, true, fog_noise_texture, linear_sampler);
        if fog_sample.density > 0.0 {
            sample += fog_sample.density;
        }
    }

    if volumes_inside.clouds {
        let cloud_sample = sample_clouds(pos, dist, time, cloud_base_texture, cloud_details_texture, cloud_motion_texture, cloud_weather_texture, linear_sampler);
        if cloud_sample > 0.0 {
            sample += cloud_sample;
        }
    }

    if volumes_inside.poles {
        let poles_sample = sample_poles(pos, dist, time, true, linear_sampler);
        if poles_sample.density > 0.0 {
            sample += poles_sample.density;
        }
    }

    return min(sample, 1.0);
}

@compute @workgroup_size(8, 8, 8)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= RESOLUTION.x || id.y >= RESOLUTION.y || id.z >= RESOLUTION.z) { return; }

    // Convert to world space
    let uv = vec3<f32>(id) / vec3<f32>(RESOLUTION);
    let ndc = uv_to_ndc(uv.xy);
    let ndc_depth = view_z_to_depth_ndc(uv.z, view.clip_from_view);
    let pos = position_ndc_to_world(vec3<f32>(ndc, ndc_depth), view.world_from_clip);
    let t = distance(view.world_position, pos);

    // Sample density
    let altitude = distance(pos, view.planet_center) - view.planet_radius;
    var volumes_inside: VolumesInside; // TODO simplify testing which volumes
    volumes_inside.clouds = true;
    volumes_inside.fog = true;
    volumes_inside.poles = true;
    let density = sample_volume(pos, t, view.time, volumes_inside);

    // Lightmarching for self-shadowing
    var density_sunwards = max(density, 0.0);
    var lightmarch_pos = pos;
    var light_altitude: f32;
    for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
        lightmarch_pos += (view.sun_direction + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE[j];
        light_altitude = distance(lightmarch_pos, view.planet_center) - view.planet_radius;
        density_sunwards += sample_volume(vec3<f32>(lightmarch_pos.x, light_altitude, lightmarch_pos.z), t, view.time, volumes_inside);
    }

    // Captures the direct lighting from the sun
    let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE[1]);
    let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE[1] * 0.25) * 0.7;
    let beers_total = max(beers, beers2);

    // textureStore(output, vec3<i32>(id), vec4<f32>(density, beers_total, 0.0, 0.0));
    textureStore(output, vec3<i32>(id), vec4<f32>(pos.x * 0.001, pos.y * 0.001, pos.z * 0.001, 0.0));
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_clip * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near(clip_from_view: mat4x4<f32>) -> f32 {
    return clip_from_view[3][2];
}

/// Convert linear view z to ndc depth.
fn view_z_to_depth_ndc(view_z: f32, clip_from_view: mat4x4<f32>) -> f32 {
    return -perspective_camera_near(clip_from_view) / view_z;
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}
