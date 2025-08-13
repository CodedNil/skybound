#import skybound::utils::View
#import skybound::clouds::{CLOUDS_BOTTOM_HEIGHT, CLOUDS_TOP_HEIGHT, sample_clouds}
#import skybound::aur_fog::{FOG_START_HEIGHT, FOG_BOTTOM_HEIGHT}

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var linear_sampler: sampler;

@group(0) @binding(3) var cloud_base_texture: texture_3d<f32>;
@group(0) @binding(4) var cloud_details_texture: texture_3d<f32>;
@group(0) @binding(5) var cloud_motion_texture: texture_2d<f32>;
@group(0) @binding(6) var cloud_weather_texture: texture_2d<f32>;

const RESOLUTION: vec3<u32> = vec3<u32>(256, 144, 1024);
const FROXEL_NEAR: f32 = 1.0; // Near plane of froxel frustum
const FROXEL_FAR: f32 = 1000000.0; // Far plane of froxel frustum

const LIGHT_STEPS: u32 = 40; // How many steps to take along the sun direction
const LIGHT_STEPS_AUR: u32 = 4; // How many steps to take along the aur direction
const LIGHT_STEP_SIZE: f32 = 150.0;
const AUR_DIRECTION: vec3<f32> = vec3<f32>(0.0, -1.0, 0.0);

fn sample_volume(pos: vec3<f32>, dist: f32, time: f32) -> f32 {
    return sample_clouds(pos, dist, time, true, cloud_base_texture, cloud_details_texture, cloud_motion_texture, cloud_weather_texture, linear_sampler);
}

@compute @workgroup_size(8, 8, 16)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= RESOLUTION.x || id.y >= RESOLUTION.y || id.z >= RESOLUTION.z) { return; }

    // Convert froxel index to normalized uvw
    let texture_coord = vec3<f32>(id) / vec3<f32>(RESOLUTION);

    // Convert uvw to world space position
    let ndc_xy = uv_to_ndc(texture_coord.xy); // Converts UV [0,1] to NDC [-1,1]
    let logarithmic_depth = FROXEL_NEAR * pow(FROXEL_FAR / FROXEL_NEAR, texture_coord.z);
    let ndc_depth = view_z_to_depth_ndc(-logarithmic_depth, view.clip_from_view);
    let ndc_pos = vec3<f32>(ndc_xy.x, ndc_xy.y, ndc_depth);
    let world_pos = position_ndc_to_world(ndc_pos, view.world_from_clip);
    let t = distance(view.world_position, world_pos);

    // Early exit if outside of volumes
    let inside_clouds = world_pos.y >= CLOUDS_BOTTOM_HEIGHT && world_pos.y <= CLOUDS_TOP_HEIGHT;
    let inside_fog = world_pos.y >= FOG_BOTTOM_HEIGHT && world_pos.y <= FOG_START_HEIGHT;
    if !(inside_clouds || inside_fog) { return; }

    // Sample density
    let density = sample_volume(world_pos + view.camera_offset, t, view.time);

    // Lightmarching for self-shadowing
    var density_sunwards = max(density, 0.0);
    var lightmarch_pos = world_pos;
    for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
        lightmarch_pos += view.sun_direction * LIGHT_STEP_SIZE;
        density_sunwards += sample_volume(lightmarch_pos + view.camera_offset, t, view.time);
    }

    var density_aurwards = max(density, 0.0);
    lightmarch_pos = world_pos;
    for (var j: u32 = 0; j <= LIGHT_STEPS_AUR; j++) {
        lightmarch_pos += AUR_DIRECTION * LIGHT_STEP_SIZE;
        density_aurwards += sample_volume(lightmarch_pos + view.camera_offset, t, view.time);
    }

    // Store the calculated data into the froxel texture
    textureStore(output, id, vec4(density, density_sunwards * 0.1, density_aurwards * 0.1, 0.0));
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: vec3<f32>, world_from_clip: mat4x4<f32>) -> vec3<f32> {
    let world_pos = world_from_clip * vec4(ndc_pos, 1.0);
    return world_pos.xyz / world_pos.w;
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near(clip_from_view: mat4x4<f32>) -> f32 {
    return clip_from_view[3][2];
}

/// Convert linear view z to ndc depth.
/// Note: View z input should be negative for values in front of the camera as -z is forward
fn view_z_to_depth_ndc(view_z: f32, clip_from_view: mat4x4<f32>) -> f32 {
    return -perspective_camera_near(clip_from_view) / view_z;
}
