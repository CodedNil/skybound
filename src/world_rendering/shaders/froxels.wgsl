#define_import_path skybound::froxels
#import skybound::utils::View

@group(0) @binding(3) var froxels_texture: texture_3d<f32>;
@group(0) @binding(4) var froxels_sampler: sampler;

struct FroxelData {
    density: f32,
    sun_light: f32,
}
fn get_froxel_data(pos: vec3<f32>, view: View) -> FroxelData {
    let ndc = position_world_to_ndc(pos, view.clip_from_world);
    let uv = ndc_to_uv(ndc.xy);
    let view_z = depth_ndc_to_view_z(ndc.z, view.clip_from_view, view.view_from_clip);

    // Sample the texture
    let sample = textureSample(froxels_texture, froxels_sampler, vec3<f32>(uv, view_z));

    return FroxelData(sample.x, sample.y);
}

/// Convert a world space position to ndc space
fn position_world_to_ndc(world_pos: vec3<f32>, clip_from_world: mat4x4<f32>) -> vec3<f32> {
    let ndc_pos = clip_from_world * vec4(world_pos, 1.0);
    return ndc_pos.xyz / ndc_pos.w;
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2(0.5, -0.5) + vec2(0.5);
}

/// Retrieve the perspective camera near clipping plane
fn perspective_camera_near(clip_from_view: mat4x4<f32>) -> f32 {
    return clip_from_view[3][2];
}

/// Convert ndc depth to linear view z.
fn depth_ndc_to_view_z(ndc_depth: f32, clip_from_view: mat4x4<f32>, view_from_clip: mat4x4<f32>) -> f32 {
    return -perspective_camera_near(clip_from_view) / ndc_depth;
}
