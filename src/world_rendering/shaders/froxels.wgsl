#define_import_path skybound::froxels
#import skybound::utils::View

@group(0) @binding(3) var froxels_texture: texture_3d<f32>;
@group(0) @binding(4) var froxels_sampler: sampler;

const CLIP_NEAR: f32 = 0.1;
const CLIP_FAR: f32 = 1000.0;

struct FroxelData {
    density: f32,
    sun_light: f32,
}
fn get_froxel_data(pos: vec3<f32>, view: View) -> FroxelData {
    let clip = view.clip_from_world * vec4<f32>(pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * 0.5 + vec2<f32>(0.5);

    // Convert linear view-space depth to [0..1] Z tex coord using near/far
    let vz = -(view.view_from_world * vec4<f32>(pos, 1.0)).z;
    let log_z = log(vz / CLIP_NEAR) / log(CLIP_FAR / CLIP_NEAR);
    let zf = clamp(log_z, 0.0, 1.0);

    let sample = textureSample(froxels_texture, froxels_sampler, vec3<f32>(uv, zf));

    return FroxelData(sample.r, sample.g);
}
