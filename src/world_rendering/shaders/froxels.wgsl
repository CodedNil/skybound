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

    let uv = ndc.xy * 0.5 + 0.5;
    let view_pos = view.view_from_world * vec4<f32>(pos, 1.0);
    let uvw = vec3<f32>(
        uv.x,
        uv.y,
        clamp((length(view_pos.xyz) - CLIP_NEAR) / (CLIP_FAR - CLIP_NEAR), 0.0, 1.0)
    );

    let sample = textureSample(froxels_texture, froxels_sampler, uvw);

    return FroxelData(0.0, sample.r);
}
