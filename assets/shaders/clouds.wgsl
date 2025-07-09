#import bevy_render::view::View
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
struct VolumetricClouds {
    intensity: f32,
}
@group(0) @binding(2) var<uniform> settings: VolumetricClouds;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let offset_strength = settings.intensity;

    var col = textureSample(screen_texture, texture_sampler, in.uv);
    let offsetCol = textureSample(screen_texture, texture_sampler, in.uv + vec2<f32>(offset_strength, -offset_strength));

    if (length(col.rgb - offsetCol.rgb) > 0.1) {
        col = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return col;
}
