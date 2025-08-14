#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var cloudsTexture: texture_2d<f32>;
@group(0) @binding(1) var cloudsSampler: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let clouds = textureSample(cloudsTexture, cloudsSampler, in.uv);
    return vec4<f32>(clouds.xyz, 1.0);
}
