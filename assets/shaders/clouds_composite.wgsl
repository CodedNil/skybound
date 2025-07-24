#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screenTexture: texture_2d<f32>; // Original scene color
@group(0) @binding(1) var cloudsTexture: texture_2d<f32>; // Low-res clouds with alpha
@group(0) @binding(2) var cloudsSampler: sampler;         // Sampler for cloudsTexture

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let pix = in.position.xy;

    // Sample original scene color and depth
    let sceneCol: vec3<f32> = textureLoad(screenTexture, vec2<i32>(pix), 0).xyz;

    // Sample low-resolution clouds texture
    let clouds = textureSample(cloudsTexture, cloudsSampler, uv);

    // Blend over the clouds
    let final_color = mix(sceneCol, clouds.xyz, clouds.a);

    return vec4<f32>(final_color, 1.0);
}
