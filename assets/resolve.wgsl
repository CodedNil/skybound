@group(0) @binding(0)
var color: texture_2d<f32>;
@group(0) @binding(1)
var motion: texture_2d<f32>;
@group(0) @binding(2)
var normal: texture_2d<f32>;
@group(0) @binding(3)
var depth: texture_depth_2d;
@group(0) @binding(4)
var linear_sampler: sampler;

struct Output {
    @location(0) color: vec4<f32>,
    @location(1) motion: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(@location(0) uv: vec2<f32>) -> Output {
    let size = textureDimensions(depth);
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), vec2<i32>(size) - 1);
    return Output(
            textureSample(color, linear_sampler, uv),
            textureLoad(motion, pixel, 0),
            textureLoad(normal, pixel, 0),
            textureLoad(depth, pixel, 0),
        );
}
