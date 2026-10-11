struct Parameters {
    world_from_clip: mat4x4<f32>,
    origin: vec4<f32>,
    size: vec4<f32>,
    native: vec4<f32>,
}
@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var volume: texture_2d<f32>;
@group(0) @binding(2) var motion: texture_2d<f32>;
@group(0) @binding(3) var metadata: texture_2d<f32>;
@group(0) @binding(4) var history: texture_2d<f32>;
@group(0) @binding(5) var history_metadata: texture_2d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;

struct FilterOutput {
    @location(0) color: vec4<f32>,
    @location(1) metadata: vec4<f32>,
}
@fragment
fn reconstruct(@location(0) uv: vec2<f32>) -> FilterOutput {
    let size = vec2<i32>(parameters.size.xy);
    let p = clamp(vec2<i32>(uv * parameters.size.xy), vec2<i32>(0), size - 1);
    let current = textureLoad(volume, p, 0);
    let sample_data = textureLoad(metadata, p, 0);
    let previous_uv = uv - textureLoad(motion, p, 0).xy;
    var weight = 0.0;
    var old = current;
    if parameters.size.z > 0.5 && all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0)) {
        old = textureSampleLevel(history, linear_sampler, previous_uv, 0.0);
        let previous_p = clamp(vec2<i32>(previous_uv * parameters.size.xy), vec2<i32>(0), size - 1);
        let old_meta = textureLoad(history_metadata, previous_p, 0);
        let difference = abs(sample_data.xy - old_meta.xy);
        let tolerance = max(vec2<f32>(10.0), sample_data.xy * 0.04);
        if all(difference < tolerance) {
            weight = 0.92;
            let luminance_change = abs(dot(old.rgb - current.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)));
            weight *= 1.0 - smoothstep(0.12, 0.65, luminance_change);
        }
    }
    var minimum = current;
    var maximum = current;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let neighbour = textureLoad(volume, clamp(p + vec2<i32>(x, y), vec2<i32>(0), size - 1), 0);
            minimum = min(minimum, neighbour); maximum = max(maximum, neighbour);
        }
    }
    old = clamp(old, minimum, maximum);
    return FilterOutput(mix(current, old, weight), sample_data);
}
