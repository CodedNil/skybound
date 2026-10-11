struct Parameters {
    world_from_clip: mat4x4<f32>, origin: vec4<f32>, size: vec4<f32>, native: vec4<f32>,
}
@group(0) @binding(0) var<uniform> parameters: Parameters;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var volume: texture_2d<f32>;
@group(0) @binding(3) var metadata: texture_2d<f32>;
@group(0) @binding(4) var depth: texture_depth_2d;
@group(0) @binding(5) var thin_color: texture_2d<f32>;
@group(0) @binding(6) var thin_transmission: texture_2d<f32>;
@group(0) @binding(7) var thin_prefix: texture_2d<f32>;

@fragment
fn composite(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let native_size = textureDimensions(scene);
    let p = clamp(vec2<i32>(uv * vec2<f32>(native_size)), vec2<i32>(0), vec2<i32>(native_size) - 1);
    let background = textureLoad(scene, p, 0).rgb;
    if parameters.native.z > 0.5 { return vec4<f32>(background, 1.0); }
    let d = textureLoad(depth, p, 0);
    let world = parameters.world_from_clip * vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), d, 1.0);
    let native_distance = select(1000000.0, length(world.xyz / world.w - parameters.origin.xyz), d > 0.0);
    let coordinates = uv * parameters.size.xy - 0.5;
    let base = vec2<i32>(floor(coordinates));
    let fraction = fract(coordinates);
    let size = vec2<i32>(parameters.size.xy);
    var accumulated = vec4<f32>(0.0); var total = 0.0;
    for (var y = 0; y < 2; y += 1) {
        for (var x = 0; x < 2; x += 1) {
            let q = clamp(base + vec2<i32>(x, y), vec2<i32>(0), size - 1);
            let sample_data = textureLoad(metadata, q, 0);
            let spatial = select(1.0 - fraction.x, fraction.x, x == 1) * select(1.0 - fraction.y, fraction.y, y == 1);
            let weight = spatial / (1.0 + abs(sample_data.y - native_distance) / max(2.0, native_distance * 0.02));
            accumulated += textureLoad(volume, q, 0) * weight; total += weight;
        }
    }
    let fog = accumulated / max(total, 0.000001);
    let thin = textureLoad(thin_color, p, 0);
    let transmission = textureLoad(thin_transmission, p, 0).rgb;
    if thin.a < 0.0 { return vec4<f32>(thin.rgb + transmission * background, 1.0); }
    let prefix = textureLoad(thin_prefix, p, 0);
    let remainder_color = max(vec3<f32>(0.0), fog.rgb - prefix.rgb) / max(prefix.a, 0.0001);
    let remainder_transmission = clamp(fog.a / max(prefix.a, 0.0001), 0.0, 1.0);
    return vec4<f32>(thin.rgb + transmission * (remainder_color + remainder_transmission * background), 1.0);
}
