struct Parameters {
    world_from_clip: mat4x4<f32>,
    origin: vec4<f32>,
    // Full size, whether history is usable, and which pixel of each 2x2
    // block the volume traced this frame.
    size: vec4<f32>,
    native: vec4<f32>,
}
@group(0) @binding(0) var<uniform> parameters: Parameters;
// Traced at one pixel of each 2x2 block.
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

fn block_offset() -> vec2<i32> {
    switch u32(parameters.size.w) & 3u {
        case 0u: { return vec2<i32>(0, 0); }
        case 1u: { return vec2<i32>(1, 1); }
        case 2u: { return vec2<i32>(1, 0); }
        default: { return vec2<i32>(0, 1); }
    }
}

// Rebuild the volume at full resolution: the pixel traced this frame keeps
// its fresh value, softened by history; the other three come from where
// history saw them, held within the range of the fresh blocks around them.
@fragment
fn reconstruct(@location(0) uv: vec2<f32>) -> FilterOutput {
    let size = vec2<i32>(parameters.size.xy);
    let blocks = (size + 1) / 2;
    let p = clamp(vec2<i32>(uv * parameters.size.xy), vec2<i32>(0), size - 1);
    let b = p / 2;
    let sample_data = textureLoad(metadata, b, 0);
    let fresh = textureLoad(volume, b, 0);
    var minimum = fresh;
    var maximum = fresh;
    var filtered = vec4<f32>(0.0);
    var total = 0.0;
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let neighbour = clamp(b + vec2<i32>(x, y), vec2<i32>(0), blocks - 1);
            let geometry = textureLoad(metadata, neighbour, 0).y;
            if abs(geometry - sample_data.y) <= max(10.0, sample_data.y * 0.04) {
                let weight = f32((2 - abs(x)) * (2 - abs(y)));
                let value = textureLoad(volume, neighbour, 0);
                filtered += value * weight;
                total += weight;
                minimum = min(minimum, value);
                maximum = max(maximum, value);
            }
        }
    }
    filtered /= max(total, 1.0);
    let traced = all(b * 2 + block_offset() == p);
    let current = select(filtered, mix(fresh, filtered, 0.25), traced);
    let previous_uv = uv - textureLoad(motion, b, 0).xy;
    var usable = false;
    var old = current;
    if parameters.size.z > 0.5 && all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0)) {
        old = textureSampleLevel(history, linear_sampler, previous_uv, 0.0);
        let previous_p = clamp(vec2<i32>(previous_uv * parameters.size.xy), vec2<i32>(0), size - 1);
        let old_meta = textureLoad(history_metadata, previous_p, 0);
        let difference = abs(sample_data.xy - old_meta.xy);
        let tolerance = max(vec2<f32>(1500.0, 10.0), sample_data.xy * vec2<f32>(0.35, 0.04));
        let thin_edge = max(current.a, old.a) > 0.8;
        usable = difference.y < tolerance.y && (difference.x < tolerance.x || thin_edge);
    }
    if !usable {
        return FilterOutput(current, sample_data);
    }
    old = clamp(old, minimum, maximum);
    // A traced pixel takes most of its fresh value; the others rely on
    // history, refreshed every fourth frame.
    let weight = select(0.9, 0.35, traced);
    return FilterOutput(mix(current, old, weight), sample_data);
}
