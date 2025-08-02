#define_import_path skybound::clouds
#import skybound::functions::{remap, perlin_fbm31, worley_fbm21}

@group(0) @binding(4) var noise_texture: texture_3d<f32>;
@group(0) @binding(5) var noise_sampler: sampler;

fn sample_texture(pos: vec3<f32>) -> vec4<f32> {
    return textureSample(noise_texture, noise_sampler, pos);
}

const COVERAGE = 0.5; // Adjust between 0 and 1 for desired cloud density

/// Sample from the clouds
fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32) -> f32 {
    let altitude = pos.y;
    var sample: f32;

    // Height gradient
    let height_weight = smoothstep(1800.0, 2000.0, altitude) * smoothstep(5000.0, 2000.0, altitude) + smoothstep(18800.0, 19000.0, altitude) * smoothstep(21000.0, 19000.0, altitude);
    if height_weight <= 0.0 { return sample; }

    // // Worley noise for base shape
    // let worley = worley_fbm21(pos.xz * 0.0005 + vec2(time * 0.02, 0.0)) * pow(height_weight, 5.0) - COVERAGE;
    // if worley <= -0.4 { return sample; }

    // // Sample puff noise
    // let perlin_octaves = u32(round(mix(3.0, 6.0, smoothstep(8000.0, 500.0, dist))));
    // var pfbm: f32 = mix(1.0, perlin_fbm31(pos * 0.003 + vec3(time * 0.1, 0.0, 0.0), perlin_octaves), 0.5);
    // pfbm = abs(pfbm * 2.0 - 1.0); // Billowy perlin noise

    // sample = clamp(worley + (worley + 0.4) * pfbm, 0.0, 1.0);

    let scaled_pos = pos * 0.00001;
    let perlin_worley = mix(-0.5, 0.5, textureSample(noise_texture, noise_sampler, scaled_pos * 0.5 - vec3(time * 0.0018)).r);

    let wfbm = (sample_texture(scaled_pos).g + perlin_worley) *
        		 sample_texture(scaled_pos + vec3(time * 0.01)).b *
                 sample_texture(scaled_pos + vec3(time * 0.02)).a;

    sample = remap(wfbm, 0.4, 1.0, 0.0, 1.0) * height_weight;

    return sample;
}
