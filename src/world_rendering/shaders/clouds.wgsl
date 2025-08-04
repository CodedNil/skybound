#define_import_path skybound::clouds
#import skybound::functions::{remap, perlin_fbm31, worley_fbm21}

@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(4) var noise_texture: texture_3d<f32>;

fn sample_texture(pos: vec3<f32>) -> vec4<f32> {
    return textureSample(noise_texture, linear_sampler, vec3<f32>(pos.x, pos.z, pos.y * 0.125));
}

const COVERAGE = 0.5; // Adjust between 0 and 1 for desired cloud density

/// Sample from the clouds
fn sample_clouds(pos: vec3<f32>, dist: f32, time: f32) -> f32 {
    let altitude = pos.y;
    var sample: f32;

    // Height gradient
    let low_altitude = smoothstep(1400.0, 1500.0, altitude) * smoothstep(2500.0, 1600.0, altitude);
    let high_altitude = smoothstep(7800.0, 8000.0, altitude) * smoothstep(8500.0, 8000.0, altitude);
    let height_weight = clamp(low_altitude + high_altitude, 0.0, 1.0);
    if height_weight <= 0.0 { return sample; }

    let scaled_pos = pos * 0.00001;
    let perlin_worley = mix(-0.5, 0.5, sample_texture(scaled_pos * 0.5 - vec3(time * 0.0018)).r);

    let wfbm = (sample_texture(scaled_pos).g + perlin_worley) *
        		 sample_texture(scaled_pos + vec3(time * 0.001)).b *
                 sample_texture(scaled_pos + vec3(time * 0.002)).a;

    sample = remap(wfbm * height_weight, 0.4, 1.0, 0.0, 1.0);

    return sample;
}
