#import skybound::functions::{remap, perlin_fbm31_tiled, worley_fbm31_tiled}

@group(0) @binding(0) var output: texture_storage_3d<rgba32float, write>;

const RESOLUTION_XZ: f32 = 1024.0;
const RESOLUTION_Y: f32 = 128.0;

@compute @workgroup_size(4, 4, 4)
fn generate_noise(@builtin(global_invocation_id) id: vec3<u32>) {
    let pos = vec3<f32>(id) / vec3<f32>(RESOLUTION_XZ, RESOLUTION_XZ, RESOLUTION_Y);

    let freq = 4.0;

    var pfbm = mix(1.0, perlin_fbm31_tiled(pos, 4.0, 7u), 0.5);
    pfbm = abs(pfbm * 2.0 - 1.0);

    let worley1 = worley_fbm31_tiled(pos, freq);
    let worley2 = worley_fbm31_tiled(pos, freq * 2.0);
    let worley3 = worley_fbm31_tiled(pos, freq * 4.0);
    let perlin_worley = remap(pfbm, 0.0, 1.0, worley1, 1.0);

    let color = vec4<f32>(perlin_worley, worley1, worley2, worley3);
    textureStore(output, vec3<i32>(id), color);
}
