#import skybound::utils::View

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> view: View;

const RESOLUTION = vec3<u32>(128, 80, 512);

@compute @workgroup_size(8, 8, 8)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= RESOLUTION.x || id.y >= RESOLUTION.y || id.z >= RESOLUTION.z) {
        return;
    }
    let pos = vec3<f32>(id) / vec3<f32>(RESOLUTION);

    let color = vec4<f32>(pos.z, pos.x, pos.y, 0.0);
    textureStore(output, vec3<i32>(id), color);
}
