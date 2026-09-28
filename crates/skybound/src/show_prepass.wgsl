struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct ShowPrepassUniform {
    depth_power: f32,
    delta_time: f32,
    mode: u32,
}

@group(0) @binding(0) var<uniform> show_prepass: ShowPrepassUniform;
@group(0) @binding(1) var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(2) var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(3) var motion_vector_prepass_texture: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel_position = vec2<i32>(in.position.xy);

    switch show_prepass.mode {
        case 0u: {
            let raw_depth = textureLoad(depth_prepass_texture, pixel_position, 0);
            let depth = pow(raw_depth, show_prepass.depth_power);
            return vec4<f32>(depth, depth, depth, 1.0);
        }
        case 1u: {
            return textureLoad(normal_prepass_texture, pixel_position, 0);
        }
        default: {
            let motion = textureLoad(motion_vector_prepass_texture, pixel_position, 0).rg;
            return vec4<f32>(abs(motion) / show_prepass.delta_time, 0.0, 1.0);
        }
    }
}
