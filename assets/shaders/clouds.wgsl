@group(0) @binding(0)
var<uniform> camera: Camera;
struct Camera {
    view_proj: mat4x4<f32>,
    world_position: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> mesh: Mesh;
struct Mesh {
    model: mat4x4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * mesh.model * vec4<f32>(input.position, 1.0);
    out.world_position = (mesh.model * vec4<f32>(input.position, 1.0)).xyz;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let camera_pos = camera.world_position.xyz;
    let ray_origin = camera_pos;
    let ray_direction = normalize(in.world_position - camera_pos);

    let cloud_center = vec3<f32>(0.0, 0.0, 0.0);
    let cloud_radius = 5.0;

    let oc = ray_origin - cloud_center;
    let a = dot(ray_direction, ray_direction);
    let b = 2.0 * dot(oc, ray_direction);
    let c = dot(oc, oc) - cloud_radius * cloud_radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        discard;
    }

    let sqrt_disc = sqrt(discriminant);
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);
    let t_min = max(0.0, min(t0, t1));
    let t_max = max(t0, t1);

    if t_min > t_max {
        discard;
    }

    let num_steps = 30u;
    let step_size = (t_max - t_min) / f32(num_steps);
    var acc_alpha = 0.0;
    var acc_color = vec3<f32>(0.0);

    for (var i = 0u; i < num_steps; i = i + 1u) {
        let t = t_min + f32(i) * step_size;
        let pos = ray_origin + ray_direction * t;
        let dist = length(pos - cloud_center);

        // Use smoothstep for a smooth fade from center to edge:
        // 0 opacity at radius, full "density" at 0
        let density = 1.0 - smoothstep(cloud_radius * 0.7, cloud_radius, dist);

        // Clamp density for safety (0 to 1)
        let clamped_density = clamp(density, 0.0, 1.0);

        // Alpha affected by density and step size. You can tweak 1.0 to adjust opacity
        let alpha = clamped_density * step_size * 1.0;

        let color = vec3<f32>(1.0, 1.0, 1.0); // white cloud

        // Front-to-back alpha blending
        acc_color += color * alpha * (1.0 - acc_alpha);
        acc_alpha += alpha * (1.0 - acc_alpha);

        if acc_alpha >= 0.95 {
            break;
        }
    }

    return vec4<f32>(acc_color, acc_alpha);
}
