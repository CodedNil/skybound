#import skybound::utils::View
#import skybound::raymarch::{VolumesInside, sample_volume}

@group(0) @binding(0) var output: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var linear_sampler: sampler;

const RESOLUTION = vec3<u32>(128, 80, 512);

const DENSITY: f32 = 0.05; // Base density for lighting
const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE = array<f32, 6>(30.0, 50.0, 80.0, 160.0, 300.0, 500.0);
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));
const MIN_SUN_DOT: f32 = sin(radians(-8.0));
const AUR_DIR: vec3<f32> = vec3(0.0, -1.0, 0.0);


@compute @workgroup_size(8, 8, 8)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    let view_pos_o = vec3<f32>(id) / vec3<f32>(RESOLUTION); // View space coordinates
    let view_pos = vec3<f32>(view_pos_o.x, view_pos_o.z, view_pos_o.y); // Y up

    let wp4 = view.world_from_view * vec4<f32>(view_pos, 1.0);
    let pos = wp4.xyz / wp4.w; // World space coordinates

    // Get sun direction and intensity, mix between aur light (straight up) and sun
    let sun_dot = view.sun_direction.y;
    let sun_t = clamp((sun_dot - MIN_SUN_DOT) / -MIN_SUN_DOT, 0.0, 1.0);
    let sun_dir = normalize(mix(AUR_DIR, view.sun_direction, sun_t));

    // Sample density
    let altitude = distance(pos, view.planet_center) - view.planet_radius;
    var volumes_inside: VolumesInside; // TODO simplify testing which volumes
    volumes_inside.clouds = true;
    volumes_inside.fog = true;
    volumes_inside.poles = true;
    // let density = sample_volume(vec3<f32>(pos.x, altitude, pos.z), view_pos.z, view.time, volumes_inside, true, linear_sampler).density;

    // Lightmarching for self-shadowing
    // var density_sunwards = max(density, 0.0);
    // var lightmarch_pos = pos;
    // var light_altitude: f32;
    // for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
    //     lightmarch_pos += (sun_dir + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE[j];
    //     light_altitude = distance(lightmarch_pos, view.planet_center) - view.planet_radius;
    //     density_sunwards += sample_volume(vec3<f32>(lightmarch_pos.x, light_altitude, lightmarch_pos.z), view_pos.z, view.time, volumes_inside, true, linear_sampler).density;
    // }

    // // Captures the direct lighting from the sun
    // let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE[1]);
    // let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE[1] * 0.25) * 0.7;
    // let beers_total = max(beers, beers2);
    let density = 0.0;
    let beers_total = 0.0;

    textureStore(output, vec3<i32>(id), vec4<f32>(density, beers_total, 0.0, 0.0));
}
