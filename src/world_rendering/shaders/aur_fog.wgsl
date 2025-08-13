#define_import_path skybound::aur_fog
#import skybound::utils::{View, mod1, hash12, hash13, intersect_sphere}

const COLOR_A: vec3<f32> = vec3(0.6, 0.3, 0.8);
const COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6);
const FLASH_COLOR: vec3<f32> = vec3(0.6, 0.6, 1.0) * 5.0;
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);

const FOG_START_HEIGHT: f32 = 0.0;
const FOG_BOTTOM_HEIGHT: f32 = -500.0;


// Fog turbulence calculations
const ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const TURB_ROTS: array<mat2x2<f32>, 8> = array<mat2x2<f32>,8>(
    mat2x2<f32>(vec2<f32>(0.600, -0.800), vec2<f32>(0.800, 0.600)),
    mat2x2<f32>(vec2<f32>(-0.280, -0.960), vec2<f32>(0.960, -0.280)),
    mat2x2<f32>(vec2<f32>(-0.936, -0.352), vec2<f32>(0.352, -0.936)),
    mat2x2<f32>(vec2<f32>(-0.843, 0.538), vec2<f32>(-0.538, -0.843)),
    mat2x2<f32>(vec2<f32>(-0.076, 0.997), vec2<f32>(-0.997, -0.076)),
    mat2x2<f32>(vec2<f32>(0.752, 0.659), vec2<f32>(-0.659, 0.752)),
    mat2x2<f32>(vec2<f32>(0.978, -0.206), vec2<f32>(0.206, 0.978)),
    mat2x2<f32>(vec2<f32>(0.422, -0.907), vec2<f32>(0.907, 0.422))
);
const TURB_AMP: f32 = 0.6; // Turbulence amplitude
const TURB_SPEED: f32 = 0.5; // Turbulence speed
const TURB_FREQ: f32 = 0.4; // Initial turbulence frequency
const TURB_EXP: f32 = 1.6; // Frequency multiplier per iteration
fn compute_turbulence(initial_pos: vec2<f32>, iters: i32, time: f32) -> vec2<f32> {
    var pos = initial_pos;
    var freq = TURB_FREQ;
    for (var i = 0; i < iters; i++) {
        // Compute phase using rotated y-coordinate, time, and iteration offset
        let rot = TURB_ROTS[i];
        let phase = freq * (pos * rot).y + TURB_SPEED * time + f32(i);
        pos += TURB_AMP * rot[0] * sin(phase) / freq; // Add perpendicular sine offset
        freq *= TURB_EXP; // Increase frequency
    }

    return pos;
}

// Use Poisson disk sampling to create lightning flashes in a grid
const FLASH_GRID: f32 = 10000.0; // Grid cell size
const FLASH_FREQUENCY: f32 = 0.05; // Chance per cell per second
const FLASH_DURATION_MIN: f32 = 1.0; // Seconds per cycle
const FLASH_DURATION_MAX: f32 = 4.0; // Seconds per cycle
const FLASH_FLICKER: f32 = 120.0; // Hz on/off
const FLASH_SCALE: f32 = 0.002; // Fall-off

const POISSON_SAMPLES: u32 = 4u;
const POISSON_OFFSETS: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(0.0000, 0.5000), vec2<f32>(0.3621, 0.3536), vec2<f32>(0.4755, 0.1545), vec2<f32>(0.2939, -0.1545),
    vec2<f32>(0.0955, -0.4045), vec2<f32>(-0.0955, -0.4045), vec2<f32>(-0.2939, -0.1545), vec2<f32>(-0.4755, 0.1545),
    vec2<f32>(-0.3621, 0.3536), vec2<f32>(-0.0000, 0.0000), vec2<f32>(0.1545, 0.4755), vec2<f32>(0.4045, 0.0955),
    vec2<f32>(0.4045, -0.0955), vec2<f32>(0.1545, -0.4755), vec2<f32>(-0.1545, -0.4755), vec2<f32>(-0.4045, -0.0955)
);
const CELL_OFFSETS: array<vec2<f32>, 9> = array<vec2<f32>,9>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, -1.0)
);
fn flash_emission(pos: vec2<f32>, time: f32) -> vec3<f32> {
    // Cell coords and fractional part
    let uv = pos / FLASH_GRID;
    let cell = floor(uv);
    var total = vec3<f32>(0.0);
    let max_effective_dist = 1.0 / (FLASH_SCALE * 0.1); // Max possible scale

    // For each of the 9 cells around pos
    for (var c = 0u; c < 9u; c++) {
        let cell_pos = cell + CELL_OFFSETS[c];
        let seed = dot(cell_pos, vec2<f32>(127.1, 311.7));

        // Check if this cell's flash is currently active
        let period = 1.0 / FLASH_FREQUENCY;
        let h_cell = hash12(seed);
        let start_time = h_cell.x * period;
        let tmod = mod1(time - start_time, period);
        let duration_jitter = mix(FLASH_DURATION_MIN, FLASH_DURATION_MAX, h_cell.y);
        if tmod > duration_jitter { continue; }

        // Normalised flash life progress [0..1]
        let life = clamp(tmod / duration_jitter, 0.0, 1.0);

        // Poisson-disc points
        for (var i = 0u; i < POISSON_SAMPLES; i++) {
            // Per-point randoms
            let h = hash13(seed + f32(i) * 17.0);
            let offset = h.x; // Random phase offset
            let rate = mix(0.8, 1.2, h.y); // Small flicker-rate jitter

            // Calculate the position of the poisson disc point with subtle motion
            let motion = vec2<f32>(
                sin(time * 0.5 + h.x * 10.0),
                cos(time * 0.4 + h.y * 9.0)
            ) * 0.3;
            let jit = POISSON_OFFSETS[i] + motion;
            let gp = (cell_pos + jit) * FLASH_GRID;

            // Distance fall-off
            let d = distance(pos, gp);
            if d > max_effective_dist { continue; }

            // Animate scale
            let base_scale = mix(0.5, 3.0, h.z);
            let pulse = 0.5 + 0.5 * sin(life * 3.14159);
            let scaled_distance = d * FLASH_SCALE * base_scale * pulse;

            // Fade in/out
            let fade = smoothstep(0.0, 0.1, life) * (1.0 - smoothstep(0.9, 1.0, life));

            // Subtle flicker
            let flicker_time = time * FLASH_FLICKER * rate + offset * 100.0;
            let flicker = mix(0.8, 1.0, step(0.5, fract(flicker_time)));

            total += exp(-scaled_distance) * FLASH_COLOR * fade * flicker;
        }
    }

    return total;
}

/// Sample from the fog
struct FogSample {
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_fog(pos: vec3<f32>, dist: f32, time: f32, noise_texture: texture_3d<f32>, linear_sampler: sampler) -> FogSample {
    var sample: FogSample;

    if pos.y > FOG_START_HEIGHT { return sample; }

    let height_noise = textureSampleLevel(noise_texture, linear_sampler, vec3<f32>(pos.xz * 0.00004, time * 0.01,), 0.0).r * -300.0;
    let altitude = pos.y - height_noise;
    let density = smoothstep(0.0, -500.0, altitude);
    if density <= 0.0 { return sample; }

    // Use turbulent position for density
    var fbm_value: f32 = -1.0;
    if density >= 1.0 {
        sample.density = 1.0;
    } else {
        let turb_iters = i32(round(mix(2.0, 6.0, smoothstep(50000.0, 1000.0, dist))));
        let turb_pos: vec2<f32> = compute_turbulence(pos.xz * 0.01 + vec2<f32>(time, 0.0), turb_iters, time);
        fbm_value = textureSampleLevel(noise_texture, linear_sampler, vec3<f32>(turb_pos.xy * 0.1, altitude * 0.001), 0.0).g * 2.0 - 1.0;
        sample.density = pow(fbm_value, 2.0) * density + smoothstep(-50.0, -1000.0, altitude);
    }
    if sample.density <= 0.0 { return sample; }

    // Compute fog color based on turbulent flow
    sample.color = mix(COLOR_A, COLOR_B, fbm_value);
    // Apply artificial shadowing: darken towards black as altitude decreases
    let shadow_factor = 1.0 - smoothstep(-30.0, -500.0, altitude);
    sample.color = mix(sample.color * 0.1, sample.color, shadow_factor);

    // Add emission from the fog color and lightning flashes
    let emission_amount = smoothstep(-200.0, -1000.0, altitude);
    sample.emission = (sample.color * 0.5 + FLASH_COLOR * 0.5) * emission_amount * sample.density;
    sample.emission += flash_emission(pos.xz, time) * smoothstep(-100.0, -800.0, altitude) * sample.density;

    return sample;
}

// Returns vec2(entry_t, exit_t), or vec2(max, 0.0) if no hit
fn fog_raymarch_entry(ro: vec3<f32>, rd: vec3<f32>, view: View, t_max: f32) -> vec2<f32> {
    let cam_pos = vec3<f32>(0.0, view.planet_radius + ro.y, 0.0);
    let altitude = distance(ro, view.planet_center) - view.planet_radius;

    let shell_dist = intersect_sphere(cam_pos, rd, view.planet_radius + FOG_START_HEIGHT);

    var t_start: f32;
    var t_end: f32;
    if altitude > FOG_START_HEIGHT {
        // We are above the fog, only raymarch if the intersects the sphere, start at the shell_dist
        if shell_dist <= 0.0 { return vec2<f32>(t_max, 0.0); }
        t_start = shell_dist;
        t_end = t_max;
    } else {
        // We are inside the fog, start raymarching at the camera and end at the shell_dist
        t_start = 0.0;
        if shell_dist <= 0.0 {
            t_end = t_max;
        } else {
            t_end = min(shell_dist, t_max);
        }
    }

    return vec2<f32>(t_start, t_end);
}
