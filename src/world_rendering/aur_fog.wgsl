#define_import_path skybound::aur_fog
#import skybound::functions::{fbm_3, hash_12}

// Turbulence parameters for fog
const COLOR_A: vec3<f32> = vec3(0.3, 0.2, 0.8); // Deep blue
const COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6); // Deep purple
const ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const TURB_AMP: f32 = 0.6; // Turbulence amplitude
const TURB_SPEED: f32 = 0.5; // Turbulence speed
const TURB_FREQ: f32 = 0.4; // Initial turbulence frequency
const TURB_EXP: f32 = 1.6; // Frequency multiplier per iteration

// Fog lightning parameters
const FLASH_FREQUENCY: f32 = 0.05; // Chance of a flash per second per cell
const FLASH_GRID: f32 = 5000.0; // Grid cell size
const FLASH_POINTS: i32 = 4; // How many points per cell
const FLASH_COLOR: vec3<f32> = vec3(0.6, 0.6, 1.0) * 20.0;
const FLASH_SCALE: f32 = 0.002;
const FLASH_DURATION: f32 = 2.0; // Seconds
const FLASH_FLICKER_SPEED: f32 = 20.0; // Hz of the on/off cycles

// Fog turbulence and lightning calculations
fn compute_turbulence(initial_pos: vec2<f32>, iters: f32, time: f32) -> vec2<f32> {
    var pos = initial_pos;
    var freq = TURB_FREQ;
    var rot = ROTATION_MATRIX;
    for (var i = 0.0; i < iters; i += 1.0) {
        // Compute phase using rotated y-coordinate, time, and iteration offset
        let phase = freq * (pos * rot).y + TURB_SPEED * time + i;
        pos = pos + TURB_AMP * rot[0] * sin(phase) / freq; // Add perpendicular sine offset
        rot = rot * ROTATION_MATRIX; // Rotate for next iteration
        freq = freq * TURB_EXP; // Increase frequency
    }

    return pos;
}

// Voronoi-style closest point calculation for fog
fn flash_emission(pos: vec3<f32>, time: f32) -> vec3<f32> {
    let cell = floor(pos.xz / FLASH_GRID);
    let t_block = floor(time / FLASH_DURATION);
    let in_dur = (time - t_block * FLASH_DURATION) < FLASH_DURATION;
    var emission = vec3(0.0);

    if !in_dur { return emission; }

    let flicker_t = floor(time * FLASH_FLICKER_SPEED);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let nbr = cell + vec2(f32(x), f32(y));
            let seed = dot(nbr, vec2(127.1, 311.7)) + t_block;

            // Per-neighbour flicker trigger
            if hash_12(seed + flicker_t).x > 0.5 && hash_12(seed).x <= FLASH_FREQUENCY {
                for (var k = 0; k < FLASH_POINTS; k++) {
                    let off_seed = seed + f32(k) * 17.0;
                    let h = hash_12(off_seed);
                    if h.y <= 0.3 { continue; }

                    let phase = time * (FLASH_FLICKER_SPEED * 0.5) + 6.2831 * h.x;
                    let offset = h * 0.5 + 0.5 * sin(phase);
                    let gp = (nbr + offset) * FLASH_GRID;
                    // Smooth 2D fall-off
                    let d = distance(pos.xz, gp);
                    emission += exp(-d * FLASH_SCALE) * FLASH_COLOR;
                }
            }
        }
    }

    return emission;
}

/// Sample from the fog
struct FogSample {
    contribution: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_fog(pos: vec3<f32>, dist: f32, time: f32) -> FogSample {
    var sample: FogSample;
    if pos.y > 1000.0 { return sample; }

    let height_noise = fbm_3(vec3(pos.xz * 0.0001, time * 0.2), 1);
    let altitude = pos.y - height_noise * 1000.0;
    let density = smoothstep(20.0, -100.0, altitude);

    if density > 0.0 {
        // Use turbulent position for density
        let turb_iters = round(mix(4.0, 8.0, smoothstep(10000.0, 1000.0, dist)));
        let turb_pos = compute_turbulence(pos.xz * 0.01 + vec2(time, 0.0), turb_iters, time);
        let fbm_octaves = u32(round(mix(3.0, 5.0, smoothstep(10000.0, 1000.0, dist))));
        let fbm_value = fbm_3(vec3(turb_pos.x, altitude * 0.05, turb_pos.y), fbm_octaves);
        sample.contribution = pow(fbm_value, 2.0) * density + smoothstep(-50.0, -200.0, altitude);

        if sample.contribution > 0.0 {
            // Compute fog color based on turbulent flow, with a larger scale noise for color variation
            let color_noise = fbm_3(vec3(pos.xz * 0.0001, 0.0), 3);
            sample.color = mix(COLOR_A, COLOR_B, fbm_value * 0.4 + color_noise * 0.6);
            sample.emission = sample.contribution * sample.color * 6.0;

            // Apply artificial shadowing: darken towards black as altitude decreases
            let shadow_factor = 1.0 - smoothstep(0.0, -100.0, altitude);
            sample.color = mix(sample.color * 0.1, sample.color, shadow_factor);

            // Compute lightning emission using Voronoi grid
            sample.emission += flash_emission(pos, time) * smoothstep(-5.0, -100.0, altitude) * sample.contribution;
        }
    }

    return sample;
}
