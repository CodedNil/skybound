#define_import_path skybound::aur_fog
#import skybound::functions::{fbm_3, hash_12}

// Turbulence parameters for fog
const FOG_COLOR_A: vec3<f32> = vec3(0.3, 0.2, 0.8); // Deep blue
const FOG_COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6); // Deep purple
const FOG_ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const FOG_TURB_ITERS: f32 = 8.0; // Number of turbulence iterations
const FOG_TURB_AMP: f32 = 0.6; // Turbulence amplitude
const FOG_TURB_SPEED: f32 = 0.5; // Turbulence speed
const FOG_TURB_FREQ: f32 = 0.4; // Initial turbulence frequency
const FOG_TURB_EXP: f32 = 1.6; // Frequency multiplier per iteration

// Fog lightning parameters
const FOG_FLASH_FREQUENCY: f32 = 0.05; // Chance of a flash per second per cell
const FOG_FLASH_GRID: f32 = 2000.0; // Grid cell size
const FOG_FLASH_POINTS: i32 = 4; // How many points per cell
const FOG_FLASH_COLOR: vec3<f32> = vec3(0.6, 0.6, 1.0) * 30.0;
const FOG_FLASH_SCALE: f32 = 0.01;
const FOG_FLASH_DURATION: f32 = 2.0; // Seconds
const FOG_FLASH_FLICKER_SPEED: f32 = 20.0; // Hz of the on/off cycles

// Fog turbulence and lightning calculations
fn fog_compute_turbulence(initial_pos: vec2<f32>, time: f32) -> vec2<f32> {
    var pos = initial_pos;
    var freq = FOG_TURB_FREQ;
    var rot = FOG_ROTATION_MATRIX;
    for (var i = 0.0; i < FOG_TURB_ITERS; i = i + 1.0) {
        // Compute phase using rotated y-coordinate, time, and iteration offset
        let phase = freq * (pos * rot).y + FOG_TURB_SPEED * time + i;
        pos = pos + FOG_TURB_AMP * rot[0] * sin(phase) / freq; // Add perpendicular sine offset
        rot = rot * FOG_ROTATION_MATRIX; // Rotate for next iteration
        freq = freq * FOG_TURB_EXP; // Increase frequency
    }

    return pos;
}

// Voronoi-style closest point calculation for fog
fn fog_flash_emission(pos: vec3<f32>, time: f32) -> vec3<f32> {
    let cell = floor(pos.xz / FOG_FLASH_GRID);
    let t_block = floor(time / FOG_FLASH_DURATION);
    let in_dur = (time - t_block * FOG_FLASH_DURATION) < FOG_FLASH_DURATION;
    var emission = vec3(0.0);

    if !in_dur { return emission; }

    let flicker_t = floor(time * FOG_FLASH_FLICKER_SPEED);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let nbr = cell + vec2(f32(x), f32(y));
            let seed = dot(nbr, vec2(127.1, 311.7)) + t_block;

            // Per-neighbour flicker trigger
            if hash_12(seed + flicker_t).x > 0.5 && hash_12(seed).x <= FOG_FLASH_FREQUENCY {
                for (var k = 0; k < FOG_FLASH_POINTS; k++) {
                    let off_seed = seed + f32(k) * 17.0;
                    let h = hash_12(off_seed);
                    if h.y <= 0.3 { continue; }

                    let phase = time * (FOG_FLASH_FLICKER_SPEED * 0.5) + 6.2831 * h.x;
                    let offset = h * 0.5 + 0.5 * sin(phase);
                    let gp = (nbr + offset) * FOG_FLASH_GRID;
                    // Smooth 2D fall-off
                    let d = distance(pos.xz, gp);
                    emission += exp(-d * FOG_FLASH_SCALE) * FOG_FLASH_COLOR;
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
    let altitude = pos.y;
    let fog_density = smoothstep(20.0, -100.0, altitude);

    if fog_density > 0.01 && pos.y > -250.0 {
        // Use turbulent position for density
        let turb_pos = fog_compute_turbulence(pos.xz * 0.01, time);
        let fbm_octaves = u32(round(mix(2.0, 5.0, smoothstep(10000.0, 1000.0, dist))));
        let fbm_value = fbm_3(vec3(turb_pos.x, pos.y * 0.05, turb_pos.y), fbm_octaves);
        sample.contribution = pow(fbm_value, 2.0) * fog_density + smoothstep(-50.0, -200.0, altitude);

        // Compute fog color based on turbulent flow
        sample.color = mix(FOG_COLOR_A, FOG_COLOR_B, fbm_value * 0.6);
        sample.emission = sample.contribution * sample.color * 6.0;

        // Apply artificial shadowing: darken towards black as altitude decreases
        let shadow_factor = 1.0 - smoothstep(0.0, -100.0, altitude);
        sample.color = mix(sample.color * 0.1, sample.color, shadow_factor);

        // Compute lightning emission using Voronoi grid
        sample.emission += fog_flash_emission(pos, time) * smoothstep(-10.0, -60.0, altitude) * sample.contribution;
    }

    return sample;
}
