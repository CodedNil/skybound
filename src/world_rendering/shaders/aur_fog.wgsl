#define_import_path skybound::aur_fog
#import skybound::functions::{mod1, hash12, hash13}
#import skybound::sky::AtmosphereColors

@group(0) @binding(7) var fog_noise_texture: texture_3d<f32>;

// Colours
const COLOR_A: vec3<f32> = vec3(0.6, 0.3, 0.8);
const COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6);
const FLASH_COLOR: vec3<f32> = vec3(0.6, 0.6, 1.0) * 5.0;
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping
const DENSITY: f32 = 0.05; // Base density for lighting

const RAYMARCH_START_HEIGHT: f32 = 0.0;
const MAX_STEPS: i32 = 512;
const STEP_SIZE: f32 = 48.0;
const FAST_RENDER_START: f32 = 20000.0; // Distance to begin fading to fast mode
const FAST_RENDER_END: f32 = 100000.0; // Distance to switch to fast mode

const STEP_SCALING_START: f32 = 1000.0; // Distance from camera to start scaling step size
const STEP_SCALING_END: f32 = 50000.0; // Distance from camera to use max step size
const STEP_SCALING_MAX: f32 = 4.0; // Maximum scaling factor to increase by

const LIGHT_STEPS: u32 = 2; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 30.0;
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));


// Fog turbulence calculations
const ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const TURB_ROTS: array<mat2x2<f32>, 8> = array<mat2x2<f32>,8>(
    mat2x2<f32>(vec2<f32>( 0.600, -0.800), vec2<f32>( 0.800,  0.600)),
    mat2x2<f32>(vec2<f32>(-0.280, -0.960), vec2<f32>( 0.960, -0.280)),
    mat2x2<f32>(vec2<f32>(-0.936, -0.352), vec2<f32>( 0.352, -0.936)),
    mat2x2<f32>(vec2<f32>(-0.843,  0.538), vec2<f32>(-0.538, -0.843)),
    mat2x2<f32>(vec2<f32>(-0.076,  0.997), vec2<f32>(-0.997, -0.076)),
    mat2x2<f32>(vec2<f32>( 0.752,  0.659), vec2<f32>(-0.659,  0.752)),
    mat2x2<f32>(vec2<f32>( 0.978, -0.206), vec2<f32>( 0.206,  0.978)),
    mat2x2<f32>(vec2<f32>( 0.422, -0.907), vec2<f32>( 0.907,  0.422))
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
const FLASH_FREQUENCY: f32 = 0.02; // Chance per cell per second
const FLASH_DURATION_MIN: f32 = 2.0; // Seconds per cycle
const FLASH_DURATION_MAX: f32 = 8.0; // Seconds per cycle
const FLASH_FLICKER: f32 = 8.0; // Hz on/off
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
fn flash_emission(pos: vec3<f32>, time: f32) -> vec3<f32> {
    // Cell coords and fractional part
    let uv = pos.xz / FLASH_GRID;
    let cell = floor(uv);
    var total = vec3<f32>(0.0);

    // For each of the 9 cells around pos
    for (var c = 0u; c < 9u; c++) {
        let cell_pos = cell + CELL_OFFSETS[c];
        let seed = dot(cell_pos, vec2<f32>(127.1, 311.7));

        // Check if this cell's flash is currently active
        let period = 1.0 / FLASH_FREQUENCY;
        let h = hash12(seed);
        let start_time = h.x * period;
        let tmod = mod1(time - start_time, period);
        let duration_jitter = mix(FLASH_DURATION_MIN, FLASH_DURATION_MAX, h.y);
        if (tmod > duration_jitter) { continue; }

        // Poisson-disc points
        for (var i = 0u; i < POISSON_SAMPLES; i++) {
            // Per-point randoms
            let h = hash13(seed + f32(i) * 17.0);
            let offset = h.x; // Random phase offset
            let rate = mix(0.8, 1.2, h.y); // Small flicker-rate jitter
            let flash_scale_jitter = mix(0.5, 3.0, h.z); // Use h.z to vary the scale

            // Apply individual flicker
            let flicker_time = time * FLASH_FLICKER * rate + offset * 100.0;
            let on = step(fract(flicker_time), 0.5); // ON half the time
            if (on < 0.5) { continue; }

            // Calculate the position of the poisson disc point
            let jit = POISSON_OFFSETS[i] * h.x;
            let gp = (cell_pos + jit) * FLASH_GRID;

            let d = distance(pos.xz, gp);
            let scaled_distance = d * FLASH_SCALE * flash_scale_jitter;
            total += exp(-scaled_distance) * FLASH_COLOR;
        }
    }

    return total;
}

/// Sample from the fog
struct FogSample {
    fog_height: f32,
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_fog(pos: vec3<f32>, dist: f32, time: f32, only_density: bool, fast: f32, linear_sampler: sampler) -> FogSample {
    var sample: FogSample;
    if pos.y > 0.0 { return sample; }

    let height_noise = sample_height(pos.xz, time, linear_sampler);
    sample.fog_height = height_noise;
    let altitude = pos.y - height_noise;
    let density = smoothstep(0.0, -500.0, altitude);
    if density <= 0.0 { return sample; }

    // Use turbulent position for density
    var fbm_value: f32 = -1.0;
    if fast >= 1.0 || density >= 1.0 {
        sample.density = 1.0;
    } else {
        let turb_iters = i32(round(mix(2.0, 6.0, smoothstep(50000.0, 1000.0, dist))));
        let turb_pos: vec2<f32> = compute_turbulence(pos.xz * 0.01 + vec2(time, 0.0), turb_iters, time);
        fbm_value = mix(sample_texture(vec3(turb_pos.xy * 0.1, altitude * 0.001), linear_sampler).g * 2.0 - 1.0, fbm_value, fast);
        sample.density = min(pow(fbm_value, 2.0) * density + smoothstep(-50.0, -1000.0, altitude) + fast, 1.0);
    }
    if only_density || sample.density <= 0.0 { return sample; }

    // Compute fog color based on turbulent flow
    sample.color = mix(COLOR_A, COLOR_B, fbm_value);
    // Apply artificial shadowing: darken towards black as altitude decreases
    let shadow_factor = 1.0 - smoothstep(-30.0, -500.0, altitude);
    sample.color = mix(sample.color * 0.1, sample.color, shadow_factor);

    // Add emission from the fog color and lightning flashes
    sample.emission = sample.density * smoothstep(-100.0, -500.0, altitude) * sample.color * 0.5;
    if fast <= 1.0 {
        let flash = flash_emission(pos, time);
        sample.emission += flash * smoothstep(-300.0, -500.0, altitude) * sample.density * (1.0 - fast);
    }

    return sample;
}


fn render_fog(ro: vec3<f32>, rd: vec3<f32>, atmosphere_colors: AtmosphereColors, sun_dir: vec3<f32>, t_max: f32, dither: f32, time: f32, linear_sampler: sampler) -> vec4<f32> {
    if ro.y > RAYMARCH_START_HEIGHT && rd.y > 0.0 { return vec4<f32>(0.0); } // Early exit if ray will never enter the fog

    // Start raymarching at y=RAYMARCH_START_HEIGHT intersection if above y=0, else at camera
    var t = select(0.0, (RAYMARCH_START_HEIGHT - ro.y) / rd.y, ro.y > RAYMARCH_START_HEIGHT) + dither * STEP_SIZE;
    // var t = dither * STEP_SIZE;
    var t_end = t_max;

    // Accumulation variables
    var acc_color = vec3(0.0);
    var acc_alpha = 0.0;
    var transmittance = 1.0;

    // Start raymarching
    for (var i = 0; i < MAX_STEPS; i += 1) {
        if t >= t_end || acc_alpha >= ALPHA_THRESHOLD { break; }

        // Scale step size based on distance from camera
        var step_scaler = 1.0;
        if t > STEP_SCALING_START {
            step_scaler = 1.0 + smoothstep(STEP_SCALING_START, STEP_SCALING_END, t) * STEP_SCALING_MAX;
        }
        // Reduce scaling when close to surfaces
        let close_threshold = STEP_SIZE * step_scaler;
        let distance_left = t_max - t;
        if distance_left < close_threshold {
            let norm = clamp(distance_left / close_threshold, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }
        var step = STEP_SIZE * step_scaler;

        // Sample the fog
        let pos = ro + rd * (t + dither * step);
        let fog_sample = sample_fog(pos, t, time, false, smoothstep(FAST_RENDER_START, FAST_RENDER_END, t), linear_sampler);
        let step_density = fog_sample.density;

        if step_density > 0.0 {
            let step_transmittance = exp(-DENSITY * step_density * step);
            let alpha_step = (1.0 - step_transmittance);

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var lightmarch_pos = pos;
            for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
                lightmarch_pos += (sun_dir + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE;
                density_sunwards += sample_fog(lightmarch_pos, t, time, true, 0.0, linear_sampler).density;
            }

            // Captures the direct lighting from the sun
			let beers = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE);
			let beers2 = exp(-DENSITY * density_sunwards * LIGHT_STEP_SIZE * 0.25) * 0.7;
			let beers_total = max(beers, beers2);

			// Compute in-scattering
            let ambient = atmosphere_colors.ground * DENSITY * mix(atmosphere_colors.ambient, vec3(1.0), 0.4) * (sun_dir.y);
            let in_scattering = ambient + beers_total * atmosphere_colors.sun * atmosphere_colors.phase;

			acc_alpha += alpha_step * (1.0 - acc_alpha);
			acc_color += in_scattering * transmittance * alpha_step * fog_sample.color + fog_sample.emission;

			transmittance *= step_transmittance;
        }

        t += step;

        if pos.y > RAYMARCH_START_HEIGHT { break; } // End early if we're above the fog
    }

    acc_alpha = min(acc_alpha * (1.0 / ALPHA_THRESHOLD), 1.0); // Scale alpha so ALPHA_THRESHOLD becomes 1.0

    return clamp(vec4(acc_color, acc_alpha), vec4(0.0), vec4(1.0));
}

fn sample_height(pos: vec2<f32>, time: f32, linear_sampler: sampler) -> f32 {
    return sample_texture(vec3<f32>(pos * 0.00004, time * 0.01), linear_sampler).r * -300.0;
}

fn sample_texture(pos: vec3<f32>, linear_sampler: sampler) -> vec2<f32> {
    return textureSample(fog_noise_texture, linear_sampler, pos).rg;
}
