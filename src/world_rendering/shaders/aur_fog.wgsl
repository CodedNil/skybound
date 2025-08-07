#define_import_path skybound::aur_fog
#import skybound::functions::hash12
#import skybound::sky::AtmosphereColors

@group(0) @binding(7) var fog_noise_texture: texture_3d<f32>;

// Turbulence parameters for fog
const COLOR_A: vec3<f32> = vec3(0.6, 0.4, 1.0);
const COLOR_B: vec3<f32> = vec3(0.4, 0.1, 0.6);
const COLOR_C: vec3<f32> = vec3(0.0, 0.0, 1.0);
const ROTATION_MATRIX = mat2x2<f32>(vec2<f32>(0.6, -0.8), vec2<f32>(0.8, 0.6));
const TURB_AMP: f32 = 0.6; // Turbulence amplitude
const TURB_SPEED: f32 = 0.5; // Turbulence speed
const TURB_FREQ: f32 = 0.4; // Initial turbulence frequency
const TURB_EXP: f32 = 1.6; // Frequency multiplier per iteration

// Fog lightning parameters
const FLASH_FREQUENCY: f32 = 0.05; // Chance of a flash per second per cell
const FLASH_GRID: f32 = 5000.0; // Grid cell size
const FLASH_POINTS: i32 = 4; // How many points per cell
const FLASH_COLOR: vec3<f32> = vec3(0.6, 0.6, 1.0) * 2.0;
const FLASH_SCALE: f32 = 0.002;
const FLASH_DURATION: f32 = 2.0; // Seconds
const FLASH_FLICKER_SPEED: f32 = 20.0; // Hz of the on/off cycles

// Raymarcher Parameters
const ALPHA_THRESHOLD: f32 = 0.95; // Max alpha to reach before stopping

const RAYMARCH_START_HEIGHT: f32 = 0.0;
const MAX_STEPS: i32 = 512;
const STEP_SIZE: f32 = 32.0;
const FAST_RENDER_DISTANCE: f32 = 5000000.0; // How far to render before switching to fast mode

const LIGHT_STEPS: u32 = 6; // How many steps to take along the sun direction
const LIGHT_STEP_SIZE: f32 = 30.0;
const LIGHT_RANDOM_VECTORS = array<vec3<f32>, 6>(vec3(0.38051305, 0.92453449, -0.02111345), vec3(-0.50625799, -0.03590792, -0.86163418), vec3(-0.32509218, -0.94557439, 0.01428793), vec3(0.09026238, -0.27376545, 0.95755165), vec3(0.28128598, 0.42443639, -0.86065785), vec3(-0.16852403, 0.14748697, 0.97460106));

// Lighting Parameters
const SUN_COLOR: vec3<f32> = vec3(0.99, 0.97, 0.96);
const SHADOW_EXTINCTION: f32 = 5.0; // Higher = deeper core shadows
const DENSITY: f32 = 0.05; // Base density for lighting


// Fog turbulence and lightning calculations
fn compute_turbulence(initial_pos: vec2<f32>, iters: f32, time: f32) -> vec2<f32> {
    var pos = initial_pos;
    var freq = TURB_FREQ;
    var rot = ROTATION_MATRIX;
    for (var i = 0.0; i < iters; i += 1.0) {
        // Compute phase using rotated y-coordinate, time, and iteration offset
        let phase = freq * (pos * rot).y + TURB_SPEED * time + i;
        pos += TURB_AMP * rot[0] * sin(phase) / freq; // Add perpendicular sine offset
        rot *= ROTATION_MATRIX; // Rotate for next iteration
        freq *= TURB_EXP; // Increase frequency
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
            if hash12(seed + flicker_t).x > 0.5 && hash12(seed).x <= FLASH_FREQUENCY {
                for (var k = 0; k < FLASH_POINTS; k++) {
                    let off_seed = seed + f32(k) * 17.0;
                    let h = hash12(off_seed);
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
    fog_height: f32,
    density: f32,
    color: vec3<f32>,
    emission: vec3<f32>,
}
fn sample_fog(pos: vec3<f32>, dist: f32, time: f32, fast: bool, very_fast: bool, linear_sampler: sampler) -> FogSample {
    var sample: FogSample;
    if pos.y > 0.0 { return sample; }

    let height_noise = sample_height(pos.xz, time, linear_sampler);
    sample.fog_height = height_noise;
    let altitude = pos.y - height_noise;
    let density = smoothstep(0.0, -100.0, altitude);
    if density <= 0.0 { return sample; }

    // Use turbulent position for density
    var fbm_value: f32;
    if very_fast {
        sample.density = 1.0;
    } else {
        let turb_iters = round(mix(4.0, 8.0, smoothstep(10000.0, 1000.0, dist)));
        let turb_pos: vec2<f32> = compute_turbulence(pos.xz * 0.01 + vec2(time, 0.0), turb_iters, time);
        fbm_value = sample_texture(vec3(turb_pos.x, turb_pos.y, altitude * 0.01) * 0.1, linear_sampler).g;
        sample.density = pow(fbm_value, 2.0) * density + smoothstep(-50.0, -200.0, altitude);
    }
    if sample.density <= 0.0 { return sample; }
    if fast { return sample; }

    // Compute fog color based on turbulent flow, with a larger scale noise for color variation
    let color_noise = sample_texture(vec3(pos.xz * 0.0001, 0.0), linear_sampler).b;
    if very_fast {
        sample.color = mix(COLOR_A, COLOR_B, color_noise);
    } else {
        sample.color = mix(COLOR_A, COLOR_B, fbm_value * 0.4 + color_noise * 0.6);
    }
    sample.emission = sample.density * smoothstep(-5.0, -100.0, altitude) * COLOR_C * 0.1;

    // Apply artificial shadowing: darken towards black as altitude decreases
    let shadow_factor = 1.0 - smoothstep(0.0, -100.0, altitude);
    sample.color = mix(sample.color * 0.1, sample.color, shadow_factor);

    // Compute lightning emission using Voronoi grid
    sample.emission += flash_emission(pos, time) * smoothstep(-5.0, -100.0, altitude) * sample.density;

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

        // Reduce scaling when close to surfaces
        var step_scaler = 1.0;
        let close_threshold = STEP_SIZE * step_scaler;
        let distance_left = t_max - t;
        if distance_left < close_threshold {
            let norm = clamp(distance_left / close_threshold, 0.0, 1.0);
            step_scaler = mix(step_scaler, 0.5, 1.0 - norm);
        }
        var step = STEP_SIZE * step_scaler;

        // Sample the fog
        let pos = ro + rd * t;
        let fog_sample = sample_fog(pos, t, time, false, t > FAST_RENDER_DISTANCE, linear_sampler);
        let step_density = fog_sample.density;

        if step_density > 0.0 {
            let step_transmittance = exp(-DENSITY * step_density * step);
            let alpha_step = (1.0 - step_transmittance);

            // Lightmarching for self-shadowing
            var density_sunwards = max(step_density, 0.0);
            var lightmarch_pos = pos;
            for (var j: u32 = 0; j <= LIGHT_STEPS; j++) {
                lightmarch_pos += (sun_dir + LIGHT_RANDOM_VECTORS[j] * f32(j)) * LIGHT_STEP_SIZE;
                density_sunwards += sample_fog(lightmarch_pos, t, time, true, false, linear_sampler).density;
            }
            // Take a single distant sample
            lightmarch_pos += sun_dir * LIGHT_STEP_SIZE * 18.0;
            density_sunwards += pow(sample_fog(lightmarch_pos, t, time, true, false, linear_sampler).density, 1.5);

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
    return sample_texture(vec3<f32>(pos * 0.00004, time * 0.03), linear_sampler).r * -100.0;
}

fn sample_texture(pos: vec3<f32>, linear_sampler: sampler) -> vec3<f32> {
    return textureSample(fog_noise_texture, linear_sampler, pos).rgb;
}
