#define_import_path skybound::clouds
#import skybound::utils::{View, intersect_sphere}

const BASE_SCALE = 0.005;
const BASE_TIME = 0.01;

// Base Parameters
const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.2) * 0.1 * BASE_TIME; // Main wind for base shape

// Detail Parameters
const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: vec3<f32> = vec3<f32>(1.0, 0.0, -1.0) * 0.2 * BASE_TIME; // Details move faster


// --- Cloud packed as four u32s matching the CPU layout ---
// data_a: len(14) | x_pos (signed 18)
// data_b: height(14) | y_pos (signed 18)
// data_c: seed(7)| width(5)| form(2)| density(4)| detail(4)| brightness(4)| yaw(6)
// data_d: altitude(15)
struct Cloud {
    data_a: u32,
    data_b: u32,
    data_c: u32,
    data_d: u32,
}

// Bit layout constants
const SEED_BITS: u32 = 7u;
const WIDTH_BITS: u32 = 5u;
const FORM_BITS: u32 = 2u;
const DENSITY_BITS: u32 = 4u;
const DETAIL_BITS: u32 = 4u;
const BRIGHTNESS_BITS: u32 = 4u;
const YAW_BITS: u32 = 6u;
const ALT_BITS: u32 = 15u;

// size bits
const SIZE_LEN_BITS: u32 = 14u;
const SIZE_HEIGHT_BITS: u32 = 14u;
const SIZE_WIDTH_BITS: u32 = WIDTH_BITS;

// shifts
const SIZE_LEN_SHIFT: u32 = 0u;
const X_SHIFT: u32 = SIZE_LEN_SHIFT + SIZE_LEN_BITS; // in data_a

const SIZE_HEIGHT_SHIFT: u32 = 0u;
const Y_SHIFT: u32 = SIZE_HEIGHT_SHIFT + SIZE_HEIGHT_BITS; // in data_b

// data_c shifts
const SEED_SHIFT: u32 = 0u;
const WIDTH_SHIFT: u32 = SEED_SHIFT + SEED_BITS; // 7
const FORM_SHIFT: u32 = WIDTH_SHIFT + SIZE_WIDTH_BITS; // 12
const DENSITY_SHIFT: u32 = FORM_SHIFT + FORM_BITS; // 14
const DETAIL_SHIFT: u32 = DENSITY_SHIFT + DENSITY_BITS; // 18
const BRIGHTNESS_SHIFT: u32 = DETAIL_SHIFT + DETAIL_BITS; // 22
const YAW_SHIFT: u32 = BRIGHTNESS_SHIFT + BRIGHTNESS_BITS; // 26

// altitude in data_d
const ALT_SHIFT: u32 = 0u;

// masks
const SIZE_LEN_MASK: u32 = (1u << SIZE_LEN_BITS) - 1u;
const SIZE_HEIGHT_MASK: u32 = (1u << SIZE_HEIGHT_BITS) - 1u;
const SIZE_WIDTH_MASK: u32 = (1u << SIZE_WIDTH_BITS) - 1u;
const SEED_MASK: u32 = (1u << SEED_BITS) - 1u;
const FORM_MASK: u32 = (1u << FORM_BITS) - 1u;
const DENSITY_MASK: u32 = (1u << DENSITY_BITS) - 1u;
const DETAIL_MASK: u32 = (1u << DETAIL_BITS) - 1u;
const BRIGHTNESS_MASK: u32 = (1u << BRIGHTNESS_BITS) - 1u;
const YAW_MASK: u32 = (1u << YAW_BITS) - 1u;
const ALT_MASK: u32 = (1u << ALT_BITS) - 1u;

// Precomputed float inverse of the width max so shader does one multiply instead of a division
const SIZE_WIDTH_INV_F: f32 = 1.0 / f32(SIZE_WIDTH_MASK);

// Small inline helper using precomputed masks
fn get_bits_field(container: u32, mask: u32, shift: u32) -> u32 {
    return (container >> shift) & mask;
}

// Signed extractor for two's complement signed fields stored in bits
fn get_signed_field(container: u32, bits: u32, shift: u32) -> i32 {
    let raw = get_bits_field(container, (1u << bits) - 1u, shift);
    let sign_bit = 1u << (bits - 1u);
    if (raw & sign_bit) != 0u {
        return i32(raw) - i32(1u << bits);
    }
    return i32(raw);
}

// Decode cloud position
fn get_cloud_pos(c: Cloud) -> vec3<f32> {
    let alt_raw = get_bits_field(c.data_d, ALT_MASK, ALT_SHIFT);
    let x_raw = get_signed_field(c.data_a, 18u, X_SHIFT);
    let y_raw = get_signed_field(c.data_b, 18u, Y_SHIFT);
    return vec3<f32>(f32(x_raw), f32(y_raw), f32(alt_raw));
}

// Decode cloud scale as a vec3<f32>
fn get_cloud_scale(c: Cloud) -> vec3<f32> {
    let len_raw = get_bits_field(c.data_a, SIZE_LEN_MASK, SIZE_LEN_SHIFT);
    let width_raw = get_bits_field(c.data_c, SIZE_WIDTH_MASK, WIDTH_SHIFT);
    let height_raw = get_bits_field(c.data_b, SIZE_HEIGHT_MASK, SIZE_HEIGHT_SHIFT);
    let length_m = f32(len_raw);
    let width_m = length_m * (f32(width_raw) * SIZE_WIDTH_INV_F);
    let height_m = f32(height_raw);
    return vec3<f32>(length_m, width_m, height_m);
}

// Get cloud yaw in radians (0..2PI)
const TWO_PI: f32 = 6.283185307179586;
fn get_cloud_yaw(c: Cloud) -> f32 {
    let yaw_raw = get_bits_field(c.data_c, YAW_MASK, YAW_SHIFT);
    return f32(yaw_raw) * (TWO_PI / f32(1u << YAW_BITS));
}

// Returns (near, far), the intersection points
fn cloud_intersect(ro: vec3<f32>, rd: vec3<f32>, cloud: Cloud) -> vec2<f32> {
    // Transform ray into the cloud local frame, applying yaw then scaling into unit-sphere space
    let center = get_cloud_pos(cloud);
    let scale = get_cloud_scale(cloud); // (length, width, height) in meters

    // Rotate world -> cloud local by -yaw (so cloud's local X aligns with unrotated axis)
    let yaw = get_cloud_yaw(cloud);
    let cy = cos(yaw);
    let sy = sin(yaw);

    let to_local = ro - center;
    // inverse rotation R(-yaw): [ c  s; -s  c ]
    let lx = cy * to_local.x + sy * to_local.y;
    let ly = -sy * to_local.x + cy * to_local.y;
    let lz = to_local.z;

    let dir_lx = cy * rd.x + sy * rd.y;
    let dir_ly = -sy * rd.x + cy * rd.y;
    let dir_lz = rd.z;

    // Scale to unit sphere based on half-extents
    let inv_radius = 1.0 / (scale * 0.65);
    let local_origin = vec3<f32>(lx, ly, lz) * inv_radius;
    let local_dir = vec3<f32>(dir_lx, dir_ly, dir_lz) * inv_radius;

    // Build the quadratic
    let a = dot(local_dir, local_dir);
    let b = dot(local_origin, local_dir);
    let c = dot(local_origin, local_origin) - 1.0;

    // If the ray origin is outside the sphere (c>0) and the ray is pointing away from it (b>0), there is no intersection
    if c > 0.0 && b > 0.0 {
        return vec2<f32>(1.0, 0.0); // No intersection
    }

    // Compute the discriminant
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return vec2<f32>(1.0, 0.0); // No real roots → no intersection
    }

    // Solve for the two roots
    let sqrt_disc = sqrt(disc);
    let inv_a = 1.0 / a;
    let near = (-b - sqrt_disc) * inv_a;
    let far = (-b + sqrt_disc) * inv_a;
    return vec2<f32>(near, far);
}

fn remap(v: f32, low: f32, high: f32, new_low: f32, new_high: f32) -> f32 {
    return new_low + (v - low) * (new_high - new_low) / (high - low);
}

// Sample a single cloud (ellipsoid) at a world position. Keeps the noise sampling simple
// and offsets the base noise by the cloud's seed so different clouds sample different regions.
fn sample_cloud(cloud: Cloud, pos: vec3<f32>, view: View, time: f32, simple: bool, base_texture: texture_3d<f32>, details_texture: texture_3d<f32>, linear_sampler: sampler) -> f32 {
    let center = get_cloud_pos(cloud);
    let scale = get_cloud_scale(cloud);
    let yaw = get_cloud_yaw(cloud);

    // Rotate world -> cloud local by -yaw
    let cy = cos(yaw);
    let sy = sin(yaw);
    let to_local = pos - center;
    let lx = cy * to_local.x + sy * to_local.y;
    let ly = -sy * to_local.x + cy * to_local.y;
    let lz = to_local.z;

    // 1. Seed and Wind
    let seed = get_bits_field(cloud.data_c, SEED_MASK, SEED_SHIFT);
    let seed_f = f32(seed);
    let seed_offset = vec3<f32>(seed_f * 0.123, seed_f * 0.456, seed_f * 0.789);
    let wind = WIND_DIRECTION_BASE * time;

    // 2. Coordinate Warping (Domain Warping)
    // This breaks the geometric silhouette by distorting the lookup space.
    let warp_coord = vec3<f32>(lx, ly, lz) * 0.003 + seed_offset + wind * 0.5;
    let warp = textureSampleLevel(base_texture, linear_sampler, warp_coord, 0.0).rgb * 2.0 - 1.0;

    // Distort the local position
    let distorted_p = vec3<f32>(lx, ly, lz) + warp * 120.0;
    let inv_radius = 1.0 / (scale * 0.5);
    let p = distorted_p * inv_radius;

    // 3. Height-based shaping
    let height_grad = p.z * 0.5 + 0.5;

    // Dramatic flat bottom and billowy top
    // We use a power function on the top falloff to make it more "dome-like"
    let bottom_shape = smoothstep(0.0, 0.2, height_grad);
    let top_shape = saturate(pow(1.0 - height_grad, 0.5) * 1.5);
    let vertical_shape = bottom_shape * top_shape;

    // Horizontal shape with noise-driven radius
    let horizontal_dist = length(p.xy);
    let horizontal_shape = smoothstep(1.0, 0.5, horizontal_dist);

    let container = vertical_shape * horizontal_shape;
    if container <= 0.01 { return 0.0; }

    // 4. Multi-scale Noise for Density
    let noise_coord = p * 0.6 + seed_offset + wind;
    let noise = textureSampleLevel(base_texture, linear_sampler, noise_coord, 0.0).r;

    // 5. Density Construction
    // Threshold increases with height to make the top more "shredded"
    let threshold = mix(0.1, 0.45, height_grad);
    var density = saturate(remap(noise * container, threshold, 1.0, 0.0, 1.0));
    density = smoothstep(0.0, 1.0, density);

    if simple { return density; }

    // 6. Detail Erosion
    let detail_coord = p * 3.5 + seed_offset * 0.2 + WIND_DIRECTION_DETAIL * time;
    let detail_noise = textureSampleLevel(details_texture, linear_sampler, detail_coord, 0.0).r;
    density = saturate(density - detail_noise * 0.4 * (1.0 - density));

    return density;
}
