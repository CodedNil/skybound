use core::f32::consts::PI;
use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

// Precomputed Constants
const THREE_OVER_SIXTEEN_PI: f32 = 3.0 / (16.0 * PI);

// Atmosphere Physical Properties
const SUN_INTENSITY: f32 = 22.0;
const EXPOSURE: f32 = 3.0;

// Atmosphere dimensions (in meters)
const ATMOSPHERE_HEIGHT: f32 = 200_000.0;
const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0; // ~8km
const MIE_SCALE_HEIGHT: f32 = 1200.0; // ~1.2km

// Wavelength-dependent scattering coefficients (in meters^-1) for Rayleigh (air), Mie (aerosols), and Ozone
const C_RAYLEIGH: Vec3 = vec3(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE: Vec3 = vec3(3.996e-6, 3.996e-6, 3.996e-6);
const C_OZONE: Vec3 = vec3(0.650e-6, 1.881e-6, 0.085e-6);

// Tweakable Parameters
const RAYLEIGH_STRENGTH: f32 = 1.0;
const MIE_STRENGTH: f32 = 0.2;
const OZONE_STRENGTH: f32 = 1.0;
const MIE_G: f32 = 0.85; // Mie scattering directionality (-1 = backscatter, 0 = uniform, 1 = forward scatter)

// Raymarching settings
const PRIMARY_STEPS: i32 = 16; // Steps along the main view ray
const LIGHT_STEPS: i32 = 8; // Steps for the secondary light rays (to the sun)

// Planet emissive glow
const PLANET_EMISSION_COLOR: Vec3 = vec3(0.3, 0.15, 0.4); // Gentle purple
const PLANET_GLOW_ANGULAR_WIDTH: f32 = 0.03; // Radians, soft angular falloff
const PLANET_EMISSION_CHROMA_PRESERVE: f32 = 0.9;

// Returns the near (x) and far (y) intersection distances
fn intersect_sphere(ro: Vec3, rd: Vec3, radius: f32) -> Vec2 {
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(ro);
    let c = ro.dot(ro) - (radius * radius);
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return Vec2::splat(-1.0); // No real intersection
    }

    let sqrt_disc = disc.sqrt();
    vec2(-b - sqrt_disc, -b + sqrt_disc) / (2.0 * a)
}

// Rayleigh phase function
fn phase_rayleigh(cos_theta: f32) -> f32 {
    THREE_OVER_SIXTEEN_PI * (1.0 + cos_theta * cos_theta)
}

// Henyey-Greenstein phase function for Mie scattering, adapted from reference for robustness
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let numerator = (1.0 - g2) / (4.0 * PI);
    let denominator_term = 1.0 + g2 - 2.0 * g * cos_theta;
    let denominator = denominator_term.powf(1.5);
    numerator / denominator
}

// Get altitude of a point in world space
fn atmosphere_height(position: Vec3, planet_radius: f32) -> f32 {
    (position.length() - planet_radius).max(0.0)
}

// Density functions for each atmospheric component, based on altitude
fn density_rayleigh(h: f32) -> f32 {
    (h.max(0.0) / -RAYLEIGH_SCALE_HEIGHT).exp()
}

fn density_mie(h: f32) -> f32 {
    (h.max(0.0) / -MIE_SCALE_HEIGHT).exp()
}

// Ozone layer is concentrated around 25km
fn density_ozone(h: f32) -> f32 {
    (1.0 - (h - 25000.0).abs() / 15000.0).max(0.0)
}

// Combines densities into a single vector for easier calculation
fn density_atmosphere(h: f32) -> Vec3 {
    vec3(density_rayleigh(h), density_mie(h), density_ozone(h))
}

// Converts optical depth to transmittance via Beer-Lambert law
fn calculate_transmittance(optical_depth: Vec3) -> Vec3 {
    let mie_extinction = optical_depth.y * C_MIE * MIE_STRENGTH;
    let rayleigh_extinction = optical_depth.x * C_RAYLEIGH * RAYLEIGH_STRENGTH;
    let ozone_extinction = optical_depth.z * C_OZONE * OZONE_STRENGTH;

    let total_extinction = rayleigh_extinction + mie_extinction + ozone_extinction;
    (-total_extinction).exp()
}

// Calculates the optical depth (amount of atmosphere) along a ray
fn integrate_optical_depth(ray_start: Vec3, ray_dir: Vec3, view: &ViewUniform) -> Vec3 {
    let atmosphere_radius = view.planet_radius + ATMOSPHERE_HEIGHT;
    let intersection = intersect_sphere(ray_start, ray_dir, atmosphere_radius);
    let ray_length = intersection.y.max(0.0);
    let step_size = ray_length / LIGHT_STEPS as f32;
    let mut optical_depth = Vec3::ZERO;
    for i in 0..LIGHT_STEPS {
        let local_pos = ray_start + ray_dir * (i as f32 + 0.5) * step_size;
        let altitude = atmosphere_height(local_pos, view.planet_radius);
        if altitude <= ATMOSPHERE_HEIGHT {
            optical_depth += density_atmosphere(altitude) * step_size;
        }
    }
    optical_depth
}

// Smooth occlusion factor in [0,1]. Returns 1.0 for fully occluded, 0.0 for fully visible.
// This avoids hard binary transitions that create banding when sampling sparsely.
fn occlusion_factor_smooth(pos: Vec3, dir: Vec3, planet_radius: f32) -> f32 {
    // t_closest is the param (along dir) where the ray comes closest to the planet center
    let t_closest = -pos.dot(dir);
    if t_closest <= 0.0 {
        return 0.0; // closest approach is behind the start -> no occlusion in front
    }

    // distance from center to ray (closest approach)
    let closest_point = pos + dir * t_closest;
    let d_closest = closest_point.length();

    // penetration depth under the planetary radius
    let penetration = planet_radius - d_closest;
    if penetration <= 0.0 {
        return 0.0;
    }

    // softening width (meters) - smooth over a small region around the terminator
    let soft_width = 4000.0; // ~4 km soft transition to hide banding
    let f = (penetration / soft_width).saturate();
    // cubic smoothstep for nicer falloff
    f * f * (3.0 - 2.0 * f)
}

// Main function to calculate scattered light (in-scattering)
fn integrate_scattering(
    ray_start: Vec3,
    ray_dir: Vec3,
    ray_length: f32,
    sun_dir: Vec3,
    view: &ViewUniform,
) -> Vec3 {
    let cos_theta = ray_dir.dot(sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, MIE_G);
    let mut optical_depth_view = Vec3::ZERO;
    let mut rayleigh_scatter = Vec3::ZERO;
    let mut mie_scatter = Vec3::ZERO;

    // Non-uniform sampling: concentrate samples closer to the camera for better quality
    let ray_height = atmosphere_height(ray_start, view.planet_radius);
    let sample_distribution_exponent =
        1.0 + (1.0 - ray_height / ATMOSPHERE_HEIGHT).saturate() * 8.0;

    let mut prev_ray_time = 0.0;
    for i in 0..PRIMARY_STEPS {
        // Distribute samples non-linearly
        let ray_time =
            ((i + 1) as f32 / PRIMARY_STEPS as f32).powf(sample_distribution_exponent) * ray_length;
        let step_size = ray_time - prev_ray_time;
        let local_pos = ray_start + ray_dir * ray_time;
        let altitude = atmosphere_height(local_pos, view.planet_radius);
        if altitude > ATMOSPHERE_HEIGHT {
            continue;
        }

        let local_density = density_atmosphere(altitude);
        optical_depth_view += local_density * step_size;

        // Transmittance from sample point to camera
        let view_transmittance = calculate_transmittance(optical_depth_view);

        // Transmittance from sun to sample point (light travelling through atmosphere)
        // Smooth occlusion to avoid banding: compute a soft occlusion factor in [0,1]
        let occ = occlusion_factor_smooth(local_pos, sun_dir, view.planet_radius);
        let light_transmittance = if occ < 1.0 {
            let optical_depth_light = integrate_optical_depth(local_pos, sun_dir, view);
            let lt = calculate_transmittance(optical_depth_light);
            // scale light by visibility (1 - occ)
            lt * (1.0 - occ)
        } else {
            Vec3::splat(0.0)
        };

        // Accumulate scattered light
        rayleigh_scatter +=
            view_transmittance * light_transmittance * phase_r * local_density.x * step_size;
        mie_scatter +=
            view_transmittance * light_transmittance * phase_m * local_density.y * step_size;

        prev_ray_time = ray_time;
    }

    // Slightly warm spectral tint for more appealing sunsets
    let sun_color = vec3(1.0, 0.97, 0.90) * SUN_INTENSITY;
    let radiance = sun_color
        * (rayleigh_scatter * C_RAYLEIGH * RAYLEIGH_STRENGTH + mie_scatter * C_MIE * MIE_STRENGTH);

    // Apply exposure to the final radiance
    radiance * EXPOSURE
}

pub fn render_sky(rd: Vec3, view: &ViewUniform, sun_dir: Vec3) -> Vec3 {
    let ro_relative = view.world_position - view.planet_center;
    let atmosphere_radius = view.planet_radius + ATMOSPHERE_HEIGHT;

    // Find intersection distances with the atmosphere and planet from the cameras origin
    let atmosphere_isect = intersect_sphere(ro_relative, rd, atmosphere_radius);
    let planet_isect = intersect_sphere(ro_relative, rd, view.planet_radius - 2000.0);

    // If the ray completely misses the atmosphere, there's nothing to render
    if atmosphere_isect.y < 0.0 {
        return Vec3::ZERO;
    }

    // Determine the integration interval [ray_start_dist, ray_end_dist] along the ray
    let ray_start_dist = atmosphere_isect.x.max(0.0);
    let mut ray_end_dist = atmosphere_isect.y;

    // If the ray hits the planet, clamp the ray's end point to the planet surface
    if planet_isect.x > 0.0 {
        ray_end_dist = ray_end_dist.min(planet_isect.x);
    }

    // Calculate the final length and start position for the raymarching
    let ray_length = (ray_end_dist - ray_start_dist).max(0.0);
    if ray_length <= 0.0 {
        return Vec3::ZERO;
    }
    let ray_start = ro_relative + rd * ray_start_dist;

    // Get the final scattered light color
    let mut final_color = integrate_scattering(ray_start, rd, ray_length, sun_dir, view);

    // Subtle planet emissive glow (scattered)
    let to_center = -ro_relative.normalize();
    let dist = ro_relative.length();
    let disk_angle = (view.planet_radius / dist).saturate().asin();
    let angle = (rd.dot(to_center).clamp(-1.0, 1.0)).acos();

    // Rim distance = angle outside the disk; apply a smooth gaussian-like falloff
    let rim = (angle - disk_angle).max(0.0);
    let glow = (-(rim * rim) / (PLANET_GLOW_ANGULAR_WIDTH * PLANET_GLOW_ANGULAR_WIDTH)).exp();

    // Attenuate by atmospheric transmittance along the view ray
    let trans = calculate_transmittance(integrate_optical_depth(ro_relative, rd, view));
    let trans_lum = trans.dot(vec3(0.2126, 0.7152, 0.0722));
    let atten = trans * (1.0 - PLANET_EMISSION_CHROMA_PRESERVE)
        + Vec3::splat(trans_lum) * PLANET_EMISSION_CHROMA_PRESERVE;
    final_color += PLANET_EMISSION_COLOR * glow * atten;

    // Tonemapping & Gamma Correction
    final_color = 1.0 - (-final_color).exp(); // Reinhard tonemapping
    final_color = final_color.powf(1.0 / 2.2); // Gamma correction

    final_color
}

/// Calculates the direct sunlight color after it has passed through the atmosphere to the camera
pub fn get_sun_light_color(view: &ViewUniform, sun_dir: Vec3) -> Vec3 {
    let ro_relative = view.world_position - view.planet_center;

    // Use a smooth occlusion factor for the camera as well to avoid hard edges at the horizon
    let cam_occ = occlusion_factor_smooth(ro_relative, sun_dir, view.planet_radius);

    // Calculate the optical depth from the camera position towards the sun
    let optical_depth_light = integrate_optical_depth(ro_relative, sun_dir, view);

    // Calculate the transmittance (how much light makes it through) based on the optical depth
    let light_transmittance = calculate_transmittance(optical_depth_light) * (1.0 - cam_occ);

    // Slightly warm spectral tint for the sun (more pleasing sunsets)
    let sun_base = vec3(1.0, 0.97, 0.90) * SUN_INTENSITY;

    // The final sun color is the base intensity filtered by the atmospheric transmittance
    let sun_color = sun_base * light_transmittance;

    // Return the raw HDR color multiplied by the exposure
    sun_color * EXPOSURE
}
