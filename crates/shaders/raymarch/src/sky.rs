use crate::utils::{View, intersect_sphere};
use core::f32::consts::PI;
use spirv_std::glam::Vec3;
use spirv_std::num_traits::Float;

const THREE_OVER_SIXTEEN_PI: f32 = 3.0 / (16.0 * PI);
const SUN_INTENSITY: f32 = 22.0;
const EXPOSURE: f32 = 3.0;
const ATMOSPHERE_HEIGHT: f32 = 200_000.0;
const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0;
const MIE_SCALE_HEIGHT: f32 = 1200.0;

const C_RAYLEIGH: Vec3 = Vec3::new(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE: Vec3 = Vec3::new(3.996e-6, 3.996e-6, 3.996e-6);
const C_OZONE: Vec3 = Vec3::new(0.650e-6, 1.881e-6, 0.085e-6);

const RAYLEIGH_STRENGTH: f32 = 1.0;
const MIE_STRENGTH: f32 = 0.2;
const OZONE_STRENGTH: f32 = 1.0;
const MIE_G: f32 = 0.85;

const PRIMARY_STEPS: i32 = 16;
const LIGHT_STEPS: i32 = 8;

const PLANET_EMISSION_COLOR: Vec3 = Vec3::new(0.3, 0.15, 0.4); // 0.6, 0.3, 0.8 * 0.5
const PLANET_GLOW_ANGULAR_WIDTH: f32 = 0.03;
const PLANET_EMISSION_CHROMA_PRESERVE: f32 = 0.9;

fn phase_rayleigh(cos_theta: f32) -> f32 {
    THREE_OVER_SIXTEEN_PI * (1.0 + cos_theta * cos_theta)
}

fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let numerator = (1.0 - g2) / (4.0 * PI);
    let denominator = (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5);
    numerator / denominator
}

fn atmosphere_height(position: Vec3, planet_radius: f32) -> f32 {
    (position.length() - planet_radius).max(0.0)
}

fn density_atmosphere(h: f32) -> Vec3 {
    Vec3::new(
        (-(h / RAYLEIGH_SCALE_HEIGHT).max(0.0)).exp(),
        (-(h / MIE_SCALE_HEIGHT).max(0.0)).exp(),
        (1.0 - (h - 25000.0).abs() / 15000.0).max(0.0),
    )
}

fn calculate_transmittance(optical_depth: Vec3) -> Vec3 {
    let total_extinction = optical_depth.x * C_RAYLEIGH * RAYLEIGH_STRENGTH
        + optical_depth.y * C_MIE * MIE_STRENGTH
        + optical_depth.z * C_OZONE * OZONE_STRENGTH;
    (-total_extinction).exp()
}

fn integrate_optical_depth(ray_start: Vec3, ray_dir: Vec3, view: &View) -> Vec3 {
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

fn occlusion_factor_smooth(pos: Vec3, dir: Vec3, planet_radius: f32) -> f32 {
    let t_closest = -pos.dot(dir);
    if t_closest <= 0.0 {
        return 0.0;
    }
    let d_closest = (pos + dir * t_closest).length();
    let penetration = planet_radius - d_closest;
    if penetration <= 0.0 {
        return 0.0;
    }
    let soft_width = 4000.0;
    let f = (penetration / soft_width).clamp(0.0, 1.0);
    f * f * (3.0 - 2.0 * f)
}

fn integrate_scattering(
    ray_start: Vec3,
    ray_dir: Vec3,
    ray_length: f32,
    sun_dir: Vec3,
    view: &View,
) -> Vec3 {
    let cos_theta = ray_dir.dot(sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = phase_mie(cos_theta, MIE_G);

    let mut optical_depth_view = Vec3::ZERO;
    let mut rayleigh_scatter = Vec3::ZERO;
    let mut mie_scatter = Vec3::ZERO;

    let ray_height = atmosphere_height(ray_start, view.planet_radius);
    let sample_distribution_exponent =
        1.0 + (1.0 - ray_height / ATMOSPHERE_HEIGHT).clamp(0.0, 1.0) * 8.0;

    let mut prev_ray_time = 0.0;
    for i in 0..PRIMARY_STEPS {
        let ray_time = ((i as f32 + 1.0) / PRIMARY_STEPS as f32).powf(sample_distribution_exponent)
            * ray_length;
        let step_size = ray_time - prev_ray_time;
        let local_pos = ray_start + ray_dir * ray_time;
        let altitude = atmosphere_height(local_pos, view.planet_radius);

        if altitude <= ATMOSPHERE_HEIGHT {
            let local_density = density_atmosphere(altitude);
            optical_depth_view += local_density * step_size;
            let view_transmittance = calculate_transmittance(optical_depth_view);

            let occ = occlusion_factor_smooth(local_pos, sun_dir, view.planet_radius);
            let light_transmittance = if occ < 1.0 {
                calculate_transmittance(integrate_optical_depth(local_pos, sun_dir, view))
                    * (1.0 - occ)
            } else {
                Vec3::ZERO
            };

            rayleigh_scatter +=
                view_transmittance * light_transmittance * phase_r * local_density.x * step_size;
            mie_scatter +=
                view_transmittance * light_transmittance * phase_m * local_density.y * step_size;
        }
        prev_ray_time = ray_time;
    }

    let sun_color = Vec3::new(1.0, 0.97, 0.90) * SUN_INTENSITY;
    let radiance = sun_color
        * (rayleigh_scatter * C_RAYLEIGH * RAYLEIGH_STRENGTH + mie_scatter * C_MIE * MIE_STRENGTH);
    radiance * EXPOSURE
}

pub fn render_sky(rd: Vec3, view: &View, sun_dir: Vec3) -> Vec3 {
    let ro_relative = view.world_position - view.planet_center;
    let atmosphere_radius = view.planet_radius + ATMOSPHERE_HEIGHT;

    let atmosphere_isect = intersect_sphere(ro_relative, rd, atmosphere_radius);
    let planet_isect = intersect_sphere(ro_relative, rd, view.planet_radius - 2000.0);

    if atmosphere_isect.y < 0.0 {
        return Vec3::ZERO;
    }

    let ray_start_dist = atmosphere_isect.x.max(0.0);
    let mut ray_end_dist = atmosphere_isect.y;
    if planet_isect.x > 0.0 {
        ray_end_dist = ray_end_dist.min(planet_isect.x);
    }

    let ray_length = (ray_end_dist - ray_start_dist).max(0.0);
    if ray_length <= 0.0 {
        return Vec3::ZERO;
    }
    let ray_start = ro_relative + rd * ray_start_dist;

    let mut final_color = integrate_scattering(ray_start, rd, ray_length, sun_dir, view);

    // Subtle planet emissive glow (scattered)
    let to_center = (-ro_relative).normalize();
    let dist = ro_relative.length();
    let disk_angle = (view.planet_radius / dist).clamp(0.0, 1.0).asin();
    let angle = rd.dot(to_center).clamp(-1.0, 1.0).acos();

    // Rim distance = angle outside the disk; apply a smooth gaussian-like falloff
    let rim = (angle - disk_angle).max(0.0);
    let glow = (-(rim * rim) / (PLANET_GLOW_ANGULAR_WIDTH * PLANET_GLOW_ANGULAR_WIDTH)).exp();

    // Attenuate by atmospheric transmittance along the view ray
    let trans = calculate_transmittance(integrate_optical_depth(ro_relative, rd, view));
    let trans_lum = trans.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let atten = trans * (1.0 - PLANET_EMISSION_CHROMA_PRESERVE)
        + Vec3::splat(trans_lum) * PLANET_EMISSION_CHROMA_PRESERVE;
    final_color += PLANET_EMISSION_COLOR * glow * atten;

    // Tonemapping & Gamma Correction
    final_color = Vec3::ONE - (-final_color).exp(); // Reinhard tonemapping
    final_color = final_color.powf(1.0 / 2.2); // Gamma correction

    final_color
}

/// Calculates the direct sunlight color after it has passed through the atmosphere to the camera
pub fn get_sun_light_color(_ro: Vec3, view: &View, sun_dir: Vec3) -> Vec3 {
    let ro_relative = view.world_position - view.planet_center;

    // Use a smooth occlusion factor for the camera as well to avoid hard edges at the horizon
    let cam_occ = occlusion_factor_smooth(ro_relative, sun_dir, view.planet_radius);

    // Calculate the optical depth from the camera position towards the sun
    let optical_depth_light = integrate_optical_depth(ro_relative, sun_dir, view);

    // Calculate the transmittance (how much light makes it through) based on the optical depth
    let light_transmittance = calculate_transmittance(optical_depth_light) * (1.0 - cam_occ);

    // Slightly warm spectral tint for the sun (more pleasing sunsets)
    let sun_base = Vec3::new(1.0, 0.97, 0.90) * SUN_INTENSITY;

    // The final sun color is the base intensity filtered by the atmospheric transmittance
    let sun_color = sun_base * light_transmittance;

    // Return the raw HDR color multiplied by the exposure
    sun_color * EXPOSURE
}
