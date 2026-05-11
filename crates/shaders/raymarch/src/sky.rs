use crate::lighting::henyey_greenstein;
use crate::utils::intersect_sphere;
use core::f32::consts::PI;
use skybound_shared::PLANET_RADIUS;
use spirv_std::glam::{FloatExt, Vec3, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const THREE_OVER_SIXTEEN_PI: f32 = 3.0 / (16.0 * PI);

const SUN_INTENSITY: f32 = 22.0;
const EXPOSURE: f32 = 3.0;
const ATMOSPHERE_HEIGHT: f32 = 200_000.0;
const ATMOSPHERE_RADIUS: f32 = PLANET_RADIUS + ATMOSPHERE_HEIGHT;
const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0;
const MIE_SCALE_HEIGHT: f32 = 1200.0;

const C_RAYLEIGH: Vec3 = vec3(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE: Vec3 = vec3(3.996e-6, 3.996e-6, 3.996e-6);
const C_OZONE: Vec3 = vec3(0.650e-6, 1.881e-6, 0.085e-6);

const RAYLEIGH_STRENGTH: f32 = 1.0;
const MIE_STRENGTH: f32 = 0.2;
const OZONE_STRENGTH: f32 = 1.0;
const MIE_G: f32 = 0.85;

const PRIMARY_STEPS: i32 = 16;
const LIGHT_STEPS: i32 = 8;

const PLANET_EMISSION_COLOR: Vec3 = vec3(0.3, 0.15, 0.4);
const PLANET_GLOW_ANGULAR_WIDTH: f32 = 0.03;
const PLANET_EMISSION_CHROMA_PRESERVE: f32 = 0.9;

fn phase_rayleigh(cos_theta: f32) -> f32 {
    THREE_OVER_SIXTEEN_PI * (1.0 + cos_theta * cos_theta)
}

fn altitude(position: Vec3) -> f32 {
    (position.length() - PLANET_RADIUS).max(0.0)
}

fn density_rayleigh(h: f32) -> f32 {
    (h.max(0.0) / -RAYLEIGH_SCALE_HEIGHT).exp()
}

fn density_mie(h: f32) -> f32 {
    (h.max(0.0) / -MIE_SCALE_HEIGHT).exp()
}

fn density_ozone(h: f32) -> f32 {
    (1.0 - (h - 25000.0).abs() / 15000.0).max(0.0)
}

fn density_atmosphere(h: f32) -> Vec3 {
    vec3(density_rayleigh(h), density_mie(h), density_ozone(h))
}

fn calculate_transmittance(optical_depth: Vec3) -> Vec3 {
    let extinction = optical_depth.x * C_RAYLEIGH * RAYLEIGH_STRENGTH
        + optical_depth.y * C_MIE * MIE_STRENGTH
        + optical_depth.z * C_OZONE * OZONE_STRENGTH;
    (-extinction).exp()
}

fn integrate_optical_depth(ray_start: Vec3, ray_dir: Vec3) -> Vec3 {
    let intersection = intersect_sphere(ray_start, ray_dir, ATMOSPHERE_RADIUS);
    let ray_length = intersection.y.max(0.0);
    let step_size = ray_length / LIGHT_STEPS as f32;
    let mut optical_depth = Vec3::ZERO;
    for i in 0..LIGHT_STEPS {
        let local_pos = ray_start + ray_dir * (i as f32 + 0.5) * step_size;
        let h = altitude(local_pos);
        if h <= ATMOSPHERE_HEIGHT {
            optical_depth += density_atmosphere(h) * step_size;
        }
    }
    optical_depth
}

fn occlusion_factor(pos: Vec3, dir: Vec3) -> f32 {
    let t_closest = -pos.dot(dir);
    if t_closest <= 0.0 {
        return 0.0;
    }
    let closest_point = pos + dir * t_closest;
    let penetration = PLANET_RADIUS - closest_point.length();
    if penetration <= 0.0 {
        return 0.0;
    }
    let f = (penetration / 4000.0).saturate();
    f * f * (3.0 - 2.0 * f)
}

fn integrate_scattering(ray_start: Vec3, ray_dir: Vec3, ray_length: f32, sun_dir: Vec3) -> Vec3 {
    let cos_theta = ray_dir.dot(sun_dir);
    let phase_r = phase_rayleigh(cos_theta);
    let phase_m = henyey_greenstein(cos_theta, MIE_G);

    let ray_height = altitude(ray_start);
    let sample_exp = 1.0 + (1.0 - ray_height / ATMOSPHERE_HEIGHT).saturate() * 8.0;

    let mut optical_depth_view = Vec3::ZERO;
    let mut rayleigh_scatter = Vec3::ZERO;
    let mut mie_scatter = Vec3::ZERO;
    let mut prev_ray_time = 0.0;

    for i in 0..PRIMARY_STEPS {
        let ray_time = ((i + 1) as f32 / PRIMARY_STEPS as f32).powf(sample_exp) * ray_length;
        let step_size = ray_time - prev_ray_time;
        let local_pos = ray_start + ray_dir * ray_time;
        let h = altitude(local_pos);
        if h > ATMOSPHERE_HEIGHT {
            continue;
        }

        let local_density = density_atmosphere(h);
        optical_depth_view += local_density * step_size;
        let view_transmittance = calculate_transmittance(optical_depth_view);

        let occ = occlusion_factor(local_pos, sun_dir);
        let light_transmittance = if occ < 1.0 {
            let optical_depth_light = integrate_optical_depth(local_pos, sun_dir);
            calculate_transmittance(optical_depth_light) * (1.0 - occ)
        } else {
            Vec3::splat(0.0)
        };

        rayleigh_scatter +=
            view_transmittance * light_transmittance * phase_r * local_density.x * step_size;
        mie_scatter +=
            view_transmittance * light_transmittance * phase_m * local_density.y * step_size;

        prev_ray_time = ray_time;
    }

    let sun_color = vec3(1.0, 0.97, 0.90) * SUN_INTENSITY;
    let radiance = sun_color
        * (rayleigh_scatter * C_RAYLEIGH * RAYLEIGH_STRENGTH + mie_scatter * C_MIE * MIE_STRENGTH);
    radiance * EXPOSURE
}

pub fn render_sky(rd: Vec3, ro_relative: Vec3, sun_dir: Vec3) -> Vec3 {
    let atmosphere_isect = intersect_sphere(ro_relative, rd, ATMOSPHERE_RADIUS);
    let planet_isect = intersect_sphere(ro_relative, rd, PLANET_RADIUS - 2000.0);

    if atmosphere_isect.y < 0.0 {
        return Vec3::ZERO;
    }

    let ray_start_dist = atmosphere_isect.x.max(0.0);
    let mut ray_end_dist = atmosphere_isect.y;
    if planet_isect.x > 0.0 {
        ray_end_dist = ray_end_dist.min(planet_isect.x + 35_000.0);
    }

    let ray_length = (ray_end_dist - ray_start_dist).max(0.0);
    if ray_length <= 0.0 {
        return Vec3::ZERO;
    }
    let ray_start = ro_relative + rd * ray_start_dist;

    let mut final_color = integrate_scattering(ray_start, rd, ray_length, sun_dir);

    // Planet emissive glow
    let to_center = -ro_relative.normalize();
    let dist = ro_relative.length();
    let disk_angle = (PLANET_RADIUS / dist).saturate().asin();
    let angle = (rd.dot(to_center).clamp(-1.0, 1.0)).acos();
    let rim = (angle - disk_angle).max(0.0);
    let glow = (-(rim * rim) / (PLANET_GLOW_ANGULAR_WIDTH * PLANET_GLOW_ANGULAR_WIDTH)).exp();

    let trans = calculate_transmittance(integrate_optical_depth(ro_relative, rd));
    let trans_lum = trans.dot(vec3(0.2126, 0.7152, 0.0722));
    let atten = trans * (1.0 - PLANET_EMISSION_CHROMA_PRESERVE)
        + Vec3::splat(trans_lum) * PLANET_EMISSION_CHROMA_PRESERVE;
    final_color += PLANET_EMISSION_COLOR * glow * atten;

    final_color = 1.0 - (-final_color).exp();
    final_color.powf(1.0 / 2.2)
}

pub fn get_sun_light_color(ro_relative: Vec3, sun_dir: Vec3) -> Vec3 {
    let cam_occ = occlusion_factor(ro_relative, sun_dir);
    let optical_depth_light = integrate_optical_depth(ro_relative, sun_dir);
    let light_transmittance = calculate_transmittance(optical_depth_light) * (1.0 - cam_occ);
    let sun_base = vec3(1.0, 0.97, 0.90) * SUN_INTENSITY;
    sun_base * light_transmittance * EXPOSURE
}
