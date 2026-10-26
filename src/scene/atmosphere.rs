use crate::scene::{
    PLANET_RADIUS,
    math::{henyey_greenstein, intersect_sphere, square},
    planet::REACTION_HEIGHT,
};
use core::f32::consts::PI;
use spirv_std::glam::{FloatExt, Vec2, Vec3, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

const THREE_OVER_SIXTEEN_PI: f32 = 3.0 / (16.0 * PI);

const POLE_INTENSITY: f32 = 12.0;

const EXPOSURE: f32 = 3.0;

pub const POLE_RADIANCE: Vec3 = vec3(
    POLE_INTENSITY * EXPOSURE,
    0.97 * POLE_INTENSITY * EXPOSURE,
    0.90 * POLE_INTENSITY * EXPOSURE,
);

const ATMOSPHERE_HEIGHT: f32 = REACTION_HEIGHT;

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

const AUR_RADIANCE: Vec3 = vec3(0.08, 0.012, 0.16);

pub const SKY_SIZE: u32 = 128;

pub const TRANSMITTANCE_SIZE: u32 = 512;

const AIR_SCALE_HEIGHT: f32 = 11_000.0;

const AIR_EXTINCTION: f32 = 0.000_008;

pub fn transmittance_coordinates(position: Vec3, direction: Vec3) -> Vec2 {
    let radius = position.length().max(PLANET_RADIUS);
    let height = ((radius - PLANET_RADIUS) / ATMOSPHERE_HEIGHT)
        .clamp(0.0, 1.0)
        .sqrt();
    let cosine = (position / radius).dot(direction).clamp(-1.0, 1.0);
    let horizon = -(1.0 - square(PLANET_RADIUS / radius)).max(0.0).sqrt();
    let delta = cosine - horizon;
    let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
    let angle = (delta.abs() / (1.0 - horizon * sign).max(0.0001)).sqrt() * sign;
    let uv = vec2(angle * 0.5 + 0.5, height);
    (uv * (TRANSMITTANCE_SIZE - 1) as f32 + 0.5) / TRANSMITTANCE_SIZE as f32
}

fn transmittance_ray(uv: Vec2) -> (Vec3, Vec3) {
    let radius = PLANET_RADIUS + uv.y * uv.y * ATMOSPHERE_HEIGHT;
    let horizon = -(1.0 - square(PLANET_RADIUS / radius)).max(0.0).sqrt();
    let angle = uv.x * 2.0 - 1.0;
    let cosine = horizon + angle * angle.abs() * (1.0 - horizon * angle.signum());
    let direction = vec3((1.0 - cosine * cosine).max(0.0).sqrt(), 0.0, cosine);
    let position = Vec3::Z * radius;
    (position, direction)
}

pub fn transmittance_texel(uv: Vec2) -> Vec3 {
    let (position, direction) = transmittance_ray(uv);
    calculate_transmittance(integrate_optical_depth(position, direction, 10_000_000.0))
        * (1.0 - occlusion_factor(position, direction))
}

pub fn air_texel(uv: Vec2) -> f32 {
    let (position, direction) = transmittance_ray(uv);
    air_optical_depth(position, direction, 10_000_000.0)
}

pub fn sky_direction(uv: Vec2, horizon: f32) -> Vec3 {
    let y = uv.y * 2.0 - 1.0;
    let z = horizon + y * y.abs() * (1.0 - horizon * (if y >= 0.0 { 1.0 } else { -1.0 }));
    let r = (1.0 - z * z).max(0.0).sqrt();
    let azimuth = uv.x * PI;
    vec3(r * azimuth.cos(), r * azimuth.sin(), z)
}

pub fn sky_uv(rd: Vec3, pole_dir: Vec3, horizon: f32) -> Vec2 {
    let horizontal = rd.truncate().length() * pole_dir.truncate().length();
    let cosine = (rd.x * pole_dir.x + rd.y * pole_dir.y) / horizontal.max(1e-6);
    let delta = rd.z - horizon;
    let sign = if delta < 0.0 { -1.0 } else { 1.0 };
    let y = (delta.abs() / (1.0 - horizon * sign)).sqrt() * sign;
    let uv = vec2(cosine.clamp(-1.0, 1.0).acos() / PI, y * 0.5 + 0.5);
    (uv * (SKY_SIZE - 1) as f32 + 0.5) / SKY_SIZE as f32
}

fn phase_rayleigh(cos_theta: f32) -> f32 {
    THREE_OVER_SIXTEEN_PI * (1.0 + cos_theta * cos_theta)
}

fn altitude(position: Vec3) -> f32 {
    position.length() - PLANET_RADIUS
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
    vec3(density_rayleigh(h), density_mie(h), density_ozone(h)) * h.smoothstep(-1500.0, 0.0)
}

fn calculate_transmittance(optical_depth: Vec3) -> Vec3 {
    let extinction = optical_depth.x * C_RAYLEIGH * RAYLEIGH_STRENGTH
        + optical_depth.y * C_MIE * MIE_STRENGTH
        + optical_depth.z * C_OZONE * OZONE_STRENGTH;
    (-extinction).exp()
}

fn integrate_optical_depth(ray_start: Vec3, ray_dir: Vec3, distance: f32) -> Vec3 {
    let intersection = intersect_sphere(ray_start, ray_dir, ATMOSPHERE_RADIUS);
    let start = intersection.x.max(0.0);
    let ray_length = (intersection.y.min(distance) - start).max(0.0);
    let step_size = ray_length / LIGHT_STEPS as f32;
    let mut optical_depth = Vec3::ZERO;
    for i in 0..LIGHT_STEPS {
        let local_pos = ray_start + ray_dir * (start + (i as f32 + 0.5) * step_size);
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

fn integrate_scattering(ray_start: Vec3, ray_dir: Vec3, ray_length: f32, source: Vec3) -> Vec3 {
    let ray_height = altitude(ray_start);
    let sample_exp = 1.0 + (1.0 - ray_height / ATMOSPHERE_HEIGHT).saturate() * 8.0;

    let mut optical_depth_view = Vec3::ZERO;
    let mut rayleigh_scatter = Vec3::ZERO;
    let mut mie_scatter = Vec3::ZERO;
    let mut prev_ray_time = 0.0;
    let mut aur_scatter = Vec3::ZERO;
    let mut aur_depth = 0.0;

    for i in 0..PRIMARY_STEPS {
        let ray_time = ((i + 1) as f32 / PRIMARY_STEPS as f32).powf(sample_exp) * ray_length;
        let step_size = ray_time - prev_ray_time;
        let local_pos = ray_start + ray_dir * (prev_ray_time + step_size * 0.5);
        prev_ray_time = ray_time;
        let h = altitude(local_pos);
        if h > ATMOSPHERE_HEIGHT {
            continue;
        }

        let local_density = density_atmosphere(h);
        let offset = source - local_pos;
        let light_distance = offset.length().max(1.0);
        let pole_dir = offset / light_distance;
        let cos_theta = ray_dir.dot(pole_dir);
        let phase_r = phase_rayleigh(cos_theta);
        let phase_m = henyey_greenstein(cos_theta, MIE_G);
        optical_depth_view += local_density * step_size;
        aur_depth += (-h / 350.0).clamp(-20.0, 12.0).exp() * 0.0008 * step_size;
        let view_transmittance = calculate_transmittance(optical_depth_view) * (-aur_depth).exp();
        if view_transmittance.max_element() < 0.0001 {
            break;
        }
        let upward_depth = vec3(
            RAYLEIGH_SCALE_HEIGHT * (1.0 - density_rayleigh(h)),
            MIE_SCALE_HEIGHT * (1.0 - density_mie(h)),
            0.0,
        );
        let incoming = calculate_transmittance(upward_depth);
        let solid_angle = square(PLANET_RADIUS / local_pos.length().max(PLANET_RADIUS));
        let up = local_pos.normalize();
        let scattering = C_RAYLEIGH * local_density.x * 0.5
            + C_MIE * MIE_STRENGTH * local_density.y * (0.5 - ray_dir.dot(up) * 0.25);
        aur_scatter += view_transmittance * incoming * scattering * (step_size * solid_angle);

        let occ = occlusion_factor(local_pos, pole_dir);
        let light_transmittance = if occ < 1.0 {
            let optical_depth_light = integrate_optical_depth(local_pos, pole_dir, light_distance);
            calculate_transmittance(optical_depth_light) * (1.0 - occ)
        } else {
            Vec3::splat(0.0)
        };

        rayleigh_scatter +=
            view_transmittance * light_transmittance * phase_r * local_density.x * step_size;
        mie_scatter +=
            view_transmittance * light_transmittance * phase_m * local_density.y * step_size;
    }

    let pole_color = vec3(1.0, 0.97, 0.90) * POLE_INTENSITY;
    let radiance = pole_color
        * (rayleigh_scatter * C_RAYLEIGH * RAYLEIGH_STRENGTH + mie_scatter * C_MIE * MIE_STRENGTH);
    (radiance + aur_scatter * AUR_RADIANCE) * EXPOSURE
}

pub fn render_sky(rd: Vec3, ro_relative: Vec3, source: Vec3) -> Vec3 {
    let atmosphere_isect = intersect_sphere(ro_relative, rd, ATMOSPHERE_RADIUS);

    if atmosphere_isect.y < 0.0 {
        return Vec3::ZERO;
    }

    let ray_start_dist = atmosphere_isect.x.max(0.0);
    let ray_end_dist = atmosphere_isect.y;

    let ray_length = (ray_end_dist - ray_start_dist).max(0.0);
    if ray_length <= 0.0 {
        return Vec3::ZERO;
    }
    let ray_start = ro_relative + rd * ray_start_dist;

    integrate_scattering(ray_start, rd, ray_length, source).max(Vec3::ZERO)
}

pub fn get_pole_light_color(ro_relative: Vec3, source: Vec3) -> Vec3 {
    let offset = source - ro_relative;
    let distance = offset.length().max(1.0);
    let pole_dir = offset / distance;
    let cam_occ = occlusion_factor(ro_relative, pole_dir);
    let optical_depth_light = integrate_optical_depth(ro_relative, pole_dir, distance);
    let light_transmittance = calculate_transmittance(optical_depth_light) * (1.0 - cam_occ);
    POLE_RADIANCE * light_transmittance
}

pub fn air_density(altitude: f32) -> f32 {
    AIR_EXTINCTION * (-altitude.max(0.0) / AIR_SCALE_HEIGHT).exp()
}

// Integrate an exponential height profile along a linear approximation to altitude. Unlike a
// midpoint sample this also resolves thin air near the end of a very long ray.
fn air_panel(origin: Vec3, direction: Vec3, start: f32, end: f32) -> f32 {
    let a = ((origin + direction * start).length() - PLANET_RADIUS).max(0.0);
    let b = ((origin + direction * end).length() - PLANET_RADIUS).max(0.0);
    let delta = (a - b).abs() / AIR_SCALE_HEIGHT;
    let mean = if delta < 0.01 {
        1.0 - delta * 0.5 + delta * delta / 6.0
    } else {
        (1.0 - (-delta).exp()) / delta
    };
    air_density(a.min(b)) * mean * (end - start)
}

pub fn air_optical_depth(origin: Vec3, direction: Vec3, length: f32) -> f32 {
    if length < 2000.0 {
        let altitude = (origin + direction * (length * 0.5)).length() - PLANET_RADIUS;
        return air_density(altitude) * length.max(0.0);
    }
    // Above twelve scale heights the remaining air_density is negligible. Clip first so orbital
    // rays spend the same fixed work on the air as rays close to the planet.
    let shell = intersect_sphere(origin, direction, PLANET_RADIUS + AIR_SCALE_HEIGHT * 12.0);
    let start = shell.x.max(0.0);
    let end = shell.y.min(length);
    if end <= start {
        return 0.0;
    }
    let closest = (-origin.dot(direction)).clamp(start, end);
    let mut depth = 0.0;
    for side in 0..2 {
        let (a, b) = if side == 0 {
            (start, closest)
        } else {
            (closest, end)
        };
        if b <= a {
            continue;
        }
        let step = (b - a) * 0.125;
        for index in 0..8 {
            let from = a + index as f32 * step;
            depth += air_panel(origin, direction, from, from + step);
        }
    }
    depth
}

pub fn air_transmittance(origin: Vec3, direction: Vec3, length: f32) -> f32 {
    (-air_optical_depth(origin, direction, length)).exp()
}

pub fn cached_air_transmittance(
    origin: Vec3,
    direction: Vec3,
    length: f32,
    lookup: impl Fn(Vec3, Vec3) -> f32,
) -> f32 {
    if length < 2000.0 {
        return air_transmittance(origin, direction, length);
    }
    let shell = intersect_sphere(origin, direction, PLANET_RADIUS + ATMOSPHERE_HEIGHT);
    let start = shell.x.max(0.0);
    let end = shell.y.min(length);
    if end <= start {
        return 1.0;
    }
    let a = origin + direction * start;
    let b = origin + direction * end;
    // Use the outward orientation to avoid subtracting large, nearly equal columns through the
    // planet when the visible segment is descending.
    let depth = if (a + b).dot(direction) >= 0.0 {
        lookup(a, direction) - lookup(b, direction)
    } else {
        lookup(b, -direction) - lookup(a, -direction)
    };
    (-depth.max(0.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::atmosphere::{TRANSMITTANCE_SIZE, air_texel, transmittance_coordinates};
    use spirv_std::glam::{Vec2, vec2};

    #[test]
    fn cached_transmittance_matches_light_integration_across_cloud_altitudes() {
        let mut error = 0.0f32;
        for height in [2700.0, 6350.0, 12_000.0, 26_100.0, 48_000.0, 78_000.0] {
            let position = Vec3::Z * (PLANET_RADIUS + height);
            for index in 0..1024 {
                let cosine = index as f32 / 1023.0 * 2.0 - 1.0;
                let direction = vec3((1.0 - cosine * cosine).max(0.0).sqrt(), 0.0, cosine);
                let uv = transmittance_coordinates(position, direction) * TRANSMITTANCE_SIZE as f32
                    - Vec2::splat(0.5);
                let base = uv.floor();
                let mix = uv - base;
                let fetch = |x: f32, y: f32| {
                    transmittance_texel(vec2(x, y) / (TRANSMITTANCE_SIZE - 1) as f32)
                };
                let cached = fetch(base.x, base.y)
                    .lerp(fetch(base.x + 1.0, base.y), mix.x)
                    .lerp(
                        fetch(base.x, base.y + 1.0).lerp(fetch(base.x + 1.0, base.y + 1.0), mix.x),
                        mix.y,
                    );
                let exact = calculate_transmittance(integrate_optical_depth(
                    position,
                    direction,
                    10_000_000.0,
                )) * (1.0 - occlusion_factor(position, direction));
                error = error.max((cached - exact).abs().max_element());
            }
        }
        assert!(error < 0.005, "maximum transmittance error: {error}");
    }

    #[test]
    fn vertical_air_matches_the_exponential_integral() {
        for height in [0.0, 5000.0, 40_000.0] {
            for length in [10.0, 1000.0, 100_000.0, 1_000_000.0] {
                let origin = Vec3::Z * (PLANET_RADIUS + height);
                let expected = (-air_density(height)
                    * AIR_SCALE_HEIGHT
                    * (1.0 - (-length / AIR_SCALE_HEIGHT).exp()))
                .exp();
                assert!((air_transmittance(origin, Vec3::Z, length) - expected).abs() < 0.00001);
            }
        }
    }

    #[test]
    fn grazing_air_is_reciprocal_and_close_to_numerical_integration() {
        let origin = Vec3::Z * (PLANET_RADIUS + 2000.0) - Vec3::X * 600_000.0;
        let length = 1_200_000.0;
        let value = air_transmittance(origin, Vec3::X, length);
        assert!(
            (value - air_transmittance(origin + Vec3::X * length, -Vec3::X, length)).abs() < 1e-5
        );
        let mut depth = 0.0;
        for index in 0..4096 {
            let point = origin + Vec3::X * ((index as f32 + 0.5) * length / 4096.0);
            depth += air_density(point.length() - PLANET_RADIUS) * length / 4096.0;
        }
        assert!((value - (-depth).exp()).abs() < 0.015);
        assert!(
            (air_transmittance(Vec3::Z * (PLANET_RADIUS + 200_000.0), Vec3::X, length) - 1.0).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn cached_air_columns_match_direct_integration_on_visible_segments() {
        let size = TRANSMITTANCE_SIZE as usize;
        let table: Vec<_> = (0..size * size)
            .map(|index| {
                air_texel(vec2((index % size) as f32, (index / size) as f32) / (size - 1) as f32)
            })
            .collect();
        let lookup = |point, direction| {
            let uv = transmittance_coordinates(point, direction) * size as f32 - 0.5;
            let uv = uv.clamp(Vec2::ZERO, Vec2::splat((size - 1) as f32));
            let column = uv.x.floor() as usize;
            let row = uv.y.floor() as usize;
            let horizontal = uv.x.fract();
            let vertical = uv.y.fract();
            let fetch = |x: usize, y: usize| table[y.min(size - 1) * size + x.min(size - 1)];
            let top = fetch(column, row) * (1.0 - horizontal) + fetch(column + 1, row) * horizontal;
            let bottom = fetch(column, row + 1) * (1.0 - horizontal)
                + fetch(column + 1, row + 1) * horizontal;
            top * (1.0 - vertical) + bottom * vertical
        };
        for height in [0.0, 2500.0, 20_000.0, 100_000.0, 350_000.0] {
            let origin = Vec3::Z * (PLANET_RADIUS + height);
            for cosine in [-1.0f32, -0.3, -0.05, 0.0, 0.1, 0.7, 1.0] {
                let direction = Vec3::new((1.0 - cosine * cosine).sqrt(), 0.0, cosine);
                for length in [500.0, 5000.0, 100_000.0, 1_000_000.0] {
                    let closest = (-origin.dot(direction)).clamp(0.0, length);
                    if (origin + direction * closest).length() < PLANET_RADIUS {
                        continue;
                    }
                    let expected = air_transmittance(origin, direction, length);
                    let cached = cached_air_transmittance(origin, direction, length, lookup);
                    assert!(
                        (cached - expected).abs() < 0.02,
                        "height={height} cosine={cosine} length={length}: {cached} vs {expected}"
                    );
                }
            }
        }
    }
}
