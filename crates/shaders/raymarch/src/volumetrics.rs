use crate::aur_ocean::{OCEAN_TOP_HEIGHT, sample_ocean};
use crate::clouds::{CLOUD_BOTTOM_HEIGHT, CLOUD_TOP_HEIGHT, get_cloud_layer_above, sample_clouds};
use crate::poles::{poles_raymarch_entry, sample_poles};
use crate::utils::{AtmosphereData, View, intersect_plane, intersect_sphere, ray_shell_intersect};
use spirv_std::glam::{Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;

const MAX_STEPS: i32 = 512;
const STEP_SIZE_INSIDE: f32 = 120.0;
const STEP_SIZE_OUTSIDE: f32 = 240.0;

const SCALING_END: f32 = 200000.0;
const SCALING_MAX: f32 = 6.0;
const SCALING_MAX_VERTICAL: f32 = 2.0;
const SCALING_MAX_OCEAN: f32 = 2.0;
const CLOSE_THRESHOLD: f32 = 2000.0;

const LIGHT_STEPS: u32 = 4;
const LIGHT_STEP_SIZE: f32 = 90.0;
const SUN_CONE_ANGLE: f32 = 0.005;
const AUR_LIGHT_DIR: Vec3 = Vec3::new(0.0, 0.0, -1.0);
const AUR_LIGHT_DISTANCE: f32 = 60000.0;
const AUR_LIGHT_COLOR: Vec3 = Vec3::new(0.36, 0.18, 0.48); // 0.6, 0.3, 0.8 * 0.6

const EXTINCTION: f32 = 0.05;
const SCATTERING_ALBEDO: f32 = 0.65;
const SHADOW_FADE_END: f32 = 80000.0;

const DISK_SAMPLES: [Vec2; 16] = [
    Vec2::new(0.0000, 0.0000),
    Vec2::new(0.7071, 0.0000),
    Vec2::new(-0.7071, 0.0000),
    Vec2::new(0.0000, 0.7071),
    Vec2::new(0.0000, -0.7071),
    Vec2::new(0.5000, 0.5000),
    Vec2::new(-0.5000, 0.5000),
    Vec2::new(0.5000, -0.5000),
    Vec2::new(-0.5000, -0.5000),
    Vec2::new(0.9238, 0.3826),
    Vec2::new(-0.9238, 0.3826),
    Vec2::new(0.3826, 0.9238),
    Vec2::new(-0.3826, 0.9238),
    Vec2::new(0.3826, -0.9238),
    Vec2::new(-0.3826, -0.9238),
    Vec2::new(0.0000, 0.0000),
];

pub struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

pub fn sample_volume(
    pos: Vec3,
    view: &View,
    time: f32,
    clouds: bool,
    ocean: bool,
    poles: bool,
    base_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    details_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
) -> VolumeSample {
    let mut sample = VolumeSample {
        density: 0.0,
        color: Vec3::ONE,
        emission: Vec3::ZERO,
    };

    let mut blended_color = Vec3::ZERO;
    if clouds {
        let cloud_val = sample_clouds(
            pos,
            view,
            time,
            false,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );
        if cloud_val > 0.0 {
            blended_color += Vec3::ONE * cloud_val;
            sample.density += cloud_val;
        }
    }
    if ocean {
        let ocean_sample = sample_ocean(pos, time, false, details_texture, sampler);
        if ocean_sample.density > 0.0 {
            blended_color += ocean_sample.color * ocean_sample.density;
            sample.density += ocean_sample.density;
            sample.emission += ocean_sample.emission * ocean_sample.density;
        }
    }
    if poles {
        let poles_sample = sample_poles(pos, time, sampler);
        if poles_sample.density > 0.0 {
            blended_color += poles_sample.color * poles_sample.density;
            sample.density += poles_sample.density;
            sample.emission += poles_sample.emission * poles_sample.density;
        }
    }

    if sample.density > 0.0001 {
        let mix_factor = sample.density.clamp(0.0, 1.0);
        sample.color = blended_color / sample.density;
        sample.emission = (sample.emission / sample.density) * mix_factor;
    }
    sample
}

fn sample_shadowing(
    world_pos: Vec3,
    view: &View,
    atmosphere: &AtmosphereData,
    step_density: f32,
    time: f32,
    sun_dir: Vec3,
    base_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    details_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
    dither: f32,
) -> Vec3 {
    let mut optical_depth = step_density * EXTINCTION * LIGHT_STEP_SIZE * 0.5;

    let up = if sun_dir.z.abs() > 0.999 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let tangent1 = sun_dir.cross(up).normalize();
    let tangent2 = sun_dir.cross(tangent1);

    for j in 0..LIGHT_STEPS {
        let step_index = j as f32 + 1.0;
        let distance_along = step_index * LIGHT_STEP_SIZE;
        let disk_radius = SUN_CONE_ANGLE.tan() * distance_along;

        let idxf = (dither + step_index * 0.618034).fract();
        let idx = (idxf * 16.0).floor() as usize;
        let sample_offset = DISK_SAMPLES[idx % 16];
        let disk_offset =
            tangent1 * (sample_offset.x * disk_radius) + tangent2 * (sample_offset.y * disk_radius);

        let lightmarch_pos = world_pos + sun_dir * distance_along + disk_offset;
        let clouds = sample_clouds(
            lightmarch_pos,
            view,
            time,
            false,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );
        let ocean = sample_ocean(lightmarch_pos, time, true, details_texture, sampler).density;
        optical_depth += (clouds + ocean).max(0.0) * EXTINCTION * LIGHT_STEP_SIZE;
    }

    if step_density > 0.0 {
        for i in 1..=16 {
            let next_layer = get_cloud_layer_above(world_pos.z, i as f32);
            if next_layer <= 0.0 {
                break;
            }

            let intersection_t = intersect_plane(world_pos, sun_dir, next_layer);
            if intersection_t <= 0.0 || intersection_t > SHADOW_FADE_END {
                continue;
            }

            let shadow_falloff = 1.0 - (intersection_t / SHADOW_FADE_END);
            let layer_disk_radius = SUN_CONE_ANGLE.tan() * intersection_t;
            let idxf = (dither + i as f32 * 0.618034).fract();
            let idx = (idxf * 16.0).floor() as usize;
            let sample_offset = DISK_SAMPLES[idx % 16];
            let disk_offset = tangent1 * (sample_offset.x * layer_disk_radius)
                + tangent2 * (sample_offset.y * layer_disk_radius);

            let layer_sample_pos = world_pos + sun_dir * intersection_t + disk_offset;
            let light_sample_density = sample_clouds(
                layer_sample_pos,
                view,
                time,
                true,
                base_texture,
                details_texture,
                weather_texture,
                sampler,
            );
            optical_depth +=
                light_sample_density.max(0.0) * EXTINCTION * LIGHT_STEP_SIZE * shadow_falloff;
        }
    }

    let sun_transmittance = (-optical_depth).exp();
    let single_scattering = sun_transmittance * atmosphere.sun;
    let multiple_scattering = atmosphere.sky * sun_transmittance;

    let shadow_boost = 1.0 - sun_transmittance;
    let ambient_base = 0.12 + 0.6 * (step_density * 3.0).clamp(0.0, 1.0);
    let ambient_floor_vec =
        atmosphere.sky * ambient_base * (0.6 + 0.4 * shadow_boost) + atmosphere.sky * 0.03;

    (single_scattering + multiple_scattering + ambient_floor_vec) * SCATTERING_ALBEDO
}

pub struct VolumetricsResult {
    pub color: Vec4,
    pub depth: f32,
}

pub fn raymarch_volumetrics(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &View,
    t_max: f32,
    dither: f32,
    time: f32,
    base_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    details_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
) -> VolumetricsResult {
    let mut color = Vec3::ZERO;
    let mut transmittance = 1.0;
    let mut first_hit_depth = t_max;

    let cloud_shell = ray_shell_intersect(
        ro,
        rd,
        view,
        CLOUD_BOTTOM_HEIGHT - 500.0,
        CLOUD_TOP_HEIGHT + 500.0,
    );
    let ocean_isect = intersect_sphere(
        ro - view.planet_center,
        rd,
        view.planet_radius + OCEAN_TOP_HEIGHT,
    );
    let poles_isect = poles_raymarch_entry(ro, rd, view, t_max);

    let mut t_start = t_max;
    let mut t_end = 0.0;

    if cloud_shell.x <= cloud_shell.y {
        t_start = t_start.min(cloud_shell.x.max(0.0));
        t_end = t_end.max(cloud_shell.y);
    }
    if cloud_shell.z <= cloud_shell.w {
        t_start = t_start.min(cloud_shell.z.max(0.0));
        t_end = t_end.max(cloud_shell.w);
    }
    if ocean_isect.y > 0.0 {
        t_start = t_start.min(ocean_isect.x.max(0.0));
        t_end = t_end.max(ocean_isect.y);
    }
    if poles_isect.1 > 0.0 {
        t_start = t_start.min(poles_isect.0.max(0.0));
        t_end = t_end.max(poles_isect.1);
    }

    t_end = t_end.min(t_max);
    if t_start >= t_end {
        return VolumetricsResult {
            color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            depth: t_max,
        };
    }

    let mut t = t_start + dither * STEP_SIZE_INSIDE;
    let sun_dir = (atmosphere.sun_pos - ro).normalize();

    for _ in 0..MAX_STEPS {
        if t >= t_end || transmittance < 0.01 {
            break;
        }

        let pos = ro + rd * t;
        let in_clouds = pos.z >= CLOUD_BOTTOM_HEIGHT - 500.0 && pos.z <= CLOUD_TOP_HEIGHT + 500.0;
        let in_ocean = pos.z <= OCEAN_TOP_HEIGHT;
        let in_poles = (pos - view.planet_center).length() > view.planet_radius + 50000.0;

        let sample = sample_volume(
            pos,
            view,
            time,
            in_clouds,
            in_ocean,
            in_poles,
            base_texture,
            details_texture,
            weather_texture,
            sampler,
        );

        let mut step_size = if sample.density > 0.001 {
            STEP_SIZE_INSIDE
        } else {
            STEP_SIZE_OUTSIDE
        };
        let dist_scale = (1.0 + (t / SCALING_END).clamp(0.0, 1.0) * (SCALING_MAX - 1.0));
        step_size *= dist_scale;

        if sample.density > 0.001 {
            if first_hit_depth == t_max {
                first_hit_depth = t;
            }

            let step_transmittance = (-sample.density * EXTINCTION * step_size).exp();
            let shadowing = sample_shadowing(
                pos,
                view,
                atmosphere,
                sample.density,
                time,
                sun_dir,
                base_texture,
                details_texture,
                weather_texture,
                sampler,
                dither,
            );

            let step_light =
                (shadowing * sample.color + sample.emission) * (1.0 - step_transmittance);
            color += step_light * transmittance;
            transmittance *= step_transmittance;
        }

        t += step_size;
    }

    VolumetricsResult {
        color: color.extend(transmittance),
        depth: first_hit_depth,
    }
}

trait Smoothstep {
    fn smoothstep(self, edge0: Self, edge1: Self) -> Self;
}

impl Smoothstep for f32 {
    fn smoothstep(self, edge0: f32, edge1: f32) -> f32 {
        let t = ((self - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}
