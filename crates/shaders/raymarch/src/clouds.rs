use skybound_shared::ViewUniform;
use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use spirv_std::num_traits::Float;

const BASE_SCALE: f32 = 0.005;
const BASE_TIME: f32 = 0.01;

const BASE_NOISE_SCALE: f32 = 0.01 * BASE_SCALE;
const WIND_DIRECTION_BASE: Vec3 = Vec3::new(1.0 * 0.1 * BASE_TIME, 0.0, 0.2 * 0.1 * BASE_TIME);

const WEATHER_NOISE_SCALE: f32 = 0.001 * BASE_SCALE;
const WIND_DIRECTION_WEATHER: Vec2 = Vec2::new(1.0 * 0.02 * BASE_TIME, 0.0);

const DETAIL_NOISE_SCALE: f32 = 0.2 * BASE_SCALE;
const WIND_DIRECTION_DETAIL: Vec3 = Vec3::new(1.0 * 0.2 * BASE_TIME, 0.0, -1.0 * 0.2 * BASE_TIME);

pub const CLOUD_BOTTOM_HEIGHT: f32 = 1000.0;
pub const CLOUD_TOP_HEIGHT: f32 = 33000.0;
pub const CLOUD_LAYER_SPACING: f32 = 2000.0;
pub const CLOUD_TOTAL_LAYERS: usize = 16;

const CLOUD_LAYER_HEIGHTS: [f32; 16] = [
    1950.0, 1900.0, 1850.0, 1800.0, 1750.0, 1600.0, 1550.0, 1500.0, 1400.0, 1350.0, 1200.0, 1100.0,
    900.0, 850.0, 700.0, 680.0,
];

const CLOUD_LAYER_OFFSETS: [Vec2; 16] = [
    Vec2::new(0.8, -0.6),
    Vec2::new(-0.4, 0.9),
    Vec2::new(0.1, -0.8),
    Vec2::new(0.6, 0.4),
    Vec2::new(-0.2, 0.2),
    Vec2::new(0.3, -0.9),
    Vec2::new(0.9, -0.7),
    Vec2::new(0.5, 0.1),
    Vec2::new(-0.7, 0.3),
    Vec2::new(0.0, -0.2),
    Vec2::new(-0.6, 0.5),
    Vec2::new(1.0, -0.5),
    Vec2::new(0.7, -0.3),
    Vec2::new(-0.1, 0.8),
    Vec2::new(-0.9, -0.4),
    Vec2::new(0.4, 0.6),
];

const CLOUD_LAYER_SCALES: [f32; 16] = [
    1.0, 0.9, 0.85, 0.85, 0.8, 0.8, 0.75, 0.75, 0.7, 0.7, 0.7, 0.6, 0.5, 0.45, 0.3, 0.3,
];

const CLOUD_LAYER_STRETCH: [f32; 16] = [
    1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.5, 1.8, 2.5, 3.5, 6.0, 10.0, 14.0, 18.0,
];

const CLOUD_LAYER_DETAILS: [f32; 16] = [
    0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95,
];

pub struct CloudLayer {
    pub index: u32,
    pub bottom: f32,
    pub height: f32,
    pub is_cirrus: bool,
    pub displacement: f32,
}

fn calculate_layer_v_offset(
    pos_xy: Vec2,
    index_u: u32,
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
) -> f32 {
    let disp_coord = pos_xy * 0.000_005 + Vec2::splat(index_u as f32 * 0.23);
    let disp_noise: f32 = weather_texture.sample_by_lod(*sampler, disp_coord, 0.0).x;
    (disp_noise - 0.5) * CLOUD_LAYER_SPACING * 0.85
}

pub fn get_cloud_layer(
    pos: Vec3,
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
    _time: f32,
) -> CloudLayer {
    let raw_index = (pos.z - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING;
    let index_u = (raw_index.floor().max(0.0) as u32).min(CLOUD_TOTAL_LAYERS as u32 - 1);

    if pos.z < CLOUD_BOTTOM_HEIGHT - CLOUD_LAYER_SPACING || index_u >= CLOUD_TOTAL_LAYERS as u32 {
        return CloudLayer {
            index: 0,
            bottom: 0.0,
            height: 0.0,
            is_cirrus: false,
            displacement: 0.0,
        };
    }

    let displacement = calculate_layer_v_offset(pos.xy(), index_u, weather_texture, sampler);
    let layer_bottom = CLOUD_BOTTOM_HEIGHT + index_u as f32 * CLOUD_LAYER_SPACING + displacement;
    let height = CLOUD_LAYER_HEIGHTS[index_u as usize];

    let is_within = pos.z >= layer_bottom && pos.z <= layer_bottom + height;
    let is_cirrus = index_u >= 12;

    CloudLayer {
        index: index_u,
        bottom: layer_bottom,
        height: if is_within { height } else { 0.0 },
        is_cirrus,
        displacement,
    }
}

pub fn get_cloud_layer_above(altitude: f32, above_count: f32) -> f32 {
    let target_index =
        ((altitude - CLOUD_BOTTOM_HEIGHT) / CLOUD_LAYER_SPACING + above_count).floor() as i32;
    if target_index < 0 || target_index >= CLOUD_TOTAL_LAYERS as i32 {
        return -1.0;
    }
    CLOUD_BOTTOM_HEIGHT + target_index as f32 * CLOUD_LAYER_SPACING
}

pub fn sample_clouds(
    pos: Vec3,
    view: &ViewUniform,
    time: f32,
    simple: bool,
    base_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    details_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    sampler: &spirv_std::Sampler,
) -> f32 {
    let cloud_layer = get_cloud_layer(pos, weather_texture, sampler, time);
    if cloud_layer.height <= 0.0 {
        return 0.0;
    }

    let layer_i = cloud_layer.index as usize;
    let weather_pos_2d = (pos.xy() + CLOUD_LAYER_OFFSETS[layer_i] * 10000.0) * WEATHER_NOISE_SCALE
        + time * WIND_DIRECTION_WEATHER;
    let weather_noise: Vec4 = weather_texture.sample_by_lod(*sampler, weather_pos_2d, 0.0);

    let mut weather_coverage = weather_noise.x * 1.2 - 0.4;
    let layer_fraction = layer_i as f32 / CLOUD_TOTAL_LAYERS as f32;
    let cloud_layer_coverage_a = 1.0
        - (weather_noise.z - layer_fraction)
            .abs()
            .smoothstep(0.0, 0.6);
    let cloud_layer_coverage_b = 1.0
        - (1.0 - layer_fraction)
            .abs()
            .smoothstep(0.0, weather_noise.z * 0.5);
    weather_coverage *= cloud_layer_coverage_a + cloud_layer_coverage_b;
    weather_coverage *= 1.0 - layer_fraction * 0.3;

    let lat_norm = (view.latitude.abs() * 0.8).clamp(0.0, 1.0);
    weather_coverage *= 0.5.lerp(1.0, lat_norm);

    if weather_coverage <= 0.0 {
        return 0.0;
    }

    let cloud_height = cloud_layer.height * (0.8 + weather_noise.w * 1.5);
    let h_coord = ((pos.z - cloud_layer.bottom) / cloud_height).clamp(0.0, 1.0);

    let mut h_profile = if cloud_layer.is_cirrus {
        (h_coord.smoothstep(0.0, 0.4) * (1.0 - h_coord).smoothstep(0.0, 0.7)).powf(1.5)
    } else if layer_i < 5 {
        let base = h_coord.smoothstep(0.0, 0.15);
        let top = (1.0 - h_coord).smoothstep(0.0, 0.6);
        (base * top).powf(0.7)
    } else {
        let base = h_coord.smoothstep(0.0, 0.2);
        let top = (1.0 - h_coord).smoothstep(0.0, 0.8);
        base * top
    };

    h_profile *= h_coord.smoothstep(0.0, 0.1) * (1.0 - h_coord).smoothstep(0.0, 0.1);
    if h_profile <= 0.0 {
        return 0.0;
    }

    let stretch = CLOUD_LAYER_STRETCH[layer_i];
    let base_scale = BASE_NOISE_SCALE * CLOUD_LAYER_SCALES[layer_i];
    let base_time = time * WIND_DIRECTION_BASE;

    let mut wind = WIND_DIRECTION_BASE.xy().normalize();
    if wind.length_squared() < 1e-12 {
        wind = Vec2::new(1.0, 0.0);
    }
    let wind_perp = Vec2::new(-wind.y, wind.x);
    let coord_along = pos.xy().dot(wind);
    let coord_perp = pos.xy().dot(wind_perp);
    let base_time_vec = Vec3::new(
        base_time.xy().dot(wind),
        base_time.xy().dot(wind_perp),
        base_time.z,
    );

    if simple {
        let shadow_pos =
            Vec3::new(coord_along * stretch, coord_perp, pos.z) * base_scale + base_time_vec;
        let shadow_noise: f32 = base_texture.sample_by_lod(*sampler, shadow_pos, 2.0).x;
        return (shadow_noise * h_profile) + weather_coverage - 1.0;
    }

    let noise_sample_pos = if cloud_layer.is_cirrus {
        Vec3::new(coord_along * stretch, coord_perp * 0.4, pos.z * 4.0) * (base_scale * 0.4)
            + base_time_vec
    } else {
        let leaning = h_coord * 150.0 * stretch;
        Vec3::new(coord_along * stretch + leaning, coord_perp, pos.z) * base_scale + base_time_vec
    };

    let base_noise: f32 = base_texture
        .sample_by_lod(*sampler, noise_sample_pos, 0.0)
        .x;
    let mut density = base_noise;
    if cloud_layer.is_cirrus {
        density = density.powf(3.5) * 0.5;
    } else if layer_i < 5 {
        density = (density * 1.4 - 0.1).clamp(0.0, 1.0);
    }

    let mut cloud_val = (density * h_profile) + weather_coverage - 1.0;
    if cloud_val <= 0.0 {
        return 0.0;
    }

    let detail_time = time * WIND_DIRECTION_DETAIL;
    let mut dw = WIND_DIRECTION_DETAIL.xy().normalize();
    if dw.length_squared() < 1e-12 {
        dw = wind;
    }
    let dperp = Vec2::new(-dw.y, dw.x);
    let dtime_vec = Vec3::new(
        detail_time.xy().dot(dw),
        detail_time.xy().dot(dperp),
        detail_time.z,
    );

    let dpos = Vec3::new(
        pos.xy().dot(dw) * stretch.max(1.0) * 0.5,
        pos.xy().dot(dperp),
        pos.z,
    ) * DETAIL_NOISE_SCALE
        * CLOUD_LAYER_SCALES[layer_i]
        - dtime_vec;
    let detail_noise: f32 = details_texture.sample_by_lod(*sampler, dpos, 0.0).x;

    let erosion_mask = if layer_i < 5 { 1.0 - h_coord } else { h_coord };
    let erosion = detail_noise * 0.4 * CLOUD_LAYER_DETAILS[layer_i] * (1.1 + erosion_mask);

    cloud_val = (cloud_val - erosion).clamp(0.0, 1.0);
    cloud_val
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
