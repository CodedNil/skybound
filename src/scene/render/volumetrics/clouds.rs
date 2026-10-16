use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4, Vec4Swizzles};

pub use crate::scene::climate::{
    CLOUD_BOTTOM as CLOUD_BOTTOM_HEIGHT, CLOUD_TOP as CLOUD_TOP_HEIGHT,
};
use crate::scene::climate::{Climate, clouds as layout};
use crate::scene::render::utils::Textures;
use crate::scene::{PLANET_RADIUS, ViewUniform};
pub use layout::LayerTraversal;

pub fn active_kinds(view: &ViewUniform) -> u32 {
    if view.weather.x < 0.0 {
        return layout::ALL_KINDS;
    }
    let climate = Climate::on_planet(Vec3::X * PLANET_RADIUS, view.time(), view.weather);
    (u32::from(climate.low_cloud > 0.001) * 3)
        | (u32::from(climate.middle_cloud > 0.001) * 12)
        | (u32::from(climate.high_cloud > 0.001) * layout::HIGH_KINDS)
        | (u32::from(climate.storm > 0.001) << layout::STORM)
}

fn region(point: Vec3, kind: usize, textures: &Textures) -> Vec4 {
    textures.weather(layout::region_coordinates(
        layout::column_point(point),
        kind,
    ))
}

fn region_seed(seed: Vec4, kind: usize) -> f32 {
    match layout::cloud_type(kind) {
        0 => seed.x,
        1 => seed.y,
        _ => seed.z,
    }
}

pub struct CloudInterval {
    pub bounds: Vec2,
    pub kinds: u32,
}

pub fn ray_event(
    origin: Vec3,
    direction: Vec3,
    distance: f32,
    view: &ViewUniform,
    active: u32,
    layers: &mut LayerTraversal,
    textures: &Textures,
) -> CloudInterval {
    layers.advance(origin, direction, distance, active);
    let point = origin + direction * distance;
    let mut interval = CloudInterval {
        bounds: Vec2::splat(layers.end - distance),
        kinds: 0,
    };
    let radius = point.length();
    let altitude = radius - PLANET_RADIUS;
    let mut pending = layers.kinds | (active & (1 << layout::STORM));
    while pending != 0 {
        let kind = pending.trailing_zeros() as usize;
        pending &= pending - 1;
        let bounds = layout::layer_bounds(kind);
        let weather = if kind != layout::STORM
            && altitude >= bounds.x - 128.0
            && altitude <= bounds.y + 128.0
        {
            region(point, kind, textures)
        } else {
            Vec4::ZERO
        };
        let mut event = layout::ray_event(
            point,
            direction,
            view.time(),
            kind,
            region_seed(weather, kind),
        );
        if kind != layout::STORM && altitude >= bounds.x && altitude <= bounds.y {
            let clearance = layout::column_clearance(altitude, weather.w, kind);
            if clearance > 16.0 {
                event = Vec2::splat(clearance.max(event.x));
            }
        }
        if event.x <= 0.0 {
            interval.kinds |= 1 << kind;
        }
        interval.bounds = interval.bounds.min(event);
    }
    interval
}

pub fn sample_clouds(point: Vec3, view: &ViewUniform, footprint: f32, textures: &Textures) -> f32 {
    sample_selected(point, view, footprint, layout::ALL_KINDS, textures)
}

pub fn sample_selected(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    kinds: u32,
    textures: &Textures,
) -> f32 {
    let radius = point.length();
    let altitude = radius - PLANET_RADIUS;
    if altitude < CLOUD_BOTTOM_HEIGHT || altitude > CLOUD_TOP_HEIGHT {
        return 0.0;
    }
    let mut pending = kinds & layout::kinds_at_altitude(altitude);
    if pending == 0 && kinds & (1 << layout::STORM) == 0 {
        return 0.0;
    }
    let weather = if view.weather.x >= 0.0 {
        view.weather
    } else {
        textures.weather_on_planet(point, view.time())
    };
    let climate = Climate::on_planet(point, view.time(), weather);
    let column_point = point * (PLANET_RADIUS / radius);
    let mut density = 0.0f32;
    while pending != 0 {
        let kind = pending.trailing_zeros() as usize;
        pending &= pending - 1;
        let strength = match layout::cloud_type(kind) {
            0 => climate.low_cloud,
            1 => climate.middle_cloud,
            _ => climate.high_cloud,
        };
        density += sample_layer(
            point.extend(altitude),
            column_point,
            view,
            footprint,
            kind,
            strength,
            textures,
        );
    }
    if kinds & (1 << layout::STORM) == 0 {
        return density;
    }
    density.max(super::storms::sample_storm(
        point,
        view.time(),
        climate.storm,
        footprint,
        textures,
    ))
}

fn sample_layer(
    point: Vec4,
    column_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    kind: usize,
    strength: f32,
    textures: &Textures,
) -> f32 {
    let altitude = point.w;
    let point = point.xyz();
    let noise = textures.noise(footprint);
    if strength <= 0.001 {
        return 0.0;
    }
    let bounds = layout::layer_bounds(kind);
    if altitude <= bounds.x || altitude >= bounds.y {
        return 0.0;
    }
    let weather = textures.weather(layout::region_coordinates(column_point, kind));
    let column = layout::column_bounds(weather.w, kind);
    if altitude <= column.x || altitude >= column.y {
        return 0.0;
    }
    let seed = region_seed(weather, kind);
    let coverage = layout::region_value(seed, view.time()).saturate() * strength;
    if coverage <= 0.0 {
        return 0.0;
    }
    let scale = layout::cloud_scale(kind);
    let flow = layout::flow(column_point, view.time(), kind);
    let mut billows = noise.clouds(flow.previous, scale);
    if flow.blend > 0.0 {
        billows = billows.lerp(noise.clouds(flow.current, scale), flow.blend);
    }
    billows.z = weather.w * 0.5 + billows.z * 0.5;
    let filter = (footprint * scale * 2.0).min(0.12);
    let mut body = layout::layer_density(altitude, coverage, kind, billows, filter) * strength;
    if body > 0.0 {
        let vertical = (point - column_point) * scale;
        let warp = Vec3::new(billows.x, billows.z, billows.w) * 1.7;
        let detail_scale = 0.000_45;
        let mut detail = noise.cloud_details(
            (flow.previous + vertical) * (detail_scale / scale) + warp,
            detail_scale,
        );
        if flow.blend > 0.0 {
            detail = detail.lerp(
                noise.cloud_details(
                    (flow.current + vertical) * (detail_scale / scale) + warp,
                    detail_scale,
                ),
                flow.blend,
            );
        }
        let erosion = detail.x * 0.5 + detail.y * 0.3 + detail.z * 0.2;
        let base = bounds.x + billows.z * layout::base_variation(kind);
        let depth = layout::column_height(billows.w, kind);
        let height = ((altitude - base) / depth).saturate();
        billows.y -= erosion * (0.02 + height * height * 0.14);
        body = layout::layer_density(altitude, coverage, kind, billows, filter) * strength;
        body *= (1.0 - detail.x * 0.35)
            * if layout::cloud_type(kind) == 2 {
                0.35
            } else {
                1.0
            };
    }
    body
}
