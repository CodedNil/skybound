use spirv_std::glam::{FloatExt, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec4};
use spirv_std::{Image, Sampler, spirv};

use super::utils::Textures;
use super::volumetrics::aur_ocean;
use crate::scene::{FrameUniform, ViewUniform, cache as grid};

fn atlas_uv(c: Vec2, cascade: u32) -> Vec2 {
    let pixel = (c / grid::spacing(cascade) + grid::SIZE as f32 * 0.5)
        .clamp(Vec2::splat(0.5), Vec2::splat(grid::SIZE as f32 - 0.5));
    (pixel + vec2(cascade as f32, 0.0) * grid::SIZE as f32)
        / vec2(
            (grid::SIZE * grid::CASCADES) as f32,
            (grid::SIZE * grid::ROWS) as f32,
        )
}

pub fn sample(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
) -> Option<(Vec4, Vec4)> {
    if view.cache_origin.w < 0.5 || !textures.cached {
        return None;
    }
    let c = grid::coordinates(point, view)?;
    let spatial = grid::cascade(c);
    let filter = grid::filter_level(footprint).min((grid::CASCADES - 1) as f32);
    let cascade = spatial.max(filter as u32);
    if cascade >= grid::CASCADES {
        return None;
    }
    let uv = atlas_uv(c, cascade);
    let mut shape = textures
        .cache_shape
        .sample_by_lod(*textures.sampler, uv, 0.0);
    let mut column = textures
        .cache_columns
        .sample_by_lod(*textures.sampler, uv, 0.0);
    if cascade + 1 < grid::CASCADES {
        let edge = c.abs().max_element() / (grid::spacing(cascade) * grid::SIZE as f32 * 0.5);
        let blend = edge
            .smoothstep(0.72, 0.90)
            .max((filter - cascade as f32).clamp(0.0, 1.0));
        if blend > 0.0 {
            let uv = atlas_uv(c, cascade + 1);
            shape = shape.lerp(
                textures
                    .cache_shape
                    .sample_by_lod(*textures.sampler, uv, 0.0),
                blend,
            );
            column = column.lerp(
                textures
                    .cache_columns
                    .sample_by_lod(*textures.sampler, uv, 0.0),
                blend,
            );
        }
    }
    Some((shape, column))
}

#[spirv(fragment(entry_point_name = "column_cache"))]
pub fn column_cache(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] _sky: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] transmittance: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] ejections: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)]
    _cloud_records: &[crate::scene::cloud::Cloud],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] _cloud_grid: &[u32],
    #[spirv(location = 0)] shape: &mut Vec4,
    #[spirv(location = 1)] column: &mut Vec4,
) {
    let cascade = pixel.x as u32 / grid::SIZE;
    let row = pixel.y as u32 / grid::SIZE;
    let spacing = grid::spacing(cascade);
    let c = (pixel.xy()
        - vec2(cascade as f32, row as f32) * grid::SIZE as f32
        - grid::SIZE as f32 * 0.5)
        * spacing;
    let view = &frame.view;
    let point = grid::point(c, view);
    let textures = Textures {
        base,
        details,
        weather,
        transmittance,
        ejections,
        sampler,
        cache_shape: _sky,
        cache_columns: _sky,
        cached: false,
    };
    let base_wave = aur_ocean::wave_base(point, view.time(), spacing * 0.5, &textures);
    let wave = base_wave + aur_ocean::wave_detail(point, view.time(), spacing * 0.5, &textures);
    let basis = crate::scene::planet::surface_frame(point);
    let epsilon = (spacing * 0.5).max(200.0);
    let east = aur_ocean::wave_base(
        point + basis.x_axis * epsilon,
        view.time(),
        spacing * 0.5,
        &textures,
    ) - base_wave;
    let north = aur_ocean::wave_base(
        point + basis.y_axis * epsilon,
        view.time(),
        spacing * 0.5,
        &textures,
    ) - base_wave;
    let normal = (basis.z_axis - (basis.x_axis * east + basis.y_axis * north) * (10.0 / epsilon))
        .normalize();
    *shape = vec4(wave / 4096.0, normal.x, normal.y, normal.z);
    let material_point = point.normalize() * (crate::scene::PLANET_RADIUS + wave - 100.0);
    let material = aur_ocean::sample_ocean(
        material_point,
        view,
        spacing * 0.5,
        spirv_std::glam::vec3(1.0e10, 0.0, 0.0),
        wave,
        normal,
        Vec4::ZERO,
        &textures,
    );
    *column = (material.emission + material.color * view.ambient_color.xyz() * 0.2)
        .extend(material.density.max(0.01));
}
