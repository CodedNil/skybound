pub(crate) mod solids;
mod utils;
mod volumetrics;

use self::{
    solids::aur_spikes::raymarch_aur_spikes,
    utils::{AtmosphereData, Textures, blue_noise},
    volumetrics::raymarch_volumetrics,
};
use crate::scene::{
    FrameUniform, atmosphere::sky_uv, geometry::intersect_sphere, life,
    lighting::henyey_greenstein, surface,
};
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

pub(crate) const T_MAX: f32 = 1_000_000.0;

fn position_ndc_to_world(ndc_pos: Vec3, world_from_clip: Mat4) -> Vec3 {
    let world_pos = world_from_clip * ndc_pos.extend(1.0);
    world_pos.xyz() / world_pos.w
}

fn uv_to_ndc(uv: Vec2) -> Vec2 {
    uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0)
}

#[spirv(fragment(depth_replacing, entry_point_name = "main"))]
pub fn main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_normal: &mut Vec4,
    #[spirv(frag_depth)] out_frag_depth: &mut f32,
) {
    let view = &frame.view;
    let frame_offset = (view.frame_count() * 0.618_034).fract();
    let dither = (blue_noise(uv * 1024.0) + frame_offset).fract();

    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);

    let ro = view.world_position.xyz();
    let rd = (world_pos_far - ro).normalize();

    let sun_pos = view.sun_position.xyz();
    let sun_dir = (sun_pos - ro).normalize();

    let cos_theta = sun_dir.dot(rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.95) * 0.003;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = hg_forward + hg_back * 0.15 + hg_silver * 0.2;

    let sky: Vec4 =
        sky_texture.sample_by_lod(*sampler, sky_uv(rd, sun_dir, view.sky_horizon.x), 0.0);
    let sky = sky.xyz();
    let atmosphere = AtmosphereData {
        sun_pos,
        sky,
        sun: (view.sun_color.xyz() * 0.45 + sky * 0.18) * phase,
        ambient: view.ambient_color.xyz() + sky * 0.15,
    };

    let textures = Textures {
        base: base_texture,
        details: details_texture,
        weather: weather_texture,
        sampler,
    };

    let mut closest = solids::ShadeResult::MISS;
    let creature = frame.player;
    let interval = intersect_sphere(ro - creature.position.xyz(), rd, creature.position.w);
    if creature.position.w > 0.0 && interval.x < interval.y && interval.y > 0.0 {
        closest = life::render_player(
            surface::Ray {
                origin: ro,
                direction: rd,
                light: sun_dir,
                max_distance: T_MAX,
            },
            &creature,
        );
    }
    let mut volumetrics = raymarch_volumetrics(
        ro,
        rd,
        &atmosphere,
        view,
        closest.distance(),
        dither,
        &textures,
    );
    let terrain = raymarch_aur_spikes(
        ro,
        rd,
        view,
        closest.distance().min(volumetrics.opaque_distance),
    );
    if terrain.distance() < closest.distance() {
        closest = terrain;
        volumetrics = raymarch_volumetrics(
            ro,
            rd,
            &atmosphere,
            view,
            closest.distance(),
            dither,
            &textures,
        );
    }
    let background = if closest.is_hit() {
        closest.color()
    } else {
        sky
    };
    let rendered_color = volumetrics.color.xyz() + background * volumetrics.color.w;
    let normal = if closest.is_hit() {
        closest.normal
    } else {
        Vec3::Z
    };
    let mut depth = closest.distance();

    let mut motion_vector = Vec2::ZERO;
    let mut frag_depth = 0.0;

    depth = depth.min(volumetrics.depth);

    let world_pos_far_unjittered =
        position_ndc_to_world(ndc.extend(0.01), view.world_from_clip_unjittered);
    let rd_unjittered = (world_pos_far_unjittered - ro).normalize();
    let world_pos_mv = ro + rd_unjittered * depth;
    let previous_pos = if depth == closest.color_depth.w && closest.is_hit() {
        closest.previous_position + (rd_unjittered - rd) * depth
    } else {
        world_pos_mv
    };
    let clip_pos_prev = view.prev_clip_from_world * previous_pos.extend(1.0);
    if clip_pos_prev.w > 0.0 {
        let ndc_prev = clip_pos_prev.xyz() / clip_pos_prev.w;
        motion_vector = uv - (ndc_prev.xy() * vec2(0.5, -0.5) + 0.5);
    }
    if depth < T_MAX {
        let clip_curr = view.clip_from_world * (ro + rd * depth).extend(1.0);
        frag_depth = clip_curr.z / clip_curr.w;
    }

    let display_color = match view.times.z as u32 {
        1 => Vec3::splat(frag_depth.max(0.0).sqrt()),
        2 => normal * 0.5 + 0.5,
        3 => (motion_vector.abs() / view.times.w.max(0.0001)).extend(0.0),
        _ => rendered_color,
    };
    *out_color = display_color.extend(1.0).saturate();
    *out_motion = motion_vector.extend(0.0).extend(0.0);
    *out_normal = (normal * 0.5 + 0.5).extend(1.0);
    *out_frag_depth = frag_depth;
}
