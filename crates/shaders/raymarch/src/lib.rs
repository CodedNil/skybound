#![no_std]

pub mod aur_ocean;
pub mod clouds;
pub mod poles;
pub mod raymarch_solids;
pub mod sky;
pub mod utils;
pub mod volumetrics;

use crate::raymarch_solids::raymarch_solids;
use crate::sky::{get_sun_light_color, render_sky};
use crate::utils::{
    AtmosphereData, View, blue_noise, get_sun_position, position_ndc_to_world, uv_to_ndc,
};
use crate::volumetrics::raymarch_volumetrics;
use spirv_std::glam::{Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

pub struct FragmentOutput {
    pub color: Vec4,
    pub motion: Vec4,
    pub frag_depth: f32,
}

const INV_4_PI: f32 = 0.079_577_47;
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    INV_4_PI * (1.0 - g2) / (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5)
}

#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_uv: &mut Vec2,
) {
    let uv = vec2(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos: Vec2 = 2.0 * uv - Vec2::ONE;

    *out_pos = pos.extend(0.0).extend(1.0);
    *out_uv = vec2(uv.x, 1.0 - uv.y);
}

#[spirv(fragment)]
pub fn main_fs(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] view: &View,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &spirv_std::Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &spirv_std::Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] linear_sampler: &spirv_std::Sampler,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    // #[spirv(frag_depth)] out_frag_depth: &mut f32,
) {
    let frame_offset = (view.frame_count as f32 * 0.618_034).fract();
    let dither = (blue_noise(uv * 1024.0) + frame_offset).fract();

    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(ndc.extend(1.0), view.world_from_clip);

    let ro = view.world_position;
    let rd = (world_pos_far - ro).normalize();
    let mut t_max = 1_000_000.0;
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - ro).normalize();

    let cos_theta = sun_dir.dot(rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.95) * 0.003;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = hg_forward + hg_back * 0.15 + hg_silver * 0.2;

    let mut atmosphere = AtmosphereData {
        sun_pos,
        sky: render_sky(rd, view, sun_dir),
        sun: Vec3::ZERO,
        ambient: Vec3::ZERO,
    };
    atmosphere.sun =
        (get_sun_light_color(ro, view, sun_dir) * 0.45 + atmosphere.sky * 0.18) * phase;
    atmosphere.ambient = atmosphere.sky * 0.8
        + render_sky(Vec3::new(1.0, 0.0, 1.0).normalize(), view, sun_dir) * 0.2;

    let solids = raymarch_solids(ro, rd, view, t_max, view.time);
    let mut rendered_color = if solids.depth < t_max {
        solids.color
    } else {
        atmosphere.sky
    };
    t_max = solids.depth;

    let volumetrics = raymarch_volumetrics(
        ro,
        rd,
        &atmosphere,
        view,
        t_max,
        dither,
        view.time,
        base_texture,
        details_texture,
        weather_texture,
        linear_sampler,
    );
    rendered_color = volumetrics.color.xyz() + rendered_color * volumetrics.color.w;

    let mut motion_vector = Vec2::ZERO;
    let mut frag_depth = 0.0;
    let depth = solids.depth.min(volumetrics.depth);
    if depth < 1_000_000.0 {
        let world_pos_far_unjittered =
            position_ndc_to_world(ndc.extend(1.0), view.world_from_clip_unjittered);
        let rd_unjittered = (world_pos_far_unjittered - ro).normalize();
        let world_pos_mv = ro + rd_unjittered * depth;

        let clip_pos_prev = view.prev_clip_from_world * world_pos_mv.extend(1.0);
        let ndc_prev = clip_pos_prev.xyz() / clip_pos_prev.w;
        let uv_prev = ndc_prev.xy() * vec2(0.5, -0.5) + 0.5;
        motion_vector = uv - uv_prev;

        let clip_curr = view.clip_from_world * world_pos_mv.extend(1.0);
        frag_depth = clip_curr.z / clip_curr.w;
    }

    *out_color = rendered_color.extend(1.0).clamp(Vec4::ZERO, Vec4::ONE);
    *out_motion = motion_vector.extend(0.0).extend(0.0);
    // *out_frag_depth = frag_depth;
}
