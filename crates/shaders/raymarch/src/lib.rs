#![no_std]

pub mod aur_ocean;
pub mod clouds;
pub mod poles;
pub mod raymarch;
pub mod sky;
pub mod utils;
pub mod volumetrics;

use crate::raymarch::raymarch_solids;
use crate::sky::{get_sun_light_color, render_sky};
use crate::utils::{AtmosphereData, blue_noise, get_sun_position};
use crate::volumetrics::raymarch_volumetrics;
use core::f32::consts::PI;
use skybound_shared::ViewUniform;
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

#[spirv(vertex)]
fn main_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_uv: &mut Vec2,
) {
    let uv = vec2(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos: Vec2 = 2.0 * uv - Vec2::ONE;

    *out_pos = pos.extend(0.0).extend(1.0);
    *out_uv = vec2(uv.x, 1.0 - uv.y);
}

#[spirv(fragment(depth_replacing))]
fn main_fs(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] view: &ViewUniform,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(frag_depth)] out_frag_depth: &mut f32,
) {
    // Spatially-varying blue noise offset by a per-frame golden-ratio step so each
    let frame_offset = (view.frame_count as f32 * 0.618_034).fract();
    let dither = (blue_noise(uv * 1024.0) + frame_offset).fract();

    // Reconstruct world-space position for a ray through the far plane
    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);

    // Ray origin & dir
    let ro = view.world_position;
    let rd = (world_pos_far - ro).normalize();
    let mut t_max = 1_000_000.0;
    let sun_pos = get_sun_position(view);
    let sun_dir = (sun_pos - ro).normalize();

    // Phase functions for silver and back scattering
    let cos_theta = sun_dir.dot(rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.95) * 0.003;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = hg_forward + hg_back * 0.15 + hg_silver * 0.2;

    // Precalculate sun, sky and ambient colors
    let sky = render_sky(rd, view, sun_dir);
    let atmosphere = AtmosphereData {
        sun_pos,
        sky,
        sun: (get_sun_light_color(view, sun_dir) * 0.45 + sky * 0.18) * phase,
        ambient: sky * 0.8 + render_sky(vec3(1.0, 0.0, 1.0).normalize(), view, sun_dir) * 0.2,
    };

    // Run solids raymarch (solids are independent of volumetrics)
    let solids = raymarch_solids(ro, rd, view, t_max);
    let mut rendered_color = if solids.depth < t_max {
        solids.color
    } else {
        atmosphere.sky
    };
    t_max = solids.depth;

    // Sample the volumetrics
    let volumetrics = raymarch_volumetrics(
        ro,
        rd,
        &atmosphere,
        view,
        t_max,
        dither,
        view.time,
        *base_texture,
        *details_texture,
        *weather_texture,
        *sampler,
    );
    rendered_color = volumetrics.color.xyz() + rendered_color * volumetrics.color.w;

    // Motion vectors + depth
    let mut motion_vector = Vec2::ZERO;
    let mut frag_depth = 0.0;
    let depth = solids.depth.min(volumetrics.depth);
    if depth < 1_000_000.0 {
        let world_pos_far_unjittered =
            position_ndc_to_world(ndc.extend(0.01), view.world_from_clip_unjittered);
        let rd_unjittered = (world_pos_far_unjittered - ro).normalize();
        let world_pos_mv = ro + rd_unjittered * depth;

        // Motion vector: current UV minus where this world point was last frame
        let clip_pos_prev = view.prev_clip_from_world * world_pos_mv.extend(1.0);
        let ndc_prev = clip_pos_prev.xyz() / clip_pos_prev.w;
        let uv_prev = ndc_prev.xy() * vec2(0.5, -0.5) + 0.5;
        motion_vector = uv - uv_prev;

        // Clip-space depth for the depth buffer (reverse-Z NDC)
        let clip_curr = view.clip_from_world * world_pos_mv.extend(1.0);
        frag_depth = clip_curr.z / clip_curr.w;
    }

    *out_color = rendered_color.extend(1.0).saturate();
    *out_motion = motion_vector.extend(0.0).extend(0.0);
    *out_frag_depth = frag_depth;

    // *out_color = Vec3::splat(frag_depth * 2000.0).extend(1.0); // DEBUG: depth
    // *out_color = (motion_vector * 200.0 + 0.5).extend(0.5).extend(1.0); // DEBUG: motion vectors
}

const INV_4_PI: f32 = 0.25 * (1.0 / PI);
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    INV_4_PI * (1.0 - g2) / (1.0 + g2 - 2.0 * g * cos_theta).powf(1.5)
}

/// Convert a ndc space position to world space
fn position_ndc_to_world(ndc_pos: Vec3, world_from_clip: Mat4) -> Vec3 {
    let world_pos = world_from_clip * ndc_pos.extend(1.0);
    world_pos.xyz() / world_pos.w
}

/// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: Vec2) -> Vec2 {
    uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0)
}
