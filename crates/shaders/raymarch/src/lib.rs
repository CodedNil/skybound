#![no_std]

mod lighting;
mod sky;
mod solids;
mod utils;
mod volumetrics;

use crate::lighting::henyey_greenstein;
use crate::sky::{get_sun_light_color, render_sky};
use crate::solids::aur_spikes::raymarch_aur_spikes;
use crate::solids::ships::raymarch_ship;
use crate::utils::{AtmosphereData, Textures, blue_noise, get_sun_position};
use crate::volumetrics::raymarch_volumetrics;
use skybound_shared::{ShipUniform, ViewUniform};
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

const T_MAX: f32 = 1_000_000.0;

fn position_ndc_to_world(ndc_pos: Vec3, world_from_clip: Mat4) -> Vec3 {
    let world_pos = world_from_clip * ndc_pos.extend(1.0);
    world_pos.xyz() / world_pos.w
}

fn uv_to_ndc(uv: Vec2) -> Vec2 {
    uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0)
}

#[spirv(fragment)]
pub fn ship_main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] view: &ViewUniform,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] ship: &ShipUniform,
    #[spirv(location = 0)] out_surface: &mut Vec4,
) {
    let ndc = uv_to_ndc(uv);
    let world_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);
    let ro = view.world_position.xyz();
    let rd = (world_far - ro).normalize();

    let shade = raymarch_ship(ro, rd, view, ship);
    *out_surface = shade.color_depth;
}

#[spirv(fragment(depth_replacing))]
pub fn main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] view: &ViewUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] extra_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] ship_surface_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(frag_depth)] out_frag_depth: &mut f32,
) {
    let frame_offset = (view.frame_count() * 0.618_034).fract();
    let dither = (blue_noise(uv * 1024.0) + frame_offset).fract();

    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);

    let ro = view.world_position.xyz();
    let rd = (world_pos_far - ro).normalize();

    let sun_pos = get_sun_position(
        view.planet_center(),
        view.planet_rotation,
        view.ro_relative(),
        view.latitude(),
    );
    let sun_dir = (sun_pos - ro).normalize();
    let ro_relative = view.ro_relative();

    let cos_theta = sun_dir.dot(rd);
    let hg_forward = henyey_greenstein(cos_theta, 0.4);
    let hg_silver = henyey_greenstein(cos_theta, 0.95) * 0.003;
    let hg_back = henyey_greenstein(cos_theta, -0.05);
    let phase = hg_forward + hg_back * 0.15 + hg_silver * 0.2;

    let sky = render_sky(rd, ro_relative, sun_dir);
    let up = (ro - view.planet_center()).normalize();
    let sky_zenith = render_sky(up, ro_relative, sun_dir);
    let atmosphere = AtmosphereData {
        sun_pos,
        sky,
        sun: (get_sun_light_color(ro_relative, sun_dir) * 0.45 + sky * 0.18) * phase,
        ambient: sky_zenith * 0.7 + render_sky(-up, ro_relative, sun_dir) * 0.15 + sky * 0.15,
    };

    let textures = Textures {
        base: base_texture,
        details: details_texture,
        weather: weather_texture,
        extra: extra_texture,
        sampler,
    };

    // Read ship pass output
    let ship_surface: Vec4 = ship_surface_tex.sample(*sampler, uv);
    let ship_t = ship_surface.w;

    // Raymarch world solids
    let solids = raymarch_aur_spikes(ro, rd, view, ship_surface.w, dither, &textures);

    // Choose what's closest: ship or solid
    let mut rendered_color = if ship_t <= solids.color_depth.w {
        ship_surface.xyz()
    } else if solids.color_depth.w <= T_MAX {
        solids.color_depth.xyz()
    } else {
        atmosphere.sky
    };

    // Volumetrics pass
    let mut depth = ship_t.min(solids.color_depth.w);
    let volumetrics = raymarch_volumetrics(ro, rd, &atmosphere, view, depth, dither, &textures);
    rendered_color = volumetrics.color.xyz() + rendered_color * volumetrics.color.w;

    // Motion vectors + depth
    let mut motion_vector = Vec2::ZERO;
    let mut frag_depth = 0.0;

    depth = depth.min(volumetrics.depth);

    if depth < T_MAX {
        let world_pos_far_unjittered =
            position_ndc_to_world(ndc.extend(0.01), view.world_from_clip_unjittered);
        let rd_unjittered = (world_pos_far_unjittered - ro).normalize();
        let world_pos_mv = ro + rd_unjittered * depth;

        let clip_pos_prev = view.prev_clip_from_world * world_pos_mv.extend(1.0);
        let ndc_prev = clip_pos_prev.xyz() / clip_pos_prev.w;
        let uv_prev = ndc_prev.xy() * vec2(0.5, -0.5) + 0.5;
        motion_vector = uv - uv_prev;

        let clip_curr = view.clip_from_world * world_pos_mv.extend(1.0);
        frag_depth = clip_curr.z / clip_curr.w;
    }

    *out_color = rendered_color.extend(1.0).saturate();
    *out_motion = motion_vector.extend(0.0).extend(0.0);
    *out_frag_depth = frag_depth;
}
