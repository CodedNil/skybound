#![no_std]

mod lighting;
mod sky;
mod solids;
mod utils;
mod volumetrics;

use crate::lighting::henyey_greenstein;
use crate::sky::{get_sun_light_color, render_sky};
use crate::solids::raymarch_solids;
use crate::solids::ships::{estimate_ship_normal, mat_color, sdf_ship};
use crate::utils::{AtmosphereData, Textures, blue_noise, get_sun_position};
use crate::volumetrics::raymarch_volumetrics;
use skybound_shared::{ShipUniform, ViewUniform};
use spirv_std::glam::{Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

// ── Shared helpers ────────────────────────────────────────────────────────────

fn position_ndc_to_world(ndc_pos: Vec3, world_from_clip: Mat4) -> Vec3 {
    let world_pos = world_from_clip * ndc_pos.extend(1.0);
    world_pos.xyz() / world_pos.w
}

fn uv_to_ndc(uv: Vec2) -> Vec2 {
    uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0)
}

// ── Ship pass entry point ─────────────────────────────────────────────────────

/// Maximum t for ship raymarching (ships are near the camera).
const SHIP_MAX_T: f32 = 4000.0;
const SHIP_MAX_STEPS: i32 = 128;
const SHIP_EPSILON: f32 = 0.08;
const SHIP_MIN_STEP: f32 = 0.05;
const ENABLE_REFLECTION_VOLUMETRICS: bool = false;

#[spirv(fragment)]
pub fn ship_main(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] view: &ViewUniform,
    #[spirv(uniform, descriptor_set = 0, binding = 1)] ship: &ShipUniform,
    #[spirv(location = 0)] out_surface: &mut Vec4, // RGB=color, A=t (0=miss)
    #[spirv(location = 1)] out_gbuf: &mut Vec4,    // RGB=normal*0.5+0.5, A=specular
) {
    // Ship invalid/hidden.
    if ship.core_position.w < 0.0 {
        *out_surface = Vec4::ZERO;
        *out_gbuf = Vec4::ZERO;
        return;
    }

    let ndc = uv_to_ndc(uv);
    let world_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);
    let ro = view.world_position.xyz();
    let rd = (world_far - ro).normalize();

    let sun_pos = get_sun_position(
        view.planet_center(),
        view.planet_rotation,
        view.ro_relative(),
        view.latitude(),
    );
    let sun_dir = (sun_pos - ro).normalize();

    // Raymarch.
    let mut t = 0.0;
    let mut hit = false;
    let mut hit_mat = 0u32;

    for _ in 0..SHIP_MAX_STEPS {
        if t >= SHIP_MAX_T {
            break;
        }
        let p = ro + rd * t;
        let (d, mat) = sdf_ship(p, ship);
        if d < SHIP_EPSILON {
            hit = true;
            hit_mat = mat;
            break;
        }
        t += d.max(SHIP_MIN_STEP);
    }

    if !hit {
        *out_surface = Vec4::ZERO;
        *out_gbuf = Vec4::ZERO;
        return;
    }

    let p = ro + rd * t;
    let normal = estimate_ship_normal(p, ship);

    // Simple diffuse + ambient lighting.
    let dot_nl = normal.dot(sun_dir).max(0.0);
    let ambient = 0.06;
    let base_color = mat_color(hit_mat);
    let lit = base_color * (dot_nl * 0.9 + ambient);

    // Specular highlights for core (metallic).
    let specular = if hit_mat == 0 { 1.0f32 } else { 0.0f32 };
    // Fresnel rim for all materials.
    let view_dir = (ro - p).normalize();
    let fresnel = (1.0 - normal.dot(view_dir).abs()).powf(3.0) * 0.4;
    let color = lit + Vec3::splat(fresnel);

    *out_surface = color.extend(t); // A = t value (non-zero = hit)
    *out_gbuf = (normal * 0.5 + Vec3::splat(0.5)).extend(specular);
}

// ── Main scene entry point ────────────────────────────────────────────────────

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
    #[spirv(descriptor_set = 0, binding = 7)] ship_gbuf_tex: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_gbuffer: &mut Vec4,
    #[spirv(frag_depth)] out_frag_depth: &mut f32,
) {
    let frame_offset = (view.frame_count() * 0.618_034).fract();
    let dither = (blue_noise(uv * 1024.0) + frame_offset).fract();

    let ndc = uv_to_ndc(uv);
    let world_pos_far = position_ndc_to_world(ndc.extend(0.01), view.world_from_clip);

    let ro = view.world_position.xyz();
    let rd = (world_pos_far - ro).normalize();
    let max_dist = 1_000_000.0f32;

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

    // ── Read ship pass output ─────────────────────────────────────────────────
    let ship_surface: Vec4 = ship_surface_tex.sample(*sampler, uv);
    let ship_t = ship_surface.w; // 0.0 = no ship hit, >0 = hit distance
    let ship_hit = ship_t > 0.001;
    let ship_gbuf: Vec4 = ship_gbuf_tex.sample(*sampler, uv);

    // Limit solid march so it stops at the ship hull.
    let solid_t_max = if ship_hit { ship_t } else { max_dist };

    // ── Raymarch world solids (aurora spikes etc.) ────────────────────────────
    let solids = raymarch_solids(ro, rd, view, solid_t_max, dither, &textures);

    // ── Choose what's closest: ship or solid ─────────────────────────────────
    let (rendered_solid, solid_depth) = if ship_hit && ship_t <= solids.depth {
        (ship_surface.xyz(), ship_t)
    } else if solids.hit >= 1.0 {
        (solids.color, solids.depth)
    } else {
        (atmosphere.sky, max_dist)
    };

    let mut rendered_color = rendered_solid;

    // Reflected volumetrics for world solids (not ships).
    if ENABLE_REFLECTION_VOLUMETRICS
        && solids.hit >= 1.0
        && solids.refl_weight > 0.0
        && solids.depth <= solid_depth + 0.1
    {
        let refl_pos = ro + rd * solids.depth;
        let refl_rd = rd - 2.0 * solids.normal.dot(rd) * solids.normal;
        let refl_vols = raymarch_volumetrics(
            refl_pos,
            refl_rd,
            &atmosphere,
            view,
            solids.refl_depth,
            dither,
            &textures,
        );
        let refl_background = if solids.refl_hit > 0.5 {
            solids.refl_color
        } else {
            render_sky(refl_rd, ro_relative, sun_dir)
        };
        let effective_refl = refl_vols.color.xyz() + refl_background * refl_vols.color.w;
        rendered_color = rendered_color.lerp(effective_refl, solids.refl_weight);
    }

    // Volumetrics stop at whichever surface (ship or solid) is closer.
    let vol_t_max = if ship_hit {
        ship_t.min(solids.depth)
    } else {
        solids.depth
    };
    let volumetrics = raymarch_volumetrics(ro, rd, &atmosphere, view, vol_t_max, dither, &textures);
    rendered_color = volumetrics.color.xyz() + rendered_color * volumetrics.color.w;

    // ── Motion vectors + depth ────────────────────────────────────────────────
    let mut motion_vector = Vec2::ZERO;
    let mut gbuffer = Vec4::ZERO;
    let mut frag_depth = 0.0;

    let depth = {
        let geom = if ship_hit {
            ship_t.min(solids.depth)
        } else {
            solids.depth
        };
        geom.min(volumetrics.depth)
    };

    if depth < max_dist {
        let world_pos_far_unjittered =
            position_ndc_to_world(ndc.extend(0.01), view.world_from_clip_unjittered);
        let rd_unjittered = (world_pos_far_unjittered - ro).normalize();
        let world_pos_mv = ro + rd_unjittered * depth;

        let clip_pos_prev = view.prev_clip_from_world * world_pos_mv.extend(1.0);
        let ndc_prev = clip_pos_prev.xyz() / clip_pos_prev.w;
        let uv_prev = ndc_prev.xy() * vec2(0.5, -0.5) + 0.5;
        motion_vector = uv - uv_prev;

        // G-buffer: prefer ship if it's the closest surface.
        if ship_hit && ship_t <= depth + 0.1 {
            gbuffer = ship_gbuf; // RGB=normal*0.5+0.5, A=specular
        } else if solids.hit >= 1.0 && solids.depth <= depth + 0.1 {
            gbuffer = (solids.normal * 0.5 + Vec3::splat(0.5)).extend(solids.specular);
        }

        let clip_curr = view.clip_from_world * world_pos_mv.extend(1.0);
        frag_depth = clip_curr.z / clip_curr.w;
    }

    *out_color = rendered_color.extend(1.0).saturate();
    *out_motion = motion_vector.extend(0.0).extend(0.0);
    *out_gbuffer = gbuffer;
    *out_frag_depth = frag_depth;
}
