use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

use crate::scene::atmosphere::{render_sky, sky_direction, sky_uv};
use crate::scene::aur::{
    CURRENT, EJECTION_GRID_HALF, EJECTION_GRID_WIDTH, ejection, ejection_clearance, energy_color,
};
use crate::scene::climate::{CLOUD_TOP, cloud_wind};
use crate::scene::life::surface::{Hit, Ray};
use crate::scene::math::intersect_sphere;
use crate::scene::planet::surface_frame;
use crate::scene::render::clouds::short_volume;
use crate::scene::render::textures::{AtmosphereData, Textures};
use crate::scene::{FrameUniform, PLANET_RADIUS, ViewUniform, life};
pub mod clouds;
mod march;
pub mod ocean;
mod poles;
mod spikes;
mod textures;

pub(crate) const T_MAX: f32 = 1_000_000.0;

type Depth = Image!(2D, type=f32, depth=true, sampled=true, components=1);

struct ThinLayers {
    color: [Vec4; 8],
    transmission: [Vec4; 8],
    count: u32,
}

impl ThinLayers {
    fn add(&mut self, hit: Hit) {
        if !hit.is_hit() {
            return;
        }
        let mut color = hit.color_depth;
        let mut transmission =
            hit.transmittance
                .extend(if hit.part == u32::MAX { 1.0 } else { 0.0 });
        for index in 0..8 {
            if color.w < self.color[index].w {
                core::mem::swap(&mut color, &mut self.color[index]);
                core::mem::swap(&mut transmission, &mut self.transmission[index]);
            }
        }
        self.count = (self.count + 1).min(8);
    }
}

#[spirv(fragment(entry_point_name = "ejection_lookup"))]
pub fn ejection_lookup(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(location = 0)] out: &mut Vec4,
) {
    let x = pixel.x as u32;
    let cell = vec2(
        (x % EJECTION_GRID_WIDTH) as f32 - EJECTION_GRID_HALF as f32,
        pixel.y.floor() - 1.0,
    );
    if x >= EJECTION_GRID_WIDTH * 2 {
        *out = Vec4::splat(ejection_clearance(cell, frame.view.time()));
        return;
    }
    let mut event = ejection(cell, frame.view.time());
    if x < EJECTION_GRID_WIDTH {
        let mut active = 0u32;
        for index in 0..9 {
            let neighbour = cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
            active |= u32::from(ejection(neighbour, frame.view.time())[0].w > 0.0) << index;
        }
        event[0].w += (active << 1) as f32;
    }
    *out = if x < EJECTION_GRID_WIDTH {
        event[0]
    } else {
        event[1]
    };
}

fn world_point(uv: Vec2, depth: f32, matrix: Mat4) -> Vec3 {
    let p = matrix
        * (uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0))
            .extend(depth)
            .extend(1.0);
    p.xyz() / p.w
}

fn scene_ray(uv: Vec2, frame: &FrameUniform, jittered: bool) -> Ray {
    let view = &frame.view;
    let origin = view.world_position.xyz();
    Ray {
        origin,
        direction: view.ray_direction(uv, jittered),
        light: view.pole_direction(),
        ambient: view.ambient_color.xyz().min(Vec3::splat(0.8)),
        radiance: view.pole_color.xyz().min(Vec3::splat(1.5)),
        max_distance: T_MAX,
        footprint: 2.0 / frame.viewport.w,
    }
}

fn opaque_hit(ray: Ray, frame: &FrameUniform) -> Hit {
    let mut closest = Hit::MISS;
    for i in 0..2 {
        let c = &frame.players[i];
        let bounds = intersect_sphere(ray.origin - c.position.xyz(), ray.direction, c.position.w);
        if c.position.w > 0.0 && bounds.x <= bounds.y && bounds.y > 0.0 {
            let hit = life::render_opaque(
                Ray {
                    max_distance: closest.distance(),
                    ..ray
                },
                c,
            );
            if hit.distance() < closest.distance() {
                closest = hit;
            }
        }
    }
    if frame.view.camera_position.w as u32 & 1024 != 0 {
        return closest;
    }
    let spire = spikes::trace(
        ray.origin,
        ray.direction,
        &frame.view,
        closest.distance(),
        ray.footprint,
    );
    if spire.distance() < closest.distance() {
        closest = spire;
    }
    closest
}

fn energy_layers(ray: Ray, view: &ViewUniform, layers: &mut ThinLayers) {
    for index in 0..32 {
        let bolt = view.aur_bolts[index];
        if bolt[0].w <= 0.0 {
            break;
        }
        let start = bolt[0].xyz();
        let end = bolt[1].xyz();
        let axis = end - start;
        let length = axis.length_squared().max(0.001);
        let center = (start + end) * 0.5;
        let radius = (bolt[0].w * 12.0).max(ray.footprint * center.distance(ray.origin));
        let bound = axis.length() * 0.5 + radius * 6.0;
        let bounds = intersect_sphere(ray.origin - center, ray.direction, bound);
        if bounds.y < 0.0 || bounds.x > ray.max_distance {
            continue;
        }
        let offset = ray.origin - start;
        let along_ray = axis.dot(ray.direction);
        let along_origin = axis.dot(offset);
        let along = ((along_origin - along_ray * ray.direction.dot(offset))
            / (length - along_ray * along_ray).max(0.001))
        .clamp(0.0, 1.0);
        let point_on_bolt = start + axis * along;
        let distance = (point_on_bolt - ray.origin).dot(ray.direction);
        if distance <= 0.0 || distance >= ray.max_distance {
            continue;
        }
        let point = ray.origin + ray.direction * distance;
        let separation = point.distance_squared(point_on_bolt);
        let core_width = bolt[0].w.max(ray.footprint * distance * 0.5);
        let core = (-separation / (core_width * core_width)).exp();
        // A Lorentzian halo keeps the long, soft tails of light in haze.
        let glow = 1.0 / (1.0 + separation / (radius * radius) * 4.0);
        let energy = bolt[1].w * bolt[0].w / core_width;
        let phase = view.time() * 3.0 - along * 18.0 + start.x * 0.0002;
        let tint = energy_color(0.5 + phase.sin() * 0.2, 0.55 + (phase * 0.7).cos() * 0.2);
        let color = (tint.lerp(Vec3::ONE, 0.5) * core + tint * glow * 0.3) * energy;
        if color.max_element() < 0.001 {
            continue;
        }
        layers.add(Hit {
            color_depth: color.extend(distance),
            normal: Vec3::ZERO,
            previous_position: point,
            transmittance: Vec3::ONE,
            part: u32::MAX,
        });
    }
}

fn depth_limit(uv: Vec2, frame: &FrameUniform, depth: &Depth, ray: Ray) -> f32 {
    let pixel = (uv * frame.viewport.zw()).as_ivec2();
    let d = depth.fetch(pixel);
    if d > 0.0 {
        world_point(uv, d, frame.view.world_from_clip)
            .distance(ray.origin)
            .min(T_MAX)
    } else {
        T_MAX
    }
}

fn atmosphere(
    uv: Vec2,
    frame: &FrameUniform,
    sampler: &Sampler,
    sky_texture: &Image!(2D, type=f32, sampled=true),
) -> AtmosphereData {
    let ray = scene_ray(uv, frame, false);
    let local = surface_frame(frame.view.up(ray.origin)).transpose();
    let direction = local * ray.direction;
    let light = local * ray.light;
    let horizon = frame.view.sky_horizon.x;
    let lookup = |direction: Vec3| {
        sky_texture
            .sample_by_lod(*sampler, sky_uv(direction, light, horizon), 0.0)
            .xyz()
    };
    // Air in front of the ocean scatters horizon light and the aur glow
    // rising beneath it, never the sky behind the planet.
    let level = direction
        .with_z(direction.z.max(horizon + 0.03))
        .normalize();
    let glow = vec3(0.30, 0.24, 0.42)
        * frame.view.sky_horizon.y
        * (-frame.view.altitude().max(0.0) / 30_000.0).exp();
    AtmosphereData {
        sky: lookup(direction),
        haze: lookup(level) + glow * (0.5 - direction.z * 0.5).clamp(0.0, 1.0) * 0.35,
    }
}

fn motion(uv: Vec2, position: Vec3, frame: &FrameUniform) -> Vec2 {
    let clip = frame.view.prev_clip_from_world * position.extend(1.0);
    if clip.w > 0.0 {
        uv - (clip.xy() / clip.w * vec2(0.5, -0.5) + 0.5)
    } else {
        Vec2::ZERO
    }
}

/// Solids, with the sky behind them; every pixel writes depth and motion.
#[spirv(fragment(depth_replacing, entry_point_name = "opaque"))]
pub fn opaque(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 4)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_normal: &mut Vec4,
    #[spirv(frag_depth)] out_depth: &mut f32,
) {
    let ray = scene_ray(uv, frame, true);
    let hit = opaque_hit(ray, frame);
    let point = ray.origin + ray.direction * hit.distance();
    let (depth, mv, color) = if hit.is_hit() {
        let clip = frame.view.clip_from_world * point.extend(1.0);
        (
            clip.z / clip.w,
            motion(uv, hit.previous_position, frame),
            hit.color(),
        )
    } else {
        let sky = atmosphere(uv, frame, sampler, sky_texture).sky;
        (0.0, motion(uv, point, frame), sky)
    };
    *out_color = match frame.view.times.z as u32 {
        1 => Vec3::splat(depth.max(0.0).sqrt()),
        2 => hit.normal * 0.5 + 0.5,
        3 => (mv.abs() / frame.view.times.w.max(0.0001)).extend(0.0),
        _ => color,
    }
    .extend(1.0);
    *out_motion = mv.extend(0.0).extend(0.0);
    *out_normal = (hit.normal * 0.5 + 0.5).extend(if hit.is_hit() { 1.0 } else { 0.5 });
    *out_depth = depth;
}

/// The clouds along every pixel's ray, marched apart from the rest of the
/// volume so that each shader stays small.
#[spirv(fragment(entry_point_name = "clouds"))]
pub fn clouds(
    #[spirv(frag_coord)] fragment: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] depth_texture: &Depth,
    #[spirv(descriptor_set = 0, binding = 6)] transmittance_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] ejection_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 8)] cache_shape: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 9)] cloud_cover: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 10)] cloud_light: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 13)] cloud_bounds: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 14)] cloud_towers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 11)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] cloud_cells: &[Vec4],
    #[spirv(location = 0)] out_slab: &mut Vec4,
    #[spirv(location = 1)] out_front: &mut Vec4,
) {
    // Each frame traces one pixel of every 2x2 block; the resolve pass
    // rebuilds the others from history.
    let traced = fragment.xy().floor() * 2.0 + block_offset(frame);
    let uv = (traced + 0.5) / frame.history.zw();
    let ray = scene_ray(uv, frame, false);
    let geometry_limit = depth_limit(uv, frame, depth_texture, ray);
    let limit = if geometry_limit >= T_MAX - 1.0 {
        10_000_000.0
    } else {
        geometry_limit
    };
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        transmittance: transmittance_texture,
        ejections: ejection_texture,
        cache_shape,
        cloud_cover,
        cloud_layers,
        cloud_light,
        cloud_bounds,
        cloud_towers,
        cached: true,
        sampler,
    };
    // White noise per pixel and frame: a phase for the cloud steps that the
    // cloud history averages away without leaving a pattern.
    let pixel = traced.as_uvec2();
    let seed = pixel.x.wrapping_mul(73_856_093)
        ^ pixel.y.wrapping_mul(19_349_663)
        ^ (frame.view.frame_count() as u32).wrapping_mul(83_492_791);
    let slab = march::march_clouds(
        f32::from_bits((seed >> 9) | 0x3f80_0000) - 1.0,
        ray.origin,
        ray.direction,
        &frame.view,
        limit,
        ray.footprint,
        &textures,
        cloud_cells,
    );
    *out_slab = slab.radiance.extend(slab.transmittance);
    *out_front = Vec4::splat(slab.front);
}

/// Which pixel of each 2x2 block the clouds trace this frame.
fn block_offset(frame: &FrameUniform) -> Vec2 {
    match frame.history.y as u32 & 3 {
        0 => vec2(0.0, 0.0),
        1 => vec2(1.0, 1.0),
        2 => vec2(1.0, 0.0),
        _ => vec2(0.0, 1.0),
    }
}

/// Rebuild the clouds at full resolution: the pixel traced this frame is
/// kept, the other three come from where history saw them, held within
/// the range of the freshly traced clouds around them.
#[spirv(fragment(entry_point_name = "cloud_resolve"))]
pub fn cloud_resolve(
    #[spirv(frag_coord)] fragment: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] fresh: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] fresh_front: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] history: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] history_front: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_slab: &mut Vec4,
    #[spirv(location = 1)] out_front: &mut Vec4,
) {
    let pixel = fragment.xy().floor();
    let block = (pixel * 0.5).floor();
    let own: Vec4 = fresh.fetch(block.as_ivec2());
    let own_front: Vec4 = fresh_front.fetch(block.as_ivec2());
    let traced = (pixel - (block * 2.0 + block_offset(frame)))
        .abs()
        .max_element()
        < 0.5;
    if frame.history.x < 0.5 {
        *out_slab = own;
        *out_front = own_front;
        return;
    }
    // Where this pixel's clouds were last frame: they drift with the wind.
    let size = frame.history.zw();
    let uv = (pixel + 0.5) / size;
    let ray = scene_ray(uv, frame, false);
    let depth = own_front.x.min(400_000.0);
    let position = ray.origin + ray.direction * depth;
    let wind = frame.view.planet_rotation * cloud_wind(frame.view.to_planet(position).normalize());
    let previous = uv - motion(uv, position - wind * frame.view.times.w, frame);
    if previous.cmplt(Vec2::ZERO).any() || previous.cmpgt(Vec2::ONE).any() {
        *out_slab = own;
        *out_front = own_front;
        return;
    }
    let past: Vec4 = history.sample_by_lod(*sampler, previous, 0.0);
    let past_front: Vec4 = history_front.fetch((previous * size).as_ivec2());
    // The range of the fresh clouds in the blocks around this pixel.
    let mut low = own;
    let mut high = own;
    for index in 0..9 {
        let offset = vec2((index % 3) as f32 - 1.0, (index / 3) as f32 - 1.0);
        let neighbour = (block + offset)
            .clamp(Vec2::ZERO, (size * 0.5).ceil() - 1.0)
            .as_ivec2();
        let value: Vec4 = fresh.fetch(neighbour);
        low = low.min(value);
        high = high.max(value);
    }
    // The pixel traced this frame also averages with its history, so the
    // noise of each frame's step phases settles instead of shimmering.
    let held = past.clamp(low, high);
    if traced {
        *out_slab = held.lerp(own, 0.35);
        *out_front = own_front;
        return;
    }
    *out_slab = held;
    *out_front = if past_front.x < own_front.x * 4.0 {
        past_front
    } else {
        own_front
    };
}

/// The aur, its arcs and the pole lights, with the clouds slotted in; like
/// the clouds, traced at one pixel of each 2x2 block per frame.
#[spirv(fragment(entry_point_name = "volume"))]
pub fn volume(
    #[spirv(frag_coord)] fragment: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] depth_texture: &Depth,
    #[spirv(descriptor_set = 0, binding = 6)] transmittance_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] ejection_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 8)] cache_shape: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 9)] cloud_cover: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 10)] cloud_light: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 13)] cloud_bounds: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 14)] cloud_towers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 11)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 15)] cloud_slab: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 16)] cloud_front: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_metadata: &mut Vec4,
) {
    let traced = fragment.xy().floor() * 2.0 + block_offset(frame);
    let uv = (traced + 0.5) / frame.history.zw();
    let ray = scene_ray(uv, frame, false);
    let geometry_limit = depth_limit(uv, frame, depth_texture, ray);
    let limit = if geometry_limit >= T_MAX - 1.0 {
        10_000_000.0
    } else {
        geometry_limit
    };
    let atmosphere = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        transmittance: transmittance_texture,
        ejections: ejection_texture,
        cache_shape,
        cloud_cover,
        cloud_layers,
        cloud_light,
        cloud_bounds,
        cloud_towers,
        cached: true,
        sampler,
    };
    let texel = traced.as_ivec2();
    let clouds: Vec4 = cloud_slab.fetch(texel);
    let front: Vec4 = cloud_front.fetch(texel);
    let volume = march::integrate(
        ray.origin,
        ray.direction,
        &frame.view,
        limit,
        ray.footprint,
        &atmosphere,
        &textures,
        march::CloudSlab {
            radiance: clouds.xyz(),
            transmittance: clouds.w,
            front: front.x,
        },
    );
    let position = ray.origin + ray.direction * volume.depth;
    let altitude = position.distance(frame.view.planet_center()) - PLANET_RADIUS;
    let current = if altitude <= 3700.0 {
        frame.view.planet_rotation * CURRENT
    } else if altitude > CLOUD_TOP {
        Vec3::ZERO
    } else {
        frame.view.planet_rotation * cloud_wind(frame.view.to_planet(position).normalize())
    };
    let previous = position - current * frame.view.times.w;
    *out_color = volume.color;
    *out_motion = motion(uv, previous, frame).extend(0.0).extend(0.0);
    *out_metadata = vec3(volume.depth, limit, volume.color.xyz().max_element()).extend(1.0);
}

#[spirv(fragment(entry_point_name = "transparent"))]
pub fn transparent(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] depth_texture: &Depth,
    #[spirv(descriptor_set = 0, binding = 6)] transmittance_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] ejection_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 8)] cache_shape: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 9)] cloud_cover: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 10)] cloud_light: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 13)] cloud_bounds: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 14)] cloud_towers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 11)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] _cloud_cells: &[Vec4],
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_transmittance: &mut Vec4,
    #[spirv(location = 2)] out_prefix: &mut Vec4,
) {
    let mut ray = scene_ray(uv, frame, false);
    let geometry_limit = depth_limit(uv, frame, depth_texture, ray);
    ray.max_distance = if geometry_limit >= T_MAX - 1.0 {
        10_000_000.0
    } else {
        geometry_limit
    };
    let mut hits = ThinLayers {
        color: [vec4(0.0, 0.0, 0.0, 10_000_000.0); 8],
        transmission: [Vec4::ONE; 8],
        count: 0,
    };
    energy_layers(ray, &frame.view, &mut hits);
    for i in 0..2 {
        let c = &frame.players[i];
        if c.position.w <= 0.0 {
            continue;
        }
        let interval = intersect_sphere(ray.origin - c.position.xyz(), ray.direction, c.position.w);
        if interval.x > interval.y || interval.y < 0.0 || interval.x > ray.max_distance {
            continue;
        }
        life::visit_thin(ray, c, |hit| hits.add(hit));
    }
    if hits.count == 0 && ray.max_distance >= 1000.0 {
        *out_color = Vec4::ZERO;
        *out_transmittance = Vec4::ONE;
        *out_prefix = Vec4::W;
        return;
    }
    let environment = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        transmittance: transmittance_texture,
        ejections: ejection_texture,
        cache_shape,
        cloud_cover,
        cloud_layers,
        cloud_light,
        cloud_bounds,
        cloud_towers,
        cached: true,
        sampler,
    };
    let mut color = Vec3::ZERO;
    let mut transmission = Vec3::ONE;
    let mut prefix = Vec3::ZERO;
    let mut prefix_t = 1.0;
    let mut cursor = 0.0;
    for i in 0..hits.count as usize {
        let hit = hits.color[i];
        let length = hit.w - cursor;
        let fog = short_volume(
            ray.origin + ray.direction * cursor,
            ray.direction,
            &environment,
            &frame.view,
            length,
            &textures,
        );
        if hits.transmission[i].w > 0.5 {
            color += transmission * fog.color.w * hit.xyz();
            continue;
        }
        color += transmission * fog.color.xyz();
        transmission *= fog.color.w;
        prefix += prefix_t * fog.color.xyz();
        prefix_t *= fog.color.w;
        color += transmission * hit.xyz();
        transmission *= hits.transmission[i].xyz();
        cursor = hit.w;
    }
    if ray.max_distance < 1000.0 {
        let fog = short_volume(
            ray.origin + ray.direction * cursor,
            ray.direction,
            &environment,
            &frame.view,
            (ray.max_distance - cursor).max(0.0),
            &textures,
        );
        color += transmission * fog.color.xyz();
        transmission *= fog.color.w;
        cursor = -1.0;
    }
    *out_color = color.extend(cursor);
    *out_transmittance = transmission.extend(if hits.count > 0 { hits.color[0].w } else { 0.0 });
    *out_prefix = prefix.extend(prefix_t);
}

#[spirv(fragment(entry_point_name = "sky"))]
pub fn sky(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(location = 0)] out_color: &mut Vec4,
) {
    let view = &frame.view;
    let light =
        surface_frame(view.up(view.world_position.xyz())).transpose() * view.pole_direction();
    let direction = sky_direction(uv, view.sky_horizon.x);
    let light = vec3((1.0 - light.z * light.z).max(0.0).sqrt(), 0.0, light.z);
    let origin = vec3(0.0, 0.0, PLANET_RADIUS + view.altitude());
    let source = origin + light * view.pole_position.xyz().distance(view.world_position.xyz());
    *out_color = (render_sky(direction, origin, source) * view.sky_horizon.y).extend(1.0);
}
