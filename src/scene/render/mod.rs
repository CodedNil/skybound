pub(crate) mod solids;
mod utils;
mod volumetrics;

use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec3};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

use self::solids::aur_spikes::raymarch_aur_spikes;
use self::utils::{AtmosphereData, Textures, blue_noise, intersect_sphere};
use self::volumetrics::{raymarch_volumetrics, short_volume};
use crate::scene::atmosphere::{render_sky, sky_direction, sky_uv};
use crate::scene::surface::{Hit, Ray};
use crate::scene::{FrameUniform, life};

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
    let matrix = if jittered {
        view.world_from_clip
    } else {
        view.world_from_clip_unjittered
    };
    Ray {
        origin,
        direction: (world_point(uv, 0.01, matrix) - origin).normalize(),
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
        if c.position.w > 0.0 {
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
    // The pure aur ocean hides terrain from observers above the veil.
    if frame.view.surface_position(ray.origin).z > 0.0 {
        return closest;
    }
    let terrain = raymarch_aur_spikes(ray.origin, ray.direction, &frame.view, closest.distance());
    if terrain.distance() < closest.distance() {
        closest = terrain;
    }
    closest
}

fn energy_layers(ray: Ray, view: &crate::scene::ViewUniform, layers: &mut ThinLayers) {
    for index in 0..32 {
        let bolt = view.aur_bolts[index];
        if bolt[0].w <= 0.0 {
            break;
        }
        let start = energy_point(bolt[0].xyz(), view);
        let end = energy_point(bolt[1].xyz(), view);
        let axis = end - start;
        let length = axis.length_squared().max(0.001);
        let center = (start + end) * 0.5;
        let radius = (bolt[0].w * 12.0).max(ray.footprint * center.distance(ray.origin));
        let bound = axis.length() * 0.5 + radius * 3.0;
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
        let glow = (-separation / (radius * radius)).exp();
        let energy = bolt[1].w * bolt[0].w / core_width;
        let color = (vec3(0.70, 0.92, 1.0) * core + vec3(0.08, 0.38, 0.65) * glow * 0.12) * energy;
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

fn energy_point(point: Vec3, view: &crate::scene::ViewUniform) -> Vec3 {
    view.from_planet(crate::scene::planet::surface_point(point))
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
    let sky = sky_texture
        .sample_by_lod(
            *sampler,
            sky_uv(
                crate::scene::planet::surface_frame(frame.view.up(ray.origin)).transpose()
                    * ray.direction,
                crate::scene::planet::surface_frame(frame.view.up(ray.origin)).transpose()
                    * ray.light,
                frame.view.sky_horizon.x,
            ),
            0.0,
        )
        .xyz();
    AtmosphereData {
        sky,
        ambient: frame.view.ambient_color.xyz() + sky * 0.12,
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
#[spirv(fragment(depth_replacing, entry_point_name = "opaque"))]
pub fn opaque(
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
    #[spirv(frag_depth)] out_depth: &mut f32,
) {
    let ray = scene_ray(uv, frame, true);
    let hit = opaque_hit(ray, frame);
    let environment = atmosphere(uv, frame, sampler, sky_texture);
    let _ = (base_texture, details_texture, weather_texture);
    let point = ray.origin + ray.direction * hit.distance();
    let clip = frame.view.clip_from_world * point.extend(1.0);
    let depth = if hit.is_hit() { clip.z / clip.w } else { 0.0 };
    let mv = motion(
        uv,
        if hit.is_hit() {
            hit.previous_position
        } else {
            point
        },
        frame,
    );
    let color = if hit.is_hit() {
        hit.color()
    } else {
        environment.sky
    };
    *out_color = match frame.view.times.z as u32 {
        1 => Vec3::splat(depth.max(0.0).sqrt()),
        2 => hit.normal * 0.5 + 0.5,
        3 => (mv.abs() / frame.view.times.w.max(0.0001)).extend(0.0),
        _ => color,
    }
    .extend(1.0);
    *out_motion = mv.extend(0.0).extend(0.0);
    *out_normal = (hit.normal * 0.5 + 0.5).extend(1.0);
    *out_depth = depth;
}
#[spirv(fragment(depth_replacing, entry_point_name = "volume"))]
pub fn volume(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] depth_texture: &Depth,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_normal: &mut Vec4,
    #[spirv(frag_depth)] out_depth: &mut f32,
) {
    let ray = scene_ray(uv, frame, false);
    let limit = depth_limit(uv, frame, depth_texture, ray);
    let atmosphere = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        weather: weather_texture,
        sampler,
    };
    let dither =
        (blue_noise(uv * frame.volume_size.xy()) + frame.view.frame_count() * 0.618_034).fract();
    let volume = raymarch_volumetrics(
        ray.origin,
        ray.direction,
        &atmosphere,
        &frame.view,
        limit,
        dither,
        2.0 / frame.volume_size.y,
        &textures,
    );
    let position = ray.origin + ray.direction * volume.depth;
    let altitude = position.distance(frame.view.planet_center()) - crate::scene::PLANET_RADIUS;
    let current = if altitude <= 3700.0 {
        frame.view.planet_rotation * crate::scene::aur::CURRENT
    } else if altitude > crate::scene::climate::CLOUD_TOP {
        Vec3::ZERO
    } else {
        let weather = if frame.view.weather.x >= 0.0 {
            frame.view.weather
        } else {
            textures.weather_on_planet(frame.view.to_planet(position), frame.view.time())
        };
        frame.view.planet_rotation
            * crate::scene::climate::Climate::on_planet(
                frame.view.to_planet(position),
                frame.view.time(),
                weather,
            )
            .wind
    };
    let previous = position - current * frame.view.times.w;
    *out_color = volume.color;
    *out_motion = motion(uv, previous, frame).extend(0.0).extend(0.0);
    *out_normal = vec3(volume.depth, limit, volume.color.xyz().max_element()).extend(1.0);
    let clip = frame.view.clip_from_world * position.extend(1.0);
    *out_depth = if volume.depth < T_MAX {
        clip.z / clip.w
    } else {
        0.0
    };
}
#[spirv(fragment(entry_point_name = "transparent"))]
pub fn transparent(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] weather_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] sky_texture: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] depth_texture: &Depth,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_transmittance: &mut Vec4,
    #[spirv(location = 2)] out_prefix: &mut Vec4,
) {
    let mut ray = scene_ray(uv, frame, false);
    ray.max_distance = depth_limit(uv, frame, depth_texture, ray);
    let mut hits = ThinLayers {
        color: [Hit::MISS.color_depth; 8],
        transmission: [Vec4::ONE; 8],
        count: 0,
    };
    energy_layers(ray, &frame.view, &mut hits);
    for i in 0..2 {
        let c = &frame.players[i];
        if c.position.w <= 0.0 {
            continue;
        }
        let interval = crate::scene::geometry::intersect_sphere(
            ray.origin - c.position.xyz(),
            ray.direction,
            c.position.w,
        );
        if interval.x > interval.y || interval.y < 0.0 || interval.x > ray.max_distance {
            continue;
        }
        life::visit_thin(ray, c, |hit| hits.add(hit));
    }
    let environment = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        weather: weather_texture,
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
            0.5,
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
            0.5,
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
    let light = crate::scene::planet::surface_frame(view.up(view.world_position.xyz())).transpose()
        * view.pole_direction();
    let direction = sky_direction(uv, view.sky_horizon.x);
    let light = vec3((1.0 - light.z * light.z).max(0.0).sqrt(), 0.0, light.z);
    let origin = vec3(0.0, 0.0, crate::scene::PLANET_RADIUS + view.altitude());
    let source = origin + light * view.pole_position.xyz().distance(view.world_position.xyz());
    *out_color = (render_sky(direction, origin, source) * view.sky_horizon.y).extend(1.0);
}
