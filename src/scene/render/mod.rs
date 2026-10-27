use crate::scene::{
    FrameUniform, PLANET_RADIUS, ViewUniform,
    atmosphere::{render_sky, sky_direction, sky_uv},
    aur::{CURRENT, EJECTION_CELL, EJECTION_WINDOW, ejection, ejection_clearance, energy_color},
    climate::{CLOUD_TOP, cloud_wind},
    life::{
        self, CreatureInstance, HERD,
        surface::{Hit, Ray},
    },
    math::intersect_sphere,
    planet::{surface_coordinates, surface_frame},
    render::{
        clouds::short_volume,
        textures::{AtmosphereData, Textures},
    },
};
use spirv_std::{
    Image, Sampler,
    glam::{FloatExt, Mat4, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3, vec4},
    num_traits::Float,
    spirv,
};

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

// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(fragment(entry_point_name = "ejection_lookup"))]
pub fn ejection_lookup(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(location = 0)] out: &mut Vec4,
) {
    let x = pixel.x as u32;
    // The cell nearest the camera among those sharing this texel.
    let view = &frame.view;
    let (centre, _) = view.place(Vec3::ZERO).arc();
    let window = EJECTION_WINDOW as f32;
    let texel = vec2((x % EJECTION_WINDOW) as f32, pixel.y.floor());
    let offset = texel - centre;
    let wrapped = offset - (offset / window).floor() * window;
    let cell = centre + wrapped
        - Vec2::select(
            wrapped.cmpge(Vec2::splat(window * 0.5)),
            Vec2::splat(window),
            Vec2::ZERO,
        );
    let mut event = ejection(cell, frame.view.time());
    if x >= EJECTION_WINDOW * 2 {
        // The cell's clearance shares a texel with its loop's shape.
        event[2].x = ejection_clearance(cell, frame.view.time());
        *out = event[2];
        return;
    }
    if x < EJECTION_WINDOW {
        let mut active = 0u32;
        for index in 0..9 {
            let neighbour = cell + vec2((index / 3) as f32 - 1.0, (index % 3) as f32 - 1.0);
            active |= u32::from(ejection(neighbour, frame.view.time())[0].w > 0.0) << index;
        }
        event[0].w += (active << 1) as f32;
    }
    *out = if x < EJECTION_WINDOW {
        event[0]
    } else {
        event[1]
    };
}

fn scene_ray(uv: Vec2, frame: &FrameUniform, jittered: bool) -> Ray {
    let view = &frame.view;
    let origin = view.world_position.xyz();
    // The bright veil below lights everything above it from beneath, fading with height as it does
    // into the haze.
    let veil =
        vec3(0.30, 0.26, 0.40) * view.sky_horizon.y * (-view.altitude().max(0.0) / 3000.0).exp();
    Ray {
        origin,
        direction: view.ray_direction(uv, jittered),
        light: view.pole_direction(),
        ambient: (view.ambient_color.xyz() + veil).min(Vec3::splat(0.8)),
        radiance: view.pole_color.xyz().min(Vec3::splat(1.5)),
        max_distance: T_MAX,
        footprint: 2.0 / frame.viewport.w,
    }
}

/// Whether a ray passes within a creature's bounding sphere before the nearest hit so far.
fn reaches(ray: Ray, c: &CreatureInstance, before: f32) -> bool {
    let bounds = intersect_sphere(ray.origin - c.position.xyz(), ray.direction, c.position.w);
    c.position.w > 0.0 && bounds.x <= bounds.y && bounds.y > 0.0 && bounds.x < before
}

fn opaque_hit(ray: Ray, frame: &FrameUniform) -> Hit {
    // The herd is drawn after this pass, each creature by its species' own pipeline; see
    // [`creature`]. Only the player and their mount are traced here, by their own tracers.
    let mut closest = Hit::MISS;
    for i in 0..2 {
        let c = &frame.players[i];
        let bounds = intersect_sphere(ray.origin - c.position.xyz(), ray.direction, c.position.w);
        if c.position.w > 0.0 && bounds.x <= bounds.y && bounds.y > 0.0 {
            let hit = life::render_player(
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

/// How far the opaque scene lies from the camera at a pixel.
fn depth_limit(uv: Vec2, frame: &FrameUniform, depth: &Depth) -> f32 {
    let pixel = (uv * frame.viewport.zw()).as_ivec2();
    let d = depth.fetch(pixel);
    if d > 0.0 {
        let p = frame.view.world_from_clip
            * (uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0))
                .extend(d)
                .extend(1.0);
        (p.xyz() / p.w)
            .distance(frame.view.world_position.xyz())
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
    // Air in front of the ocean scatters horizon light and the aur glow rising beneath it, never
    // the sky behind the planet.
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

/// Where a point lies on the screen in UV, and whether it lies in front.
fn screen(clip_from_world: Mat4, point: Vec3) -> Vec3 {
    let clip = clip_from_world * point.extend(1.0);
    (clip.xy() / clip.w.max(1.0e-6) * vec2(0.5, -0.5) + 0.5).extend(clip.w)
}

/// How far a point has moved across the screen since last frame, from where it was then. Both are
/// projected without jitter: upscalers expect motion to hold only movement.
fn motion(position: Vec3, previous: Vec3, frame: &FrameUniform) -> Vec2 {
    let now = screen(frame.view.unjittered_clip_from_world, position);
    let then = screen(frame.view.prev_clip_from_world, previous);
    if now.z > 0.0 && then.z > 0.0 {
        now.xy() - then.xy()
    } else {
        Vec2::ZERO
    }
}

/// A solid hit's colour (or a debug view of it), motion, normal and depth, as the opaque targets
/// take them.
fn solid_output(hit: Hit, ray: Ray, frame: &FrameUniform) -> (Vec4, Vec4, Vec4, f32) {
    let point = ray.origin + ray.direction * hit.distance();
    let clip = frame.view.clip_from_world * point.extend(1.0);
    let depth = clip.z / clip.w;
    let mv = motion(point, hit.previous_position, frame);
    let color = match frame.view.times.z as u32 {
        1 => Vec3::splat(depth.max(0.0).sqrt()),
        2 => hit.normal * 0.5 + 0.5,
        3 => (mv.abs() / frame.view.times.w.max(0.0001)).extend(0.0),
        _ => hit.color(),
    };
    (
        color.extend(1.0),
        mv.extend(0.0).extend(0.0),
        (hit.normal * 0.5 + 0.5).extend(1.0),
        depth,
    )
}

/// Solids, with the sky behind them; every pixel writes depth and motion.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
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
    if hit.is_hit() {
        (*out_color, *out_motion, *out_normal, *out_depth) = solid_output(hit, ray, frame);
        return;
    }
    let point = ray.origin + ray.direction * hit.distance();
    let sky = atmosphere(uv, frame, sampler, sky_texture).sky;
    *out_color = match frame.view.times.z as u32 {
        1 => Vec3::ZERO,
        2 => hit.normal * 0.5 + 0.5,
        3 => Vec3::ZERO,
        _ => sky,
    }
    .extend(1.0);
    *out_motion = motion(point, point, frame).extend(0.0).extend(0.0);
    *out_normal = (hit.normal * 0.5 + 0.5).extend(0.5);
    *out_depth = 0.0;
}

/// A creature of the herd, drawn as a quad over its bounding sphere.
///
/// The quad faces the camera at the sphere's centre, just wide enough to cover its outline, or
/// covers the whole view when the camera is close enough to be inside it or nearly so. Each
/// species is drawn by its own pipeline over the creatures of that species alone, so its fragment
/// shader holds one species' tracer and not every one's.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(vertex(entry_point_name = "creature_quad"))]
pub fn creature_quad(
    #[spirv(vertex_index)] vertex: u32,
    #[spirv(instance_index)] instance: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] creatures: &[CreatureInstance],
    #[spirv(position)] out_position: &mut Vec4,
    #[spirv(flat, location = 0)] out_instance: &mut u32,
) {
    *out_instance = instance;
    let corner = vec2(
        if vertex == 1 || vertex == 2 || vertex == 4 {
            1.0
        } else {
            -1.0
        },
        if vertex == 2 || vertex == 4 || vertex == 5 {
            1.0
        } else {
            -1.0
        },
    );
    let c = &creatures[instance as usize];
    let radius = c.position.w;
    if radius <= 0.0 {
        *out_position = Vec4::ZERO;
        return;
    }
    let to = c.position.xyz() - frame.view.world_position.xyz();
    let distance = to.length();
    if distance < radius * 1.5 {
        *out_position = corner.extend(0.5).extend(1.0);
        return;
    }
    let forward = to / distance;
    let across = if forward.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = forward.cross(across).normalize();
    let up = right.cross(forward);
    // Wide enough at the centre's distance to hold the cone of rays that graze the sphere.
    let half = radius * distance / (distance * distance - radius * radius).sqrt() * 1.05;
    let point = c.position.xyz() + (right * corner.x + up * corner.y) * half;
    *out_position = frame.view.clip_from_world * point.extend(1.0);
}

/// The creature beneath a [`creature_quad`], traced by the tracer of the one species this
/// pipeline is specialised to and depth-tested against the solids already drawn.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(fragment(depth_replacing, entry_point_name = "creature"))]
pub fn creature(
    #[spirv(frag_coord)] fragment: Vec4,
    #[spirv(flat, location = 0)] instance: u32,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] creatures: &[CreatureInstance],
    #[spirv(spec_constant(id = 0, default = 0))] species: u32,
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
    #[spirv(location = 2)] out_normal: &mut Vec4,
    #[spirv(frag_depth)] out_depth: &mut f32,
) {
    let uv = (fragment.xy() - frame.viewport.xy()) / frame.viewport.zw();
    let ray = scene_ray(uv, frame, true);
    let c = &creatures[instance as usize];
    let hit = if reaches(ray, c, T_MAX) {
        life::render_opaque_of(species, ray, c)
    } else {
        Hit::MISS
    };
    if !hit.is_hit() {
        spirv_std::arch::kill();
    }
    (*out_color, *out_motion, *out_normal, *out_depth) = solid_output(hit, ray, frame);
}

/// The clouds along every pixel's ray, marched apart from the rest of the volume so that each
/// shader stays small.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(fragment(entry_point_name = "clouds"))]
pub fn clouds(
    #[spirv(frag_coord)] fragment: Vec4,
    #[spirv(location = 0)] uv: Vec2,
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
    #[spirv(descriptor_set = 0, binding = 15)] shapes_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 11)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] cloud_cells: &[Vec4],
    #[spirv(location = 0)] out_slab: &mut Vec4,
    #[spirv(location = 1)] out_front: &mut Vec4,
) {
    let ray = scene_ray(uv, frame, true);
    let geometry_limit = depth_limit(uv, frame, depth_texture);
    let limit = if geometry_limit >= T_MAX - 1.0 {
        10_000_000.0
    } else {
        geometry_limit
    };
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        shapes: shapes_texture,
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
    // A phase for the cloud steps: interleaved gradient noise across the screen, so neighbouring
    // pixels sample different depths, advanced by the golden ratio each frame, so every pixel
    // visits its phases evenly and the upscaler's history averages them without leaving speckle.
    let pixel = fragment.xy().floor();
    let gradient =
        (52.982_918 * (0.067_110_56 * pixel.x + 0.005_837_15 * pixel.y).fract_gl()).fract_gl();
    let turn = (frame.view.frame_count() as u32 % 256) as f32 * 0.618_034;
    let slab = march::march_clouds(
        (gradient + turn).fract_gl(),
        ray.origin,
        ray.direction,
        &frame.view,
        limit,
        ray.footprint,
        &textures,
        cloud_cells,
    );
    *out_slab = slab.radiance.extend(slab.transmittance);
    *out_front = slab.depths;
}

/// Lay the volume over the scene and slot the thin layers in front of it; all three are drawn at
/// every pixel. The view's motion follows whichever the pixel mostly shows: an upscaler given the
/// solids' motion beneath a cloud smears the cloud.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(fragment(entry_point_name = "volume_composite"))]
pub fn volume_composite(
    #[spirv(location = 0)] uv: Vec2,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] scene: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] volume: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] thin_color: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] thin_transmission: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] thin_prefix: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] scene_motion: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 7)] volume_motion: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
) {
    let size = frame.viewport.zw();
    let pixel = (uv * size).floor().clamp(Vec2::ZERO, size - 1.0).as_ivec2();
    let background = scene.fetch(pixel).xyz();
    let solids: Vec4 = scene_motion.fetch(pixel);
    *out_color = background.extend(1.0);
    *out_motion = solids;
    if frame.view.times.z > 0.5 {
        return;
    }
    let fog: Vec4 = volume.fetch(pixel);
    let drifting: Vec4 = volume_motion.fetch(pixel);
    let cover = ((0.7 - fog.w) / 0.4).clamp(0.0, 1.0);
    *out_motion = solids.lerp(drifting, cover * cover * (3.0 - 2.0 * cover));
    let profile = frame.view.camera_position.w as u32;
    if profile & 32 != 0 {
        *out_color = (fog.xyz() + fog.w * background).extend(1.0);
        return;
    }
    let thin: Vec4 = thin_color.fetch(pixel);
    if profile & 64 != 0 {
        *out_color = thin.xyz().extend(1.0);
        return;
    }
    let transmission = thin_transmission.fetch(pixel).xyz();
    // The thin layers lie in front of the volume's leading part, which is taken out of it so it is
    // not counted twice.
    let prefix: Vec4 = thin_prefix.fetch(pixel);
    let opening = prefix.w.max(0.0001);
    let remainder = (fog.xyz() - prefix.xyz()).max(Vec3::ZERO) / opening;
    let through = (fog.w / opening).clamp(0.0, 1.0);
    *out_color = (thin.xyz() + transmission * (remainder + through * background)).extend(1.0);
}

/// The aur, its arcs and the pole lights, with the clouds slotted in: along the same jittered ray
/// as the solids at every pixel, so an upscaler is given the whole scene alike.
// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
#[spirv(fragment(entry_point_name = "volume"))]
pub fn volume(
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
    #[spirv(descriptor_set = 0, binding = 15)] shapes_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 17)] cloud_slab: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 18)] cloud_front: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_motion: &mut Vec4,
) {
    let ray = scene_ray(uv, frame, true);
    let geometry_limit = depth_limit(uv, frame, depth_texture);
    let limit = if geometry_limit >= T_MAX - 1.0 {
        10_000_000.0
    } else {
        geometry_limit
    };
    let atmosphere = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        shapes: shapes_texture,
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
    let size = frame.viewport.zw();
    let texel = (uv * size).floor().clamp(Vec2::ZERO, size - 1.0).as_ivec2();
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
            depths: front,
        },
    );
    let position = ray.origin + ray.direction * volume.depth;
    let altitude = position.distance(frame.view.planet_center()) - PLANET_RADIUS;
    let current = if altitude <= ocean::BODY_TOP_HEIGHT {
        frame.view.planet_rotation * CURRENT
    } else if altitude > CLOUD_TOP {
        Vec3::ZERO
    } else {
        frame.view.planet_rotation * cloud_wind(frame.view.to_planet(position).normalize())
    };
    let previous = position - current * frame.view.times.w;
    *out_color = volume.color;
    *out_motion = motion(position, previous, frame).extend(0.0).extend(0.0);
}

// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
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
    #[spirv(descriptor_set = 0, binding = 15)] shapes_texture: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 11)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 12)] _cloud_cells: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 16)] creatures: &[CreatureInstance],
    #[spirv(location = 0)] out_color: &mut Vec4,
    #[spirv(location = 1)] out_transmittance: &mut Vec4,
    #[spirv(location = 2)] out_prefix: &mut Vec4,
) {
    let mut ray = scene_ray(uv, frame, true);
    let geometry_limit = depth_limit(uv, frame, depth_texture);
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
    for i in 0..HERD {
        let c = &creatures[i];
        if reaches(ray, c, ray.max_distance) {
            life::visit_thin(ray, c, |hit| hits.add(hit));
        }
    }
    for i in 0..2 {
        let c = &frame.players[i];
        if c.position.w <= 0.0 {
            continue;
        }
        let interval = intersect_sphere(ray.origin - c.position.xyz(), ray.direction, c.position.w);
        if interval.x > interval.y || interval.y < 0.0 || interval.x > ray.max_distance {
            continue;
        }
        life::visit_player_thin(ray, c, |hit| hits.add(hit));
    }
    if hits.count == 0 {
        *out_color = Vec4::ZERO;
        *out_transmittance = Vec4::ONE;
        *out_prefix = Vec4::W;
        return;
    }
    let environment = atmosphere(uv, frame, sampler, sky_texture);
    let textures = Textures {
        base: base_texture,
        details: details_texture,
        shapes: shapes_texture,
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
    *out_color = color.extend(1.0);
    *out_transmittance = transmission.extend(if hits.count > 0 { hits.color[0].w } else { 0.0 });
    *out_prefix = prefix.extend(prefix_t);
}

// The game's own passes, left out of the preview module.
#[cfg(not(feature = "preview"))]
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
