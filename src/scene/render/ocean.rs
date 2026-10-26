use crate::scene::{
    FrameUniform, PLANET_RADIUS, ViewUniform,
    atmosphere::POLE_RADIANCE,
    aur::{
        CURRENT, EJECTION_HEIGHT, EJECTION_WIDTH, aur_climate, energy_color, nearest_spire_at,
        spire_axes,
    },
    cache as grid,
    math::{cube, fourth, rotated, smoothstep_mean, square},
    planet::{light_position, surface_frame},
    render::{march::VolumeSample, textures::Textures},
};
use core::f32::consts::TAU;
use spirv_std::{
    Image, Sampler,
    glam::{FloatExt, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, vec2, vec3, vec4},
    num_traits::Float,
    spirv,
};

pub const OCEAN_TOP_HEIGHT: f32 = EJECTION_HEIGHT + EJECTION_WIDTH * 3.0;

/// The body and its veil stay below this altitude; only arcs rise above it.
pub const BODY_TOP_HEIGHT: f32 = 462.5;

/// Cached wave heights are stored as a fraction of this, in half precision.
pub const WAVE_RANGE: f32 = 512.0;

/// Everything about a column that varies over kilometres, not metres: lightning, sunlight reaching
/// the veil and the regional climate. A ray reuses it across the short stretch of veil it crosses.
#[derive(Clone, Copy)]
pub struct Column {
    pub flash: Vec3,
    pub sun: Vec3,
    pub climate: Vec4,
}

/// How the ocean meets a spire: the veil climbs it, the current parts around it and sheds eddies
/// downstream, and energy gathers at the contact.
#[derive(Clone, Copy)]
pub struct Wrap {
    pub lift: f32,
    pub warp: Vec3,
    pub glow: f32,
}

impl Wrap {
    pub const NONE: Self = Self {
        lift: 0.0,
        warp: Vec3::ZERO,
        glow: 0.0,
    };
}

/// Wisps are thin ridges where the detail noise crosses its mean. Filtering pulls the noise towards
/// that mean, so a distant footprint would otherwise turn every column into a ridge and wash the
/// ocean out to a pale glow.
fn wisp(noise: f32, footprint: f32) -> f32 {
    let resolved = cube((1.0 - noise.abs() * 2.0).max(0.0));
    let unresolved = (footprint * 0.16).max(1.0).log2().smoothstep(1.0, 3.5);
    resolved.lerp(0.2, unresolved)
}

pub(crate) fn wave_base(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let storm = aur_climate(position.normalize(), time).x;
    let noise = textures.noise(footprint);
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    let drift = noise
        .billows(flowing * 0.00002 + Vec3::Z * time * 0.001, 0.00002)
        .xyz()
        * 2.0
        - 1.0;
    let mut warped = flowing * 0.000112 + drift * 0.8;
    warped += vec3(
        (warped.y + time * 0.07).sin(),
        (warped.z - time * 0.06).sin(),
        (warped.x + time * 0.05).sin(),
    ) * 0.4;
    let broad = noise.billows(warped + Vec3::Z * time * 0.004, 0.000176).y;
    let folds = noise
        .billows(
            warped * 0.61 + warped.yzx() * 0.46 - Vec3::Z * time * 0.006,
            0.000136,
        )
        .z;
    ((broad - 0.5) * 525.0 + (folds - 0.5) * 412.5) * (0.55 + storm * 0.9) - 15.0
}

/// Fine relief changes the volume boundary, independently of broad lighting.
pub(crate) fn wave_detail(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position.normalize() * PLANET_RADIUS - CURRENT * time;
    (textures
        .noise(footprint)
        .billows(flowing * 0.00128 + Vec3::Z * time * 0.012, 0.00128)
        .y
        - 0.5)
        * 30.0
}

/// Blend differently oriented samples so the periodic detail texture cannot impose a visible grid.
/// Altitude stays in the coordinates: this is a volume.
fn ocean_detail(point: Vec3, time: f32, footprint: f32, textures: &Textures) -> Vec4 {
    let altitude = point.length() - PLANET_RADIUS;
    let coordinates = ((point - CURRENT * time) * 0.00048
        + Vec3::Z * (altitude * 0.0024 - time * 0.025))
        * vec3(3.0, 3.0, 1.0);
    (textures.details_filtered(coordinates, footprint * 0.00288)
        + textures.details_filtered(rotated(coordinates), footprint * 0.00216))
        * 0.5
}

/// The net of light the rippling veil focuses onto anything above it, in units of about 75 m. Mean
/// near 0.25, peaks of 1 along the threads.
pub(crate) fn caustics(position: Vec2, time: f32) -> f32 {
    let mut point = position;
    let mut sum = 1.0;
    for index in 0..4 {
        let phase = time * (1.0 - 3.5 / (index as f32 + 1.0));
        point = position
            + vec2(
                (phase - point.x).cos() + (phase + point.y).sin(),
                (phase - point.y).sin() + (phase + point.x).cos(),
            );
        sum += 1.0
            / vec2(
                position.x / ((point.x + phase).sin() / 0.005),
                position.y / ((point.y + phase).cos() / 0.005),
            )
            .length();
    }
    (1.17 - (sum / 4.0).powf(1.4)).abs().powi(8).min(1.0)
}

pub(crate) fn column(point: Vec3, view: &ViewUniform, textures: &Textures) -> Column {
    let mut flash = Vec3::ZERO;
    for index in 0..32 {
        let event = view.aur_flashes[index];
        if event[1].w <= 0.0 {
            break;
        }
        let delta = (point - event[0].xyz()) * event[0].w;
        flash += event[1].xyz() * (-delta.length_squared()).exp();
    }
    let surface = point.normalize() * (PLANET_RADIUS + 25.0);
    Column {
        flash,
        sun: textures.pole_light(surface, light_position(surface)),
        climate: aur_climate(point.normalize(), view.time()),
    }
}

/// The depth-lit column colour and base density, 12 m inside the veil.
fn material(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    normal: Vec3,
    detail: Vec4,
    column: Column,
    textures: &Textures,
) -> Vec4 {
    let time = view.time();
    let climate = column.climate;
    let flowing = point - CURRENT * time;
    let flow = textures.billows(
        flowing * vec3(0.00048, 0.00048, 0.0008) - Vec3::Z * time * 0.003,
        footprint * 0.0008,
    );
    let noise = detail.w * 2.0 - 1.0;
    let wisp = wisp(noise, footprint);
    let granules = flow.w.smoothstep(0.25, 0.75);
    let facing = normal.dot(vec3(-0.48, 0.36, 0.8)).max(0.0);
    // Warm seas lean rose, charged seas teal, storms sharpen the energy.
    let rose = (flow.y + (climate.y - 0.5) * 0.6).smoothstep(0.35, 0.65);
    let hue = vec3(0.12, 0.09, 0.23).lerp(vec3(0.38, 0.24, 0.32), rose);
    let teal = (flow.z + (climate.z - 0.5) * 0.5).smoothstep(0.4, 0.65);
    let core = hue.lerp(vec3(0.40, 0.61, 0.57), teal);
    let energy = (flow.w * 0.7 + detail.w * 0.3 + climate.x * 0.12).smoothstep(0.30, 0.70);
    let depth_glow = 1.0 - (-16.875_f32 / 22.5).exp();
    let pulse = 0.85 + 0.15 * (time * (1.4 + climate.x * 2.2) + flow.y * 9.0).sin();
    let emitted = (hue * (0.65 + facing * 0.85)
        + core * depth_glow * (0.65 + cube(energy) * 3.5) * (0.35 + facing * 1.5)
        + vec3(0.26, 0.40, 0.42) * wisp * energy * 2.0)
        * pulse
        * climate.w;
    // Like the clouds, the veil also scatters the pole's light: it warms at the sunset band and
    // dims a little beyond it, towards the equator.
    let day = (column.sun.max_element() / POLE_RADIANCE.max_element())
        .sqrt()
        .min(1.0);
    let scattered = (hue + core) * column.sun * 0.06;
    let light = emitted * (0.8 + 0.2 * day)
        + scattered
        + column.flash * (0.35 + granules * 0.45 + wisp * 0.2);
    let density = (1.0 - (-12.5 + noise * 8.125).smoothstep(-21.25, 11.25))
        * (0.075 + granules * 0.09 + wisp * 0.06 + 0.03)
        + fourth(granules) * wisp * 0.035;
    (light * view.sky_horizon.y + hue * view.ambient_color.xyz() * 0.2).extend(density.max(0.01))
}

/// Four values in [0, 1) from a loop's seed.
fn arc_traits(seed: f32) -> Vec4 {
    (vec4(1.0, 1.618_034, 2.414_214, 3.302_776) * (seed * 97.31 + 0.13)).fract_gl()
}

/// A prominence: braided strands twisting around the loop, a sheath, and bright plasma draining
/// from the apex down both legs. Each loop draws its own number of strands, their twist and
/// thickness, how knotted its plasma is, its brightness and its tint, from its seed.
pub(crate) fn ejections(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec4,
    seed: f32,
    textures: &Textures,
) -> VolumeSample {
    if curve.z <= 0.0 || curve.x > EJECTION_WIDTH * 3.0 {
        return VolumeSample::default();
    }
    let looks = arc_traits(seed);
    let more = arc_traits(seed + 0.37);
    let time = view.time();
    let flowing = planet_point - CURRENT * time;
    let turbulence = textures.details_filtered(
        flowing * 0.0024 + Vec3::Z * (curve.y * 2.0 - time * 0.12),
        footprint * 0.0024,
    );
    let width = EJECTION_WIDTH * (0.55 + looks.x * 0.45) * (0.7 + turbulence.w * 0.6);
    // Strands narrower than a pixel widen and dim rather than alias.
    let thin = width * (0.08 + more.x * 0.14);
    let strand_width = thin.max(footprint * 0.6);
    let count = 2 + (looks.y * 4.0) as u32;
    let spin = if more.y > 0.5 { 1.0 } else { -1.0 };
    let twist = (curve.y * (5.0 + looks.z * 22.0) + time * (0.2 + more.z * 0.6)) * spin
        + turbulence.x * 2.0;
    let mut strands = 0.0;
    for index in 0..5 {
        if index >= count {
            break;
        }
        let offset = index as f32 * TAU / count as f32;
        let orbit = width
            * (0.2 + more.w * 0.25 + 0.14 * (curve.y * 31.0 + offset * 2.0 + time * 0.7).sin());
        let gap = (square(curve.x) + square(orbit)
            - 2.0 * curve.x * orbit * (curve.w - twist - offset).cos())
        .max(0.0)
        .sqrt();
        strands += (-square(gap / strand_width)).exp();
    }
    strands *= thin / strand_width * 3.0 / count as f32;
    // Some loops carry their plasma in knots that slide down the legs.
    let knots = 1.0
        - looks.w.smoothstep(0.4, 0.9)
            * (0.5 + 0.5 * (curve.y * (12.0 + more.x * 30.0) - time * (0.8 + looks.z))).sin();
    let sheath = (-square(curve.x / width)).exp();
    let drain = 0.5
        + 0.5
            * ((curve.y - 0.5).abs() * (20.0 + more.w * 30.0)
                + time * (2.0 + looks.x * 3.0)
                + turbulence.w * 6.0)
                .sin();
    let density = (sheath * (0.006 + more.z * 0.014) + strands * 0.07 * knots) * curve.z;
    let flow = textures.billows(
        flowing * vec3(0.00048, 0.00048, 0.0008) - Vec3::Z * time * 0.003,
        footprint * 0.0008,
    );
    let tint = energy_color(
        flow.y + (turbulence.w - 0.5) * 0.4 + (looks.y - 0.5) * 0.9,
        flow.z + turbulence.y * 0.3 + (more.y - 0.5) * 0.9,
    );
    let brightness = 0.55 + looks.w * 1.1;
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: (tint * (2.0 + strands * 8.0 + drain * 3.0)
            + vec3(0.85, 1.1, 1.0) * strands * drain * 3.0)
            * brightness
            * view.sky_horizon.y,
    }
}

/// `offset` is the point's from the camera: spires are placed relative to it, as they are traced.
pub(crate) fn wrap(point: Vec3, offset: Vec3, view: &ViewUniform, footprint: f32) -> Wrap {
    let time = view.time();
    let nearest = nearest_spire_at(view.place(offset));
    let gap = nearest.x.max(0.0);
    if gap > 625.0 {
        return Wrap::NONE;
    }
    let frame = surface_frame(point);
    let (first, second) = spire_axes(point);
    let away = (first * nearest.y + second * nearest.z).normalize();
    let around = frame.z_axis.cross(away);
    let current = (CURRENT - frame.z_axis * CURRENT.dot(frame.z_axis)).normalize();
    let wake = away.dot(current).smoothstep(0.0, 0.9) * (-gap / 325.0).exp();
    let shedding = (gap * 0.0192 - time * 0.55 + around.dot(current) * 2.0).sin();
    // A distant footprint cannot resolve the collar; widen it but keep the volume of fog it lifts,
    // rather than aliasing against long steps.
    let collar = 52.5 + footprint * 1.5;
    let contact = 20.0 + footprint;
    Wrap {
        lift: 95.0 * 52.5 / collar * (-gap / collar).exp() + 27.5 * wake,
        warp: away * (87.5 * (-gap / 100.0).exp())
            + around
                * (around.dot(current) * 162.5 * (-gap / 162.5).exp() + shedding * 225.0 * wake),
        // Energy pools around every spire and streams away in its wake.
        glow: 30.0 / contact * (-gap / contact).exp() + 0.35 * (-gap / 200.0).exp() + wake * 0.35,
    }
}

pub fn sample_body(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec4,
    wave: f32,
    height_span: f32,
    normal: Vec3,
    column: Column,
    wrap: Wrap,
    textures: &Textures,
) -> VolumeSample {
    let altitude = planet_point.length() - PLANET_RADIUS;
    let height = altitude - wave;
    if altitude > BODY_TOP_HEIGHT || height - height_span > 112.5 {
        return VolumeSample::default();
    }
    let time = view.time();
    // The flow parts around spires; textures follow it, geometry does not.
    let flowing = planet_point + wrap.warp;
    let detail = ocean_detail(flowing, time, footprint, textures);
    let noise = detail.w * 2.0 - 1.0;
    // Storms and lightning churn the veil: its edge boils far deeper and plumes are thrown high
    // above the surface.
    let foot = (-square(curve.x / (EJECTION_WIDTH * 2.5))).exp() * curve.z;
    let churn =
        1.0 + column.climate.x * 1.2 + foot * 2.0 + (column.flash.max_element() * 0.6).min(5.0);
    let edge = noise * 8.125 * churn;
    let low = (height + edge - height_span + 21.25) / 32.5;
    let high = (height + edge + height_span + 21.25) / 32.5;
    let body = 1.0 - smoothstep_mean(low, high);
    let wisp = wisp(noise, footprint);
    let depth = (-height).max(0.0);
    let plume = (-height.max(0.0) / (22.5 * churn)).exp();
    let material = material(
        flowing.normalize() * (PLANET_RADIUS + wave - 12.5),
        view,
        footprint,
        normal,
        detail,
        column,
        textures,
    );
    let base = (material.w - 0.03).max(0.075);
    let density = (body * (base + depth.min(150.0) * 0.0024) + plume * wisp * 0.008)
        * (1.0 - altitude.smoothstep(412.5, BODY_TOP_HEIGHT));
    let depth_glow = 1.0 - (-(depth + 4.375) / 22.5).exp();
    // Loops are rooted in bright, churning pools of the veil.
    let arc_glow = (-square(curve.x / (EJECTION_WIDTH * 1.8))).exp() * curve.z;
    let source = material.xyz() * (0.65 + 0.35 * depth_glow / 0.5276) * (0.8 + detail.w * 0.4)
        + vec3(0.3, 0.5, 0.48)
            * (wisp * plume * 0.15 + arc_glow * (0.6 + wisp) + wrap.glow * (0.6 + wisp))
            * view.sky_horizon.y;
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: source,
    }
}

fn atlas_uv(c: Vec2, cascade: u32) -> Vec2 {
    let pixel = (c / grid::spacing(cascade) + grid::SIZE as f32 * 0.5)
        .clamp(Vec2::splat(0.5), Vec2::splat(grid::SIZE as f32 - 0.5));
    (pixel + vec2(cascade as f32, 0.0) * grid::SIZE as f32)
        / vec2((grid::SIZE * grid::CASCADES) as f32, grid::SIZE as f32)
}

/// The cached wave height and normal; a zero normal marks a miss.
pub fn sample(point: Vec3, view: &ViewUniform, footprint: f32, textures: &Textures) -> Vec4 {
    if view.cache_origin.w < 0.5 || !textures.cached {
        return Vec4::ZERO;
    }
    let Some(c) = grid::coordinates(point, view) else {
        return Vec4::ZERO;
    };
    let spatial = grid::cascade(c);
    let filter = grid::filter_level(footprint).min((grid::CASCADES - 1) as f32);
    let cascade = spatial.max(filter as u32);
    if cascade >= grid::CASCADES {
        return Vec4::ZERO;
    }
    let shape = textures
        .cache_shape
        .sample_by_lod(*textures.sampler, atlas_uv(c, cascade), 0.0);
    if cascade + 1 >= grid::CASCADES {
        return shape;
    }
    let edge = c.abs().max_element() / (grid::spacing(cascade) * grid::SIZE as f32 * 0.5);
    let blend = edge
        .smoothstep(0.72, 0.90)
        .max((filter - cascade as f32).clamp(0.0, 1.0));
    if blend <= 0.0 {
        return shape;
    }
    shape.lerp(
        textures
            .cache_shape
            .sample_by_lod(*textures.sampler, atlas_uv(c, cascade + 1), 0.0),
        blend,
    )
}

#[spirv(fragment(entry_point_name = "column_cache"))]
pub fn column_cache(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] details: &Image!(3D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 4)] _sky: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 5)] transmittance: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 6)] ejections: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] shape: &mut Vec4,
) {
    let cascade = pixel.x as u32 / grid::SIZE;
    let spacing = grid::spacing(cascade);
    let c = (pixel.xy() - vec2(cascade as f32, 0.0) * grid::SIZE as f32 - grid::SIZE as f32 * 0.5)
        * spacing;
    let view = &frame.view;
    if cascade < grid::first_level(view.altitude(), 2.0 / frame.viewport.w) {
        *shape = Vec4::ZERO;
        return;
    }
    let point = grid::point(c, view);
    let textures = Textures {
        base,
        details,
        transmittance,
        ejections,
        sampler,
        cache_shape: _sky,
        cloud_cover: _sky,
        cloud_layers: _sky,
        cloud_light: base,
        shapes: base,
        cloud_bounds: _sky,
        cloud_towers: _sky,
        cached: false,
    };
    let base_wave = wave_base(point, view.time(), spacing * 0.5, &textures);
    let wave = base_wave + wave_detail(point, view.time(), spacing * 0.5, &textures);
    let basis = surface_frame(point);
    let epsilon = (spacing * 0.5).max(25.0);
    let east = wave_base(
        point + basis.x_axis * epsilon,
        view.time(),
        spacing * 0.5,
        &textures,
    ) - base_wave;
    let north = wave_base(
        point + basis.y_axis * epsilon,
        view.time(),
        spacing * 0.5,
        &textures,
    ) - base_wave;
    let normal = (basis.z_axis - (basis.x_axis * east + basis.y_axis * north) * (10.0 / epsilon))
        .normalize();
    *shape = vec4(wave / WAVE_RANGE, normal.x, normal.y, normal.z);
}
