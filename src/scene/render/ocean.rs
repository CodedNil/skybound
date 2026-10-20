use crate::scene::{
    FEATURE_RADIUS, FEATURE_SCALE, FrameUniform, PLANET_RADIUS, ViewUniform,
    atmosphere::POLE_RADIANCE,
    aur::{
        CURRENT, EJECTION_HEIGHT, EJECTION_WIDTH, aur_climate, energy_color, nearest_spire,
        spire_axes, spire_coordinates,
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
pub const BODY_TOP_HEIGHT: f32 = 3700.0;

/// Everything about a column that varies over kilometres, not metres:
/// lightning, sunlight reaching the veil and the regional climate. A ray
/// reuses it across the short stretch of veil it crosses.
#[derive(Clone, Copy)]
pub struct Column {
    pub flash: Vec3,
    pub sun: Vec3,
    pub climate: Vec4,
}

/// How the ocean meets a spire: the veil climbs it, the current parts
/// around it and sheds eddies downstream, and energy gathers at the contact.
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

/// Wisps are thin ridges where the detail noise crosses its mean. Filtering
/// pulls the noise towards that mean, so a distant footprint would otherwise
/// turn every column into a ridge and wash the ocean out to a pale glow.
fn wisp(noise: f32, footprint: f32) -> f32 {
    let resolved = cube((1.0 - noise.abs() * 2.0).max(0.0));
    let unresolved = (footprint * 0.02).max(1.0).log2().smoothstep(1.0, 3.5);
    resolved.lerp(0.2, unresolved)
}

pub(crate) fn wave_base(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let storm = aur_climate(position.normalize(), time).x;
    let noise = textures.noise(footprint);
    let flowing = position.normalize() * FEATURE_RADIUS - CURRENT * time;
    let drift = noise
        .billows(flowing * 0.0000025 + Vec3::Z * time * 0.001, 0.0000025)
        .xyz()
        * 2.0
        - 1.0;
    let mut warped = flowing * 0.000014 + drift * 0.8;
    warped += vec3(
        (warped.y + time * 0.07).sin(),
        (warped.z - time * 0.06).sin(),
        (warped.x + time * 0.05).sin(),
    ) * 0.4;
    let broad = noise.billows(warped + Vec3::Z * time * 0.004, 0.000022).y;
    let folds = noise
        .billows(
            warped * 0.61 + warped.yzx() * 0.46 - Vec3::Z * time * 0.006,
            0.000017,
        )
        .z;
    ((broad - 0.5) * 4200.0 + (folds - 0.5) * 3300.0) * (0.55 + storm * 0.9) - 120.0
}

/// Fine relief changes the volume boundary, independently of broad lighting.
pub(crate) fn wave_detail(position: Vec3, time: f32, footprint: f32, textures: &Textures) -> f32 {
    let flowing = position.normalize() * FEATURE_RADIUS - CURRENT * time;
    (textures
        .noise(footprint)
        .billows(flowing * 0.00016 + Vec3::Z * time * 0.012, 0.00016)
        .y
        - 0.5)
        * 240.0
}

/// Blend differently oriented samples so the periodic detail texture cannot
/// impose a visible grid. Altitude stays in the coordinates: this is a volume.
fn ocean_detail(point: Vec3, time: f32, footprint: f32, textures: &Textures) -> Vec4 {
    let altitude = point.length() - FEATURE_RADIUS;
    let coordinates = ((point - CURRENT * time) * 0.00006
        + Vec3::Z * (altitude * 0.0003 - time * 0.025))
        * vec3(3.0, 3.0, 1.0);
    (textures.details_filtered(coordinates, footprint * 0.00036)
        + textures.details_filtered(rotated(coordinates), footprint * 0.00027))
        * 0.5
}

/// The net of light the rippling veil focuses onto anything above it,
/// in units of about 600 m. Mean near 0.25, peaks of 1 along the threads.
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
    let surface = point.normalize() * (PLANET_RADIUS + 200.0 / FEATURE_SCALE);
    Column {
        flash,
        sun: textures.pole_light(surface, light_position(surface)),
        climate: aur_climate(point.normalize(), view.time()),
    }
}

/// The depth-lit column colour and base density, 100 m inside the veil.
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
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
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
    let depth_glow = 1.0 - (-135.0_f32 / 180.0).exp();
    let pulse = 0.85 + 0.15 * (time * (1.4 + climate.x * 2.2) + flow.y * 9.0).sin();
    let emitted = (hue * (0.65 + facing * 0.85)
        + core * depth_glow * (0.65 + cube(energy) * 3.5) * (0.35 + facing * 1.5)
        + vec3(0.26, 0.40, 0.42) * wisp * energy * 2.0)
        * pulse
        * climate.w;
    // Like the clouds, the veil also scatters the pole's light: it warms at
    // sunset and dims a little through the night.
    let day = (column.sun.max_element() / POLE_RADIANCE.max_element())
        .sqrt()
        .min(1.0);
    let scattered = (hue + core) * column.sun * 0.06;
    let light = emitted * (0.8 + 0.2 * day)
        + scattered
        + column.flash * (0.35 + granules * 0.45 + wisp * 0.2);
    let density = (1.0 - (-100.0 + noise * 65.0).smoothstep(-170.0, 90.0))
        * (0.075 + granules * 0.09 + wisp * 0.06 + 0.03)
        + fourth(granules) * wisp * 0.035;
    (light * view.sky_horizon.y + hue * view.ambient_color.xyz() * 0.2).extend(density.max(0.01))
}

/// A prominence: braided strands twisting around the loop, a faint sheath,
/// and bright plasma draining from the apex down both legs.
pub(crate) fn ejections(
    planet_point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    curve: Vec4,
    textures: &Textures,
) -> VolumeSample {
    if curve.z <= 0.0 || curve.x > EJECTION_WIDTH * 3.0 {
        return VolumeSample::default();
    }
    let time = view.time();
    let flowing = planet_point - CURRENT * time;
    let turbulence = textures.details_filtered(
        flowing * 0.0003 + Vec3::Z * (curve.y * 2.0 - time * 0.12),
        footprint * 0.0003,
    );
    let width = EJECTION_WIDTH * (0.7 + turbulence.w * 0.6);
    // Strands narrower than a pixel widen and dim rather than alias.
    let thin = width * 0.16;
    let strand_width = thin.max(footprint * 0.6);
    let twist = curve.y * 14.0 + time * 0.4 + turbulence.x * 2.0;
    let mut strands = 0.0;
    for index in 0..3 {
        let offset = index as f32 * TAU / 3.0;
        let orbit = width * (0.32 + 0.14 * (curve.y * 31.0 + offset * 2.0 + time * 0.7).sin());
        let gap = (square(curve.x) + square(orbit)
            - 2.0 * curve.x * orbit * (curve.w - twist - offset).cos())
        .max(0.0)
        .sqrt();
        strands += (-square(gap / strand_width)).exp();
    }
    strands *= thin / strand_width;
    let sheath = (-square(curve.x / width)).exp();
    let drain = 0.5 + 0.5 * ((curve.y - 0.5).abs() * 36.0 + time * 3.5 + turbulence.w * 6.0).sin();
    let density = (sheath * 0.012 + strands * 0.07) * curve.z;
    let flow = textures.billows(
        flowing * vec3(0.00006, 0.00006, 0.0001) - Vec3::Z * time * 0.003,
        footprint * 0.0001,
    );
    let tint = energy_color(
        flow.y + (turbulence.w - 0.5) * 0.4,
        flow.z + turbulence.y * 0.3,
    );
    VolumeSample {
        density,
        color: Vec3::ZERO,
        emission: (tint * (2.0 + strands * 8.0 + drain * 3.0)
            + vec3(0.85, 1.1, 1.0) * strands * drain * 3.0)
            * view.sky_horizon.y,
    }
}

pub(crate) fn wrap(point: Vec3, time: f32, footprint: f32) -> Wrap {
    let nearest = nearest_spire(spire_coordinates(point));
    let gap = nearest.x.max(0.0);
    if gap > 5000.0 {
        return Wrap::NONE;
    }
    let frame = surface_frame(point);
    let (first, second) = spire_axes(point);
    let away = (first * nearest.y + second * nearest.z).normalize();
    let around = frame.z_axis.cross(away);
    let current = (CURRENT - frame.z_axis * CURRENT.dot(frame.z_axis)).normalize();
    let wake = away.dot(current).smoothstep(0.0, 0.9) * (-gap / 2600.0).exp();
    let shedding = (gap * 0.0024 - time * 0.55 + around.dot(current) * 2.0).sin();
    // A distant footprint cannot resolve the collar; widen it but keep the
    // volume of fog it lifts, rather than aliasing against long steps.
    let collar = 420.0 + footprint * 1.5;
    let contact = 160.0 + footprint;
    Wrap {
        lift: 760.0 * 420.0 / collar * (-gap / collar).exp() + 220.0 * wake,
        warp: away * (700.0 * (-gap / 800.0).exp())
            + around
                * (around.dot(current) * 1300.0 * (-gap / 1300.0).exp() + shedding * 1800.0 * wake),
        // Energy pools around every spire and streams away in its wake.
        glow: 240.0 / contact * (-gap / contact).exp() + 0.35 * (-gap / 1600.0).exp() + wake * 0.35,
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
    let altitude = planet_point.length() - FEATURE_RADIUS;
    let height = altitude - wave;
    if altitude > BODY_TOP_HEIGHT || height - height_span > 900.0 {
        return VolumeSample::default();
    }
    let time = view.time();
    // The flow parts around spires; textures follow it, geometry does not.
    let flowing = planet_point + wrap.warp;
    let detail = ocean_detail(flowing, time, footprint, textures);
    let noise = detail.w * 2.0 - 1.0;
    // Storms and lightning churn the veil: its edge boils far deeper and
    // plumes are thrown high above the surface.
    let foot = (-square(curve.x / (EJECTION_WIDTH * 2.5))).exp() * curve.z;
    let churn =
        1.0 + column.climate.x * 1.2 + foot * 2.0 + (column.flash.max_element() * 0.6).min(5.0);
    let edge = noise * 65.0 * churn;
    let low = (height + edge - height_span + 170.0) / 260.0;
    let high = (height + edge + height_span + 170.0) / 260.0;
    let body = 1.0 - smoothstep_mean(low, high);
    let wisp = wisp(noise, footprint);
    let depth = (-height).max(0.0);
    let plume = (-height.max(0.0) / (180.0 * churn)).exp();
    let material = material(
        flowing.normalize() * (FEATURE_RADIUS + wave - 100.0),
        view,
        footprint,
        normal,
        detail,
        column,
        textures,
    );
    let base = (material.w - 0.03).max(0.075);
    let density = (body * (base + depth.min(1200.0) * 0.0003) + plume * wisp * 0.008)
        * (1.0 - altitude.smoothstep(3300.0, 3700.0));
    let depth_glow = 1.0 - (-(depth + 35.0) / 180.0).exp();
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
    if cascade < grid::first_level(view.altitude() * FEATURE_SCALE, 2.0 / frame.viewport.w) {
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
        cloud_bounds: _sky,
        cloud_towers: _sky,
        cached: false,
    };
    let base_wave = wave_base(point, view.time(), spacing * 0.5, &textures);
    let wave = base_wave + wave_detail(point, view.time(), spacing * 0.5, &textures);
    let basis = surface_frame(point);
    let epsilon = (spacing * 0.5).max(200.0);
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
    *shape = vec4(wave / 4096.0, normal.x, normal.y, normal.z);
}
