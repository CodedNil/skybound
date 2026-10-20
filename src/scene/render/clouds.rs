use crate::scene::{
    FEATURE_RADIUS, FEATURE_SCALE, FrameUniform, PLANET_RADIUS, ViewUniform,
    aur::aur_climate,
    climate::{
        CLOUD_BOTTOM, CLOUD_FACE, CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CLOUD_TOP, CloudCover,
        CloudParcel, LIGHT_FACE, LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH, RAIN_DEPTH, TOWER,
        atlas_direction, atlas_texel, cloud_cell, cloud_wind, cube_direction, cube_face,
        light_altitude,
    },
    math::{half, henyey_greenstein, intersect_sphere, rotated, simpson},
    planet::{light_position, light_towards},
    render::{
        march::RaymarchResult,
        textures::{AtmosphereData, Textures},
    },
};
use core::f32::consts::{FRAC_PI_2, PI, TAU};
use spirv_std::{
    Image, Sampler,
    glam::{FloatExt, IVec2, UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4},
    num_traits::Float,
    spirv,
};

/// Seconds between restarts of the noise carried along by the wind. Two
/// copies half a period apart crossfade, so detail never stretches far.
const FLOW_PERIOD: f32 = 900.0;

/// Peak extinction of each cloud family, per metre.
const CONVECTIVE_EXTINCTION: f32 = 0.02;

/// Cells across which each cumulus heap, and each storm tower, stands.
/// The size of a typical cumulus footprint, the tile its Worley noise
/// repeats over, and the scale of the clusters fields gather into.
const FIELD_CELL: f32 = 4500.0;

const FIELD_TILE: f32 = FIELD_CELL * 3.0;

const CLUSTER_SCALE: f32 = 70_000.0;

/// The size of one cloud map texel at a face's centre.
const CLOUD_TEXEL: f32 = PLANET_RADIUS * FRAC_PI_2 / CLOUD_FACE as f32;

const DECK_EXTINCTION: f32 = 0.006;

const CIRRUS_EXTINCTION: f32 = 0.000_06;

const RAIN_EXTINCTION: f32 = 0.0003;

/// The share of a layer's peak extinction that shadows are traced through,
/// where the noise is not evaluated. Light threads the gaps between clouds,
/// so a patchy layer passes far more than its mean density would.
const MEAN_FILL: f32 = 0.15;

/// Each march sample traces its own shadow this far towards the light; the
/// light volume supplies the rest of the path.
pub const NEAR_SHADOW: f32 = 1500.0;

/// Cloud droplet diameter in micrometres, which sets the shape of the
/// forward peak, the fogbow and the glory.
const DROPLET: f32 = 16.0;

/// Wavelengths of the red, green and blue channels, in nanometres.
const WAVELENGTH: Vec3 = vec3(650.0, 550.0, 450.0);

/// Cloud at a point: droplet and rain extinction, and how far up its cloud
/// the point sits, from a dark base at 0 to a sunlit crown at 1.
pub struct CloudSample {
    pub density: f32,
    pub rain: f32,
    pub height: f32,
    /// How far from the point any cloud certainly is, in metres.
    pub free: f32,
}

/// What a march step needs to know about the cloud over one column, read
/// once and shared by every density it evaluates nearby.
pub struct CloudColumn {
    cover: Vec4,
    layers: Vec4,
    /// The convective cover, top and base cloud anywhere near could reach.
    bounds: Vec4,
    /// The storm tower over this column, if any: its centre scaled by its
    /// top altitude, and its column radius.
    tower: Vec4,
    /// Offsets placing a point in the two drifting copies of the noise.
    near: Vec3,
    far: Vec3,
    /// How much the nearer copy counts in the crossfade.
    weight: f32,
    /// Direction of the wind, along which cirrus is combed and rain leans.
    along: Vec3,
}

impl CloudColumn {
    pub const CLEAR: Self = Self {
        cover: Vec4::ZERO,
        layers: Vec4::ZERO,
        bounds: Vec4::ZERO,
        tower: Vec4::ZERO,
        near: Vec3::ZERO,
        far: Vec3::ZERO,
        weight: 0.0,
        along: Vec3::X,
    };

    /// How far a ray can run from a point before it could enter any of
    /// this column's layers, zero within one; and whether the column holds a
    /// thin deck that long strides could slip through.
    pub fn room(&self, point: Vec3, direction: Vec3) -> (f32, bool) {
        let (cover, layers, bounds) = (self.cover, self.layers, self.bounds);
        let altitude = point.length() - FEATURE_RADIUS;
        let mut room = 1.0e9f32;
        if bounds.x > 0.001 {
            let floor = if storm(cover, layers) > 0.0 || self.tower.w > 0.0 {
                bounds.z - RAIN_DEPTH
            } else {
                bounds.z
            };
            room = room.min(approach(
                point,
                direction,
                altitude,
                floor - 300.0,
                bounds.y + 300.0,
            ));
        }
        if cover.z > 0.001 {
            let (low, high) = deck_span(layers);
            room = room.min(approach(
                point,
                direction,
                altitude,
                low - 300.0,
                high + 300.0,
            ));
        }
        if cover.w > 0.001 {
            room = room.min(approach(
                point,
                direction,
                altitude,
                layers.w - 2100.0,
                layers.w + 2100.0,
            ));
        }
        (room, cover.z > 0.001)
    }

    pub fn sample(
        &self,
        point: Vec3,
        footprint: f32,
        textures: &Textures,
        detailed: bool,
    ) -> CloudSample {
        let radius = point.length();
        let altitude = radius - FEATURE_RADIUS;
        let mut sample = CloudSample {
            density: 0.0,
            rain: 0.0,
            height: 0.5,
            free: 1.0e9,
        };
        if altitude < CLOUD_BOTTOM || altitude > CLOUD_TOP {
            return sample;
        }
        let (cover, layers) = (self.cover, self.layers);

        let near = point + self.near;
        let far = point + self.far;
        let weight = self.weight;
        // Only the large shapes crossfade between the two copies; smaller
        // structure follows the nearer copy and simply boils as it restarts.
        let lead = if weight > 0.5 { near } else { far };
        // A probe only asks whether cloud is near, and may only overstate it.
        let blended = detailed && weight > 0.0 && weight < 1.0;
        let shape = |offset: Vec3, scale: f32| {
            if !blended {
                return textures.base_filtered((lead + offset) / scale, footprint / scale);
            }
            textures
                .base_filtered((near + offset) / scale, footprint / scale)
                .lerp(
                    textures.base_filtered((far + offset) / scale, footprint / scale),
                    1.0 - weight,
                )
        };
        // Probes need only the large shapes.
        let small = |offset: Vec3, scale: f32| {
            if detailed {
                textures.base_filtered((lead + offset) / scale, footprint / scale)
            } else {
                Vec4::splat(0.5)
            }
        };
        let top = cover.y;
        let deep = storm(cover, layers);
        let window = (self.bounds.z - 400.0 - altitude).max(altitude - self.bounds.y - 400.0);
        if self.bounds.x <= 0.001 || window > 0.0 {
            // Outside the altitudes nearby cloud fills; the bounds hold for
            // about a cloud map texel around.
            let vertical = if self.bounds.x > 0.001 { window } else { 1.0e9 };
            sample.free = vertical.min(CLOUD_TEXEL);
        } else {
            // Every cloud is a smooth dimensional profile, positive inside,
            // that noise erodes only near its boundary: billowy towards the
            // crowns, wispy at the bases (Schneider, Nubis).
            let rise = altitude - layers.x;
            let depth = (cover.y - layers.x).max(400.0);
            let up = point / radius;
            // Footprints are columns: sampled level with the cloud base.
            let column = lead - up * rise;
            let mut shape =
                field_profile(column, cover.x, rise, depth, footprint, textures, detailed);
            let mut fraction = (rise / depth).clamp(0.0, 1.0);
            let mut scale = FIELD_CELL;
            if self.tower.w > 0.0 {
                // Storm towers are the weather's own: placed and grown by it.
                let top = self.tower.xyz().length();
                let centre = self.tower.xyz() / top;
                let horizontal = (up - centre) * FEATURE_RADIUS;
                let local = vec3(
                    horizontal.dot(self.along),
                    horizontal.dot(up.cross(self.along)),
                    rise,
                );
                let (profile, along) =
                    tower_profile(local, self.tower.w, (top - layers.x).max(4000.0));
                if profile > shape {
                    shape = profile;
                    fraction = along;
                    scale = self.tower.w;
                }
            }
            let density = erode(shape, fraction, scale, lead, footprint, textures, detailed);
            sample.density += density * CONVECTIVE_EXTINCTION;
            sample.height = fraction;
            // Profiles are no distances: within the layer, only a stride
            // short of the smallest footprint is safe.
            sample.free = 0.0;
        }
        if deep > 0.0 && altitude < layers.x && altitude > layers.x - RAIN_DEPTH {
            // Rain hangs below deep storms in shafts that thin on their way
            // down and lean with the wind.
            let fall = (layers.x - altitude) / RAIN_DEPTH;
            let leaning = self.along * ((layers.x - altitude) * 0.4);
            let shafts = if detailed {
                small(leaning, 6000.0).x
            } else {
                0.95
            };
            let rain = remap(shafts, 0.72, 0.92) * (1.0 - fall * fall) * deep;
            sample.rain += rain * RAIN_EXTINCTION;
        }
        // Decks, cirrus and rain are noise, not bodies: bound only by how far
        // above or below their layers the point lies.
        if deep > 0.0 {
            sample.free = sample.free.min((altitude - layers.x).max(0.0));
        }
        let (floor, ceiling) = deck_span(layers);
        if cover.z > 0.001 {
            sample.free = sample
                .free
                .min((floor - altitude).max(altitude - ceiling).max(0.0));
        }
        if cover.w > 0.001 {
            sample.free = sample
                .free
                .min(((altitude - layers.w).abs() - 1800.0).max(0.0));
        }
        if cover.z > 0.001 && altitude > floor && altitude < ceiling {
            // Decks and anvils: sheets broken into cells and rolls, whose
            // floor and ceiling heave in slow waves.
            let thickness = layers.z.max(300.0);
            let base = layers.y - thickness * 0.5;
            // Stratocumulus: rounded cells of cloud a few kilometres across
            // with clear lanes between them, from cellular noise.
            let waves = shape(Vec3::ZERO, 48_000.0);
            // Cells of several sizes, grouped by broad waves into patches and
            // rolls, so the deck never reads as a sheet of equal bubbles.
            let cellular = if detailed {
                let worley = textures.details_filtered(lead / 15_000.0, footprint / 15_000.0);
                1.0 - (worley.x * 0.65 + worley.y * 0.35)
            } else {
                0.9
            };
            let height = (altitude - base) / thickness + (waves.y - 0.5) * 0.6;
            let gradient = (height * (1.0 - height) * 4.0).max(0.0);
            let fill = cellular * 0.7 + waves.x * 0.3 + (cover.z - 0.6) * 0.6;
            let edge = ((fill - 0.62) / 0.2).clamp(0.0, 1.0);
            let density = edge * edge * (3.0 - 2.0 * edge) * gradient * cover.z.sqrt();

            sample.density += density * DECK_EXTINCTION;
            sample.height = sample.height.max(height.clamp(0.0, 1.0));
        }
        if cover.w > 0.001 && (altitude - layers.w).abs() < 1800.0 {
            // Thin cirrus: long smooth streaks combed out along the wind,
            // with hooked ends where the noise bends them; mostly see-through.
            let combed = near - self.along * (near.dot(self.along) * 0.96);
            let noise = textures.base_filtered(combed / 30_000.0, footprint / 30_000.0);
            let bend = textures.base_filtered(near / 90_000.0, footprint / 90_000.0);
            let height = (altitude - layers.w) / 1800.0 + (bend.y - 0.5) * 0.8;
            let gradient = (1.0 - height * height).max(0.0);
            let streak = noise.w * 0.6 + noise.y * 0.4 + (bend.x - 0.5) * 0.3;
            let edge = ((streak - 0.76 + cover.w * 0.08) / 0.08).clamp(0.0, 1.0);
            let density = edge * edge * (3.0 - 2.0 * edge) * gradient * (cover.w * 0.5);
            sample.density += density * CIRRUS_EXTINCTION;
        }
        sample
    }
}

/// The dimensional profile of fields of cumulus over a column at a height:
/// positive inside a cloud, rising to 1 in its heart. Footprints are cells
/// of Worley noise grouped into clusters, their outlines warped and two
/// rotated copies combined so no pattern repeats; each footprint rises
/// highest at its middle, into a dome on a flat base.
fn field_profile(
    column: Vec3,
    cover: f32,
    rise: f32,
    depth: f32,
    footprint: f32,
    textures: &Textures,
    detailed: bool,
) -> f32 {
    if rise < 0.0 {
        return rise / 200.0;
    }
    let clusters = textures.base_filtered(column / CLUSTER_SCALE, footprint / CLUSTER_SCALE);
    let warp = (clusters.yzw() - 0.5) * (FIELD_CELL * 0.8);
    // Each noise is spread over the whole range (its measured 5th to 95th
    // percentiles), so cover means the share of the sky that is cloud.
    let cells = |p: Vec3| {
        let worley = textures.details_filtered(p / FIELD_TILE, footprint / FIELD_TILE);
        remap(1.0 - (worley.x * 0.7 + worley.y * 0.3), 0.25, 0.72)
    };
    let warped = column + warp;
    let cells = cells(warped).max(cells(rotated(warped / FIELD_TILE) * FIELD_TILE * 1.37));
    let grouped = remap(clusters.x, 0.55, 0.78);
    // Clusters raise both how many footprints break the surface and how
    // tall they grow; cover sets the share of the sky they fill.
    let field = cells * 0.7 + grouped * 0.3;
    let profile = remap(field, 1.0 - cover * 0.65, 1.0);
    // Heaps about as tall as they are wide, larger in the clusters; storm
    // towers come from the weather itself.
    // A footprint spans about half a cell; heaps stand a little less tall
    // than they are wide, taller in the clusters.
    let tallest = (FIELD_CELL * (0.18 + 0.32 * grouped)).min(depth);
    let u = rise / tallest.max(400.0);
    let probe = if detailed { 0.0 } else { 0.15 };
    // A dome: highest in the footprint's heart, rounding down to its rim.
    // The profile falls about linearly to the rim, so this is a round dome.
    profile * (2.0 - profile) - u * u + probe
}

/// The dimensional profile of a storm tower about a point relative to its
/// foot, and how far up it the point sits: a flaring base drawing in air, a
/// column narrowing as it climbs, and an anvil spreading downwind.
fn tower_profile(q: Vec3, radius: f32, height: f32) -> (f32, f32) {
    let along = (q.z / height).clamp(0.0, 1.0);
    if q.z < 0.0 {
        return (q.z / 200.0, 0.0);
    }
    let girth = radius * (0.6 + 0.4 * (1.0 - along).powi(3));
    // The column leans downwind and wavers as it climbs.
    let lean = vec2(
        along * along * radius * 0.6,
        (along * 9.0).sin() * radius * 0.08,
    );
    let column = 1.0 - (q.truncate() - lean).length() / girth;
    let anvil_base = height * 0.82;
    let spread = q.truncate() - vec2(radius * (0.9 + 0.8 * along), 0.0);
    let extent = vec2(radius * 3.2, radius * 2.3);
    let outward = (spread / extent).length();
    let thickness = (height - anvil_base) * (1.0 - outward * outward).max(0.0);
    let anvil = (1.0 - outward)
        .min(1.0 - (q.z - anvil_base - thickness * 0.4).abs() / (thickness * 0.6 + 1.0));
    let top = 1.0 - q.z / height;
    (column.min(top * 4.0).max(anvil * 0.8), along)
}

/// Erode a dimensional profile into cloud. Large lobes at a fraction of the
/// cloud's size heap its silhouette; Perlin-Worley billows crown its tops
/// and fine wisps fray its base, both stronger towards its edge and faded
/// out where they are smaller than a pixel, so distant clouds stay soft.
fn erode(
    profile: f32,
    fraction: f32,
    scale: f32,
    position: Vec3,
    footprint: f32,
    textures: &Textures,
    detailed: bool,
) -> f32 {
    if profile <= -0.3 {
        return 0.0;
    }
    if !detailed {
        return (profile + 0.3).clamp(0.0, 1.0);
    }
    // Detail that pixels cannot resolve fades rather than aliasing.
    let resolve = |size: f32| 1.0 - (footprint / size).smoothstep(0.15, 0.6);
    let lobe_size = scale * 0.45;
    let lobes = textures.base_filtered(position / lobe_size, footprint / lobe_size);
    let mut shape = profile + (lobes.x - 0.75) * 0.6 * resolve(lobe_size);
    let detail = textures.details_filtered(position / 1500.0, footprint / 1500.0);
    let billows = 1.0 - (detail.x * 0.5 + detail.y * 0.3 + detail.z * 0.2);
    let wisps = detail.w;
    let erosion = wisps.lerp(billows, fraction.smoothstep(0.1, 0.5));
    let edge = (1.0 - shape).clamp(0.0, 1.0);
    shape -= (1.0 - erosion) * 0.45 * edge * resolve(1500.0);
    // A soft rim whose width follows the pixel, so outlines never flicker.
    let width = 0.06 + footprint / scale;
    (shape / width).clamp(0.0, 1.0).powf(1.5)
}

fn remap(value: f32, low: f32, high: f32) -> f32 {
    ((value - low) / (high - low).max(0.0001)).clamp(0.0, 1.0)
}

/// How much a column's convection is a deep, raining storm.
fn storm(cover: Vec4, layers: Vec4) -> f32 {
    cover.x.smoothstep(0.45, 0.8) * (cover.y - layers.x).smoothstep(12_000.0, 22_000.0)
}

/// The altitudes a deck can fill once its waves heave it.
fn deck_span(layers: Vec4) -> (f32, f32) {
    let thickness = layers.z.max(300.0);
    let base = layers.y - thickness * 0.5;
    // Its waves can heave it by up to half its thickness either way.
    (base - thickness * 0.6, base + thickness * 1.6)
}

/// Mean extinction over a column at an altitude, without its noise: what
/// shadows are traced through.
fn mean_density(cover: Vec4, layers: Vec4, altitude: f32) -> f32 {
    let mut density = 0.0;
    if cover.x > 0.001 && altitude > layers.x && altitude < cover.y {
        let height = (altitude - layers.x) / (cover.y - layers.x).max(400.0);
        density += cover.x * cover.x * (1.0 - height * height * 0.5) * CONVECTIVE_EXTINCTION;
    }
    let (floor, ceiling) = deck_span(layers);
    if cover.z > 0.001 && altitude > floor && altitude < ceiling {
        density += cover.z * cover.z * DECK_EXTINCTION;
    }
    if cover.w > 0.001 && (altitude - layers.w).abs() < 1800.0 {
        density += cover.w * CIRRUS_EXTINCTION;
    }
    density * MEAN_FILL
}

/// Mean optical depth of a column beneath an altitude, from the aur up.
fn depth_below(cover: Vec4, layers: Vec4, altitude: f32) -> f32 {
    let span = |low: f32, high: f32| (altitude.min(high) - low).max(0.0);
    let mut depth = 0.0;
    if cover.x > 0.001 {
        depth += cover.x * cover.x * 0.8 * CONVECTIVE_EXTINCTION * span(layers.x, cover.y);
        depth +=
            storm(cover, layers) * 0.4 * RAIN_EXTINCTION * span(layers.x - RAIN_DEPTH, layers.x);
    }
    if cover.z > 0.001 {
        let (floor, ceiling) = deck_span(layers);
        depth += cover.z * cover.z * 0.7 * DECK_EXTINCTION * span(floor, ceiling);
    }
    if cover.w > 0.001 {
        depth += cover.w * CIRRUS_EXTINCTION * span(layers.w - 1800.0, layers.w + 1800.0);
    }
    depth * MEAN_FILL
}

/// Forced benchmark weather replaces the weather systems everywhere.
pub fn forced_weather(view: &ViewUniform) -> bool {
    view.weather.x >= 0.0
}

/// Distance along a ray from a point to where it enters a layer spanning
/// `low` to `high` altitude: zero within it, and far when it never will.
fn approach(point: Vec3, direction: Vec3, altitude: f32, low: f32, high: f32) -> f32 {
    if altitude >= low && altitude <= high {
        return 0.0;
    }
    let target = if altitude < low { low } else { high };
    let crossing = intersect_sphere(point, direction, FEATURE_RADIUS + target);
    if altitude < low {
        crossing.y.max(0.0)
    } else if crossing.x <= crossing.y && crossing.x >= 0.0 {
        crossing.x
    } else {
        1.0e9
    }
}

pub fn cloud_column(point: Vec3, view: &ViewUniform, textures: &Textures) -> CloudColumn {
    let up = point.normalize();
    let [cover, layers, bounds, tower] = textures.cloud_column(up);
    let time = view.time();
    // Carry the noise with the large-scale flow, which moves through the
    // world, restarting each copy before the converging wind can stretch it
    // out of shape.
    let wind = cloud_wind(up) * FEATURE_SCALE;
    let phase = time / FLOW_PERIOD - (time / FLOW_PERIOD).floor();
    let later = phase + 0.5 - (phase + 0.5).floor();
    CloudColumn {
        cover,
        layers,
        bounds,
        tower,
        // A slow rise through the noise keeps every cloud evolving.
        near: Vec3::Z * (time * 0.5) - wind * (phase * FLOW_PERIOD),
        far: Vec3::Z * (time * 0.5) - wind * (later * FLOW_PERIOD),
        // Each copy holds alone for most of its life; they blend briefly.
        weight: (1.0 - (1.0 - phase * 2.0).abs()).smoothstep(0.45, 0.55),
        along: wind / wind.length().max(0.001),
    }
}

/// Draine's phase function, which adds the side and back lobes of real
/// droplets to Henyey-Greenstein.
fn draine(cosine: f32, g: f32, alpha: f32) -> f32 {
    henyey_greenstein(cosine, g) * (1.0 + alpha * cosine * cosine)
        / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0)
}

/// Approximate Mie scattering by water droplets of a diameter in
/// micrometres (Jendersie and d'Eon, 2023), valid from 5 to 50.
fn droplet_mie(cosine: f32, diameter: f32) -> f32 {
    // The diffraction peak is narrower than any pixel; keep its energy in a
    // lobe a little wider so thin cloud glows around the light.
    let g_forward = (-0.099_056_7 / (diameter - 1.671_54)).exp().min(0.82);
    let g_draine = (-2.206_79 / (diameter + 3.910_29) - 0.428_934).exp();
    let alpha = (3.624_89 - 8.292_88 / (diameter + 5.528_25)).exp();
    let blend = (-0.599_085 / (diameter - 0.641_583) - 0.665_888).exp();
    henyey_greenstein(cosine, g_forward) * (1.0 - blend) + draine(cosine, g_draine, alpha) * blend
}

/// Scattering by cloud droplets per colour: approximate Mie, with the fogbow
/// and the glory, the rings round the shadow of anyone looking down on
/// sunlit cloud, both spaced in proportion to wavelength.
pub fn droplet_phase(cosine: f32) -> Vec3 {
    // The broad lobes barely differ between channels; the colour lies in
    // the rings and bows.
    let mie = Vec3::splat(droplet_mie(cosine, DROPLET));
    // Angle from the antisolar point.
    let back = (-cosine).clamp(-1.0, 1.0).acos();
    let ring = WAVELENGTH / 550.0 * (0.035 * 16.0 / DROPLET);
    let glory = (Vec3::splat(back * 2.0 * PI) / ring).cos() * 0.5 + Vec3::splat(0.5);
    // A broad pale bow near 39 degrees, faintly red outside.
    let bow = Vec3::splat(back) - vec3(0.695, 0.68, 0.665);
    let fogbow = (-(bow * bow) / 0.0016).exp();
    mie + (glory * ((-back / 0.05).exp() * 0.5) + fogbow * 0.035) / (4.0 * PI)
}

/// Scattering by raindrops, large enough for geometric optics: the primary
/// bow near 42 degrees from the antisolar point with red outside, the
/// fainter secondary near 51 with its colours reversed, more light inside
/// the primary, and Alexander's dark band between the two.
pub fn rain_phase(cosine: f32) -> Vec3 {
    let back = Vec3::splat((-cosine).clamp(-1.0, 1.0).acos());
    let lobe = |angle: Vec3, width: f32| {
        let offset = back - angle;
        (-(offset * offset) / (width * width)).exp()
    };
    let primary = lobe(vec3(0.738, 0.726, 0.709), 0.009);
    let secondary = lobe(vec3(0.880, 0.896, 0.924), 0.012) * 0.43;
    let inside = Vec3::ONE - back.smoothstep(Vec3::splat(0.62), Vec3::splat(0.72));
    let forward = Vec3::splat(henyey_greenstein(cosine, 0.8));
    forward * 0.7 + (primary * 0.9 + secondary + inside * 0.06) / (4.0 * PI)
}

/// What a ray's cloud lighting shares along its length: the pole lights are
/// so far off that the scattering angle barely changes along a ray, and the
/// aur's glow changes only over thousands of kilometres.
///
/// Every member is four wide: nested in a march's state, three-wide
/// members would be laid out differently by the shader compiler and the
/// graphics driver's validator.
pub struct CloudOptics {
    droplets: Vec4,
    rain: Vec4,
    /// Phases of the second and third orders of scattering.
    spread: Vec4,
    glow: Vec4,
}

pub fn cloud_optics(
    point: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    raining: bool,
) -> CloudOptics {
    let light = light_towards(point);
    let cosine = direction.dot(light);
    let climate = aur_climate(point.normalize(), view.time());
    CloudOptics {
        droplets: droplet_phase(cosine).extend(0.0),
        rain: if raining {
            rain_phase(cosine).extend(0.0)
        } else {
            Vec4::ZERO
        },
        spread: vec4(
            henyey_greenstein(cosine, 0.3),
            henyey_greenstein(cosine, 0.15),
            0.0,
            0.0,
        ),
        // The aur below glows up into cloud bases: warmer in calm seas,
        // brighter over storms.
        glow: (vec3(0.42, 0.34, 0.58).lerp(vec3(0.55, 0.36, 0.44), climate.y * 0.5)
            * climate.w
            * view.sky_horizon.y)
            .extend(0.0),
    }
}

/// Light scattered towards the eye by cloud droplets and by rain at a
/// point, per unit of each one's extinction. `shadow` is the optical depth
/// the sample traced towards the light itself, and `height` how far up its
/// cloud it sits.
pub fn cloud_light(
    point: Vec3,
    view: &ViewUniform,
    textures: &Textures,
    optics: &CloudOptics,
    shadow: f32,
    height: f32,
) -> (Vec3, Vec3) {
    let depth = textures.light_depth(point);
    let optical = depth.x + shadow;
    // Light scattered again and again: each order reaches deeper into the
    // cloud, less attenuated and less forward (Wrenninge's octaves).
    let direct = (-optical).exp();
    let orders = optics.spread.x * (-optical * 0.5).exp() * 0.5
        + optics.spread.y * (-optical * 0.25).exp() * 0.25;
    // Deep inside, light that has scattered many times keeps the interior
    // from going black.
    let diffuse = (-optical * 0.25).exp() * 0.07 + (-optical * 0.06).exp() * 0.05 + orders;
    // Thin edges facing the light hold too few droplets to brighten much.
    let powder = 1.0 - (-shadow * 2.0 - 0.15).exp();
    let world = point / FEATURE_SCALE;
    let sun = textures.pole_light(world, light_position(world)) * view.sky_horizon.y * 1.5;
    // Seen from above at night, clouds over the aur glow with its light
    // diffused up through them.
    let altitude = point.length() - FEATURE_RADIUS;
    // A cloud's own body hides the glow from all but its base.
    let upward = optics.glow.xyz()
        * (-altitude.max(0.0) / 22_000.0).exp()
        * ((-depth.z).exp() * 0.6 + (-depth.z * 0.15).exp() * 0.4)
        * (1.0 - 0.75 * height);
    // Skylight: crowns see the open sky, bases sit in their own shadow.
    let sky =
        (view.ambient_color.xyz() * 0.45 + vec3(0.025, 0.035, 0.055)) * (0.45 + 0.55 * height);
    (
        sun * (optics.droplets.xyz() * (direct * (0.55 + 0.45 * powder)) + Vec3::splat(diffuse))
            + upward
            + sky,
        sun * (optics.rain.xyz() * direct + Vec3::splat(diffuse * 0.5)) + upward * 0.5 + sky,
    )
}

pub fn short_volume(
    ro: Vec3,
    rd: Vec3,
    atmosphere: &AtmosphereData,
    view: &ViewUniform,
    length: f32,
    textures: &Textures,
) -> RaymarchResult {
    let world = view.to_planet(ro);
    let origin = world * FEATURE_SCALE;
    let direction = view.planet_rotation.conjugate() * rd;
    let world_length = length;
    let length = length * FEATURE_SCALE;
    let middle = origin + direction * (length * 0.5);
    let footprint = (length / 8.0).max(4.0);
    let density = |point| {
        let sample = cloud_column(point, view, textures).sample(point, footprint, textures, false);
        sample.density + sample.rain
    };
    let a = density(origin);
    let m = density(middle);
    let b = density(origin + direction * length);
    let optical = simpson(a, m, b, length);
    let air = textures.air_transmittance(world, direction, world_length);
    let transmission = (-optical).exp() * air;
    let lighting = if optical > 0.0001 {
        cloud_light(
            middle,
            view,
            textures,
            &cloud_optics(middle, direction, view, false),
            0.0,
            0.5,
        )
        .0
    } else {
        atmosphere.haze
    };
    RaymarchResult {
        color: (lighting * (1.0 - transmission)).extend(transmission),
        depth: world_length,
    }
}

/// Splat the weather systems over each column of the planet: the cover,
/// heights and layers that cloud density is shaped from.
#[spirv(fragment(entry_point_name = "cloud_map"))]
pub fn cloud_map(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base: &Image!(3D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] parcels: &[CloudParcel],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] index: &[u32],
    #[spirv(location = 0)] out_cover: &mut Vec4,
    #[spirv(location = 1)] out_layers: &mut Vec4,
    #[spirv(location = 2)] out_tower: &mut Vec4,
) {
    let view = &frame.view;
    *out_tower = Vec4::ZERO;
    if forced_weather(view) {
        let weather = view.weather;
        *out_cover = vec4(
            weather.x,
            9000.0 + weather.y * 36_000.0,
            weather.w * 0.8,
            weather.z,
        );
        *out_layers = vec4(5000.0, 20_000.0, 2500.0, 42_000.0);
        return;
    }
    let up = atlas_direction(pixel.xy(), CLOUD_FACE);
    let cell = cloud_cell(up);
    let mut cover = CloudCover::default();
    // The nearest storm tower whose anvil reaches this texel, by how far
    // inside its reach the texel lies.
    let mut nearest = 0.0f32;
    for offset in index[cell]..index[cell + 1] {
        let parcel = parcels[index[offset as usize] as usize];
        cover.add(parcel, up);
        // Only the strongest, deepest systems build a full thunderhead; the
        // rest are fields of towering cumulus.
        if parcel.kind() == TOWER && parcel.shape.y > 0.6 && parcel.shape.w > 32_000.0 {
            let centre = parcel.centre.xyz();
            let distance = (up - centre).length() * PLANET_RADIUS;
            let column = parcel.detail.x * 0.35;
            let reach = column * 4.2 / FEATURE_SCALE + CLOUD_TEXEL * 1.5;
            let inside = 1.0 - distance / reach;
            if inside > nearest {
                nearest = inside;
                *out_tower = (centre * parcel.shape.w).extend(column);
            }
        }
    }
    let [mut a, b] = cover.texels();
    // Mesoscale weather within each system: clusters, open cells and gaps
    // that ride the wind with it.
    let point = up * PLANET_RADIUS;
    let time = view.time();
    let phase = time / FLOW_PERIOD - (time / FLOW_PERIOD).floor();
    let later = phase + 0.5 - (phase + 0.5).floor();
    let weight = 1.0 - (1.0 - phase * 2.0).abs();
    let wind = cloud_wind(up);
    let noise = |offset: f32| {
        base.sample_by_lod(
            *sampler,
            (point - wind * (offset * FLOW_PERIOD)) / 90_000.0,
            0.0,
        )
    };
    let meso = noise(phase).lerp(noise(later), 1.0 - weight);
    let clusters = (meso.x * 0.6 + meso.y * 0.4).smoothstep(0.25, 0.7);
    a.x *= clusters;
    a.z *= (meso.z * 0.7 + 0.45).min(1.0);
    a.w *= (meso.w * 0.9 + 0.3).min(1.0);
    *out_cover = a;
    *out_layers = b;
}

/// How far convective cloud near each column can reach: the most cover,
/// highest top and lowest base among its neighbours, and any storm tower.
#[spirv(fragment(entry_point_name = "cloud_bounds"))]
pub fn cloud_bounds(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(descriptor_set = 0, binding = 0)] cloud_cover: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] cloud_towers: &Image!(2D, type=f32, sampled=true),
    #[spirv(location = 0)] out_bounds: &mut Vec4,
) {
    let texel = pixel.xy().as_ivec2();
    let limit = IVec2::new(CLOUD_MAP_WIDTH as i32 - 1, CLOUD_MAP_HEIGHT as i32 - 1);
    let mut bounds = vec4(0.0, 0.0, 1.0e6, 0.0);
    for index in 0..9 {
        let at = (texel + IVec2::new(index % 3 - 1, index / 3 - 1)).clamp(IVec2::ZERO, limit);
        let cover: Vec4 = cloud_cover.fetch(at);
        if cover.x > 0.001 {
            let layers: Vec4 = cloud_layers.fetch(at);
            // Towers, added below, climb past the convective top of the
            // columns around them.
            bounds = vec4(
                bounds.x.max(cover.x),
                bounds.y.max(cover.y),
                bounds.z.min(layers.x),
                0.0,
            );
        }
    }
    let tower: Vec4 = cloud_towers.fetch(texel);
    if tower.w > 0.0 {
        let layers: Vec4 = cloud_layers.fetch(texel);
        bounds = vec4(
            1.0,
            bounds.y.max(tower.xyz().length()),
            bounds.z.min(layers.x),
            0.0,
        );
    }
    *out_bounds = bounds;
}

/// Trace the cloud's shadow through every texel of the light volume: its
/// optical depth towards the pole light beyond each sample's own near
/// shadow and in full, and down to the aur.
#[spirv(compute(threads(8, 8, 1), entry_point_name = "cloud_light"))]
pub fn cloud_shadow(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] cloud_cover: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] cloud_layers: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] output: &mut [UVec2],
) {
    // Shadows change slowly: each frame retraces alternate layers.
    let layer = id.z * 2 + (frame.view.frame_count() as u32 & 1);
    if id.x >= LIGHT_WIDTH || id.y >= LIGHT_HEIGHT || layer >= LIGHT_LAYERS {
        return;
    }
    let id = UVec3::new(id.x, id.y, layer);
    let column = |point: Vec3| {
        let uv = atlas_texel(point.normalize(), CLOUD_FACE)
            / vec2(CLOUD_MAP_WIDTH as f32, CLOUD_MAP_HEIGHT as f32);
        (
            cloud_cover.sample_by_lod(*sampler, uv, 0.0),
            cloud_layers.sample_by_lod(*sampler, uv, 0.0),
        )
    };
    let up = atlas_direction(id.truncate().as_vec2() + 0.5, LIGHT_FACE);
    let altitude = light_altitude(id.z as f32);
    let start = up * (FEATURE_RADIUS + altitude);
    let light = light_towards(start);
    let (cover, layers) = column(start);
    let below = depth_below(cover, layers, altitude);
    // The near stretch, which samples resolve finely themselves, then
    // lengthening steps far out towards the light.
    let middle = start + light * (NEAR_SHADOW * 0.5);
    let (cover, layers) = column(middle);
    let near = mean_density(cover, layers, middle.length() - FEATURE_RADIUS) * NEAR_SHADOW;
    let mut far = 0.0;
    let mut distance = NEAR_SHADOW;
    let mut step = 1500.0;
    for _ in 0..7 {
        let point = start + light * (distance + step * 0.5);
        let height = point.length() - FEATURE_RADIUS;
        if height > CLOUD_TOP {
            break;
        }
        let (cover, layers) = column(point);
        far += mean_density(cover, layers, height) * step;
        distance += step;
        step *= 1.8;
    }
    let index = (id.z * LIGHT_HEIGHT + id.y) * LIGHT_WIDTH + id.x;
    output[index as usize] = UVec2::new(half(far) | (half(far + near) << 16), half(below));
}
