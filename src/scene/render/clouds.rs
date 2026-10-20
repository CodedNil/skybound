use crate::scene::{
    FEATURE_RADIUS, FEATURE_SCALE, FrameUniform, PLANET_RADIUS, ViewUniform,
    aur::aur_climate,
    climate::{
        CLOUD_BOTTOM, CLOUD_FACE, CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CLOUD_TOP, CloudCover,
        CloudParcel, LAYER_DRIFT, LAYERS, LIGHT_FACE, LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH,
        RAIN_DEPTH, TOWER, atlas_direction, atlas_texel, cloud_cell, cloud_wind, layer_altitude,
        layer_depth, light_altitude,
    },
    math::{half, henyey_greenstein, intersect_sphere, rotated, simpson, square},
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

/// The spacing of cumulus heaps, about a kilometre and a half apart in the
/// world, and the scale of the clusters they gather into.
const CUMULUS_CELL: f32 = 12_000.0;

const CLUSTER_SCALE: f32 = 50_000.0;

/// The size of the broad swells on which every layer rises and falls
/// together, and how far they lift or lower it; and how far each cloud's
/// own base strays from the swell.
const SWELL_SCALE: f32 = 120_000.0;

const SWELL: f32 = LAYER_DRIFT * 0.7;

const STRAY: f32 = LAYER_DRIFT * 0.3;

/// The size of one cloud map texel at a face's centre, in feature metres.
const CLOUD_TEXEL: f32 = FEATURE_RADIUS * FRAC_PI_2 / CLOUD_FACE as f32;

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

/// Peak extinction of each layer's cloud, per metre: dense heaps of
/// cumulus, through thinner middle cloud, to ice crystals that barely veil
/// the sky.
fn extinction(layer: u32) -> f32 {
    match layer {
        0 => 0.02,
        1 => 0.012,
        2 => 0.01,
        3 => 0.006,
        4 => 0.012,
        5 => 0.0025,
        _ => 0.000_3,
    }
}

/// How a layer's genus is shaped. Cells are its clouds' spacing; streaks
/// smear them out along the wind; a lens rounds their bases as well as
/// their tops; puffs heap lobes on them and erosion frays them; ripples lay
/// billows across the wind.
#[derive(Clone, Copy)]
struct Genus {
    cell: f32,
    streaks: f32,
    lens: f32,
    puffs: f32,
    erosion: f32,
    ripples: f32,
}

fn genus(layer: u32) -> Genus {
    let genus = |cell, streaks, lens, puffs, erosion, ripples| Genus {
        cell,
        streaks,
        lens,
        puffs,
        erosion,
        ripples,
    };
    match layer {
        // Stratocumulus: broad lumpy rolls along the wind.
        1 => genus(13_000.0, 0.0, 0.25, 0.45, 0.4, 0.0),
        // Altocumulus: small rounded puffs, often in waves.
        2 => genus(4500.0, 0.0, 0.5, 0.4, 0.4, 0.3),
        // Lenticular lenses and altostratus: smooth and drawn out.
        3 => genus(32_000.0, 3.0, 0.5, 0.08, 0.12, 0.0),
        // Castellanus: turrets rising from a common base.
        4 => genus(6000.0, 0.0, 0.0, 0.55, 0.45, 0.0),
        // Cirrocumulus: fine grains in ripples.
        5 => genus(2600.0, 0.0, 0.5, 0.2, 0.3, 0.55),
        // Cirrus: fibres combed far out along the wind.
        _ => genus(2600.0, 12.0, 0.35, 0.0, 0.5, 0.0),
    }
}

/// How finely a density is evaluated: in full; as a cheap probe that may
/// only overstate the cloud, to decide whether a step needs the full
/// density at all; as a cheap estimate of it, for shadows; or, for the
/// upper layers far off, as each layer's whole depth through its column.
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum Detail {
    Full,
    Probe,
    Shadow,
    Slab,
}

/// The lowest layer crossed whole far off, and how far off: thin layers
/// seen from afar cover few pixels in depth, and marching through them is
/// most of a ray's cost.
pub const SLAB_LAYER: u32 = 2;

pub const SLAB_DISTANCE: f32 = 60_000.0;

/// Cloud at a point: droplet and rain extinction, and how far up its cloud
/// the point sits, from a dark base at 0 to a sunlit crown at 1.
///
/// Crossed whole, the density is the layers' optical depth straight up
/// through the column, and `free` the deepest of their clouds.
pub struct CloudSample {
    pub density: f32,
    pub rain: f32,
    pub height: f32,
    /// How far from the point any cloud certainly is, in metres.
    pub free: f32,
}

/// What a march step needs to know about the cloud over one column, read
/// once and shared by every density it evaluates nearby.
///
/// Every member is four wide, so a march can keep one in its state.
#[derive(Clone, Copy)]
pub struct CloudColumn {
    /// The cover of layers 0 to 3, then of 4 to 6 and the convective top.
    lower: Vec4,
    upper: Vec4,
    /// Which layers hold cloud anywhere near, as bits; then the highest
    /// convective top and the most convective cover near.
    bounds: Vec4,
    /// The storm tower over this column, if any: its centre scaled by its
    /// top altitude, and its column radius.
    tower: Vec4,
    /// Offsets placing a point in the two drifting copies of the noise, and
    /// how much the nearer copy counts in the crossfade.
    near: Vec4,
    far: Vec4,
    /// Direction of the wind, along which rain leans and towers lean.
    along: Vec4,
}

/// The noise one density evaluation reads: positions are taken in the
/// planet and carried into the two drifting copies of the noise, crossfaded
/// for the large shapes where they meet.
struct Drift<'a, 'b> {
    textures: &'a Textures<'b>,
    point: Vec3,
    /// Offsets into the two copies, and into the one that leads.
    near: Vec3,
    far: Vec3,
    lead: Vec3,
    weight: f32,
    footprint: f32,
    detail: Detail,
}

impl Drift<'_, '_> {
    fn sample(&self, p: Vec3, scale: f32) -> Vec4 {
        self.textures
            .shapes_filtered(p / scale, self.footprint / scale)
    }

    /// A smooth field varying over features of about `size`: the noise's
    /// coarse octaves alone, read from a mip that has blurred the fine ones
    /// away.
    fn smooth(&self, p: Vec3, size: f32) -> Vec4 {
        let scale = size * 8.0;
        let footprint = self.footprint.max(scale / 32.0);
        let sample = |offset: Vec3| {
            self.textures
                .shapes_filtered((p + offset) / scale, footprint / scale)
        };
        if !matches!(self.detail, Detail::Full) || self.weight <= 0.0 || self.weight >= 1.0 {
            return sample(self.lead);
        }
        sample(self.near).lerp(sample(self.far), 1.0 - self.weight)
    }

    /// The large shapes about a position, crossfaded between the copies.
    fn shapes(&self, p: Vec3, scale: f32) -> Vec4 {
        if !matches!(self.detail, Detail::Full) || self.weight <= 0.0 || self.weight >= 1.0 {
            return self.sample(p + self.lead, scale);
        }
        self.sample(p + self.near, scale)
            .lerp(self.sample(p + self.far, scale), 1.0 - self.weight)
    }

    /// Cells of cloud: Worley cells, warped, and for the most prominent
    /// clouds a second rotated copy, so that no grid shows through their
    /// outlines. Their field changes by at most about five per cell
    /// crossed, which bounds how far a point outside them is from one.
    /// Squared distances round
    /// each cell's crown where plain ones would peak in a cusp; and the
    /// result is spread over the whole range by its measured 5th and 95th
    /// percentiles.
    fn cells(&self, column: Vec3, scale: f32, twice: bool) -> f32 {
        let tile = scale * 8.0;
        let warp = self.smooth(column * 0.37, scale * 2.7);
        let warped = column + (warp.wzy() - 0.5) * (scale * 1.6);
        let a = self.shapes(warped, tile).x;
        if !twice {
            return remap(a * (2.0 - a), 0.34, 0.95);
        }
        let turned = rotated((warped + self.lead) / (tile * 1.31));
        let b = self
            .textures
            .shapes_filtered(turned, self.footprint / (tile * 1.31))
            .x;
        let nearest = a.max(b);
        remap(nearest * (2.0 - nearest), 0.56, 0.97)
    }

    /// Structure about a position with features of about `size`, from the
    /// leading copy alone; the cells repeat eight times across the texture.
    fn detail(&self, p: Vec3, size: f32) -> Vec4 {
        self.sample(p + self.lead, size * 8.0)
    }

    /// Detail that pixels cannot resolve fades rather than aliasing.
    fn resolve(&self, size: f32) -> f32 {
        1.0 - (self.footprint / size).smoothstep(0.15, 0.6)
    }
}

impl CloudColumn {
    pub const CLEAR: Self = Self {
        lower: Vec4::ZERO,
        upper: Vec4::ZERO,
        bounds: Vec4::ZERO,
        tower: Vec4::ZERO,
        near: Vec4::ZERO,
        far: Vec4::ZERO,
        along: Vec4::X,
    };

    fn cover(&self, layer: u32) -> f32 {
        match layer {
            0 => self.lower.x,
            1 => self.lower.y,
            2 => self.lower.z,
            3 => self.lower.w,
            4 => self.upper.x,
            5 => self.upper.y,
            _ => self.upper.z,
        }
    }

    /// Altitudes the cloud of a layer near this column can fill.
    fn band(&self, layer: u32) -> Vec2 {
        // The lowest and highest the swell lifts the layers near here, and
        // how far any cloud's own base strays from it.
        let low = self.bounds.z - STRAY;
        let high = self.bounds.w + STRAY;
        let base = layer_altitude(layer);
        if layer == 0 {
            // Cumulus bases wander half as far; the tallest crowns bulge a
            // little past their dome.
            let floor = if self.raining() {
                base - RAIN_DEPTH
            } else {
                base + low * 0.5 - 300.0
            };
            let crown = (self.bounds.y - base).max(0.0) * 1.2;
            return vec2(floor, base + high * 0.5 + crown);
        }
        if layer == 6 {
            // Fall streaks hang below the cirrus.
            return vec2(base + low - 2500.0, base + high + layer_depth(6));
        }
        let depth = layer_depth(layer) * 1.55;
        vec2(base + low - depth * 0.2, base + high + depth * 1.2)
    }

    fn raining(&self) -> bool {
        storm(self.lower, self.upper) > 0.0 || self.tower.w > 0.0
    }

    /// How far a ray can run from a point before it could enter any of
    /// this column's layers, zero within one; and how far it may climb in
    /// one step without slipping past the thinnest cloud there.
    pub fn room(&self, point: Vec3, direction: Vec3) -> (f32, f32) {
        let mask = self.bounds.x as u32;
        if mask == 0 {
            return (1.0e9, 1.0e9);
        }
        let altitude = point.length() - FEATURE_RADIUS;
        // The nearest band above and below the point.
        let mut above = vec2(1.0e9, 1.0e9);
        let mut below = vec2(-1.0e9, -1.0e9);
        let mut rise = 1.0e9f32;
        for layer in 0..LAYERS {
            if mask & (1 << layer) == 0 {
                continue;
            }
            let band = self.band(layer);
            if altitude >= band.x && altitude <= band.y {
                rise = rise.min(climb(layer));
                continue;
            }
            if band.x > altitude && band.x < above.x {
                above = band;
            }
            if band.y < altitude && band.y > below.y {
                below = band;
            }
        }
        if rise < 1.0e8 {
            return (0.0, rise);
        }
        let mut room = 1.0e9f32;
        if above.x < 1.0e8 {
            room = room.min(approach(point, direction, altitude, above.x, above.y));
        }
        if below.y > -1.0e8 {
            room = room.min(approach(point, direction, altitude, below.x, below.y));
        }
        (room, rise)
    }

    /// Where a ray leaves the upper layers it is within, to cross them whole
    /// in one step; zero where it is within a lower layer, which is always
    /// marched.
    pub fn crossing(&self, point: Vec3, direction: Vec3) -> f32 {
        let mask = self.bounds.x as u32;
        let altitude = point.length() - FEATURE_RADIUS;
        let mut exit = 0.0f32;
        let mut entry = 1.0e9f32;
        for layer in 0..LAYERS {
            if mask & (1 << layer) == 0 {
                continue;
            }
            let band = self.band(layer);
            let inside = altitude >= band.x && altitude <= band.y;
            if layer < SLAB_LAYER {
                if inside {
                    return 0.0;
                }
                // Stop short of any lower layer the ray would enter.
                entry = entry.min(approach(point, direction, altitude, band.x, band.y));
                continue;
            }
            if inside {
                let top = intersect_sphere(point, direction, FEATURE_RADIUS + band.y).y;
                let bottom = intersect_sphere(point, direction, FEATURE_RADIUS + band.x);
                let leave = if bottom.x <= bottom.y && bottom.x >= 0.0 {
                    bottom.x
                } else {
                    top
                };
                exit = exit.max(leave);
            }
        }
        exit.min(entry)
    }

    pub fn sample(
        &self,
        point: Vec3,
        footprint: f32,
        textures: &Textures,
        detail: Detail,
    ) -> CloudSample {
        let radius = point.length();
        let altitude = radius - FEATURE_RADIUS;
        let mut sample = CloudSample {
            density: 0.0,
            rain: 0.0,
            height: 0.5,
            free: 1.0e9,
        };
        let mask = self.bounds.x as u32;
        if altitude < CLOUD_BOTTOM || altitude > CLOUD_TOP || mask == 0 {
            // The bounds hold for about a cloud map texel around.
            sample.free = CLOUD_TEXEL * 0.5;
            return sample;
        }
        let up = point / radius;
        let drift = Drift {
            textures,
            point,
            near: self.near.xyz(),
            far: self.far.xyz(),
            lead: if self.near.w > 0.5 {
                self.near.xyz()
            } else {
                self.far.xyz()
            },
            weight: self.near.w,
            footprint,
            detail,
        };
        let mut height = 0.0;
        let mut weights = 0.0;
        let mut thickest = 0.0f32;
        for layer in 0..LAYERS {
            if mask & (1 << layer) == 0 {
                continue;
            }
            let band = self.band(layer);
            let outside = (band.x - altitude).max(altitude - band.y);
            if outside > 0.0 {
                // Clear of this layer by a known height, which a level ray
                // keeps for about half a cloud map texel.
                sample.free = sample.free.min(outside.min(CLOUD_TEXEL * 0.5));
                continue;
            }
            let cover = self.cover(layer);
            if cover <= 0.001 && (layer > 0 || self.tower.w <= 0.0) {
                sample.free = 0.0;
                continue;
            }
            let slab = matches!(detail, Detail::Slab);
            if slab && layer < SLAB_LAYER {
                continue;
            }
            let found = if layer == 0 {
                self.convective(&drift, up, altitude, cover)
            } else if layer == 6 {
                cirrus(&drift, up, altitude, cover)
            } else {
                stratiform(&drift, layer, up, altitude, cover)
            };
            if slab {
                if found.x > 0.0 {
                    sample.density += found.x * extinction(layer);
                    thickest = thickest.max(found.z);
                }
                continue;
            }
            // Outside every footprint, the field's bounded slope says how
            // far the nearest cloud of this layer is.
            sample.free = sample
                .free
                .min(if found.x > 0.0 { 0.0 } else { found.z.max(0.0) });
            let (density, fraction) = (found.x * extinction(layer), found.y);
            sample.density += density;
            height += fraction * density;
            weights += density;
        }
        if matches!(detail, Detail::Slab) {
            sample.free = thickest;
            return sample;
        }
        if weights > 0.0 {
            sample.height = height / weights;
        }
        let deep = storm(self.lower, self.upper).max(if self.tower.w > 0.0 { 0.6 } else { 0.0 });
        let base = layer_altitude(0);
        if deep > 0.0 && altitude < base && altitude > base - RAIN_DEPTH {
            // Rain hangs below deep storms in shafts that thin on their way
            // down and lean with the wind.
            let fall = (base - altitude) / RAIN_DEPTH;
            let leaning = self.along.xyz() * ((base - altitude) * 0.4);
            let shafts = match detail {
                Detail::Full => drift.detail(point + leaning, 3000.0).z,
                Detail::Probe => 0.95,
                Detail::Shadow | Detail::Slab => 0.7,
            };
            let rain = remap(shafts, 0.55, 0.8) * (1.0 - fall * fall) * deep;
            sample.rain += rain * RAIN_EXTINCTION;
        }
        sample
    }

    /// Cumulus and storm towers: heaps on a flat base, each as tall as its
    /// column's convection lets it grow, cauliflowered at the crown.
    /// The density, how far up its cloud the point sits, and how far any
    /// cloud of this layer certainly is: as for every layer's density.
    fn convective(&self, drift: &Drift, up: Vec3, altitude: f32, cover: f32) -> Vec3 {
        let nominal = layer_altitude(0);
        // The column level with the layer's base, leaning downwind with
        // height as the wind strengthens aloft.
        let rise = altitude - nominal;
        let column = drift.point - up * rise - self.along.xyz() * (rise.max(0.0) * 0.25);
        // Footprints first: most samples between heaps end here.
        let clusters = drift.smooth(column * 0.7, CLUSTER_SCALE);
        let grouped = remap(clusters.w, 0.42, 0.66);
        let cells = drift.cells(column, CUMULUS_CELL, true);
        // Clusters raise both how many footprints break the surface and how
        // tall they grow; cover sets the share of the sky they fill.
        let field = cells * 0.75 + grouped * 0.25 + slack(drift.detail);
        let footprint = remap(field, 1.0 - cover, 1.45 - cover);
        let towering = self.tower.w > 0.0;
        if footprint <= 0.0 && !towering {
            return vec3(0.0, 0.5, (1.0 - cover - field) * CUMULUS_CELL * 0.2);
        }
        // Cumulus bases sit near the condensation level: they wander less.
        // A probe allows for any wander instead of reading it.
        let probe = matches!(drift.detail, Detail::Probe);
        let base = if matches!(drift.detail, Detail::Full) {
            nominal + wander(drift, column, CUMULUS_CELL, 0.0) * 0.5
        } else if probe {
            nominal + (swell(drift.textures.shapes, drift.textures.sampler, column) - STRAY) * 0.5
        } else {
            nominal
        };
        let rise = altitude - base;
        // Each heap's own height: flat humilis, heaped mediocris, and in the
        // clusters congestus towering over their neighbours, as far as the
        // column's convection reaches.
        let development = (self.upper.w - base).max(2500.0);
        let own = remap(
            drift.smooth(column * 1.13, CUMULUS_CELL * 1.1).w,
            0.32,
            0.68,
        );
        let tallest = (CUMULUS_CELL * (0.18 + 0.75 * own * own * (0.35 + 0.65 * grouped)))
            .min(development)
            .max(2000.0);
        if probe && !towering {
            // Above or below every heap here; heaps' heights vary slowly
            // enough that half the gap is clear all round.
            let gap = (-300.0 - rise).max(rise - tallest * 1.2 - STRAY);
            return if gap > 0.0 {
                vec3(0.0, 0.5, gap * 0.45)
            } else {
                vec3(1.0, 0.5, 0.0)
            };
        }
        let u = rise / tallest;
        // A dome on a flat base: highest at the heart of the footprint.
        let mut shape = footprint * (2.0 - footprint) - u * u;
        let mut fraction = u.clamp(0.0, 1.0);
        let mut scale = CUMULUS_CELL;
        if towering {
            // Storm towers are the weather's own: placed and grown by it.
            let top = self.tower.xyz().length();
            let centre = self.tower.xyz() / top;
            let horizontal = (up - centre) * FEATURE_RADIUS;
            let local = vec3(
                horizontal.dot(self.along.xyz()),
                horizontal.dot(up.cross(self.along.xyz())),
                rise,
            );
            let (profile, along) = tower_profile(local, self.tower.w, (top - base).max(20_000.0));
            if profile > shape {
                shape = profile;
                fraction = along;
                scale = self.tower.w;
            }
        }
        if rise < 0.0 {
            // A flat, slightly ragged base.
            shape = shape.min(rise / 250.0);
        }
        vec3(heap(shape, fraction, scale, drift), fraction, 0.0)
    }
}

/// How much further a probe reaches than the cloud: the full density blends
/// two drifting copies of the noise where a probe reads only one.
fn slack(detail: Detail) -> f32 {
    if matches!(detail, Detail::Probe) {
        0.12
    } else {
        0.0
    }
}

/// How far a cloud's base sits above or below its layer's at a column:
/// the broad swell, and each cloud's own stray from it.
fn wander(drift: &Drift, column: Vec3, cell: f32, layer: f32) -> f32 {
    let offset = Vec3::splat(layer * 3.7 * cell);
    let own = drift.smooth(column * 0.91 + offset, cell * 1.2).w;
    swell(drift.textures.shapes, drift.textures.sampler, column)
        + ((own - 0.5) * 3.2).clamp(-1.0, 1.0) * STRAY
}

/// The broad swell on which every layer's clouds rise and fall together,
/// in metres. It stays put rather than drifting with the clouds, so the
/// bounds pass can bound it over each column.
fn swell(shapes: &Image!(3D, type=f32, sampled=true), sampler: &Sampler, point: Vec3) -> f32 {
    let at = point.normalize() * (FEATURE_RADIUS / (SWELL_SCALE * 8.0));
    let value: Vec4 = shapes.sample_by_lod(*sampler, at, 2.0);
    ((value.w - 0.5) * 3.2).clamp(-1.0, 1.0) * SWELL
}

/// Erode a heap's dimensional profile into cloud. Lobes at a fraction of
/// the heap's size bulge its crown into cauliflower; finer billows crown
/// its tops and wisps fray its base, both stronger towards its edge and
/// faded out where they are smaller than a pixel.
fn heap(profile: f32, fraction: f32, scale: f32, drift: &Drift) -> f32 {
    if profile <= -0.45 {
        return 0.0;
    }
    match drift.detail {
        Detail::Full => {}
        Detail::Probe => return (profile + 0.45).clamp(0.0, 1.0),
        Detail::Shadow | Detail::Slab => return (profile * 3.0).clamp(0.0, 0.8),
    }
    // Cauliflower: lobes a third of the heap across, with smaller lobes on
    // them, bulging from the crown and flanks but not the flat base.
    let crown = fraction.smoothstep(0.03, 0.3);
    let lobe = scale * 0.33;
    let mut shape = profile;
    let resolve = drift.resolve(lobe);
    if resolve > 0.01 {
        let lobes = remap(drift.detail(drift.point, lobe).y, 0.3, 0.67);
        shape += (lobes - 0.45) * 0.9 * crown * resolve;
    }
    let resolve = drift.resolve(1500.0);
    if resolve > 0.01 && shape < 1.0 {
        let billows = remap(
            drift.detail(drift.point + Vec3::splat(700.0), 1500.0).z,
            0.55,
            0.9,
        );
        let wisps = drift
            .textures
            .details_filtered(
                (drift.point + drift.lead) / 1600.0,
                drift.footprint / 1600.0,
            )
            .w;
        let erosion = remap(wisps, 0.23, 0.77).lerp(billows, crown);
        shape -= (1.0 - erosion) * 0.45 * (1.0 - shape) * resolve;
    }
    // A soft rim whose width follows the pixel, so outlines never flicker.
    let width = 0.07 + drift.footprint / scale;
    (shape / width).clamp(0.0, 1.0).powf(1.5)
}

/// The cloud of one of the layers above the cumulus at a point, and how
/// far up it the point sits.
fn stratiform(drift: &Drift, layer: u32, up: Vec3, altitude: f32, cover: f32) -> Vec3 {
    let genus = genus(layer);
    let nominal = layer_altitude(layer);
    let level = drift.point - up * (altitude - nominal);
    // Footprints first: most samples between clouds end here.
    let cells = if genus.streaks > 0.0 {
        streaks(drift, level, up, genus, layer)
    } else {
        drift.cells(level, genus.cell, false)
    };
    // Cells gather into patches with open sky between them.
    let patches = remap(drift.smooth(level * 0.61, genus.cell * 4.0).w, 0.32, 0.68);
    let mut field = cells * 0.6 + patches * 0.4 + slack(drift.detail);
    if genus.ripples > 0.0 {
        // Billows laid in rows, bending where the noise does, across one of
        // the fixed axes for the same reason streaks are drawn along them.
        let bend = drift.smooth(level * 0.5, genus.cell * 1.5).z;
        let weights = streak_axes(up);
        let along = if weights.x > weights.y {
            Vec3::X
        } else {
            Vec3::Z
        };
        let phase = level.dot(along) / (genus.cell * 1.3) + bend * 5.0;
        let rows = ((phase * TAU).sin() * 0.5 - 0.1) * (weights.x - weights.y).abs();
        field += rows * genus.ripples * drift.resolve(genus.cell);
    }
    let footprint = remap(field, 1.0 - cover, 1.3 - cover);
    if footprint <= 0.0 {
        return vec3(0.0, 0.5, (1.0 - cover - field) * genus.cell * 0.2);
    }
    let full = matches!(drift.detail, Detail::Full);
    let base = if full {
        nominal + wander(drift, level, genus.cell, layer as f32)
    } else {
        nominal + swell(drift.textures.shapes, drift.textures.sampler, level)
    };
    // Denser cover thickens the layer, and each cloud has its own depth.
    let own = remap(drift.smooth(level * 1.07, genus.cell * 1.3).w, 0.32, 0.68);
    let depth = layer_depth(layer) * (0.3 + 0.7 * cover) * (0.45 + 1.1 * own);
    if matches!(drift.detail, Detail::Slab) {
        // The whole depth of the cloud over this column, its edge frayed
        // by the same detail as up close.
        let scale = (genus.cell * 0.15).max(800.0);
        let detail = drift.detail(level + Vec3::splat(1300.0), scale);
        let frayed =
            footprint - (1.0 - remap(detail.w, 0.23, 0.77)) * genus.erosion * (1.0 - footprint);
        let fill = (frayed * (2.0 - frayed) * 1.6).clamp(0.0, 1.0);
        return vec3(fill * fill * depth * 0.42, 0.5, depth);
    }
    if matches!(drift.detail, Detail::Probe) {
        // Allowing for any stray of the base from the swell. The swell, the
        // stray and the depth all vary slowly enough that well under half
        // the gap to the layer is clear all round.
        let rise = altitude - base;
        let gap = (-STRAY - 0.2 * depth - rise).max(rise - STRAY - 1.2 * depth);
        return if gap > 0.0 {
            vec3(0.0, 0.5, gap * 0.45)
        } else {
            vec3(1.0, 0.5, 0.0)
        };
    }
    let h = (altitude - base) / depth.max(500.0);
    if h < -0.2 || h > 1.2 {
        return vec3(0.0, 0.5, 0.0);
    }
    // A rounded top, and a rounded base too where the genus is a lens.
    let centre = genus.lens * 0.5;
    let v = if h > centre {
        (h - centre) / (1.0 - centre)
    } else if centre > 0.0 {
        (centre - h) / centre
    } else {
        -h * 4.0
    };
    let mut shape = footprint * (2.0 - footprint) - v * v;
    if layer == 3 {
        // Lenticular clouds stack like plates where the wave flow layers.
        shape -= ((h * 3.0 * PI).sin().abs() - 0.6).max(0.0) * 0.5 * footprint;
    }
    let fraction = h.clamp(0.0, 1.0);
    if !full {
        return vec3((shape * 2.5).clamp(0.0, 0.8), fraction, 0.0);
    }
    if genus.puffs > 0.0 {
        let lobe = genus.cell * 0.35;
        let resolve = drift.resolve(lobe);
        if resolve > 0.01 {
            let lobes = drift
                .detail(drift.point + Vec3::splat(layer as f32 * 917.0), lobe)
                .y;
            shape += (remap(lobes, 0.3, 0.67) - 0.5)
                * genus.puffs
                * footprint.smoothstep(0.0, 0.3)
                * resolve;
        }
    }
    let scale = (genus.cell * 0.15).max(800.0);
    let resolve = drift.resolve(scale);
    if resolve > 0.01 {
        let detail = drift.detail(drift.point + Vec3::splat(1300.0), scale);
        let erosion = remap(detail.w, 0.23, 0.77)
            .lerp(remap(detail.z, 0.55, 0.9), fraction.smoothstep(0.1, 0.6));
        let edge = (1.0 - shape).clamp(0.0, 1.0);
        shape -= (1.0 - erosion) * genus.erosion * edge * resolve;
    }
    let width = 0.1 + drift.footprint / genus.cell;
    vec3((shape / width).clamp(0.0, 1.0).powf(1.3), fraction, 0.0)
}

/// How much each of the two fixed axes along which streaks are drawn out
/// lies level at a column. A coordinate along a fixed axis runs straight
/// through the noise, where the wind's own direction, turning from column to
/// column, would give none; and no single direction can lie level all over a
/// sphere, so two blend, crossing where neither lies level.
fn streak_axes(up: Vec3) -> Vec2 {
    let level = vec2(1.0 - up.x * up.x, 1.0 - up.z * up.z);
    let weights = vec2(level.x.smoothstep(0.45, 0.8), level.y.smoothstep(0.45, 0.8));
    weights / (weights.x + weights.y).max(0.0001)
}

/// A position squeezed along a fixed axis, so noise sampled there draws out
/// that way.
fn squeezed(p: Vec3, axis: Vec3, stretch: f32) -> Vec3 {
    p - axis * (p.dot(axis) * (1.0 - 1.0 / stretch))
}

/// A field drawn out into streaks along the fixed axes, bending as it goes
/// into fibres and hooks.
fn streaks(drift: &Drift, level: Vec3, up: Vec3, genus: Genus, layer: u32) -> f32 {
    let weights = streak_axes(up);
    // Hooks: the streaks bend with a slow warp.
    let warp = drift.smooth(level * 0.53, genus.cell * 2.0);
    let bent = level + (warp.yzw() - 0.5) * (genus.cell * 7.0);
    let mut sum = 0.0;
    for axis in 0..2 {
        let weight = if axis == 0 { weights.x } else { weights.y };
        if weight < 0.01 {
            continue;
        }
        let direction = if axis == 0 { Vec3::X } else { Vec3::Z };
        let at = squeezed(bent, direction, genus.streaks);
        let value = if layer == 3 {
            // Lenses are smooth swells of the wave flow, not cells.
            remap(drift.smooth(at, genus.cell * 0.6).w, 0.32, 0.68)
        } else {
            remap(drift.shapes(at, genus.cell * 8.0).x, 0.2, 0.75)
        };
        sum += value * weight;
    }
    sum
}

/// Cirrus: a thin sheet of ice combed into fibres along the fixed axes,
/// bent into hooks, and sagging into fall streaks where the fibres are
/// thickest.
fn cirrus(drift: &Drift, up: Vec3, altitude: f32, cover: f32) -> Vec3 {
    let nominal = layer_altitude(6);
    let level = drift.point - up * (altitude - nominal);
    let weights = streak_axes(up);
    let warp = drift.smooth(level * 0.53, 20_000.0);
    let bent = level + (warp.yzw() - 0.5) * 24_000.0;
    let mut fibres = 0.0;
    for axis in 0..2 {
        let weight = if axis == 0 { weights.x } else { weights.y };
        if weight < 0.01 {
            continue;
        }
        let direction = if axis == 0 { Vec3::X } else { Vec3::Z };
        let at = squeezed(bent, direction, 16.0);
        let broad = remap(drift.smooth(at, 4000.0).w, 0.32, 0.68);
        let fine = if matches!(drift.detail, Detail::Full) {
            remap(drift.detail(at * 1.7, 1250.0).x, 0.2, 0.8)
        } else {
            1.0
        };
        fibres += (broad * 0.65 + fine * 0.35) * weight;
    }
    let patches = remap(drift.smooth(level * 0.61, 12_000.0).w, 0.32, 0.68);
    let field = fibres * 0.75 + patches * 0.25;
    let presence = remap(field, 1.0 - cover, 1.3 - cover);
    if presence <= 0.0 {
        return vec3(0.0, 0.5, (1.0 - cover - field) * 1500.0);
    }
    let depth = layer_depth(6) * (0.4 + 0.6 * presence);
    let sag = presence * presence * 2500.0;
    if matches!(drift.detail, Detail::Slab) {
        return vec3(presence * presence.sqrt() * depth * 0.4, 0.5, depth);
    }
    if matches!(drift.detail, Detail::Probe) {
        let rise = altitude - nominal - swell(drift.textures.shapes, drift.textures.sampler, level);
        let gap = (-STRAY - sag - rise).max(rise - STRAY - depth);
        return if gap > 0.0 {
            vec3(0.0, 0.5, gap * 0.45)
        } else {
            vec3(1.0, 0.5, 0.0)
        };
    }
    let base = nominal + wander(drift, level, 12_000.0, 6.0);
    // Thicker fibres hang lower, trailing fall streaks beneath them.
    let h = (altitude - base + sag) / depth;
    let envelope = (1.0 - square(h * 2.0 - 1.0)).max(0.0);
    let density = presence * envelope;
    vec3(density * density.sqrt(), h.clamp(0.0, 1.0), 0.0)
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

/// The furthest a step may climb through a layer's band: a fraction of its
/// thinnest clouds, or steep rays would slice them into terraces or step
/// clean over them.
fn climb(layer: u32) -> f32 {
    match layer {
        0 => 1000.0,
        1 | 3 | 4 => 650.0,
        2 => 450.0,
        5 => 300.0,
        _ => 500.0,
    }
}

fn remap(value: f32, low: f32, high: f32) -> f32 {
    ((value - low) / (high - low).max(0.0001)).clamp(0.0, 1.0)
}

/// How much a column's convection is a deep, raining storm.
fn storm(lower: Vec4, upper: Vec4) -> f32 {
    lower.x.smoothstep(0.45, 0.8) * (upper.w - layer_altitude(0)).smoothstep(32_000.0, 56_000.0)
}

/// The share of a layer's band its cloud fills on average.
fn fill(layer: u32) -> f32 {
    match layer {
        0 => 0.35,
        3 | 6 => 0.5,
        _ => 0.65,
    }
}

/// Mean extinction of a layer's cloud over a column, without its noise:
/// what shadows are traced through.
fn mean_density(layer: u32, cover: f32) -> f32 {
    cover * cover * extinction(layer) * fill(layer) * MEAN_FILL
}

/// The cover of a layer in the two cloud map texels.
fn layer_cover(lower: Vec4, upper: Vec4, layer: u32) -> f32 {
    match layer {
        0 => lower.x,
        1 => lower.y,
        2 => lower.z,
        3 => lower.w,
        4 => upper.x,
        5 => upper.y,
        _ => upper.z,
    }
}

/// The altitudes a layer's cloud fills on average over a column. Most
/// cumulus stops far short of the convection's top; only storms reach it.
fn mean_band(layer: u32, lower: Vec4, upper: Vec4) -> Vec2 {
    let base = layer_altitude(layer);
    if layer == 0 {
        let development = (upper.w - base).max(0.0);
        let heaps = development.min(CUMULUS_CELL * 0.9);
        return vec2(base, base + heaps.lerp(development, storm(lower, upper)));
    }
    vec2(base, base + layer_depth(layer))
}

/// How a layer's mean density varies up through its band: cumulus thins
/// towards the crowns that only its tallest heaps reach.
fn taper(layer: u32, band: Vec2, altitude: f32) -> f32 {
    if layer > 0 {
        return 1.0;
    }
    let up = ((altitude - band.x) / (band.y - band.x).max(1.0)).clamp(0.0, 1.0);
    2.0 * (1.0 - up)
}

/// Mean optical depth of a column beneath an altitude, from the aur up.
fn depth_below(lower: Vec4, upper: Vec4, altitude: f32) -> f32 {
    let mut depth = 0.0;
    for layer in 0..LAYERS {
        let cover = layer_cover(lower, upper, layer);
        if cover > 0.001 {
            let band = mean_band(layer, lower, upper);
            // The taper averages to one over the band; beneath an altitude
            // within it, the lower part holds more than its share.
            let share = ((altitude - band.x) / (band.y - band.x).max(1.0)).clamp(0.0, 1.0);
            let share = if layer == 0 {
                share * (2.0 - share)
            } else {
                share
            };
            depth += mean_density(layer, cover) * (band.y - band.x) * share;
        }
    }
    let base = layer_altitude(0);
    depth
        + storm(lower, upper)
            * 0.4
            * RAIN_EXTINCTION
            * MEAN_FILL
            * (altitude.min(base) - (base - RAIN_DEPTH)).max(0.0)
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

/// Where a ray from an altitude, at a cosine to the vertical, crosses
/// another altitude: both distances, nearer first, or an empty range. Kept
/// in altitudes so the planet's radius never cancels in single precision.
fn crossing(altitude: f32, cosine: f32, target: f32) -> Vec2 {
    let b = (FEATURE_RADIUS + altitude) * cosine;
    let c = (altitude - target) * (2.0 * FEATURE_RADIUS + altitude + target);
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return vec2(1.0, -1.0);
    }
    let root = discriminant.sqrt();
    let q = -(b + if b >= 0.0 { root } else { -root });
    if q.abs() < 1.0e-6 {
        return vec2(0.0, 0.0);
    }
    let (a, b) = (q, c / q);
    vec2(a.min(b), a.max(b))
}

/// The stretches of a ray from an altitude, at a cosine to the vertical,
/// that lie within the band between two altitudes: one, or two where the
/// ray dips through the band and below it before rising out again.
fn within(altitude: f32, cosine: f32, band: Vec2) -> Vec4 {
    let empty = vec4(0.0, 0.0, 0.0, 0.0);
    let top = crossing(altitude, cosine, band.y);
    if top.x > top.y || top.y <= 0.0 {
        return empty;
    }
    let outer = vec2(top.x.max(0.0), top.y);
    let bottom = crossing(altitude, cosine, band.x);
    if bottom.x > bottom.y || bottom.y <= 0.0 {
        return outer.extend(0.0).extend(0.0);
    }
    vec4(
        outer.x,
        bottom.x.max(outer.x),
        bottom.y.min(outer.y),
        outer.y,
    )
}

pub fn cloud_column(point: Vec3, view: &ViewUniform, textures: &Textures) -> CloudColumn {
    let up = point.normalize();
    let [lower, upper, bounds, tower] = textures.cloud_column(up);
    let time = view.time();
    // Carry the noise with the large-scale flow, which moves through the
    // world, restarting each copy before the converging wind can stretch it
    // out of shape.
    let wind = cloud_wind(up) * FEATURE_SCALE;
    let phase = time / FLOW_PERIOD - (time / FLOW_PERIOD).floor();
    let later = phase + 0.5 - (phase + 0.5).floor();
    CloudColumn {
        lower,
        upper,
        bounds,
        tower,
        // A slow rise through the noise keeps every cloud evolving. Each
        // copy holds alone for most of its life; they blend briefly.
        near: (Vec3::Z * (time * 0.5) - wind * (phase * FLOW_PERIOD))
            .extend((1.0 - (1.0 - phase * 2.0).abs()).smoothstep(0.45, 0.55)),
        far: (Vec3::Z * (time * 0.5) - wind * (later * FLOW_PERIOD)).extend(0.0),
        along: (wind / wind.length().max(0.001)).extend(0.0),
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
        let sample =
            cloud_column(point, view, textures).sample(point, footprint, textures, Detail::Shadow);
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

/// Splat the weather systems over each column of the planet: the cover of
/// each layer and the top of the convection.
#[spirv(fragment(entry_point_name = "cloud_map"))]
pub fn cloud_map(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] base: &Image!(3D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] parcels: &[CloudParcel],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] index: &[u32],
    #[spirv(location = 0)] out_lower: &mut Vec4,
    #[spirv(location = 1)] out_upper: &mut Vec4,
    #[spirv(location = 2)] out_tower: &mut Vec4,
) {
    let view = &frame.view;
    *out_tower = Vec4::ZERO;
    let up = atlas_direction(pixel.xy(), CLOUD_FACE);
    let point = up * PLANET_RADIUS;
    let time = view.time();
    let phase = time / FLOW_PERIOD - (time / FLOW_PERIOD).floor();
    let later = phase + 0.5 - (phase + 0.5).floor();
    let weight = 1.0 - (1.0 - phase * 2.0).abs();
    let wind = cloud_wind(up);
    // Mesoscale weather within each system: clusters, open cells and gaps
    // that ride the wind with it, different in every layer.
    let noise = |offset: f32, scale: f32| {
        base.sample_by_lod(
            *sampler,
            (point - wind * (offset * FLOW_PERIOD)) / scale,
            0.0,
        )
    };
    let meso = noise(phase, 90_000.0).lerp(noise(later, 90_000.0), 1.0 - weight);
    let fine = noise(phase, 37_000.0).lerp(noise(later, 37_000.0), 1.0 - weight);
    let breaks = |value: f32, floor: f32| (value * (1.4 - floor) + floor).min(1.0);
    let (mut lower, mut upper) = if forced_weather(view) {
        // Cover of the cumulus, how deep it grows, the high cloud and the
        // middle layers.
        let weather = view.weather;
        (
            vec4(weather.x, weather.w, weather.w * 0.8, weather.w * 0.6),
            vec4(
                weather.w * 0.7,
                weather.z * 0.8,
                weather.z,
                layer_altitude(0) + 3000.0 + weather.y * 60_000.0,
            ),
        )
    } else {
        let cell = cloud_cell(up);
        let mut cover = CloudCover::default();
        // The nearest storm tower whose anvil reaches this texel, by how far
        // inside its reach the texel lies.
        let mut nearest = 0.0f32;
        for offset in index[cell]..index[cell + 1] {
            let parcel = parcels[index[offset as usize] as usize];
            cover.add(parcel, up);
            // Only the strongest, deepest systems build a full thunderhead;
            // the rest are fields of towering cumulus.
            if parcel.kind() == TOWER && parcel.shape.y > 0.6 && parcel.shape.w > 62_000.0 {
                let centre = parcel.centre.xyz();
                let distance = (up - centre).length() * PLANET_RADIUS;
                let column = parcel.detail.x * 0.35;
                let reach = column * 4.2 / FEATURE_SCALE + CLOUD_TEXEL / FEATURE_SCALE * 1.5;
                let inside = 1.0 - distance / reach;
                if inside > nearest {
                    nearest = inside;
                    *out_tower = (centre * parcel.shape.w).extend(column);
                }
            }
        }
        let [lower, upper] = cover.texels();
        (lower, upper)
    };
    lower.x *= (meso.x * 0.6 + meso.y * 0.4).smoothstep(0.25, 0.7);
    lower.y *= breaks(meso.z, 0.45);
    lower.z *= breaks(fine.w, 0.25);
    lower.w *= breaks(meso.y, 0.5);
    upper.x *= breaks(fine.z, 0.3);
    upper.y *= breaks(fine.y, 0.3);
    upper.z *= breaks(meso.w, 0.35);
    *out_lower = lower;
    *out_upper = upper;
}

/// Which layers hold cloud near each column, the highest any convection
/// near it climbs, and the lowest and highest the swell lifts the layers
/// there.
#[spirv(fragment(entry_point_name = "cloud_bounds"))]
pub fn cloud_bounds(
    #[spirv(frag_coord)] pixel: Vec4,
    #[spirv(descriptor_set = 0, binding = 0)] cloud_lower: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 1)] cloud_upper: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 2)] cloud_towers: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 4)] shapes: &Image!(3D, type=f32, sampled=true),
    #[spirv(location = 0)] out_bounds: &mut Vec4,
) {
    let texel = pixel.xy().as_ivec2();
    let limit = IVec2::new(CLOUD_MAP_WIDTH as i32 - 1, CLOUD_MAP_HEIGHT as i32 - 1);
    let mut mask = 0u32;
    let mut top = 0.0f32;
    let mut swells = vec2(1.0e9, -1.0e9);
    for index in 0..9 {
        let offset = IVec2::new(index % 3 - 1, index / 3 - 1);
        let at = (texel + offset).clamp(IVec2::ZERO, limit);
        let lower: Vec4 = cloud_lower.fetch(at);
        let upper: Vec4 = cloud_upper.fetch(at);
        for layer in 0..LAYERS {
            if layer_cover(lower, upper, layer) > 0.001 {
                mask |= 1 << layer;
            }
        }
        if lower.x > 0.001 {
            // Only storms build to the convection's top; fair cumulus stops
            // at the tallest heap.
            let base = layer_altitude(0);
            let heaps = base + (upper.w - base).min(CUMULUS_CELL * 0.95);
            top = top.max(if storm(lower, upper) > 0.0 {
                upper.w
            } else {
                heaps
            });
        }
        let swell = swell(
            shapes,
            sampler,
            atlas_direction(pixel.xy() + offset.as_vec2(), CLOUD_FACE),
        );
        swells = vec2(swells.x.min(swell), swells.y.max(swell));
    }
    let tower: Vec4 = cloud_towers.fetch(texel);
    if tower.w > 0.0 {
        // Towers climb past the convective top of the columns around them.
        mask |= 1;
        top = top.max(tower.xyz().length());
    }
    // The swell also varies between the texels it was read at.
    let margin = SWELL * 0.15;
    *out_bounds = vec4(mask as f32, top, swells.x - margin, swells.y + margin);
}

/// Trace the cloud's shadow through every texel of the light volume: its
/// optical depth towards the pole light beyond each sample's own near
/// shadow and in full, and down to the aur. Each layer's stretch of the ray
/// is integrated whole, so shadows are smooth however low the light.
#[spirv(compute(threads(8, 8, 1), entry_point_name = "cloud_light"))]
pub fn cloud_shadow(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] frame: &FrameUniform,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] cloud_lower: &Image!(2D, type=f32, sampled=true),
    #[spirv(descriptor_set = 0, binding = 3)] cloud_upper: &Image!(2D, type=f32, sampled=true),
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] output: &mut [UVec2],
) {
    // Shadows change slowly: each frame retraces every fourth layer.
    let layer = id.z * 4 + (frame.view.frame_count() as u32 & 3);
    if id.x >= LIGHT_WIDTH || id.y >= LIGHT_HEIGHT || layer >= LIGHT_LAYERS {
        return;
    }
    let column = |point: Vec3| {
        let uv = atlas_texel(point.normalize(), CLOUD_FACE)
            / vec2(CLOUD_MAP_WIDTH as f32, CLOUD_MAP_HEIGHT as f32);
        (
            cloud_lower.sample_by_lod(*sampler, uv, 0.0),
            cloud_upper.sample_by_lod(*sampler, uv, 0.0),
        )
    };
    let up = atlas_direction(vec2(id.x as f32, id.y as f32) + 0.5, LIGHT_FACE);
    let altitude = light_altitude(layer as f32);
    let start = up * (FEATURE_RADIUS + altitude);
    let light = light_towards(start);
    let cosine = up.dot(light);
    let (lower, upper) = column(start);
    let below = depth_below(lower, upper, altitude);
    let mut far = 0.0;
    let mut near = 0.0;
    // The convective top varies from column to column; bound it by the
    // highest it reaches here or where the light leaves the cloud.
    let (ahead, beyond) = column(start + light * 60_000.0);
    let reach = mean_band(0, lower, upper)
        .y
        .max(mean_band(0, ahead, beyond).y);
    for band in 0..LAYERS {
        let span = if band == 0 {
            vec2(layer_altitude(0), reach)
        } else {
            mean_band(band, lower, upper)
        };
        let stretches = within(altitude, cosine, span);
        for piece in 0..2 {
            let stretch = if piece == 0 {
                stretches.xy()
            } else {
                stretches.zw()
            };
            if stretch.y <= stretch.x {
                continue;
            }
            // Two samples of the cover across the stretch, which may run for
            // hundreds of kilometres where the light skims the layer.
            let length = stretch.y - stretch.x;
            let mut depth = 0.0;
            for tap in 0..2 {
                let t = stretch.x + length * (0.25 + 0.5 * tap as f32);
                let point = start + light * t;
                let (lower, upper) = column(point);
                let mut density = mean_density(band, layer_cover(lower, upper, band));
                if band == 0 {
                    // Thinning up to this column's own convective top.
                    let height = point.length() - FEATURE_RADIUS;
                    density *= taper(0, mean_band(0, lower, upper), height);
                }
                depth += density * length * 0.5;
            }
            // The share of the stretch beyond the near shadow.
            let beyond = ((stretch.y - NEAR_SHADOW.max(stretch.x)) / length).clamp(0.0, 1.0);
            far += depth * beyond;
            near += depth * (1.0 - beyond);
        }
    }
    let index = (layer * LIGHT_HEIGHT + id.y) * LIGHT_WIDTH + id.x;
    output[index as usize] = UVec2::new(half(far) | (half(far + near) << 16), half(below));
}
