use core::f32::consts::{FRAC_PI_2, PI, TAU};

use spirv_std::glam::{
    FloatExt, IVec2, UVec2, UVec3, Vec2, Vec3, Vec4, Vec4Swizzles, vec2, vec3, vec4,
};
use spirv_std::num_traits::Float;
use spirv_std::{Image, Sampler, spirv};

use crate::scene::aur::aur_climate;
use crate::scene::climate::{
    CLOUD_BOTTOM, CLOUD_FACE, CLOUD_MAP_HEIGHT, CLOUD_MAP_WIDTH, CLOUD_TOP, CloudCover,
    CloudParcel, LIGHT_FACE, LIGHT_HEIGHT, LIGHT_LAYERS, LIGHT_WIDTH, RAIN_DEPTH, TOWER,
    atlas_direction, atlas_texel, cloud_cell, cloud_wind, cube_direction, cube_face,
    light_altitude,
};
use crate::scene::life::smooth_min;
use crate::scene::math::{half, henyey_greenstein, intersect_sphere, simpson};
use crate::scene::planet::light_position;
use crate::scene::render::march::RaymarchResult;
use crate::scene::render::textures::{AtmosphereData, Textures};
use crate::scene::{FrameUniform, PLANET_RADIUS, ViewUniform};

/// Seconds between restarts of the noise carried along by the wind. Two
/// copies half a period apart crossfade, so detail never stretches far.
const FLOW_PERIOD: f32 = 900.0;

/// Peak extinction of each cloud family, per metre.
const CONVECTIVE_EXTINCTION: f32 = 0.02;

/// Cells across which each cumulus heap, and each storm tower, stands.
const HEAP_CELL: f32 = 6000.0;

const LARGE_CELL: f32 = 14_000.0;

/// The tallest a heap can stand: the widest heap, at its tallest shape.
const HEAP_TALLEST: f32 = LARGE_CELL * 0.27 * 1.5 * 1.3;

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
        let altitude = point.length() - PLANET_RADIUS;
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
        let altitude = radius - PLANET_RADIUS;
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
            // Solid heaps with flat bases and rounded crowns, frayed only in
            // a thin shell at their surface; and storm towers where the
            // weather raises them.
            let rise = altitude - layers.x;
            let ceiling = self.bounds.x;
            let field = |offset: Vec3| {
                let plane = face_plane((point + offset).normalize());
                // Each heap takes its cover from the column at its centre, so
                // it is never cut by the cloud map's texels.
                let cover_at = |centre: Vec2| {
                    textures.cloud_column(plane_direction(centre) - offset / PLANET_RADIUS)[0]
                };
                let a = heap(
                    plane,
                    rise,
                    HEAP_CELL,
                    Vec2::ZERO,
                    0.0,
                    1.35,
                    ceiling,
                    layers.x,
                    detailed,
                    cover_at,
                );
                let b = heap(
                    plane,
                    rise,
                    HEAP_CELL,
                    Vec2::splat(0.5),
                    17.3,
                    1.35,
                    ceiling,
                    layers.x,
                    detailed,
                    cover_at,
                );
                let mut best = if a.x > b.x { a } else { b };
                let free = a.w.min(b.w);
                // Now and then a far larger heap rises among the field.
                let large = heap(
                    plane,
                    rise,
                    LARGE_CELL,
                    Vec2::splat(0.75),
                    29.1,
                    0.55,
                    1.0,
                    layers.x,
                    detailed,
                    cover_at,
                );
                if large.x > best.x {
                    best = large;
                }
                best.w = free.min(large.w);
                best
            };
            let mut heaped = field(if weight > 0.5 { self.near } else { self.far });
            if blended {
                let (near, far) = (field(self.near), field(self.far));
                heaped = near.lerp(far, 1.0 - weight);
                heaped.w = near.w.min(far.w);
            }
            if self.tower.w > 0.0 {
                // Storm towers are the weather's own: placed and grown by it.
                let top = self.tower.xyz().length();
                let centre = self.tower.xyz() / top;
                let horizontal = (point / radius - centre) * PLANET_RADIUS;
                let up = point / radius;
                let local = vec3(
                    horizontal.dot(self.along),
                    horizontal.dot(up.cross(self.along)),
                    altitude - layers.x,
                );
                let column = tower(
                    local,
                    self.tower.w,
                    (top - layers.x).max(4000.0),
                    centre.x * 97.1,
                    detailed,
                );
                let around = (-column.x - self.tower.w * 0.3).max(0.0);
                if column.x > heaped.x {
                    heaped = column.extend(0.0);
                }
                heaped.w = heaped.w.min(around);
            }
            // Each heap's solid core fades through a zone around it where
            // noise at the heap's own scale carves the irregular silhouette
            // (the heap is the coverage the noise is remapped by), so clouds
            // stay masses with fluffy edges rather than drifting noise.
            // A fringe a fraction of the heap's size, always inside its cell;
            // the cloud history averages away what pixels cannot resolve.
            let reach = (heaped.z * (0.15 + 0.1 * heaped.y)).max(1.0);
            let occupancy = ((heaped.x + reach) / (reach * 1.4)).clamp(0.0, 1.0);
            let density = if occupancy <= 0.0 {
                0.0
            } else if detailed {
                // Between heaps, a merged mass takes the scale of a mid-sized
                // one. Each heap reads its own stretch of the noise, so
                // neighbours never repeat one another.
                let scale = heaped.z.max(HEAP_CELL * 0.3) * 1.6;
                let own = Vec3::splat(heaped.z * 7.31);
                let noise = small(own, scale);
                // A gentle, fixed-width edge: steep remaps of the 8-bit noise
                // would magnify its texel interpolation into stacked plates.
                let bulbs = noise.x * 0.6 + noise.y * 0.4;
                let edge = ((bulbs - 0.62 + occupancy * 0.75) / 0.3).clamp(0.0, 1.0);
                let mut density = edge * edge * (3.0 - 2.0 * edge) * occupancy.sqrt();
                if detailed && density > 0.0 {
                    let wisps = textures.details_filtered(lead / 1400.0, footprint / 1400.0);
                    let billows = wisps.x * 0.625 + wisps.y * 0.25 + wisps.z * 0.125;
                    // Wisps fray the bases and billows the crowns, softly and
                    // only in the fringe: thresholds here turn into grain.
                    let fade = 1.0 - (footprint / 700.0).min(1.0);
                    let erode = billows.lerp(1.0 - billows, (heaped.y * 3.0).min(1.0)) * fade;
                    density *= 1.0 - 0.7 * erode * (1.0 - occupancy);
                }
                density
            } else {
                occupancy
            };
            sample.density += density * CONVECTIVE_EXTINCTION;
            sample.height = heaped.y;
            // Plane distances overstate world ones by up to the cube warp;
            // and every heap's fringe reaches beyond its surface.
            sample.free = (heaped.w * 0.7 - 1000.0).max(0.0);
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

fn cell_hash(cell: Vec2) -> Vec4 {
    let mut q = (Vec4::new(cell.x, cell.y, cell.x, cell.y)
        * Vec4::new(0.1031, 0.1030, 0.0973, 0.1099))
    .fract_gl();
    q += q.dot(q.wzxy() + 33.33);
    ((q.xxyz() + q.yzzw()) * q.zywx()).fract_gl()
}

/// Metres across the cube faces of the planet, laid side by side.
fn face_plane(direction: Vec3) -> Vec2 {
    let (face, position) = cube_face(direction);
    let span = PLANET_RADIUS * FRAC_PI_2;
    (position * 0.5 + 0.5) * span + vec2(face as f32 * (span + HEAP_CELL * 4.0), 0.0)
}

/// The heap a grid cell holds, if any: the signed distance to its surface in
/// metres, positive inside; how far up it a point sits; and its radius. Each
/// is a cauliflower of merged lobes on a flat base, sized by the cover at its
/// centre, and stays inside its own cell, so neither neighbouring cells nor
/// face edges need checking.
fn heap(
    plane: Vec2,
    rise: f32,
    size: f32,
    shift: Vec2,
    seed: f32,
    chance: f32,
    ceiling: f32,
    base: f32,
    bulging: bool,
    cover_at: impl Fn(Vec2) -> Vec4,
) -> Vec4 {
    let cell = (plane / size + shift).floor();
    let hash = cell_hash(cell + seed);
    // No nearby column is cloudier than its bound: most empty cells end
    // here, before the cover at their centre is read.
    let middle = (cell - shift + 0.5) * size;
    // A heap never leaves its cell: any other is beyond the cell's walls.
    let edge = size * 0.5 - (plane - middle).abs().max_element();
    if hash.x > ceiling * chance {
        return vec4(-1.0e6, 0.0, 0.0, edge);
    }
    let cover = cover_at(middle);
    if cover.x < 0.001 || hash.x > cover.x * chance {
        return vec4(-1.0e6, 0.0, 0.0, edge);
    }
    // Many small heaps and a few large ones; fuller cover, larger heaps, but
    // never wider than a shallow layer lets them stand tall.
    let depth = (cover.y - base).max(400.0);
    let radius = (size * 0.27 * (0.3 + 0.7 * hash.y * hash.y) * (0.6 + 0.4 * cover.x.min(1.0)))
        .min(depth * 0.6);
    // Everything it holds, fringe included, stays this far from its centre.
    let reach = radius * 1.75;
    let centre = (vec2(hash.z, hash.w) - 0.5) * (size - reach * 2.0).max(0.0);
    // Its crown lobes rise about a third above its height; keep them under
    // the column's top, or they would be cut flat.
    let height = (radius * (0.8 + 0.7 * hash.w)).min(depth / 1.3);
    // Bases sit almost level across a field.
    let lift = rise - (hash.z - 0.5) * 300.0;
    let q = (plane - middle - centre).extend(lift);
    let lobe = |index: f32| cell_hash(cell + seed + index * 13.7);
    let lobes = if bulging { 4 } else { 0 };
    // A broad body with lobes heaped round and over its crown.
    let mut distance = ellipsoid(
        q - vec3(0.0, 0.0, height * 0.3),
        vec3(radius, radius, height * 0.42),
    );
    for index in 0..lobes {
        let random = lobe(index as f32);
        let angle = (random.x + index as f32 * 0.25) * TAU;
        let z = height * (0.45 + 0.3 * random.z);
        let offset = vec2(angle.cos(), angle.sin()) * (radius * 0.45 * random.y);
        let size = radius * (0.32 + 0.22 * random.w);
        distance = smooth_min(
            distance,
            (q - offset.extend(z)).length() - size,
            radius * 0.2,
        );
    }
    let crown = (q - vec3(0.0, 0.0, height * 0.72)).length() - radius * 0.42;
    distance = smooth_min(distance, crown, radius * 0.2);
    if !bulging {
        // Cover the reach of the lobes left out, so a probe never misses one.
        distance -= radius * 0.35;
    }
    distance = distance.max(-lift);
    vec4(
        -distance,
        (lift / height).clamp(0.0, 1.0),
        radius,
        distance.max(0.0).min(edge),
    )
}

/// A storm tower about a point `q` relative to its foot: a column narrowing a
/// little as it climbs, its flanks heaped with turrets, beneath an anvil
/// spreading flat under the tropopause. Distance positive inside, height up
/// it, and radius.
fn tower(q: Vec3, radius: f32, height: f32, seed: f32, turrets: bool) -> Vec3 {
    let along = (q.z / height).clamp(0.0, 1.0);
    // Far outside everything it holds, a bound is enough.
    let span = q.truncate().length() - radius * 4.0;
    if span > 0.0 {
        return vec3(-span - radius, along, radius);
    }
    // Broad where it draws in air at its base, narrowing as it climbs.
    let girth = radius * (0.65 + 0.35 * (1.0 - along).powi(3));
    let mut distance = q.truncate().length() - girth;
    // Turrets crowd its flanks, small and many, as cauliflower; near the
    // column only, and only where the detail is wanted.
    let flank = distance < radius * 1.2 && q.z < height * 0.95;
    for index in 0..if turrets && flank { 10 } else { 0 } {
        let random = cell_hash(vec2(seed, index as f32 * 13.7));
        let angle = (random.x + index as f32 * 0.382) * TAU;
        let z = height * (0.08 + 0.78 * (index as f32 + random.z) / 10.0);
        let reach = radius * (0.65 + 0.35 * (1.0 - z / height).powi(3));
        let offset = vec2(angle.cos(), angle.sin()) * reach * (0.6 + 0.25 * random.y);
        let size = reach * (0.4 + 0.2 * random.w);
        distance = smooth_min(
            distance,
            (q - offset.extend(z)).length() - size,
            radius * 0.15,
        );
    }
    if !turrets {
        // A probe must never see less cloud than there is: cover the reach
        // of every turret.
        distance -= radius * 0.45;
    }
    distance = distance.max(q.z - height * 0.9);
    // The anvil spreads downwind under the tropopause, flat on top and
    // thinning to a ragged edge.
    let spread = q - vec3(radius * 0.9, 0.0, height * 0.88);
    let extent = vec2(radius * 3.0, radius * 2.2);
    let outward = (spread.truncate() / extent).length();
    let thickness = (height * 0.07).max(1500.0) * (1.0 - outward * outward).max(0.03);
    let anvil = ((outward - 1.0) * extent.y).max(spread.z.abs() - thickness);
    distance = smooth_min(distance, anvil, radius * 0.6).max(-q.z);
    vec3(-distance, along, radius)
}

/// The planet direction at a point of the face plane; see [`face_plane`].
fn plane_direction(plane: Vec2) -> Vec3 {
    let span = PLANET_RADIUS * FRAC_PI_2;
    let face = (plane.x / (span + HEAP_CELL * 4.0)).floor().clamp(0.0, 5.0);
    let local = vec2(plane.x - face * (span + HEAP_CELL * 4.0), plane.y) / span;
    cube_direction(face as u32, local * 2.0 - 1.0)
}

/// Distance to an ellipsoid, a close bound everywhere: the fringe around a
/// cloud reaches well outside it, where a box bound would show.
fn ellipsoid(p: Vec3, radius: Vec3) -> f32 {
    let k0 = (p / radius).length();
    let k1 = (p / (radius * radius)).length().max(1.0e-9);
    k0 * (k0 - 1.0) / k1
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
    let crossing = intersect_sphere(point, direction, PLANET_RADIUS + target);
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
    let [cover, layers] = textures.cloud_column(up);
    let [bounds, tower] = textures.cloud_reach(up);
    let time = view.time();
    // Carry the noise with the large-scale flow, restarting each copy
    // before the converging wind can stretch it out of shape.
    let wind = cloud_wind(up);
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

pub fn cloud_sample(
    point: Vec3,
    view: &ViewUniform,
    footprint: f32,
    textures: &Textures,
    detailed: bool,
) -> CloudSample {
    cloud_column(point, view, textures).sample(point, footprint, textures, detailed)
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
    let light = (light_position(point) - point).normalize();
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
    let source = light_position(point);
    let sun = textures.pole_light(point, source) * view.sky_horizon.y * 1.5;
    // Seen from above at night, clouds over the aur glow with its light
    // diffused up through them.
    let altitude = point.length() - PLANET_RADIUS;
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
    let origin = view.to_planet(ro);
    let direction = view.planet_rotation.conjugate() * rd;
    let middle = origin + direction * (length * 0.5);
    let footprint = (length / 8.0).max(4.0);
    let density = |point| {
        let sample = cloud_sample(point, view, footprint, textures, false);
        sample.density + sample.rain
    };
    let a = density(origin);
    let m = density(middle);
    let b = density(origin + direction * length);
    let optical = simpson(a, m, b, length);
    let air = textures.air_transmittance(origin, direction, length);
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
        depth: length,
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
            let reach = column * 4.2 + CLOUD_TEXEL * 1.5;
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
            // Heaps stand no taller than their widest can; only towers,
            // added below, climb to the full convective top.
            bounds = vec4(
                bounds.x.max(cover.x),
                bounds.y.max(cover.y.min(layers.x + HEAP_TALLEST)),
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
    let start = up * (PLANET_RADIUS + altitude);
    let light = (light_position(start) - start).normalize();
    let (cover, layers) = column(start);
    let below = depth_below(cover, layers, altitude);
    // The near stretch, which samples resolve finely themselves, then
    // lengthening steps far out towards the light.
    let middle = start + light * (NEAR_SHADOW * 0.5);
    let (cover, layers) = column(middle);
    let near = mean_density(cover, layers, middle.length() - PLANET_RADIUS) * NEAR_SHADOW;
    let mut far = 0.0;
    let mut distance = NEAR_SHADOW;
    let mut step = 1500.0;
    for _ in 0..7 {
        let point = start + light * (distance + step * 0.5);
        let height = point.length() - PLANET_RADIUS;
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
