//! Native opacity and emission; the view grid supplies cloud illumination only.
use crate::scene::{
    FEATURE_RADIUS, FEATURE_SCALE, ViewUniform,
    aur::{EJECTION_CELL, EJECTION_EXTENT, EJECTION_WIDTH},
    climate::{CLOUD_BOTTOM, CLOUD_TOP, cloud_cell},
    math::{finish_opaque, henyey_greenstein, intersect_sphere, ray_shell_intersect, square},
    planet::{REACTION_RADIUS, feature_surface, light_position, light_towards},
    render::{
        clouds, ocean, poles,
        textures::{AtmosphereData, Textures},
    },
};
use spirv_std::{
    glam::{Vec2, Vec3, Vec4, Vec4Swizzles, vec3},
    num_traits::Float,
};

const VEIL_STEP: f32 = 112.0;

/// Extinction of the mist in which light shafts show, per metre at the aur,
/// and how much path it is gathered over at once.
const MIST: f32 = 0.000_004;

const MIST_STRIDE: f32 = 4000.0;

const MIST_SAMPLES: f32 = 6.0;

/// The narrowest a parcel index cell can be, in feature metres, so its layer
/// bounds hold for at least this far across it.
const CLOUD_CELL_REACH: f32 = 30_000.0 * FEATURE_SCALE;

/// The longest horizontal stride between cloud layers, in feature metres:
/// well under a cloud map texel, so the layers cannot change much across it.
const LAYER_STRIDE: f32 = 3000.0 * FEATURE_SCALE;

#[derive(Default)]
pub(super) struct VolumeSample {
    pub density: f32,
    pub color: Vec3,
    pub emission: Vec3,
}

#[derive(Copy, Clone)]
pub struct RaymarchResult {
    pub color: Vec4,
    pub depth: f32,
}

impl RaymarchResult {
    pub(super) fn empty(limit: f32) -> Self {
        Self {
            color: Vec4::W,
            depth: limit,
        }
    }
    pub(super) fn absorb(&mut self, radiance: Vec3, optical_depth: f32, distance: f32) {
        let transmission = (-optical_depth.max(0.0)).exp();
        let weight = self.color.w * (1.0 - transmission);
        let opacity = 1.0 - self.color.w;
        if opacity + weight > 0.00001 {
            self.depth = (self.depth * opacity + distance * weight) / (opacity + weight);
        }
        self.color = (self.color.xyz() + radiance * weight).extend(self.color.w * transmission);
        self.color = finish_opaque(self.color, radiance);
    }
}

/// The cloud half of the march: how far each step may go without skipping
/// cloud, and the extinction and light of the cloud a step crosses.
/// Textures and buffers are passed to each call: shader resources cannot be
/// held in a struct.
struct CloudMarch {
    jitter: f32,
    direction: Vec3,
    forced: bool,
    /// Empty samples in a row: strides lengthen through the gaps between
    /// clouds.
    misses: f32,
    /// Samples in a row that found cloud.
    depth: f32,
    /// How far the last empty probe found every cloud to be.
    free: f32,
    /// Whether this stretch of cloud has been entered at the pixel's phase.
    phased: bool,
}

/// The clouds along a pixel's ray, marched by their own pass: their light,
/// their transmittance, and where the first of them lies.
#[derive(Clone, Copy)]
pub struct CloudSlab {
    pub radiance: Vec3,
    pub transmittance: f32,
    pub front: f32,
}

struct CloudStep {
    sigma: f32,
    scattering: Vec3,
    /// A long stride found cloud: retake the step finely from its start.
    retry: bool,
}

impl CloudMarch {
    /// The longest safe step from a point, and whether it must be sampled.
    fn advance(
        &mut self,
        point: Vec3,
        footprint: f32,
        view: &ViewUniform,
        textures: &Textures,
        cells: &[Vec4],
    ) -> (f32, bool, clouds::CloudColumn) {
        let rd = self.direction;
        let gap = if self.forced {
            0.0
        } else {
            cloud_gap(point, rd, cells)
        };
        if gap > 0.0 {
            self.phased = false;
            return (gap.max(footprint), false, clouds::CloudColumn::CLEAR);
        }
        // Between this column's layers the ray runs on to the next one, in
        // strides short enough that neighbouring columns cannot hide a cloud
        // the stride would skip.
        let overhead = clouds::cloud_column(point, view, textures);
        let (room, thin) = overhead.room(point, rd);
        let up = point.normalize();
        if room > 0.0 {
            self.phased = false;
            let level = (1.0 - square(rd.dot(up))).max(0.0001).sqrt();
            return (
                room.min(LAYER_STRIDE.max(footprint * 6.0) / level)
                    .max(footprint),
                false,
                overhead,
            );
        }
        // Gaps between clouds are crossed in lengthening strides, but never
        // so steep that a thin deck slips between two samples.
        let climb = rd.dot(up).abs().max(0.02);
        // Within cloud, never climb more than a fraction of a small heap in
        // one step, or steep rays alias the heaps into hatching.
        let rise = if thin { 300.0 } else { 400.0 };
        // Steps enter cloud short and lengthen only through its solid body,
        // so its surface is found finely without crawling through it.
        let stride = if self.depth > 0.0 {
            (footprint * 1.2).max(80.0) * (1.0 + self.depth * 0.6)
        } else {
            (footprint * 1.5).max(200.0) * (1.0 + self.misses)
        };
        let mut step = stride.min(rise / climb);
        // Clear of every cloud by a known distance, leap across the gap.
        if self.free > step {
            step = self.free;
        }
        self.free = 0.0;
        // Each pixel steps through cloud at its own phases, so solid
        // surfaces are not sliced into contour rings.
        self.jitter = hash(self.jitter * 7919.0 + 0.618_034);
        if !self.phased {
            step *= 0.15 + 0.85 * self.jitter;
            self.phased = true;
        } else {
            step *= 0.75 + 0.5 * self.jitter;
        }
        (step, true, overhead)
    }

    fn sample(
        &mut self,
        overhead: &clouds::CloudColumn,
        point: Vec3,
        footprint: f32,
        view: &ViewUniform,
        textures: &Textures,
    ) -> CloudStep {
        let mut result = CloudStep {
            sigma: 0.0,
            scattering: Vec3::ZERO,
            retry: false,
        };
        // Crossing a gap, a cheap probe that can only overstate the cloud
        // decides whether the step needs the detailed density at all. On
        // finding cloud after a long stride, step back and enter it finely,
        // or its crown would be sliced into terraces.
        if self.misses > 0.0 {
            let probe = overhead.sample(point, footprint, textures, false);
            if probe.density + probe.rain <= 0.0 {
                self.free = probe.free;
                self.misses = (self.misses + 1.0).min(11.0);
                self.depth = 0.0;
                return result;
            }
            if self.misses > 1.0 {
                self.misses = 0.0;
                self.phased = false;
                result.retry = true;
                return result;
            }
        }
        let cloud = overhead.sample(point, footprint, textures, true);
        let sigma = cloud.density + cloud.rain;
        if sigma <= 0.000_000_1 {
            self.misses = (self.misses + 1.0).min(11.0);
            self.depth = 0.0;
            self.free = cloud.free;
            return result;
        }
        self.misses = 0.0;
        self.depth = (self.depth + 1.0).min(8.0);
        let light = light_towards(point);
        // One tap towards the light, through the shapes of the clouds alone,
        // stands for the sample's own cloud out to the light volume.
        let nearby = overhead.sample(
            point + light * (clouds::NEAR_SHADOW * 0.5),
            footprint.max(clouds::NEAR_SHADOW * 0.15),
            textures,
            false,
        );
        let shadow = (nearby.density + nearby.rain) * clouds::NEAR_SHADOW;
        let (droplets, rain) = clouds::cloud_light(
            point,
            view,
            textures,
            &clouds::cloud_optics(point, self.direction, view, cloud.rain > 0.0),
            shadow,
            cloud.height,
        );
        let mut scattering = droplets * cloud.density + rain * cloud.rain;
        // The ocean's light reaches cloud bases as a drifting caustic net;
        // too fine to resolve far away, where it averages out.
        let altitude = point.length() - FEATURE_RADIUS;
        let resolve = (-footprint / 1500.0).exp() * (-altitude / 9000.0).exp();
        if resolve > 0.01 {
            let surface = feature_surface(point);
            let net = ocean::caustics(surface.truncate() * 0.0003, view.time() * 0.35);
            scattering += vec3(0.40, 0.34, 0.52)
                * view.sky_horizon.y
                * ((net - 0.25) * resolve * 0.6 * cloud.density);
        }
        result.sigma = sigma;
        result.scattering = scattering;
        result
    }
}

impl CloudSlab {
    fn join(self, result: &mut RaymarchResult) {
        if self.transmittance < 0.9999 {
            let opacity = 1.0 - self.transmittance;
            result.absorb(
                self.radiance / opacity,
                -self.transmittance.max(1.0e-6).ln(),
                self.front,
            );
        }
    }
}

/// March the clouds alone along a ray, from its entry into the cloud shell
/// to the opaque scene or the aur's body.
pub fn march_clouds(
    jitter: f32,
    origin: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    limit: f32,
    angle: f32,
    textures: &Textures,
    cells: &[Vec4],
) -> CloudSlab {
    // Marched in feature space; the front is returned in the world.
    let ro = view.to_planet(origin) * FEATURE_SCALE;
    let rd = view.planet_rotation.conjugate() * direction;
    let limit = limit * FEATURE_SCALE;
    let profile = view.camera_position.w as u32;
    let forced = clouds::forced_weather(view);
    let clear = forced && view.weather.x + view.weather.z + view.weather.w <= 0.0;
    let mut slab = CloudSlab {
        radiance: Vec3::ZERO,
        transmittance: 1.0,
        front: limit,
    };
    if profile & 2 != 0 || clear {
        return slab;
    }
    let shell = ray_shell_intersect(ro, rd, FEATURE_RADIUS, CLOUD_BOTTOM, CLOUD_TOP);
    // Nothing beneath the aur's surface can be seen; its thin upper veil
    // is no bound, or clouds beyond the horizon would be cut off there.
    let surface = intersect_sphere(ro, rd, FEATURE_RADIUS - 200.0);
    let limit = if surface.x <= surface.y && surface.x > 0.0 {
        limit.min(surface.x)
    } else {
        limit
    };
    let mut sky = CloudMarch {
        jitter,
        direction: rd,
        forced,
        misses: 0.0,
        depth: 0.0,
        free: 0.0,
        phased: false,
    };
    let mut result = RaymarchResult::empty(limit);
    let mut front = limit;
    // A handful of mist samples spans the ray's crossing of the shell.
    let crossing = (shell.y - shell.x).max(0.0) + (shell.w - shell.z).max(0.0);
    let mist_stride = (crossing.min(limit) / MIST_SAMPLES).max(MIST_STRIDE);
    let mut mist_path = 0.0;
    let mut t = if shell.x < shell.y {
        shell.x.max(0.0)
    } else {
        limit
    };
    for _ in 0..512 {
        if t >= limit || result.color.w < 0.01 {
            break;
        }
        let inside = (t + 0.25 >= shell.x && t < shell.y) || (t + 0.25 >= shell.z && t < shell.w);
        if !inside {
            // Between the near and far crossings of the shell, beneath it.
            if t < shell.z && shell.z < limit {
                t = shell.z;
                continue;
            }
            break;
        }
        let end = boundary(shell.xy(), t, boundary(shell.zw(), t, limit));
        let point = ro + rd * t;
        let footprint = (t * angle).max(2.0);
        let (stride, cloudy, overhead) = sky.advance(point, footprint, view, textures, cells);
        let step = stride.min(end - t).max(0.5);
        let distance = t + step * 0.5;
        let sample_point = ro + rd * distance;
        if cloudy {
            let cloud = sky.sample(&overhead, sample_point, footprint, view, textures);
            if cloud.retry {
                continue;
            }
            if cloud.sigma > 0.000_000_01 {
                front = front.min(t);
                result.absorb(cloud.scattering / cloud.sigma, cloud.sigma * step, distance);
            }
        }
        mist_path += step;
        if mist_path >= mist_stride {
            // A faint mist through the cloud shell carries their shadows as
            // shafts: of the pole light, and of the aur's glow rising. Shafts
            // are broad, so it is gathered over strides of several steps.
            let altitude = sample_point.length() - FEATURE_RADIUS;
            let mist = MIST * (-altitude / 7000.0).exp();
            let depth = textures.light_depth(sample_point);
            let world = sample_point / FEATURE_SCALE;
            let light = light_towards(sample_point);
            let up = sample_point / (altitude + FEATURE_RADIUS);
            let sun = textures.pole_light(world, light_position(world))
                * henyey_greenstein(rd.dot(light), 0.7)
                * (-depth.y).exp();
            let aur = vec3(0.5, 0.42, 0.7)
                * henyey_greenstein(-rd.dot(up), 0.5)
                * (-depth.z).exp()
                * (-altitude / 14_000.0).exp()
                * 0.6;
            result.absorb((sun + aur) * view.sky_horizon.y, mist * mist_path, distance);
            mist_path = 0.0;
        }
        t += step;
    }
    slab.radiance = result.color.xyz();
    slab.transmittance = result.color.w;
    slab.front = front / FEATURE_SCALE;
    slab
}

/// How far a ray may run through the cloud shell before it could meet any
/// cloud, from the parcel index; zero where it must sample. The index holds
/// clearances in the world and layer altitudes in feature space.
fn cloud_gap(point: Vec3, rd: Vec3, cells: &[Vec4]) -> f32 {
    let radius = point.length();
    let up = point / radius;
    let altitude = radius - FEATURE_RADIUS;
    let cell = cells[cloud_cell(up)];
    if cell.x > 0.0 {
        return cell.x * FEATURE_SCALE;
    }
    // Within reach of a system, but perhaps between its layers: run on to
    // the next layer in the ray's direction, within this cell's reach.
    let low = cell.y <= cell.z;
    if (low && altitude > cell.y - 200.0 && altitude < cell.z + 200.0) || altitude > cell.w - 200.0
    {
        return 0.0;
    }
    let target = if rd.dot(up) > 0.0 {
        if low && cell.y > altitude {
            cell.y
        } else {
            cell.w
        }
    } else if low && cell.z < altitude {
        cell.z
    } else {
        return CLOUD_CELL_REACH;
    };
    let crossing = intersect_sphere(point, rd, FEATURE_RADIUS + target);
    let distance = if radius < FEATURE_RADIUS + target {
        crossing.y
    } else if crossing.x <= crossing.y {
        crossing.x
    } else {
        CLOUD_CELL_REACH
    };
    distance.clamp(0.0, CLOUD_CELL_REACH)
}

/// A white-noise value in [0, 1) from any seed.
fn hash(seed: f32) -> f32 {
    let mut value = seed.to_bits();
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value >> 8) as f32 / 16_777_216.0
}

fn boundary(range: Vec2, t: f32, end: f32) -> f32 {
    if range.y <= range.x {
        return end;
    }
    if range.x > t + 0.25 {
        range.x.min(end)
    } else if range.y > t + 0.25 {
        range.y.min(end)
    } else {
        end
    }
}

pub fn integrate(
    origin: Vec3,
    direction: Vec3,
    view: &ViewUniform,
    limit: f32,
    angle: f32,
    atmosphere: &AtmosphereData,
    textures: &Textures,
    slab: CloudSlab,
) -> RaymarchResult {
    // The aur is marched in feature space; the poles and the air stay in
    // the world, and their distances are scaled to and from it.
    let world = view.to_planet(origin);
    let ro = world * FEATURE_SCALE;
    let rd = view.planet_rotation.conjugate() * direction;
    let world_limit = limit;
    let limit = limit * FEATURE_SCALE;
    let mut slab = slab;
    slab.front *= FEATURE_SCALE;
    let profile = view.camera_position.w as u32;
    let absent = Vec2::new(limit, 0.0);
    let ocean = if profile & 4 == 0 {
        intersect_sphere(ro, rd, FEATURE_RADIUS + ocean::OCEAN_TOP_HEIGHT)
    } else {
        absent
    };
    // The body is bounded far below the arcs. Entering its shell exactly
    // replaces a geometric crawl down towards it from the arc ceiling.
    let body = if profile & 4 == 0 {
        intersect_sphere(ro, rd, FEATURE_RADIUS + ocean::BODY_TOP_HEIGHT)
    } else {
        absent
    };
    let polar = poles::poles_raymarch_entry(world, rd, world_limit) * FEATURE_SCALE;
    let (beam_light, beam_depth, beam_at) = poles::beam(world, rd, world_limit, view);
    let beam_at = beam_at * FEATURE_SCALE;
    let mut beamed = beam_depth <= 0.0;
    // No arc can be met before this distance along the ray.
    let mut arc_free = 0.0;
    let mut result = RaymarchResult::empty(limit);
    let mut t = 0.0;
    let mut steps = 0u32;
    let mut slabbed = false;
    for _ in 0..512 {
        if t >= limit || result.color.w < 0.01 {
            break;
        }
        steps += 1;
        let point = ro + rd * t;
        let footprint = (t * angle).max(2.0);
        let inside_ocean = t + 0.25 >= ocean.x && t < ocean.y;
        let inside_body = t + 0.25 >= body.x && t < body.y;
        let inside_poles = t + 0.25 >= polar.x && t < polar.y;
        // The clouds join the march as one slab where the first of them
        // lies; they and the aur interleave too rarely to march together.
        if !slabbed && t + 0.25 >= slab.front {
            slab.join(&mut result);
            slabbed = true;
        }
        if !beamed && t + 0.25 >= beam_at {
            result.absorb(beam_light, beam_depth, beam_at);
            beamed = true;
        }
        let mut end = boundary(ocean, t, boundary(body, t, boundary(polar, t, limit)));
        if !slabbed {
            end = end.min(slab.front.max(t + 0.5));
        }
        if !beamed {
            end = end.min(beam_at.max(t + 0.5));
        }
        let mut step = end - t;
        let mut ocean_height = 1.0e10;
        let mut ocean_shape = Vec4::ZERO;
        let mut curve = Vec4::new(1.0e10, 0.0, 0.0, 0.0);
        let mut arcs = false;
        if inside_ocean && profile & 128 == 0 && t < arc_free {
            step = step.min(arc_free - t);
        } else if inside_ocean && profile & 128 == 0 {
            // Longitude stretches towards the poles, so surface distances
            // overstate world distances by up to this factor.
            let metric = point.truncate().length() / point.length().max(1.0);
            let clearance = textures.ejection_clearance(feature_surface(point)) * metric;
            // The veil heaves and glows around a loop's feet out to about
            // twice its reach; only skip lookups where that has faded too.
            if clearance > EJECTION_WIDTH * 4.0 {
                arc_free = t + clearance - EJECTION_WIDTH * 3.0;
                step = step.min(clearance);
            } else {
                // The arc field is an exact distance in surface coordinates.
                // Probe ahead by half of a step through an arc, so the same
                // query also serves the sample at the middle of that step.
                let reach = EJECTION_WIDTH * 3.0;
                let inner = (EJECTION_WIDTH * 0.12).max(footprint * 0.75);
                let probe = inner * 0.5;
                let ahead = point + rd * probe;
                curve = textures.ejection_field(feature_surface(ahead));
                // The field searches the neighbouring cells; loops beyond them
                // are at least a cell, less their extent, away.
                let free = (curve.x - reach).min(EJECTION_CELL - EJECTION_EXTENT) * metric - probe;
                if free > inner {
                    arc_free = t + free - EJECTION_WIDTH * 3.0;
                    step = step.min(free);
                } else {
                    arcs = curve.z > 0.0;
                    step = step.min(inner);
                }
            }
        }
        if inside_body {
            let altitude = point.length() - FEATURE_RADIUS;
            let up = point / (altitude + FEATURE_RADIUS);
            ocean_shape = ocean::sample(point, view, footprint, textures);
            let wave = if ocean_shape.yzw().length_squared() > 0.25 {
                ocean_shape.x * 4096.0
            } else {
                ocean::wave_base(point, view.time(), footprint, textures)
                    + ocean::wave_detail(point, view.time(), footprint, textures)
            };
            let mut height = altitude - wave;
            ocean_height = height;
            if height < 1500.0 {
                // The veil climbs any spire it meets; step against that.
                height -= ocean::wrap(point, view.time(), footprint).lift;
            }
            // Above the veil, a cone bound on the wave slope gives a safe
            // step. Inside it, steps grow with depth as the medium saturates
            // and with distance as the footprint widens.
            let body_step = if height > 450.0 {
                (height - 320.0) * 0.8 / ((-rd.dot(up)).max(0.0) + 0.35)
            } else {
                VEIL_STEP + (-height).max(0.0) * 0.49 + height.max(0.0) * 0.6
            };
            step = step.min(body_step.max(footprint * 0.75));
            // A cone step lies wholly above the veil by construction; its far
            // end is too distant to extrapolate the wave to, so skip it.
            if height > 450.0 && step >= body_step {
                ocean_height = 1.0e10;
            }
        }
        if inside_poles {
            // Only the reaction shell is marched; the beams are integrated.
            let shell_distance = (point.length() / FEATURE_SCALE - REACTION_RADIUS).abs();
            // Its corona layers are kilometres thick: step a fraction of them.
            let stride =
                ((shell_distance - 45_000.0) * 0.8).max(1800.0 + footprint / FEATURE_SCALE);
            step = step.min(stride * FEATURE_SCALE);
        }
        step = step.max(0.5).min(end - t);
        if step <= 0.0 {
            t += 0.5;
            continue;
        }
        let distance = t + step * 0.5;
        let sample_point = ro + rd * distance;
        let mut sigma = 0.0;
        let mut scattering = Vec3::ZERO;
        if arcs {
            let jet = ocean::ejections(sample_point, view, footprint, curve, textures);
            let density = jet.density * 0.018;
            sigma += density;
            scattering += jet.emission * density;
        }
        if ocean_height < 1.0e9 {
            let altitude = sample_point.length() - FEATURE_RADIUS;
            // The cached height gradient extrapolates the wave across this
            // short segment instead of fetching it again at the midpoint.
            let up = point.normalize();
            let normal = if ocean_shape.yzw().length_squared() > 0.25 {
                ocean_shape.yzw().normalize()
            } else {
                up
            };
            let gradient = (up - normal / normal.dot(up).max(0.01)) * 0.1;
            let mut wave =
                point.length() - FEATURE_RADIUS - ocean_height + gradient.dot(rd) * step * 0.5;
            let span = (altitude - wave - ocean_height).abs();
            let wrap = if altitude - wave - span < 1300.0 {
                ocean::wrap(sample_point, view.time(), footprint)
            } else {
                ocean::Wrap::NONE
            };
            // The veil also heaves up around the feet of a loop.
            wave += wrap.lift + 450.0 * (-square(curve.x / (EJECTION_WIDTH * 2.2))).exp() * curve.z;
            let height = altitude - wave;
            if altitude <= ocean::BODY_TOP_HEIGHT && height - span < 450.0 {
                // Evaluated afresh: kept across steps, it would hold registers
                // through the whole march and cost more than it saves.
                let column = ocean::column(sample_point, view, textures);
                let body = ocean::sample_body(
                    sample_point,
                    view,
                    footprint.max(step * 0.5),
                    if profile & 128 == 0 {
                        curve
                    } else {
                        Vec4::new(1.0e10, 0.0, 0.0, 0.0)
                    },
                    wave,
                    span,
                    normal,
                    column,
                    wrap,
                    textures,
                );
                let density = body.density * 0.018;
                sigma += density;
                scattering += body.emission * density;
            }
        }
        if inside_poles {
            let beam = poles::sample_poles(
                sample_point / FEATURE_SCALE,
                view,
                footprint / FEATURE_SCALE,
                textures,
            );
            let density = beam.density * 0.018 / FEATURE_SCALE;
            sigma += density;
            scattering += (beam.emission * view.sky_horizon.y
                + beam.color * view.ambient_color.xyz() * 0.2)
                * density;
        }
        if sigma > 0.00000001 {
            result.absorb(scattering / sigma, sigma * step, distance);
        }
        t += step;
    }
    if !slabbed {
        slab.join(&mut result);
    }
    if !beamed {
        result.absorb(beam_light, beam_depth, beam_at);
    }
    if profile & 512 != 0 {
        // Bands: <16 blue, <32 cyan, <64 green, <128 yellow, <256 red, else white.
        let band = if steps < 16 {
            Vec4::new(0.0, 0.0, 1.0, 0.0)
        } else if steps < 32 {
            Vec4::new(0.0, 1.0, 1.0, 0.0)
        } else if steps < 64 {
            Vec4::new(0.0, 1.0, 0.0, 0.0)
        } else if steps < 128 {
            Vec4::new(1.0, 1.0, 0.0, 0.0)
        } else if steps < 256 {
            Vec4::new(1.0, 0.0, 0.0, 0.0)
        } else {
            Vec4::new(1.0, 1.0, 1.0, 0.0)
        };
        result.color = band * 0.5;
        return result;
    }
    result.depth /= FEATURE_SCALE;
    let transmission = textures.air_transmittance(world, rd, world_limit);
    let front = textures.air_transmittance(world, rd, result.depth.min(world_limit));
    let air = (1.0 - front) + result.color.w * (front - transmission).max(0.0);
    result.color =
        (result.color.xyz() * front + atmosphere.haze * air).extend(result.color.w * transmission);
    result
}
