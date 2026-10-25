use crate::scene::{
    PLANET_RADIUS, ViewUniform,
    aur::{EJECTION_CELL, EJECTION_EXTENT, EJECTION_WIDTH, SPIRE_METRIC},
    climate::{CLOUD_BOTTOM, CLOUD_TOP, cloud_cell},
    math::{finish_opaque, henyey_greenstein, intersect_sphere, ray_shell_intersect, square},
    planet::{REACTION_RADIUS, light_position, light_towards, surface_coordinates},
    render::{
        clouds, ocean, poles,
        textures::{AtmosphereData, Textures},
    },
};
use spirv_std::{
    glam::{FloatExt, Vec2, Vec3, Vec4, Vec4Swizzles, vec3},
    num_traits::Float,
};

const VEIL_STEP: f32 = 14.0;

/// Extinction of the mist in which light shafts show, per metre at the aur,
/// and how much path it is gathered over at once.
const MIST: f32 = 0.000_032;

const MIST_STRIDE: f32 = 500.0;

const MIST_SAMPLES: f32 = 6.0;

/// The narrowest a parcel index cell can be, so its layer bounds hold for
/// at least this far across it.
const CLOUD_CELL_REACH: f32 = 30_000.0;

/// The longest horizontal stride between cloud layers: well under a cloud
/// map texel, so the layers cannot change much across it.
const LAYER_STRIDE: f32 = 3000.0;

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
/// their transmittance, and where along the ray they lie, as the mean
/// distance of each quarter of opacity they gather. The rest of the volume
/// interleaves with them at those four depths, so fog in front of distant
/// cloud covers it, and a wisp in front does not pull the whole slab
/// forward.
#[derive(Clone, Copy)]
pub struct CloudSlab {
    pub radiance: Vec3,
    pub transmittance: f32,
    pub depths: Vec4,
}

/// Transmittance at the top of each quarter of the slab's opacity.
const QUARTERS: Vec4 = Vec4::new(1.0, 0.75, 0.5, 0.25);

/// The cloud march's result, with where along the ray it gathers each
/// quarter of its opacity.
struct Gathered {
    result: RaymarchResult,
    sums: Vec4,
    weights: Vec4,
    /// Where the mist's share of the opacity is placed: as far as the ray
    /// reaches, as before the slab was split, so the mist never draws over
    /// the veil it lies within.
    far: f32,
}

impl Gathered {
    fn absorb(&mut self, radiance: Vec3, optical_depth: f32, distance: f32) {
        self.record(radiance, optical_depth, distance, distance);
    }

    fn absorb_mist(&mut self, radiance: Vec3, optical_depth: f32, distance: f32) {
        self.record(radiance, optical_depth, distance, self.far);
    }

    fn record(&mut self, radiance: Vec3, optical_depth: f32, distance: f32, placed: f32) {
        let before = self.result.color.w;
        self.result.absorb(radiance, optical_depth, distance);
        let after = self.result.color.w;
        // The opacity this step adds, split among the quarters it spans.
        let overlap = (Vec4::splat(before).min(QUARTERS) - Vec4::splat(after).max(QUARTERS - 0.25))
            .max(Vec4::ZERO);
        self.sums += overlap * placed;
        self.weights += overlap;
    }
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
        t: f32,
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
        let (room, rise) = overhead.room(point, rd);
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
        // Steps enter cloud short and lengthen only through its solid body,
        // so its surface is found finely without crawling through it.
        // Far away, clouds are small on screen and strides lengthen with
        // distance: the upscaler's history averages their jittered phases.
        let stride = if self.depth > 0.0 {
            (footprint * 1.2).max(10.0) * (1.0 + self.depth * 0.6)
        } else {
            (footprint * 2.5).max(37.5) * (1.0 + self.misses)
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
        step: f32,
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
            let probe = overhead.sample(point, footprint, textures, clouds::Detail::Probe);
            if probe.density + probe.rain <= 0.0 {
                self.free = probe.free;
                self.misses = (self.misses + 1.0).min(20.0);
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
        let cloud = overhead.sample(point, footprint, textures, clouds::Detail::Full);
        let sigma = cloud.density + cloud.rain;
        if sigma <= 0.000_000_1 {
            self.misses = (self.misses + 1.0).min(20.0);
            self.depth = 0.0;
            self.free = cloud.free;
            return result;
        }
        self.misses = 0.0;
        self.depth = (self.depth + 1.0).min(8.0);
        let light = light_towards(point);
        // Taps towards the light, through the shapes of the clouds alone,
        // stand for the sample's own cloud out to the light volume: closely
        // spaced near the sample, where its own lobes shade it, and further
        // apart beyond.
        let mut shadow = 0.0;
        for tap in 0..3 {
            let (middle, length) = match tap {
                0 => (0.075, 0.15),
                1 => (0.3, 0.3),
                _ => (0.725, 0.55),
            };
            let reach = clouds::NEAR_SHADOW * length;
            let nearby = overhead.sample(
                point + light * (clouds::NEAR_SHADOW * middle),
                footprint.max(reach * 0.3),
                textures,
                clouds::Detail::Shadow,
            );
            shadow += (nearby.density + nearby.rain) * reach;
        }
        // The clouds of the layers above cast their own shadows, patchy as
        // they are, where the coarse light volume holds only their mean.
        shadow += clouds::shadow_above(point, light, view, textures);
        // And two taps straight up, for how much of the sky the cloud heaped
        // over the sample hides from it.
        let up = point.normalize();
        let mut hidden = 0.0;
        for tap in 0..2 {
            let (middle, reach) = if tap == 0 {
                (35.0, 70.0)
            } else {
                (180.0, 220.0)
            };
            let above = overhead.sample(
                point + up * middle,
                footprint.max(reach * 0.3),
                textures,
                clouds::Detail::Shadow,
            );
            hidden += above.density * reach;
        }
        let open = (-hidden * 0.6).exp();
        let (droplets, rain) = clouds::cloud_light(
            point,
            view,
            textures,
            &clouds::cloud_optics(point, self.direction, view, cloud.rain > 0.0),
            shadow,
            cloud.height,
            open,
        );
        let mut scattering = droplets * cloud.density + rain * cloud.rain;
        // The ocean's light reaches cloud bases as a drifting caustic net;
        // too fine to resolve far away, where it averages out.
        let altitude = point.length() - PLANET_RADIUS;
        let resolve = (-footprint / 187.5).exp() * (-altitude / 1125.0).exp();
        if resolve > 0.01 {
            let surface = surface_coordinates(point);
            let net = ocean::caustics(surface.truncate() * 0.0024, view.time() * 0.35);
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
    fn depth(&self, quarter: u32) -> f32 {
        match quarter {
            0 => self.depths.x,
            1 => self.depths.y,
            2 => self.depths.z,
            _ => self.depths.w,
        }
    }

    /// Join one quarter of the slab's opacity to the march at its depth,
    /// with the slab's mean light.
    fn join(&self, quarter: u32, result: &mut RaymarchResult) {
        let opacity = 1.0 - self.transmittance;
        let top = 1.0 - quarter as f32 * 0.25;
        let bottom = (top - 0.25).max(self.transmittance);
        if opacity < 0.0001 || bottom >= top {
            return;
        }
        result.absorb(
            self.radiance / opacity,
            (top / bottom.max(1.0e-6)).ln(),
            self.depth(quarter),
        );
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
    let ro = view.to_planet(origin);
    let rd = view.planet_rotation.conjugate() * direction;
    let profile = view.camera_position.w as u32;
    let forced = clouds::forced_weather(view);
    let clear = forced && view.weather.x + view.weather.z + view.weather.w <= 0.0;
    let mut slab = CloudSlab {
        radiance: Vec3::ZERO,
        transmittance: 1.0,
        depths: Vec4::splat(limit),
    };
    if profile & 2 != 0 || clear {
        return slab;
    }
    let shell = ray_shell_intersect(ro, rd, PLANET_RADIUS, CLOUD_BOTTOM, CLOUD_TOP);
    // Nothing beneath the aur's surface can be seen; its thin upper veil
    // is no bound, or clouds beyond the horizon would be cut off there.
    let surface = intersect_sphere(ro, rd, PLANET_RADIUS - 25.0);
    let limit = if surface.x <= surface.y && surface.x > 0.0 {
        limit.min(surface.x)
    } else {
        limit
    };
    // A ray that dips into the veil's band does not come out of it: past
    // its lowest point it would only climb back through the clouds beyond
    // the horizon, which would show through the veil as a mirror image of
    // the sky above it. The deeper it dips the less of that cloud remains,
    // fading rather than cut so no seam runs along the veil.
    let lowest = -ro.dot(rd);
    let bottom = (ro + rd * lowest.max(0.0)).length() - PLANET_RADIUS;
    let beyond = bottom.smoothstep(ocean::BODY_TOP_HEIGHT * 0.4, ocean::BODY_TOP_HEIGHT);
    let mut sky = CloudMarch {
        jitter,
        direction: rd,
        forced,
        misses: 0.0,
        depth: 0.0,
        free: 0.0,
        phased: false,
    };
    let mut result = Gathered {
        result: RaymarchResult::empty(limit),
        sums: Vec4::ZERO,
        weights: Vec4::ZERO,
        far: limit,
    };
    // A handful of mist samples spans the ray's crossing of the shell, each
    // pixel gathering at its own phase so their seams never line up.
    let crossing = (shell.y - shell.x).max(0.0) + (shell.w - shell.z).max(0.0);
    let mist_stride =
        (crossing.min(limit) / MIST_SAMPLES).max(MIST_STRIDE) * (0.6 + 0.8 * hash(jitter + 0.31));
    let mut mist = Mist {
        start: 0.0,
        path: 0.0,
        lowest,
        beyond,
    };
    let mut t = if shell.x < shell.y {
        shell.x.max(0.0)
    } else {
        limit
    };
    for _ in 0..512 {
        if t >= limit || result.result.color.w < 0.01 {
            break;
        }
        let inside = (t + 0.03 >= shell.x && t < shell.y) || (t + 0.03 >= shell.z && t < shell.w);
        if !inside {
            // Between the near and far crossings of the shell, beneath it.
            if t < shell.z && shell.z < limit {
                mist.gather(&mut result, ro, rd, view, textures);
                t = shell.z;
                continue;
            }
            break;
        }
        let end = boundary(shell.xy(), t, boundary(shell.zw(), t, limit));
        let point = ro + rd * t;
        let footprint = (t * angle).max(0.25);
        let (stride, cloudy, overhead) = sky.advance(point, t, footprint, view, textures, cells);
        let step = stride.min(end - t).max(0.0625);
        let distance = t + step * 0.5;
        let sample_point = ro + rd * distance;
        if cloudy {
            let cloud = sky.sample(&overhead, sample_point, step, footprint, view, textures);
            if cloud.retry {
                continue;
            }
            let seen = if lowest > 0.0 && distance > lowest {
                beyond
            } else {
                1.0
            };
            if cloud.sigma * seen > 0.000_000_01 {
                result.absorb(
                    cloud.scattering / cloud.sigma,
                    cloud.sigma * seen * step,
                    distance,
                );
            }
        }
        if mist.path <= 0.0 {
            mist.start = t;
        }
        mist.path += step;
        if mist.path >= mist_stride {
            mist.gather(&mut result, ro, rd, view, textures);
        }
        t += step;
    }
    // Whatever path is left since the last gather counts too, or the mist
    // would step wherever rays leave the shell after a different number of
    // strides.
    mist.gather(&mut result, ro, rd, view, textures);
    slab.radiance = result.result.color.xyz();
    slab.transmittance = result.result.color.w;
    // Quarters never reached lie beyond everything.
    let weights = result.weights;
    slab.depths = Vec4::select(
        weights.cmpgt(Vec4::splat(1.0e-6)),
        result.sums / weights.max(Vec4::splat(1.0e-6)),
        Vec4::splat(limit),
    );
    slab
}

/// Path through the cloud shell's mist not yet gathered into the march.
struct Mist {
    start: f32,
    path: f32,
    /// Where the ray dips lowest, and how much is seen beyond it.
    lowest: f32,
    beyond: f32,
}

impl Mist {
    /// A faint mist through the cloud shell carries their shadows as shafts:
    /// of the pole light, and of the aur's glow rising. Shafts are broad, so
    /// the light is found once, mid-way along each stretch of path.
    fn gather(
        &mut self,
        result: &mut Gathered,
        ro: Vec3,
        rd: Vec3,
        view: &ViewUniform,
        textures: &Textures,
    ) {
        if self.path <= 0.0 {
            return;
        }
        let distance = self.start + self.path * 0.5;
        let point = ro + rd * distance;
        let altitude = point.length() - PLANET_RADIUS;
        let seen = if self.lowest > 0.0 && distance > self.lowest {
            self.beyond
        } else {
            1.0
        };
        let density = MIST * (-altitude / 875.0).exp() * seen;
        let depth = textures.light_depth(point);
        let light = light_towards(point);
        let up = point / (altitude + PLANET_RADIUS);
        // Light scattered on through the cloud softens the edges of shadows
        // that the coarse light volume would otherwise draw hard.
        let lit = |depth: f32| (-depth).exp() * 0.8 + (-depth * 0.2).exp() * 0.2;
        let sun = textures.pole_light(point, light_position(point))
            * henyey_greenstein(rd.dot(light), 0.7)
            * lit(depth.y);
        let aur = vec3(0.5, 0.42, 0.7)
            * henyey_greenstein(-rd.dot(up), 0.5)
            * lit(depth.z)
            * (-altitude / 1750.0).exp()
            * 0.6;
        result.absorb_mist(
            (sun + aur) * view.sky_horizon.y,
            density * self.path,
            distance,
        );
        self.path = 0.0;
    }
}

/// How far a ray may run through the cloud shell before it could meet any
/// cloud, from the parcel index; zero where it must sample.
fn cloud_gap(point: Vec3, rd: Vec3, cells: &[Vec4]) -> f32 {
    let radius = point.length();
    let up = point / radius;
    let altitude = radius - PLANET_RADIUS;
    let cell = cells[cloud_cell(up)];
    if cell.x > 0.0 {
        return cell.x;
    }
    // Within reach of a system, but perhaps between its layers: run on to
    // the next layer in the ray's direction, within this cell's reach.
    let low = cell.y <= cell.z;
    if (low && altitude > cell.y - 25.0 && altitude < cell.z + 25.0) || altitude > cell.w - 25.0 {
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
    let crossing = intersect_sphere(point, rd, PLANET_RADIUS + target);
    let distance = if radius < PLANET_RADIUS + target {
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
    if range.x > t + 0.03 {
        range.x.min(end)
    } else if range.y > t + 0.03 {
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
    let ro = view.to_planet(origin);
    let rd = view.planet_rotation.conjugate() * direction;
    // Where the ray starts from the camera, for what is placed relative to it.
    let start = view.planet_rotation.conjugate() * (origin - view.world_position.xyz());
    let profile = view.camera_position.w as u32;
    let absent = Vec2::new(limit, 0.0);
    let ocean = if profile & 4 == 0 {
        intersect_sphere(ro, rd, PLANET_RADIUS + ocean::OCEAN_TOP_HEIGHT)
    } else {
        absent
    };
    // The body is bounded far below the arcs. Entering its shell exactly
    // replaces a geometric crawl down towards it from the arc ceiling.
    let body = if profile & 4 == 0 {
        intersect_sphere(ro, rd, PLANET_RADIUS + ocean::BODY_TOP_HEIGHT)
    } else {
        absent
    };
    let polar = poles::poles_raymarch_entry(ro, rd, limit);
    let (beam_light, beam_depth, beam_at) = poles::beam(ro, rd, limit, view);
    let mut beamed = beam_depth <= 0.0;
    // No arc can be met before this distance along the ray.
    let mut arc_free = 0.0;
    let mut result = RaymarchResult::empty(limit);
    let mut t = 0.0;
    let mut steps = 0u32;
    // How many quarters of the cloud slab have joined the march.
    let mut joined = 0u32;
    for _ in 0..512 {
        if t >= limit || result.color.w < 0.01 {
            break;
        }
        steps += 1;
        let point = ro + rd * t;
        let footprint = (t * angle).max(0.25);
        let inside_ocean = t + 0.03 >= ocean.x && t < ocean.y;
        let inside_body = t + 0.03 >= body.x && t < body.y;
        let inside_poles = t + 0.03 >= polar.x && t < polar.y;
        // The clouds join the march in quarters where each lies; they and
        // the aur interleave too rarely to march together.
        while joined < 4 && t + 0.03 >= slab.depth(joined) {
            slab.join(joined, &mut result);
            joined += 1;
        }
        if !beamed && t + 0.03 >= beam_at {
            result.absorb(beam_light, beam_depth, beam_at);
            beamed = true;
        }
        let mut end = boundary(ocean, t, boundary(body, t, boundary(polar, t, limit)));
        if joined < 4 {
            end = end.min(slab.depth(joined).max(t + 0.0625));
        }
        if !beamed {
            end = end.min(beam_at.max(t + 0.0625));
        }
        let mut step = end - t;
        let mut ocean_height = 1.0e10;
        let mut ocean_shape = Vec4::ZERO;
        let mut curve = Vec4::new(1.0e10, 0.0, 0.0, 0.0);
        let mut arc = 0.0;
        let mut arcs = false;
        if inside_ocean && profile & 128 == 0 && t < arc_free {
            step = step.min(arc_free - t);
        } else if inside_ocean && profile & 128 == 0 {
            // The face grid stretches towards the faces' edges, so distances
            // on it overstate those in the world by up to this factor.
            let metric = SPIRE_METRIC;
            let clearance = textures.ejection_clearance(view.place(start + rd * t)) * metric;
            // The veil heaves and glows around a loop's feet out to about
            // twice its reach; only skip lookups where that has faded too.
            if clearance > EJECTION_WIDTH * 4.0 {
                arc_free = t + clearance - EJECTION_WIDTH * 3.0;
                step = step.min(clearance);
            } else {
                // The arc field bounds the distance on the face grid.
                // Probe ahead by half of a step through an arc, so the same
                // query also serves the sample at the middle of that step.
                let reach = EJECTION_WIDTH * 3.0;
                let inner = (EJECTION_WIDTH * 0.12).max(footprint * 0.75);
                let probe = inner * 0.5;
                (curve, arc) = textures.ejection_field(view.place(start + rd * (t + probe)));
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
            let altitude = point.length() - PLANET_RADIUS;
            let up = point / (altitude + PLANET_RADIUS);
            ocean_shape = ocean::sample(point, view, footprint, textures);
            let wave = if ocean_shape.yzw().length_squared() > 0.25 {
                ocean_shape.x * ocean::WAVE_RANGE
            } else {
                ocean::wave_base(point, view.time(), footprint, textures)
                    + ocean::wave_detail(point, view.time(), footprint, textures)
            };
            let mut height = altitude - wave;
            ocean_height = height;
            if height < 187.5 {
                // The veil climbs any spire it meets; step against that.
                height -= ocean::wrap(point, start + rd * t, view, footprint).lift;
            }
            // Above the veil, a cone bound on the wave slope gives a safe
            // step. Inside it, steps grow with depth as the medium saturates
            // and with distance as the footprint widens.
            let body_step = if height > 56.25 {
                (height - 40.0) * 0.8 / ((-rd.dot(up)).max(0.0) + 0.35)
            } else {
                VEIL_STEP + (-height).max(0.0) * 0.49 + height.max(0.0) * 0.6
            };
            step = step.min(body_step.max(footprint * 0.75));
            // A cone step lies wholly above the veil by construction; its far
            // end is too distant to extrapolate the wave to, so skip it.
            if height > 56.25 && step >= body_step {
                ocean_height = 1.0e10;
            }
        }
        if inside_poles {
            // Only the reaction shell is marched; the beams are integrated.
            let shell_distance = (point.length() - REACTION_RADIUS).abs();
            // Its corona layers are kilometres thick: step a fraction of them.
            step = step.min(((shell_distance - 45_000.0) * 0.8).max(1800.0 + footprint));
        }
        step = step.max(0.0625).min(end - t);
        if step <= 0.0 {
            t += 0.0625;
            continue;
        }
        let distance = t + step * 0.5;
        let sample_point = ro + rd * distance;
        let mut sigma = 0.0;
        let mut scattering = Vec3::ZERO;
        if arcs {
            let jet = ocean::ejections(sample_point, view, footprint, curve, arc, textures);
            let density = jet.density * 0.144;
            sigma += density;
            scattering += jet.emission * density;
        }
        if ocean_height < 1.0e9 {
            let altitude = sample_point.length() - PLANET_RADIUS;
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
                point.length() - PLANET_RADIUS - ocean_height + gradient.dot(rd) * step * 0.5;
            let span = (altitude - wave - ocean_height).abs();
            let wrap = if altitude - wave - span < 162.5 {
                ocean::wrap(sample_point, start + rd * distance, view, footprint)
            } else {
                ocean::Wrap::NONE
            };
            // The veil also heaves up around the feet of a loop.
            wave += wrap.lift + 56.25 * (-square(curve.x / (EJECTION_WIDTH * 2.2))).exp() * curve.z;
            let height = altitude - wave;
            if altitude <= ocean::BODY_TOP_HEIGHT && height - span < 56.25 {
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
                let density = body.density * 0.144;
                sigma += density;
                scattering += body.emission * density;
            }
        }
        if inside_poles {
            let beam = poles::sample_poles(sample_point, view, footprint, textures);
            let density = beam.density * 0.018;
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
    while joined < 4 {
        slab.join(joined, &mut result);
        joined += 1;
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
    let transmission = textures.air_transmittance(ro, rd, limit);
    let front = textures.air_transmittance(ro, rd, result.depth.min(limit));
    let air = (1.0 - front) + result.color.w * (front - transmission).max(0.0);
    result.color =
        (result.color.xyz() * front + atmosphere.haze * air).extend(result.color.w * transmission);
    result
}
