use crate::scene::life::{CreatureInstance, Kind, Look, development};
use core::{
    f32::consts::{PI, TAU},
    mem::swap,
};
use spirv_std::glam::{FloatExt, Quat, Vec2, Vec3, Vec3Swizzles, Vec4, vec2, vec3};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub light: Vec3,
    pub ambient: Vec3,
    pub radiance: Vec3,
    pub max_distance: f32,
    pub footprint: f32,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub color_depth: Vec4,
    pub normal: Vec3,
    pub previous_position: Vec3,
    pub transmittance: Vec3,
    pub part: u32,
}

impl Hit {
    pub const MISS: Self = Self {
        color_depth: Vec4::new(0.0, 0.0, 0.0, 1e6),
        normal: Vec3::ZERO,
        previous_position: Vec3::ZERO,
        transmittance: Vec3::ONE,
        part: 0,
    };
    pub fn is_hit(self) -> bool {
        self.color_depth.w < Self::MISS.color_depth.w
    }
    pub fn distance(self) -> f32 {
        self.color_depth.w
    }
    pub fn color(self) -> Vec3 {
        self.color_depth.truncate()
    }
}

#[derive(Clone, Copy)]
pub struct TraceResult {
    pub opaque: Hit,
    pub layers: [Hit; 16],
    pub count: u32,
}

impl TraceResult {
    pub const MISS: Self = Self {
        opaque: Hit::MISS,
        layers: [Hit::MISS; 16],
        count: 0,
    };
    pub const fn opaque(hit: Hit) -> Self {
        Self {
            opaque: hit,
            ..Self::MISS
        }
    }
    pub fn front(self) -> Hit {
        if self.count > 0 && self.layers[0].distance() < self.opaque.distance() {
            self.layers[0]
        } else {
            self.opaque
        }
    }
    pub fn is_hit(self) -> bool {
        self.front().is_hit()
    }
    pub fn distance(self) -> f32 {
        self.front().distance()
    }
    pub fn composite(self, background: Vec3) -> Vec3 {
        let mut color = if self.opaque.is_hit() {
            self.opaque.color()
        } else {
            background
        };
        for i in (0..self.layers.len()).rev() {
            if i < self.count as usize && self.layers[i].distance() < self.opaque.distance() {
                color = self.layers[i].color() + color * self.layers[i].transmittance;
            }
        }
        color
    }
    pub fn add(&mut self, hit: Hit) {
        if !hit.is_hit() {
            return;
        }
        let mut incoming = hit;
        for i in 0..self.layers.len() {
            if incoming.distance() < self.layers[i].distance() {
                swap(&mut self.layers[i], &mut incoming);
            }
        }
        self.count = (self.count + 1).min(self.layers.len() as u32);
    }
}

pub struct Surface {
    pub distance: f32,
    pub material: u32,
}

impl Surface {
    pub const fn new(distance: f32, material: u32) -> Self {
        Self { distance, material }
    }
    pub fn join(&mut self, distance: f32, material: u32) {
        if distance < self.distance {
            *self = Self::new(distance, material);
        }
    }
}

#[derive(Clone, Copy)]
pub struct Material {
    pub color: Vec3,
    pub roughness: f32,
    pub metal: f32,
    pub emission: f32,
    pub transmission: Vec3,
    pub ior: f32,
    pub translucency: f32,
    /// How deeply the surface is worked into relief, which kind of relief (see [`relief`]) and how
    /// fine it is.
    pub relief: f32,
    pub texture: u32,
    pub grain: f32,
}

/// No relief: a smooth surface.
pub const SMOOTH: u32 = 0;

/// Soft skin, folded into wrinkles along the body and pitted with pores.
pub const WRINKLED: u32 = 1;

/// Plates and scutes split by grooves, each faintly domed and pitted.
pub const PLATED: u32 = 2;

/// A short pile of fine fibres lying back along the body.
pub const FURRED: u32 = 3;

/// Horn and bone, banded by growth rings.
pub const RINGED: u32 = 4;

/// Glistening mucus over soft, slightly lumpy tissue.
pub const SLICK: u32 = 5;

/// The height of a surface's relief about a point, of each kind.
pub fn relief(q: Vec3, texture: u32) -> f32 {
    match texture {
        WRINKLED => {
            let folds = (q.z * 5.0 + fbm(q * 0.8) * 4.0).sin().abs();
            (1.0 - folds) * 0.45 + noise(q * 22.0) * 0.25 + fbm(q * 2.5) * 0.3
        }
        PLATED => {
            let warped = q + vec3(fbm(q * 0.5), fbm(q * 0.5 + 7.0), fbm(q * 0.5 + 13.0)) * 0.9;
            let cell = (vec2(warped.x, warped.z) * 1.3).fract_gl() - Vec2::splat(0.5);
            let groove = cell.abs().max_element().smoothstep(0.38, 0.5);
            (0.5 - cell.length()) * 0.6 - groove + noise(q * 14.0) * 0.25
        }
        FURRED => noise(vec3(q.x * 34.0, q.y * 34.0, q.z * 5.0)) * 0.8 + noise(q * 70.0) * 0.2,
        RINGED => (q.y * 22.0 + q.x * 6.0 + fbm(q * 2.0) * 3.0).sin() * 0.5 + noise(q * 18.0) * 0.3,
        SLICK => fbm(q * 1.6) * 0.7 + noise(q * 9.0) * 0.15,
        _ => 0.0,
    }
}

impl Material {
    pub fn transmittance(self, cosine: f32) -> Vec3 {
        let dielectric = ((self.ior - 1.0) / (self.ior + 1.0)).powi(2);
        let fresnel = dielectric + (1.0 - dielectric) * (1.0 - cosine.clamp(0.0, 1.0)).powi(5);
        self.transmission * (1.0 - fresnel)
    }
    pub const fn skin(color: Vec3, roughness: f32) -> Self {
        Self {
            color,
            roughness,
            metal: 0.0,
            emission: 0.0,
            transmission: Vec3::ZERO,
            ior: 1.5,
            translucency: 0.0,
            relief: 0.0,
            texture: SMOOTH,
            grain: 1.0,
        }
    }
    pub fn shade(self, normal: Vec3, ray: Ray) -> Vec3 {
        let view = -ray.direction;
        let light = normalize(ray.light);
        let halfway = normalize(light + view);
        let nl = normal.dot(light).max(0.0);
        let backlight = (-normal.dot(light)).max(0.0);
        // Light wraps a little past the terminator, as it does through skin and soft tissue, scaled
        // so the lobe still carries the same energy.
        let wrap = 0.4 * (1.0 - self.metal);
        let wrapped = ((normal.dot(light) + wrap) / (1.0 + wrap)).max(0.0) / (1.0 + wrap);
        let nv = normal.dot(view).abs().max(0.0001);
        let nh = normal.dot(halfway).max(0.0);
        let vh = view.dot(halfway).max(0.0);
        let a2 = self.roughness.clamp(0.045, 1.0).powi(4);
        let distribution = a2 / (PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2)).max(1e-12);
        let visibility = 0.5
            / (nl * (nv * nv * (1.0 - a2) + a2).sqrt() + nv * (nl * nl * (1.0 - a2) + a2).sqrt())
                .max(1e-6);
        let dielectric = ((self.ior - 1.0) / (self.ior + 1.0)).powi(2);
        let f0 = Vec3::splat(dielectric).lerp(self.color, self.metal);
        let fresnel = f0 + (Vec3::ONE - f0) * (1.0 - vh).powi(5);
        let specular = fresnel * distribution * visibility * nl;
        // Sky above, darker air below.
        let fill = ray.ambient * (0.72 + normal.y * 0.28);
        let diffuse = self.color
            * (Vec3::ONE - fresnel)
            * (Vec3::ONE - self.transmission)
            * (1.0 - self.metal)
            * (fill + ray.radiance * (wrapped.lerp(backlight, self.translucency) / PI));
        let reflected = normal * (2.0 * normal.dot(view)) - view;
        let environment = ray.ambient * (0.72 + reflected.y * 0.28);
        let grazing = Vec3::splat(1.0 - self.roughness).max(f0);
        let environment_fresnel = f0 + (grazing - f0) * (1.0 - nv).powi(5);
        diffuse
            + specular * ray.radiance
            + environment * environment_fresnel
            + self.color * self.emission
    }
}

#[derive(Clone, Copy)]
/// A curved membrane, hinged at `root`: turned about the body's long axis by `stroke` (now and
/// last frame), and curling away from the stroke towards its tip by `bend`, so its root never
/// leaves the body.
pub struct Sheet {
    pub center: Vec3,
    pub previous_center: Vec3,
    pub root: Vec3,
    pub stroke: Vec2,
    pub radii: Vec2,
    pub bend: Vec2,
    /// How far the membrane arches between its edges, as a share of its chord: an
    /// aerofoil's camber.
    pub camber: f32,
    pub style: Vec4,
    /// How far out along the span the wrist joint lies, and how far the hand beyond it is turned
    /// about it, now and last frame.
    pub wrist: f32,
    pub fold: Vec2,
    /// The membrane's markings: their kind, how dense, how strong, and the second colour they are
    /// drawn in; see [`marking`].
    pub pattern: Vec4,
    pub second: Vec3,
    pub side: f32,
}

/// A membrane's markings at a point of it, out along the span and back across the chord, seen at a
/// cosine: bands across the span, an eyespot near the tip, clear windows between the veins (as a
/// negative mark, which thins the membrane), a dark leading edge, a sheen that shifts towards the
/// second colour at a glancing view, or a camouflage mottle.
fn marking(pattern: Vec4, along: f32, chord: f32, cosine: f32) -> f32 {
    match pattern.x as u32 {
        1 => (along * pattern.y * PI).sin().smoothstep(0.2, 0.5),
        2 => {
            let spot = vec2((along - 0.7) * 2.2, chord - 0.45).length();
            spot.smoothstep(0.24, 0.2) - spot.smoothstep(0.12, 0.08) * 0.6
        }
        3 => -((along * pattern.y * PI).sin() * (chord * pattern.y * PI).sin())
            .abs()
            .smoothstep(0.35, 0.7),
        4 => (1.0 - chord).smoothstep(0.55, 0.95),
        5 => (1.0 - cosine).powi(2),
        6 => ((along * pattern.y * 4.3 + (chord * 11.0).sin() * 1.7).sin()
            * (chord * pattern.y * 3.1 + (along * 7.0).sin() * 2.1).sin())
        .abs()
        .smoothstep(0.25, 0.65),
        _ => 0.0,
    }
}

/// A vector turned about the long axis.
fn roll(v: Vec3, angle: f32) -> Vec3 {
    let (sine, cosine) = angle.sin_cos();
    vec3(v.x * cosine - v.y * sine, v.x * sine + v.y * cosine, v.z)
}

/// A wing's style: how far its tip sweeps back, how rounded or pointed its tip is, how many fingers
/// or slotted feathers split its trailing edge, and how finely that edge frays into fringe.
pub const PLAIN_WING: Vec4 = Vec4::new(1.0, 0.5, 4.0, 0.0);

/// How far ahead of a wing's chord line its leading edge lies, in chords, a share of the way out
/// along it.
fn leading(along: f32, style: Vec4) -> f32 {
    let tip = (1.0 - along.clamp(0.0, 1.0).powf(2.0 + style.y * 10.0))
        .max(0.0)
        .sqrt();
    0.62 * tip
}

/// How far behind the chord line its trailing edge lies: tapering from the root to the tip, notched
/// between its fingers and frayed into fringe.
fn trailing(along: f32, style: Vec4, feathered: bool) -> f32 {
    let along = along.clamp(0.0, 1.0);
    let tip = (1.0 - along.powf(2.0 + style.y * 10.0)).max(0.0).sqrt();
    let slots = if style.z > 0.5 {
        let (from, to) = if feathered { (0.55, 0.8) } else { (0.1, 0.4) };
        1.0 - 0.18 * (along * PI * (style.z + 0.5)).sin().abs() * along.smoothstep(from, to)
    } else {
        1.0
    };
    let fringe = 1.0 - style.w * 0.22 * (along * 70.0).sin().abs() * along.smoothstep(0.05, 0.3);
    let root = 0.25 + 0.5 * style.y;
    (root + (1.15 - root) * (1.0 - along).sqrt()) * tip * slots * fringe
}

/// How far inside a wing's outline a point of it lies, in chords, from its coordinates:
/// positive within.
fn planform(uv: Vec2, side: f32, feathered: bool, style: Vec4) -> f32 {
    let along = uv.x.mul_add(side * 0.5, 0.5);
    (uv.y + leading(along, style))
        .min(trailing(along, style, feathered) - uv.y)
        .min((along + 0.02) * 2.0)
        .min((1.0 - along) * 2.0)
}

impl Sheet {
    fn coordinates(self, point: Vec3) -> Vec2 {
        let span = point.x / self.radii.x;
        let fraction = span.mul_add(self.side * 0.5, 0.5);
        Vec2::new(
            span,
            (point.z - (fraction * fraction - 0.25) * self.radii.x * 0.55 * self.style.x)
                / (self.radii.y * (1.0 - fraction * 0.55)),
        )
    }
}

pub struct Shell {
    pub center: Vec3,
    pub radii: Vec3,
    pub bottom: f32,
    pub scallop: f32,
}

/// A white-noise value in [0, 1) for a lattice point.
fn hash(p: Vec3) -> f32 {
    let mut q = (p * vec3(0.1031, 0.1030, 0.0973)).fract_gl();
    q += Vec3::splat(q.dot(q.yzx() + 33.33));
    ((q.x + q.y) * q.z).fract_gl()
}

/// Smooth value noise in [0, 1).
fn noise(p: Vec3) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let u = f * f * (Vec3::splat(3.0) - f * 2.0);
    let corner = |x: f32, y: f32, z: f32| hash(cell + vec3(x, y, z));
    let bottom = corner(0.0, 0.0, 0.0)
        .lerp(corner(1.0, 0.0, 0.0), u.x)
        .lerp(corner(0.0, 1.0, 0.0).lerp(corner(1.0, 1.0, 0.0), u.x), u.y);
    let top = corner(0.0, 0.0, 1.0)
        .lerp(corner(1.0, 0.0, 1.0), u.x)
        .lerp(corner(0.0, 1.0, 1.0).lerp(corner(1.0, 1.0, 1.0), u.x), u.y);
    bottom.lerp(top, u.z)
}

/// Three octaves of value noise, in about [0, 1].
fn fbm(p: Vec3) -> f32 {
    noise(p) * 0.55 + noise(p * 2.07 + 11.3) * 0.3 + noise(p * 4.13 + 27.1) * 0.15
}

/// Living pigment over a body.
///
/// Irregular blotches are drawn out along it, the countershading turned over for a world lit from
/// below, and it is mottled at two finer scales so no surface reads as flat paint.
pub fn coat(p: Vec3, id: u32, look: Look) -> Material {
    let pigment = look.pigment;
    let accent = look.accent;
    let warp = vec3(fbm(p * 0.6), fbm(p * 0.6 + 17.0), fbm(p * 0.6 + 31.0)) - 0.5;
    let blotches = fbm(vec3(p.x * 0.8, p.y * 0.8, p.z * 0.45) + warp * 2.2);
    let mottle = fbm(p * 3.7 + 5.0);
    let grain = noise(p * 19.0);
    let dorsal = (p.y * 0.30 + 0.5).saturate();
    let markings = blotches.smoothstep(0.5, 0.64) * dorsal.smoothstep(0.25, 0.75);
    let base = pigment.truncate().lerp(accent, markings * 0.7);
    // Countershading turned over, for a world lit from below: the back pale, lost against the
    // bright veil to anything looking down on it, and the belly dark, lost against the dim sky to
    // anything looking up.
    let back = (base * 1.25 + vec3(0.05, 0.05, 0.06)).min(Vec3::ONE);
    let belly = base * 0.55;
    // Pulsatile nutrient channels show faintly through the skin, branching back from the crop.
    let channel = fbm(vec3(p.x * 1.4, p.y * 1.4, p.z * 0.5) + warp * 1.5);
    let vein = 1.0 - (channel - 0.5).abs().smoothstep(0.0, 0.035);
    let mut m = Material::skin(
        belly
            .lerp(back, dorsal.smoothstep(0.2, 0.8))
            .lerp(accent * 0.6 + vec3(0.22, 0.04, 0.08), vein * 0.22)
            * (0.84 + mottle * 0.24)
            * (0.95 + grain * 0.08),
        0.55 + mottle * 0.18,
    );
    m.emission = pigment.w;
    // The body's own skin, as the species wears it: its kind of relief, how deep and how fine, and
    // how much duller or glossier than plain skin.
    let skin = look.skin;
    m.texture = if skin.x > 0.5 {
        skin.x as u32
    } else {
        WRINKLED
    };
    m.relief = 0.35 * skin.y;
    m.grain = skin.z;
    m.roughness = (m.roughness + skin.w).clamp(0.05, 1.0);
    tissue(m, p, id, look, vec3(blotches, mottle, grain))
}

/// A coat's skin turned to another of the body's tissues by its material id, from the coat's own
/// blotches, mottle and grain.
fn tissue(mut m: Material, p: Vec3, id: u32, look: Look, pattern: Vec3) -> Material {
    let pigment = look.pigment;
    let accent = look.accent;
    let (blotches, mottle, grain) = (pattern.x, pattern.y, pattern.z);
    match id {
        1 => {
            // Fins and feathers: the accent, darkening along vanes.
            m.color = accent * (0.8 + mottle * 0.3);
            m.roughness = 0.6;
            m.texture = RINGED;
            m.relief = 0.15;
        }
        2 => {
            m.color = accent.lerp(vec3(0.95, 0.29, 0.31), 0.32);
            m.transmission = Vec3::splat(0.35);
        }
        3 => {
            // Eyes: glassy lenses over a deep, faintly green-gold eye cup, dark but catching
            // every highlight.
            m.color = vec3(0.07, 0.085, 0.06) * (0.8 + grain * 0.4);
            m.roughness = 0.04;
            m.emission = 0.0;
            m.relief = 0.0;
        }
        4 => {
            m.color =
                accent.lerp(pigment.truncate(), blotches * 0.4 + 0.15) * (0.85 + mottle * 0.3);
            m.roughness = 0.32;
        }
        5 => {
            // Armour: mineral-hard plates, matte and dusty, split by grooves.
            m.color = accent.lerp(pigment.truncate(), blotches * 0.5) * (0.6 + mottle * 0.5);
            m.roughness = 0.68 + grain * 0.12;
            m.texture = PLATED;
            m.relief = 0.5;
            m.grain = look.skin.z * 0.8;
        }
        6 => {
            // Iron-hardened blades and crystal: bright only along worn edges, dull and
            // pitted elsewhere.
            m.color = vec3(0.50, 0.58, 0.62) * (0.8 + mottle * 0.3);
            m.metal = 0.6;
            m.roughness = 0.42 + grain * 0.15;
            m.texture = PLATED;
            m.relief = 0.2;
            m.grain = 2.0;
        }
        7 => {
            // Horn and bone: ivory, stained at the base.
            m.color = vec3(0.91, 0.87, 0.73).lerp(vec3(0.62, 0.52, 0.38), mottle * 0.5)
                * (0.9 + grain * 0.1);
            m.roughness = 0.7;
            m.texture = RINGED;
            m.relief = 0.4;
        }
        8 => {
            // Soft mouthparts: flushed, wet flesh that light passes into.
            m.color = vec3(0.62, 0.24, 0.30) * (0.85 + mottle * 0.3);
            m.roughness = 0.28;
            m.translucency = 0.3;
            m.texture = SLICK;
            m.relief = 0.25;
        }
        9 => {
            m.color = accent;
            m.emission = 2.5;
            m.roughness = 0.22;
            m.relief = 0.0;
        }
        10 => {
            m.color = vec3(0.17, 0.34, 0.21).lerp(vec3(0.43, 0.59, 0.27), mottle);
            m.roughness = 0.85;
        }
        11 => {
            // Display blades: dark iron, banded towards the tip in the species' colours, and
            // polished, for they are shown off.
            let band = ((p.y * 16.0 + p.z * 3.0).sin() * 0.5 + 0.5).smoothstep(0.3, 0.7);
            let tip = p.y.smoothstep(0.2, 0.6);
            m.color = vec3(0.16, 0.17, 0.19).lerp(accent * 1.15, band * tip * 0.85)
                * (0.85 + mottle * 0.25);
            m.metal = 0.55;
            m.roughness = 0.26 + grain * 0.1;
            m.texture = PLATED;
            m.relief = 0.1;
            m.grain = 3.0;
        }
        _ => {}
    }
    m
}

fn normalize(vector: Vec3) -> Vec3 {
    vector / vector.length_squared().max(1e-20).sqrt()
}

pub fn estimate_normal<F: FnMut(Vec3) -> f32>(point: Vec3, sdf: F) -> Vec3 {
    normal_at(point, 0.03, sdf)
}

fn normal_at<F: FnMut(Vec3) -> f32>(point: Vec3, epsilon: f32, mut sdf: F) -> Vec3 {
    let a = vec3(1.0, -1.0, -1.0);
    let b = vec3(-1.0, -1.0, 1.0);
    let c = vec3(-1.0, 1.0, -1.0);
    let d = Vec3::ONE;
    normalize(
        a * sdf(point + a * epsilon)
            + b * sdf(point + b * epsilon)
            + c * sdf(point + c * epsilon)
            + d * sdf(point + d * epsilon),
    )
}

pub fn local_ray(ray: Ray, c: &CreatureInstance) -> Ray {
    let rotation = Quat::from_array(c.rotation.to_array()).conjugate();
    let scale = development::scale(c);
    Ray {
        origin: (rotation * (ray.origin - c.position.truncate())) / scale,
        direction: (rotation * ray.direction) / scale,
        light: rotation * ray.light,
        ..ray
    }
}

pub fn make_hit(
    ray: Ray,
    c: &CreatureInstance,
    t: f32,
    normal: Vec3,
    previous: Vec3,
    material: Material,
) -> Hit {
    let rotation = Quat::from_array(c.rotation.to_array());
    let scale = development::scale(c);
    let world_normal = (rotation * (normal / scale)).normalize();
    let material = development::appearance(material, previous, c);
    let cosine = world_normal.dot(ray.direction).abs().clamp(0.0, 1.0);
    Hit {
        color_depth: material.shade(world_normal, ray).extend(t),
        normal: world_normal,
        previous_position: Quat::from_array(c.previous_rotation.to_array())
            * (previous
                * if c.previous_proportions.x > 0.0 {
                    c.previous_proportions.truncate()
                } else {
                    scale
                })
            + c.previous_position.truncate(),
        transmittance: material.transmittance(cosine),
        part: 0,
    }
}

pub fn trace<F: Fn(Vec3, &CreatureInstance) -> Surface>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    look: Look,
) -> Hit {
    let mut hit = trace_material(ray, c, surface, |point, id| {
        coat(point / (c.position.w * 0.15).max(0.7), id, look)
    });
    hit.transmittance = Vec3::ZERO;
    hit
}

pub fn trace_material<F: Fn(Vec3, &CreatureInstance) -> Surface, M: Fn(Vec3, u32) -> Material>(
    ray: Ray,
    c: &CreatureInstance,
    surface: F,
    material: M,
) -> Hit {
    let local = local_ray(ray, c);
    let scale = development::scale(c).min_element();
    let surface = |point, c: &CreatureInstance| {
        let mut hit = surface(development::body_point(point, c), c);
        // Allowing for the stretch that young and individual builds put on the body.
        hit.distance *= (1.0 - (1.0 - c.life.y) * 0.2) * 0.85;
        hit
    };
    let bounds = c.opaque_bounds.truncate();
    let (mut distance, end) = span_in_bounds(local, bounds, ray.max_distance);
    // March, then take the normal, the occlusion and the shadow, all in one loop with one
    // evaluation of the surface: on the GPU every evaluation is inlined whole, and a creature's
    // surface may switch over every species, so a single call keeps the shader small enough to
    // build. Phase 0 marches; 1 to 4 tap the gradient; 5 to 9 the occlusion; 10 marches the shadow
    // towards the light.
    let towards = normalize(local.light / development::scale(c));
    let reach = bounds.length() * 2.0;
    let ambient_scale = (c.position.w * 0.06).max(0.025);
    let mut phase = 0u32;
    let mut point = Vec3::ZERO;
    let mut epsilon = 0.0;
    let mut tap = 0.0;
    let mut id = 0;
    let mut gradient = Vec3::ZERO;
    let mut normal = Vec3::Y;
    let mut occlusion = 0.0;
    let mut lit = 1.0f32;
    let mut travel = 0.0;
    for _ in 0..(128 + 4 + 5 + 28) {
        let corner = match phase {
            1 => vec3(1.0, -1.0, -1.0),
            2 => vec3(-1.0, -1.0, 1.0),
            3 => vec3(-1.0, 1.0, -1.0),
            _ => Vec3::ONE,
        };
        let query = if phase == 0 {
            if distance >= end {
                break;
            }
            local.origin + local.direction * distance
        } else if phase <= 4 {
            point + corner * tap
        } else if phase <= 9 {
            point + normal * (ambient_scale * (phase - 4) as f32 * 0.2)
        } else {
            if travel > reach || lit < 0.02 {
                break;
            }
            point + normal * (epsilon * 2.0) + towards * travel
        };
        let sample = surface(query, c);
        if phase == 0 {
            epsilon = (ray.footprint * distance * 0.55 / scale)
                .clamp(c.position.w * 0.000_01, c.position.w * 0.002);
            if sample.distance < epsilon {
                point = query;
                id = sample.material;
                tap = epsilon.max(c.position.w * 0.000_05);
                phase = 1;
            } else {
                distance += (sample.distance * scale * 0.80).max(epsilon * scale * 0.5);
            }
        } else if phase <= 4 {
            gradient += corner * sample.distance;
            phase += 1;
            if phase == 5 {
                normal = normalize(gradient);
            }
        } else if phase <= 9 {
            let ring = (phase - 4) as f32;
            occlusion += (ambient_scale * ring * 0.2 - sample.distance).max(0.0) / ambient_scale
                * (0.6f32).powi((phase - 4).cast_signed());
            phase += 1;
            if phase == 10 {
                travel = epsilon * 3.0 + reach * 0.003;
            }
        } else {
            // The body's own shadow: soft-edged, so limbs, wings and folds shade what lies
            // beneath them.
            lit = lit.min(8.0 * sample.distance / travel);
            travel += sample.distance.clamp(reach * 0.004, reach * 0.08);
        }
    }
    if phase == 0 {
        return Hit::MISS;
    }
    if phase < 5 {
        normal = normalize(gradient);
    }
    let lit = lit.clamp(0.0, 1.0);
    let shaded_ray = Ray {
        ambient: ray.ambient * (1.0 - occlusion * 3.0).clamp(0.2, 1.0),
        radiance: ray.radiance * (lit * lit * (3.0 - 2.0 * lit)),
        ..ray
    };
    let material = material(point, id);
    let normal = relieved(normal, point, c, material, ray.footprint * distance);
    make_hit(
        shaded_ray,
        c,
        distance,
        normal,
        previous_point(point, c),
        material,
    )
}

/// Where a ray enters and leaves a creature's bounding box, in its own frame, clipped to the
/// ray's reach.
fn span_in_bounds(local: Ray, bounds: Vec3, max_distance: f32) -> (f32, f32) {
    let nonzero = |value: f32| if value.abs() < 1e-8 { 1e-8 } else { value };
    let safe = vec3(
        nonzero(local.direction.x),
        nonzero(local.direction.y),
        nonzero(local.direction.z),
    );
    let inverse = Vec3::ONE / safe;
    let near = (-bounds - local.origin) * inverse;
    let far = (bounds - local.origin) * inverse;
    (
        near.min(far).max_element().max(0.0),
        near.max(far).min_element().min(max_distance),
    )
}

/// Where a point of the body was last frame, for motion vectors: the tail swings on its own, so a
/// point along it moves with its posed chain.
fn previous_point(point: Vec3, c: &CreatureInstance) -> Vec3 {
    let tail_length = c.tail[5].z - c.tail[0].z;
    if point.z <= c.tail[0].z || tail_length <= 0.01 {
        return point;
    }
    let fraction = ((point.z - c.tail[0].z) / tail_length).clamp(0.0, 1.0) * 5.0;
    let index = (fraction as usize).min(4);
    let delta = c.previous_tail[index].lerp(c.previous_tail[index + 1], fraction - index as f32)
        - c.tail[index].lerp(c.tail[index + 1], fraction - index as f32);
    let scale = if c.kind.x == Kind::Nae as u32 {
        vec3(0.2, 0.25, 1.0)
    } else if c.kind.x == Kind::Naetu as u32 {
        vec3(3.0, 3.0, 1.0)
    } else {
        Vec3::ONE
    };
    point + delta.truncate() * scale
}

/// A normal tilted by the surface's fine relief, from the slope of its height field about the
/// point; the relief fades out where a pixel, of this `footprint`, spans many of its features.
fn relieved(
    normal: Vec3,
    point: Vec3,
    c: &CreatureInstance,
    material: Material,
    footprint: f32,
) -> Vec3 {
    if material.relief <= 0.0 {
        return normal;
    }
    let at = point / (c.position.w * 0.15).max(0.7) * material.grain;
    let step = 0.03;
    let mut slope = Vec3::ZERO;
    for corner in 0..4 {
        let offset = match corner {
            0 => vec3(1.0, -1.0, -1.0),
            1 => vec3(-1.0, -1.0, 1.0),
            2 => vec3(-1.0, 1.0, -1.0),
            _ => Vec3::ONE,
        };
        slope += offset * relief(at + offset * step, material.texture);
    }
    let slope = slope / (4.0 * step);
    let resolve = 1.0 - (footprint * material.grain * 4.0).smoothstep(0.3, 1.5);
    normalize(normal - (slope - normal * slope.dot(normal)) * material.relief * 0.12 * resolve)
}

pub fn tonemap(color: Vec3) -> Vec3 {
    let a = color * (color * 2.51 + Vec3::splat(0.03));
    let b = color * (color * 2.43 + Vec3::splat(0.59)) + Vec3::splat(0.14);
    (a / b).clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2)
}

impl Sheet {
    /// How the membrane curls along the span and arches across the chord.
    fn curvature(self) -> f32 {
        self.bend.x / (self.radii.x * self.radii.x)
    }

    fn arch(self) -> f32 {
        -self.camber / self.radii.y
    }

    /// Where the wrist lies across the span, in the wing's own frame about its root.
    fn wrist_x(self) -> f32 {
        let offset = self.center - self.root;
        offset.x + (2.0 * self.wrist - 1.0) * self.radii.x * self.side
    }

    /// How sharply the hand curls beyond the wrist, for a hand turned `fold` from the arm's line by
    /// the time it reaches the tip. The curl starts flat at the wrist, so the membrane bends
    /// smoothly through the joint rather than creasing at it.
    fn curl(self, fold: f32) -> f32 {
        let hand = ((1.0 - self.wrist) * 2.0 * self.radii.x).max(1e-3);
        fold.clamp(-1.2, 1.2).tan() / (2.0 * hand)
    }

    /// The membrane's height above the wing's plane at `x` and `z`, with this curvature along the
    /// span and this curl beyond the wrist.
    fn height(self, x: f32, z: f32, curvature: f32, curl: f32) -> f32 {
        let beyond = ((x - self.wrist_x()) * self.side).max(0.0);
        let across = z - (self.center.z - self.root.z);
        curvature * x * x + curl * beyond * beyond + self.arch() * across * across
    }

    /// The membrane's normal at a point of it, in the wing's frame with the stroke undone.
    fn normal_at(self, point: Vec3) -> Vec3 {
        let offset = self.center - self.root;
        let beyond = ((point.x - self.wrist_x()) * self.side).max(0.0);
        vec3(
            -2.0 * self.curvature() * point.x - 2.0 * self.curl(self.fold.x) * beyond * self.side,
            1.0,
            -2.0 * self.arch() * (point.z - offset.z),
        )
        .normalize()
    }

    /// The nearest point where a ray in the wing's frame crosses the membrane within `limit`: how
    /// far along, and where. The surface is quadratic in two pieces, the arm out to the wrist and
    /// the hand curled beyond it, each solved exactly.
    fn crossing(self, origin: Vec3, direction: Vec3, limit: f32, feathered: bool) -> (f32, Vec3) {
        let offset = self.center - self.root;
        let curvature = self.curvature();
        let arch = self.arch();
        let wrist_x = self.wrist_x();
        let mut distance = limit;
        let mut point = Vec3::ZERO;
        for panel in 0..2 {
            let curl = if panel == 0 {
                0.0
            } else {
                self.curl(self.fold.x)
            };
            let (from, towards) = (origin, direction);
            let across = from.z - offset.z;
            let out = from.x - wrist_x;
            let quadratic = curvature * towards.x * towards.x
                + curl * towards.x * towards.x
                + arch * towards.z * towards.z;
            let linear = 2.0 * curvature * from.x * towards.x
                + 2.0 * curl * out * towards.x
                + 2.0 * arch * across * towards.z
                - towards.y;
            let constant =
                curvature * from.x * from.x + curl * out * out + arch * across * across - from.y;
            let discriminant = linear * linear - 4.0 * quadratic * constant;
            if discriminant < 0.0 {
                continue;
            }
            let roots = if quadratic.abs() < 1e-8 {
                Vec2::splat(-constant / linear)
            } else {
                let root = discriminant.sqrt();
                Vec2::new(
                    (-linear - root) / (2.0 * quadratic),
                    (-linear + root) / (2.0 * quadratic),
                )
            };
            for index in 0..2 {
                let candidate = if index == 0 { roots.x } else { roots.y };
                if !(candidate > 0.0 && candidate < distance) {
                    continue;
                }
                let at = from + towards * candidate;
                let uv = self.coordinates(at - offset);
                let along = uv.x.mul_add(self.side * 0.5, 0.5);
                let on_panel = if panel == 0 {
                    along <= self.wrist
                } else {
                    along > self.wrist
                };
                if on_panel && planform(uv, self.side, feathered, self.style) >= 0.0 {
                    distance = candidate;
                    point = at;
                }
            }
        }
        (distance, point)
    }

    /// The membrane's material at wing coordinates `uv`, seen at `facing` with a pixel `footprint`
    /// wide in chords, and how much of the pixel it covers at its outline.
    fn membrane(self, uv: Vec2, pigment: Vec4, facing: f32, footprint: f32) -> (Material, f32) {
        let tint = pigment.truncate();
        let opacity = pigment.w;
        let feathered = opacity > 0.9;
        let along = uv.x.mul_add(self.side * 0.5, 0.5);
        // Spars fan out from the root across the chord, a thick one along the leading edge, with
        // faint cross veins between them; feathers instead lie in overlapping vanes slanting back
        // towards the trailing edge.
        let chord = ((uv.y + 0.62) / 1.4).clamp(0.0, 1.0);
        let spars = ((chord * 5.0 + along * 0.8) * PI).sin().abs();
        let cross = (along * 26.0 + chord * 3.0).sin().abs();
        let edge = chord.smoothstep(0.06, 0.0);
        let vein = if feathered {
            ((along * 18.0 - chord * 7.0) * PI).sin().abs()
        } else {
            spars.min(cross * 1.6 + 0.35)
        };
        let vein_width = footprint.max(0.012);
        let rib = (1.0 - vein.smoothstep(vein_width, vein_width * 2.5)).max(edge);
        // Darker towards the leading edge and the root, where the wing is thickest; paler and
        // clearer towards the trailing edge.
        let thickness = (1.0 - chord) * 0.6 + (1.0 - along) * 0.4;
        let colour = if feathered {
            tint.lerp(
                vec3(0.80, 0.85, 0.86),
                (chord * 0.45 - 0.05).clamp(0.0, 0.35),
            ) * (0.78 + thickness * 0.2 + ((along * 18.0 - chord * 7.0) * PI).sin() * 0.06)
        } else {
            tint.lerp(tint * tint, thickness * 0.5)
                .lerp(vec3(0.94, 0.59, 0.17), along.smoothstep(0.7, 1.0) * 0.25)
        };
        // The species' own markings over the membrane.
        let mark = marking(self.pattern, along, chord, facing);
        let colour = colour.lerp(self.second, (mark.max(0.0) * self.pattern.z).min(1.0));
        let windows = (-mark).max(0.0) * self.pattern.z;
        let coverage =
            (planform(uv, self.side, feathered, self.style) / footprint.max(0.001)).saturate();
        let alpha =
            ((opacity + rib * 0.6 + thickness * 0.1) * (1.0 - windows * 0.8)).clamp(0.0, 0.98);
        let mut material = Material::skin(colour, if feathered { 0.65 } else { 0.25 + rib * 0.35 });
        material.transmission = (Vec3::ONE.lerp(colour.sqrt(), alpha * 0.5)) * (1.0 - alpha);
        (material, coverage)
    }
}

pub fn sheet(ray: Ray, c: &CreatureInstance, shape: Sheet, pigment: Vec4) -> Hit {
    let local = local_ray(ray, c);
    // Into the wing's own frame: about its root, with its stroke undone.
    let origin = roll(local.origin - shape.root, -shape.stroke.x);
    let direction = roll(local.direction, -shape.stroke.x);
    let offset = shape.center - shape.root;
    let (distance, point) = shape.crossing(origin, direction, ray.max_distance, pigment.w > 0.9);
    if distance >= ray.max_distance {
        return Hit::MISS;
    }
    let surface_normal = shape.normal_at(point);
    let footprint = ray.footprint * distance / shape.radii.min_element();
    let (material, coverage) = shape.membrane(
        shape.coordinates(point - offset),
        pigment,
        surface_normal.dot(direction).abs(),
        footprint,
    );
    let normal = roll(surface_normal, shape.stroke.x);
    // Where this point of the membrane was last frame, bent and curled as it was then.
    let earlier = vec3(
        point.x,
        shape.height(
            point.x,
            point.z,
            shape.bend.y / (shape.radii.x * shape.radii.x),
            shape.curl(shape.fold.y),
        ),
        point.z,
    );
    let previous =
        shape.root + (shape.previous_center - shape.center) + roll(earlier, shape.stroke.y);
    let mut hit = make_hit(
        ray,
        c,
        distance,
        if normal.dot(local.direction) > 0.0 {
            -normal
        } else {
            normal
        },
        previous,
        material,
    );
    hit.color_depth = (hit.color() * coverage).extend(distance);
    hit.transmittance = Vec3::ONE.lerp(hit.transmittance, coverage);
    hit.part = 1;
    hit
}

/// One wing of a pair, hinged at its shoulder and beaten through the species' own stroke.
///
/// The arm turns at the shoulder, and the hand beyond the wrist follows a little behind, so the tip
/// trails the stroke, folds in on the upstroke and flings out on the down. Each species beats at
/// its own rate and depth, each individual a little out of step with the rest.
pub fn wing_sheet(c: &CreatureInstance, look: Look, index: usize, side: f32) -> Sheet {
    let wing = look.wings[index];
    let shoulder = look.hinges[index].extend(wing.z);
    let root = vec3(shoulder.x * side, shoulder.y, shoulder.z);
    // No two individuals' wings quite alike: a little longer or shorter, held a little higher or
    // lower at rest.
    let individual = c.identity - 0.5;
    let dimensions =
        wing.truncate().truncate() * vec2(1.0 + individual.x * 0.16, 1.0 + individual.y * 0.1);
    let rest = 0.1 + individual.z * 0.16;
    let development = if c.proportions.x > 0.0 {
        0.25 + c.proportions.w * 0.75
    } else {
        1.0
    };
    let chord = dimensions.y * development.sqrt();
    let reach = dimensions.x * development * (1.0 - c.animation.y * 0.65);
    let old_reach = dimensions.x * development * (1.0 - c.previous_animation.y * 0.65);
    let motion = look.flight;
    // The membrane is no board: the air it drives bows it along the span against the stroke, tips
    // curling up on the downstroke and down on the up, and fills it deeper across the chord on
    // the downstroke.
    let beat = |time: f32, folded: f32| {
        let phase = time * motion.x * TAU + index as f32 * 0.6;
        let effort = 1.0 - folded.clamp(0.0, 1.0);
        let push = -phase.cos() * effort;
        (
            rest + motion.y * phase.sin() * effort,
            motion.z * ((phase - motion.w).sin() - 0.3) * effort,
            0.02 + push * motion.y * 0.1,
            0.07 + push.max(0.0) * 0.06,
        )
    };
    let (stroke, fold, flex, camber) = beat(c.animation.z, c.animation.y);
    let (old_stroke, old_fold, old_flex, _) = beat(c.previous_animation.z, c.previous_animation.y);
    Sheet {
        center: root + vec3(side * reach * 0.5, 0.0, reach * 0.12),
        previous_center: root + vec3(side * old_reach * 0.5, 0.0, old_reach * 0.12),
        root,
        stroke: Vec2::new(side * stroke, side * old_stroke),
        radii: Vec2::new(reach * 0.52, chord),
        bend: Vec2::new(reach * flex, old_reach * old_flex),
        camber,
        style: look.styles[index],
        wrist: look.pattern.w,
        fold: Vec2::new(fold, old_fold),
        pattern: look.pattern,
        second: look.second,
        side,
    }
}

pub fn visit_paired_sheets<F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    look: Look,
    index: usize,
    mut emit: F,
) {
    let pigment = look.accent.extend(look.wings[index].w);
    for i in 0..2 {
        let side = if i == 0 { -1.0 } else { 1.0 };
        emit(sheet(ray, c, wing_sheet(c, look, index, side), pigment));
    }
}

/// The frame a membrane wing is stretched over, in the creature's own frame.
///
/// A thick arm runs from a fleshy root out to the wrist, a long finger on along the leading edge to
/// the tip, and fingers fan back from the wrist to the points of the trailing edge between its
/// scallops. A feathered wing has the arm and leading finger alone; its feathers carry the rest.
pub fn wing_bones(p: Vec3, c: &CreatureInstance, look: Look, index: usize) -> f32 {
    let style = look.styles[index];
    let weight = look.frames[index];
    let feathered = look.wings[index].w > 0.9;
    let mut distance = 1e6f32;
    for i in 0..2 {
        let side = if i == 0 { -1.0 } else { 1.0 };
        let shape = wing_sheet(c, look, index, side);
        let rx = shape.radii.x;
        let chord = shape.radii.y;
        let offset = shape.center - shape.root;
        let curvature = shape.bend.x / (rx * rx);
        let curl = shape.curl(shape.fold.x);
        // Far from this wing, its bounding sphere is distance enough.
        let middle = shape.root + roll(offset, shape.stroke.x);
        // Stopping a chord short of the sphere, so its own surface is never mistaken for
        // the wing's.
        let bound = (p - middle).length() - (rx * 1.35 + chord);
        if bound > chord {
            distance = distance.min(bound);
            continue;
        }
        let at = |along: f32, v: f32| {
            let x = offset.x + (2.0 * along - 1.0) * rx * side;
            let z = offset.z
                + (along * along - 0.25) * rx * 0.55 * style.x
                + v * chord * (1.0 - 0.55 * along);
            let placed = vec3(x, shape.height(x, z, curvature, curl), z);
            shape.root + roll(placed, shape.stroke.x)
        };
        let lead = |along: f32| -leading(along, style) * 0.9;
        let trail = |along: f32| trailing(along, style, feathered) * 0.95;
        let bone = |a: Vec3, b: Vec3, ra: f32, rb: f32| {
            let axis = b - a;
            let t = ((p - a).dot(axis) / axis.length_squared().max(1e-8)).saturate();
            ((p - a - axis * t).length() - ra.lerp(rb, t) * weight) * 0.9
        };
        let wrist = shape.wrist;
        let mut frame = bone(
            at(0.0, lead(0.0)),
            at(wrist, lead(wrist)),
            chord * 0.075,
            chord * 0.04,
        );
        for segment in 0..3 {
            let a = wrist + (0.97 - wrist) * segment as f32 / 3.0;
            let b = wrist + (0.97 - wrist) * (segment + 1) as f32 / 3.0;
            frame = frame.min(bone(
                at(a, lead(a)),
                at(b, lead(b)),
                chord * (0.04 - segment as f32 * 0.009),
                chord * (0.031 - segment as f32 * 0.009),
            ));
        }
        // A finger to each point of the trailing edge between its notches.
        let fingers = if feathered { 0 } else { style.z as u32 };
        for finger in 1..7 {
            let end = finger as f32 / (style.z + 0.5);
            if finger > fingers || end <= wrist + 0.06 || end > 0.97 {
                continue;
            }
            {
                let mid = wrist.lerp(end, 0.5);
                let start = at(wrist, lead(wrist));
                let bow = at(mid, lead(mid).lerp(trail(mid), 0.5));
                let tip = at(end, trail(end));
                frame = frame.min(bone(start, bow, chord * 0.024, chord * 0.017));
                frame = frame.min(bone(bow, tip, chord * 0.017, chord * 0.006));
            }
        }
        distance = distance.min(frame);
    }
    distance
}

pub fn shell(
    result: &mut TraceResult,
    ray: Ray,
    c: &CreatureInstance,
    center: Vec3,
    radii: Vec3,
    pigment: Vec4,
    bottom: f32,
) {
    visit_shell(ray, c, center, radii, pigment, bottom, |hit| {
        result.add(hit);
    });
}

pub fn visit_shell<F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    center: Vec3,
    radii: Vec3,
    pigment: Vec4,
    bottom: f32,
    emit: F,
) {
    let tint = pigment.truncate();
    let opacity = pigment.w;
    visit_shell_material(
        ray,
        c,
        Shell {
            center,
            radii,
            bottom,
            scallop: 0.0,
        },
        |_, cosine| {
            let mut material = Material::skin(tint, 0.18);
            material.ior = 1.36;
            material.transmission =
                (-(Vec3::ONE - tint) * (opacity / cosine.max(0.08))).exp() * (1.0 - opacity);
            if c.kind.x == Kind::Naetu as u32 {
                material.emission = 0.0125 + 0.4875 * (1.0 - cosine).powi(5);
            }
            material
        },
        emit,
    );
}

pub fn visit_shell_material<M: Fn(Vec3, f32) -> Material, F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    shape: Shell,
    material: M,
    mut emit: F,
) {
    let Shell {
        center,
        radii,
        bottom,
        scallop,
    } = shape;
    let squared_radii = radii * radii;
    let local = local_ray(ray, c);
    let origin = (local.origin - center) / radii;
    let direction = local.direction / radii;
    let quadratic = direction.length_squared();
    let linear = origin.dot(direction);
    let discriminant = linear * linear - quadratic * (origin.length_squared() - 1.0);
    if discriminant < 0.0 {
        return;
    }
    let root = discriminant.sqrt();
    for i in 0..2 {
        let distance = (-linear + if i == 0 { -root } else { root }) / quadratic;
        if distance <= 0.0 || distance >= ray.max_distance {
            continue;
        }
        let point = local.origin + local.direction * distance;
        if point.y < bottom + (point.x.atan2(point.z) * 14.0).cos() * scallop {
            continue;
        }
        let normal = ((point - center) / squared_radii).normalize();
        let cosine = normal.dot(local.direction).abs();
        let mut hit = make_hit(
            ray,
            c,
            distance,
            if normal.dot(local.direction) > 0.0 {
                -normal
            } else {
                normal
            },
            point,
            material(point, cosine),
        );
        hit.part = 2;
        emit(hit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: Vec3, direction: Vec3) -> Ray {
        Ray {
            origin,
            direction,
            light: Vec3::Y,
            ambient: Vec3::ONE,
            radiance: Vec3::ONE,
            max_distance: 100.0,
            footprint: 0.001,
        }
    }

    #[test]
    fn scattering_conserves_energy_under_uniform_illumination() {
        for roughness in [0.3, 0.6, 1.0] {
            for (metal, translucency) in [(0.0, 0.0), (1.0, 0.0), (0.0, 0.5), (0.0, 1.0)] {
                let mut material = Material::skin(Vec3::ONE, roughness);
                material.metal = metal;
                material.translucency = translucency;
                let mut energy = Vec3::ZERO;
                for latitude in 0..128 {
                    let cosine = (latitude as f32 + 0.5) / 64.0 - 1.0;
                    let sine = (1.0 - cosine * cosine).sqrt();
                    for longitude in 0..128 {
                        let angle = (longitude as f32 + 0.5) * (2.0 * PI / 128.0);
                        let illumination = Ray {
                            direction: -Vec3::Y,
                            light: vec3(sine * angle.cos(), cosine, sine * angle.sin()),
                            ambient: Vec3::ZERO,
                            ..ray(Vec3::ZERO, -Vec3::Y)
                        };
                        energy += material.shade(Vec3::Y, illumination) * (4.0 * PI / 16384.0);
                    }
                }
                assert!(
                    energy.max_element() <= 1.01,
                    "BRDF creates energy: {energy:?}"
                );
            }
        }
    }

    #[test]
    fn curved_sheet_keeps_second_root_when_first_is_outside_wing() {
        let instance = CreatureInstance {
            rotation: Vec4::W,
            previous_rotation: Vec4::W,
            ..CreatureInstance::default()
        };
        let hit = sheet(
            ray(vec3(-2.0, 0.25, 3.0), vec3(1.0, 0.0, -1.0).normalize()),
            &instance,
            Sheet {
                center: Vec3::ZERO,
                previous_center: Vec3::ZERO,
                root: Vec3::ZERO,
                stroke: Vec2::ZERO,
                radii: Vec2::ONE,
                camber: 0.0,
                style: PLAIN_WING,
                bend: Vec2::ONE,
                wrist: 1.0,
                fold: Vec2::ZERO,
                pattern: Vec4::ZERO,
                second: Vec3::ZERO,
                side: 0.0,
            },
            Vec3::ONE.extend(0.3),
        );
        assert!(hit.is_hit(), "the second valid wing root was discarded");
        assert!(
            (hit.distance() - 2.5 * 2.0f32.sqrt()).abs() < 0.001,
            "wrong wing root"
        );
        assert!(
            (hit.normal.length() - 1.0).abs() < 0.001,
            "invalid wing normal"
        );
        assert!(
            hit.transmittance.min_element() > 0.0,
            "membrane must transmit light"
        );
    }

    #[test]
    fn transparency_is_sorted_and_clipped_by_opaque_body() {
        let layer = |distance, color: Vec3| Hit {
            color_depth: color.extend(distance),
            transmittance: Vec3::splat(0.5),
            ..Hit::MISS
        };
        let mut result = TraceResult::opaque(layer(5.0, Vec3::Z));
        result.add(layer(7.0, Vec3::splat(100.0)));
        result.add(layer(3.0, Vec3::Y));
        result.add(layer(1.0, Vec3::X));
        assert!(
            result.front().distance() < 2.0,
            "layers must sort front to back"
        );
        assert!(
            result.composite(Vec3::ZERO).distance(vec3(1.0, 0.5, 0.25)) < 0.001,
            "transparent layers behind the opaque body must be excluded"
        );
    }
}
