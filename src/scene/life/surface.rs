use crate::scene::life::{CreatureInstance, Kind, development};
use core::{f32::consts::PI, mem::swap};
use spirv_std::glam::{FloatExt, Quat, Vec2, Vec3, Vec3Swizzles, Vec4, vec3};
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
        }
    }
    pub fn shade(self, normal: Vec3, ray: Ray) -> Vec3 {
        let view = -ray.direction;
        let light = normalize(ray.light);
        let halfway = normalize(light + view);
        let nl = normal.dot(light).max(0.0);
        let backlight = (-normal.dot(light)).max(0.0);
        // Light wraps a little past the terminator, as it does through skin
        // and soft tissue, scaled so the lobe still carries the same energy.
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
/// A curved membrane, hinged at `root`: turned about the body's long axis
/// by `stroke` (now and last frame), and curling away from the stroke
/// towards its tip by `bend`, so its root never leaves the body.
pub struct Sheet {
    pub center: Vec3,
    pub previous_center: Vec3,
    pub root: Vec3,
    pub stroke: Vec2,
    pub radii: Vec2,
    pub bend: Vec2,
    /// How far the membrane arches between its edges, as a share of its
    /// chord: an aerofoil's camber.
    pub camber: f32,
    pub side: f32,
}

/// A vector turned about the long axis.
fn roll(v: Vec3, angle: f32) -> Vec3 {
    let (sine, cosine) = angle.sin_cos();
    vec3(v.x * cosine - v.y * sine, v.x * sine + v.y * cosine, v.z)
}

/// How far inside a wing's outline a point of it lies, in chords, from
/// its coordinates: positive within. The leading edge runs nearly straight
/// out to a rounded tip; the trailing edge sweeps back from a broad root,
/// fingered between the spars of a membrane or frayed into feather tips.
fn planform(uv: Vec2, side: f32, feathered: bool) -> f32 {
    let along = uv.x.mul_add(side * 0.5, 0.5);
    let tip = (1.0 - along.max(0.0).powi(6)).max(0.0).sqrt();
    let lead = 0.62 * tip;
    let trailing = if feathered {
        1.0 - 0.07 * (along * 38.0).sin().abs()
    } else {
        1.0 - 0.16 * (along * PI * 4.5).sin().abs() * along.smoothstep(0.1, 0.4)
    };
    let trail = (0.4 + 0.75 * (1.0 - along).max(0.0).sqrt()) * tip * trailing;
    (uv.y + lead)
        .min(trail - uv.y)
        .min((along + 0.02) * 2.0)
        .min((1.0 - along) * 2.0)
}

impl Sheet {
    fn coordinates(self, point: Vec3) -> Vec2 {
        let span = point.x / self.radii.x;
        let fraction = span.mul_add(self.side * 0.5, 0.5);
        Vec2::new(
            span,
            (point.z - (fraction * fraction - 0.25) * self.radii.x * 0.55)
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

/// Living pigment over a body: irregular blotches drawn out along it,
/// darker over the back and paler beneath as in most animals that are seen
/// from above and below, mottled at two finer scales so no surface reads as
/// flat paint.
pub fn coat(p: Vec3, id: u32, pigment: Vec4, accent: Vec3) -> Material {
    let warp = vec3(fbm(p * 0.6), fbm(p * 0.6 + 17.0), fbm(p * 0.6 + 31.0)) - 0.5;
    let blotches = fbm(vec3(p.x * 0.8, p.y * 0.8, p.z * 0.45) + warp * 2.2);
    let mottle = fbm(p * 3.7 + 5.0);
    let grain = noise(p * 19.0);
    let dorsal = (p.y * 0.30 + 0.5).saturate();
    let markings = blotches.smoothstep(0.5, 0.64) * dorsal.smoothstep(0.25, 0.75);
    let base = pigment.truncate().lerp(accent, markings * 0.7);
    // Countershading turned over, for a world lit from below: the back
    // pale, lost against the bright veil to anything looking down on it, and
    // the belly dark, lost against the dim sky to anything looking up.
    let back = (base * 1.25 + vec3(0.05, 0.05, 0.06)).min(Vec3::ONE);
    let belly = base * 0.55;
    // Pulsatile nutrient channels show faintly through the skin, branching
    // back from the crop.
    let channel = fbm(vec3(p.x * 1.4, p.y * 1.4, p.z * 0.5) + warp * 1.5);
    let vein = 1.0 - (channel - 0.5).abs().smoothstep(0.0, 0.035);
    let mut m = Material::skin(
        belly
            .lerp(back, dorsal.smoothstep(0.2, 0.8))
            .lerp(accent * 0.6 + vec3(0.22, 0.04, 0.08), vein * 0.22)
            * (0.84 + mottle * 0.24)
            * (0.95 + grain * 0.08),
        0.4 + mottle * 0.18,
    );
    m.emission = pigment.w;
    match id {
        1 => {
            // Fins and feathers: the accent, darkening along vanes.
            m.color = accent * (0.8 + mottle * 0.3);
            m.roughness = 0.6;
        }
        2 => {
            m.color = accent.lerp(vec3(0.95, 0.29, 0.31), 0.32);
            m.transmission = Vec3::splat(0.35);
        }
        3 => {
            // Eyes: glassy lenses over a deep, faintly green-gold eye cup,
            // dark but catching every highlight.
            m.color = vec3(0.07, 0.085, 0.06) * (0.8 + grain * 0.4);
            m.roughness = 0.04;
            m.emission = 0.0;
        }
        4 => {
            m.color =
                accent.lerp(pigment.truncate(), blotches * 0.4 + 0.15) * (0.85 + mottle * 0.3);
            m.roughness = 0.32;
        }
        5 => {
            m.color = accent * (0.6 + grain * 0.4);
            m.metal = 0.55;
            m.roughness = 0.3;
        }
        6 => {
            m.color = vec3(0.60, 0.73, 0.78) * (0.9 + mottle * 0.15);
            m.metal = 0.85;
            m.roughness = 0.24 + grain * 0.1;
        }
        7 => {
            // Horn and bone: ivory, stained at the base.
            m.color = vec3(0.91, 0.87, 0.73).lerp(vec3(0.62, 0.52, 0.38), mottle * 0.5)
                * (0.9 + grain * 0.1);
            m.roughness = 0.7;
        }
        8 => {
            // Soft mouthparts: flushed, wet flesh that light passes into.
            m.color = vec3(0.62, 0.24, 0.30) * (0.85 + mottle * 0.3);
            m.roughness = 0.28;
            m.translucency = 0.3;
        }
        9 => {
            m.color = accent;
            m.emission = 2.5;
            m.roughness = 0.22;
        }
        10 => {
            m.color = vec3(0.17, 0.34, 0.21).lerp(vec3(0.43, 0.59, 0.27), mottle);
            m.roughness = 0.85;
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
    pigment: Vec4,
    accent: Vec3,
) -> Hit {
    let mut hit = trace_material(ray, c, surface, |point, id| {
        coat(point / (c.position.w * 0.15).max(0.7), id, pigment, accent)
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
        hit.distance *= 1.0 - (1.0 - c.life.y) * 0.2;
        hit
    };
    let bounds = c.opaque_bounds.truncate();
    let nonzero = |value: f32| if value.abs() < 1e-8 { 1e-8 } else { value };
    let safe = vec3(
        nonzero(local.direction.x),
        nonzero(local.direction.y),
        nonzero(local.direction.z),
    );
    let inv = Vec3::ONE / safe;
    let near = (-bounds - local.origin) * inv;
    let far = (bounds - local.origin) * inv;
    let mut distance = near.min(far).max_element().max(0.0);
    let end = near.max(far).min_element().min(ray.max_distance);
    // March, then take the normal, the occlusion and the shadow, all in one
    // loop with one evaluation of the surface: on the GPU every evaluation
    // is inlined whole, and a creature's surface may switch over every
    // species, so a single call keeps the shader small enough to build.
    // Phase 0 marches; 1 to 4 tap the gradient; 5 to 9 the occlusion; 10
    // marches the shadow towards the light.
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
                * (0.6f32).powi((phase - 4) as i32);
            phase += 1;
            if phase == 10 {
                travel = epsilon * 3.0 + reach * 0.003;
            }
        } else {
            // The body's own shadow: soft-edged, so limbs, wings and folds
            // shade what lies beneath them.
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
    let mut previous = point;
    let tail_length = c.tail[5].z - c.tail[0].z;
    if point.z > c.tail[0].z && tail_length > 0.01 {
        let fraction = ((point.z - c.tail[0].z) / tail_length).clamp(0.0, 1.0) * 5.0;
        let index = (fraction as usize).min(4);
        let delta = c.previous_tail[index]
            .lerp(c.previous_tail[index + 1], fraction - index as f32)
            - c.tail[index].lerp(c.tail[index + 1], fraction - index as f32);
        let scale = if c.kind.x == Kind::Nae as u32 {
            vec3(0.2, 0.25, 1.0)
        } else if c.kind.x == Kind::Naetu as u32 {
            vec3(3.0, 3.0, 1.0)
        } else {
            Vec3::ONE
        };
        previous += delta.truncate() * scale;
    }
    let lit = lit.clamp(0.0, 1.0);
    let shaded_ray = Ray {
        ambient: ray.ambient * (1.0 - occlusion * 3.0).clamp(0.2, 1.0),
        radiance: ray.radiance * (lit * lit * (3.0 - 2.0 * lit)),
        ..ray
    };
    make_hit(
        shaded_ray,
        c,
        distance,
        normal,
        previous,
        material(point, id),
    )
}

pub fn tonemap(color: Vec3) -> Vec3 {
    let a = color * (color * 2.51 + Vec3::splat(0.03));
    let b = color * (color * 2.43 + Vec3::splat(0.59)) + Vec3::splat(0.14);
    (a / b).clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2)
}

pub fn sheet(ray: Ray, c: &CreatureInstance, shape: Sheet, pigment: Vec4) -> Hit {
    let Sheet {
        center,
        previous_center,
        radii,
        bend,
        ..
    } = shape;
    let tint = pigment.truncate();
    let opacity = pigment.w;
    let local = local_ray(ray, c);
    // Into the wing's own frame: about its root, with its stroke undone.
    let origin = roll(local.origin - shape.root, -shape.stroke.x);
    let direction = roll(local.direction, -shape.stroke.x);
    let offset = center - shape.root;
    let curvature = bend.x / (radii.x * radii.x);
    // Arched across the chord as well as curled along the span.
    let arch = -shape.camber / radii.y;
    let across = origin.z - offset.z;
    let quadratic = curvature * direction.x * direction.x + arch * direction.z * direction.z;
    let linear =
        2.0 * curvature * origin.x * direction.x + 2.0 * arch * across * direction.z - direction.y;
    let constant = curvature * origin.x * origin.x + arch * across * across - origin.y;
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return Hit::MISS;
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
    let mut distance = ray.max_distance;
    for index in 0..2 {
        let candidate = if index == 0 { roots.x } else { roots.y };
        if !(candidate > 0.0 && candidate < distance) {
            continue;
        }
        let point = origin + direction * candidate;
        let uv = shape.coordinates(point - offset);
        if planform(uv, shape.side, opacity > 0.9) >= 0.0 {
            distance = candidate;
        }
    }
    if distance >= ray.max_distance {
        return Hit::MISS;
    }
    let point = origin + direction * distance;
    let uv = shape.coordinates(point - offset);
    let feathered = opacity > 0.9;
    let along = uv.x.mul_add(shape.side * 0.5, 0.5);
    // Spars fan out from the root across the chord, a thick one along the
    // leading edge, with faint cross veins between them; feathers instead
    // lie in overlapping vanes slanting back towards the trailing edge.
    let chord = ((uv.y + 0.62) / 1.4).clamp(0.0, 1.0);
    let spars = ((chord * 5.0 + along * 0.8) * PI).sin().abs();
    let cross = (along * 26.0 + chord * 3.0).sin().abs();
    let edge = chord.smoothstep(0.06, 0.0);
    let vein = if feathered {
        ((along * 18.0 - chord * 7.0) * PI).sin().abs()
    } else {
        spars.min(cross * 1.6 + 0.35)
    };
    let vein_width = (ray.footprint * distance / radii.min_element()).max(0.012);
    let rib = (1.0 - vein.smoothstep(vein_width, vein_width * 2.5)).max(edge);
    // Darker towards the leading edge and the root, where the wing is
    // thickest; paler and clearer towards the trailing edge.
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
    let coverage = (planform(uv, shape.side, feathered)
        / (ray.footprint * distance / radii.min_element()).max(0.001))
    .saturate();
    let alpha = (opacity + rib * 0.6 + thickness * 0.1).clamp(0.0, 0.98);
    let mut material = Material::skin(
        colour,
        if opacity > 0.9 {
            0.65
        } else {
            0.25 + rib * 0.35
        },
    );
    material.transmission = (Vec3::ONE.lerp(colour.sqrt(), alpha * 0.5)) * (1.0 - alpha);
    let normal = roll(
        vec3(
            -2.0 * curvature * point.x,
            1.0,
            -2.0 * arch * (point.z - offset.z),
        )
        .normalize(),
        shape.stroke.x,
    );
    let previous = shape.root
        + (previous_center - center)
        + roll(
            vec3(
                point.x,
                bend.y * point.x * point.x / (radii.x * radii.x),
                point.z,
            ),
            shape.stroke.y,
        );
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

pub fn paired_sheets(
    result: &mut TraceResult,
    ray: Ray,
    c: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    pigment: Vec4,
) {
    visit_paired_sheets(ray, c, root, dimensions, pigment, |hit| result.add(hit));
}

/// One wing of a pair, hinged at `root`: how far it reaches as the animal
/// grows and folds it, how its stroke turns it, and how its tip lags.
pub fn wing_sheet(c: &CreatureInstance, root: Vec3, dimensions: Vec2, side: f32) -> Sheet {
    let development = if c.proportions.x > 0.0 {
        0.25 + c.proportions.w * 0.75
    } else {
        1.0
    };
    let chord = dimensions.y * development.sqrt();
    let reach = dimensions.x * development * (1.0 - c.animation.y * 0.65);
    let old_reach = dimensions.x * development * (1.0 - c.previous_animation.y * 0.65);
    // The stroke turns each wing about its root, from a slight upward
    // rest; the tip lags behind, curling against the stroke.
    let stroke = 0.1 + c.animation.x * 2.0;
    let old_stroke = 0.1 + c.previous_animation.x * 2.0;
    Sheet {
        center: root + vec3(side * reach * 0.5, 0.0, reach * 0.12),
        previous_center: root + vec3(side * old_reach * 0.5, 0.0, old_reach * 0.12),
        root,
        stroke: Vec2::new(side * stroke, side * old_stroke),
        radii: Vec2::new(reach * 0.52, chord),
        bend: Vec2::new(
            reach * (0.02 - c.animation.x * 0.04),
            old_reach * (0.02 - c.previous_animation.x * 0.04),
        ),
        camber: 0.07,
        side,
    }
}

pub fn visit_paired_sheets<F: FnMut(Hit)>(
    ray: Ray,
    c: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    pigment: Vec4,
    mut emit: F,
) {
    for i in 0..2 {
        let side = if i == 0 { -1.0 } else { 1.0 };
        emit(sheet(
            ray,
            c,
            wing_sheet(c, root, dimensions, side),
            pigment,
        ));
    }
}

/// The frame a membrane wing is stretched over, in the creature's own
/// frame: a thick arm from a fleshy root out to the wrist a third of the
/// way along, a long finger on along the leading edge to the tip, and three
/// fingers fanning back from the wrist to the points of the trailing edge
/// between its scallops. A feathered wing has the arm and leading finger
/// alone; its feathers carry the rest.
pub fn wing_bones(
    p: Vec3,
    c: &CreatureInstance,
    root: Vec3,
    dimensions: Vec2,
    feathered: bool,
) -> f32 {
    let mut distance = 1e6f32;
    for i in 0..2 {
        let side = if i == 0 { -1.0 } else { 1.0 };
        let shape = wing_sheet(c, root, dimensions, side);
        let rx = shape.radii.x;
        let chord = shape.radii.y;
        let offset = shape.center - shape.root;
        let curvature = shape.bend.x / (rx * rx);
        let arch = -shape.camber / chord;
        // Far from this wing, its bounding sphere is distance enough.
        let middle = shape.root + roll(offset, shape.stroke.x);
        // Stopping a chord short of the sphere, so its own surface is never
        // mistaken for the wing's.
        let bound = (p - middle).length() - (rx * 1.15 + chord);
        if bound > chord {
            distance = distance.min(bound);
            continue;
        }
        let at = |along: f32, v: f32| {
            let x = offset.x + (2.0 * along - 1.0) * rx * side;
            let z =
                offset.z + (along * along - 0.25) * rx * 0.55 + v * chord * (1.0 - 0.55 * along);
            let across = z - offset.z;
            shape.root
                + roll(
                    vec3(x, curvature * x * x + arch * across * across, z),
                    shape.stroke.x,
                )
        };
        let tip = |along: f32| (1.0 - along.powi(6)).max(0.0).sqrt();
        let lead = |along: f32| -0.62 * tip(along) * 0.9;
        let trail = |along: f32| (0.4 + 0.75 * (1.0 - along).max(0.0).sqrt()) * tip(along) * 0.95;
        let bone = |a: Vec3, b: Vec3, ra: f32, rb: f32| {
            let axis = b - a;
            let t = ((p - a).dot(axis) / axis.length_squared().max(1e-8)).saturate();
            ((p - a - axis * t).length() - ra.lerp(rb, t)) * 0.9
        };
        let wrist = 0.33;
        let mut frame = bone(
            at(0.0, lead(0.0)),
            at(wrist, lead(wrist)),
            chord * 0.1,
            chord * 0.05,
        );
        for segment in 0..3 {
            let a = wrist + (0.97 - wrist) * segment as f32 / 3.0;
            let b = wrist + (0.97 - wrist) * (segment + 1) as f32 / 3.0;
            frame = frame.min(bone(
                at(a, lead(a)),
                at(b, lead(b)),
                chord * (0.05 - segment as f32 * 0.012),
                chord * (0.038 - segment as f32 * 0.012),
            ));
        }
        if !feathered {
            for finger in 0..3 {
                let end = 0.444 + finger as f32 * 0.2225;
                let mid = (wrist + end) * 0.5;
                let start = at(wrist, lead(wrist));
                let bow = at(mid, (lead(mid) + trail(mid)) * 0.5);
                let tip = at(end, trail(end));
                frame = frame.min(bone(start, bow, chord * 0.03, chord * 0.022));
                frame = frame.min(bone(bow, tip, chord * 0.022, chord * 0.008));
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
                bend: Vec2::ONE,
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
