use crate::scene::{
    PLANET_RADIUS,
    math::{cube, lattice, square},
    planet::surface_frame,
};
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU};
use spirv_std::glam::{
    FloatExt, IVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, ivec3, vec2, vec3, vec4,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub const WEATHER_SCALE: f32 = 1.0 / 256_000.0;

pub const WEATHER_SIZE: i32 = 64;

/// The cloud shell, from the rain beneath storms up to anvils and cirrus.
pub const CLOUD_BOTTOM: f32 = 187.5;

pub const CLOUD_TOP: f32 = 13_750.0;

/// The cloud layers, one genus to each: from 2 km up to 12 km, at even
/// spacing.
///
/// 0. cumulus, heaping up into congestus where it is unstable;
/// 1. stratocumulus, lumpy sheets broken into rolls and cells;
/// 2. altocumulus, fields of small rounded puffs;
/// 3. altostratus and lenticular lenses, smooth and broad;
/// 4. altocumulus castellanus, small turrets on a common base;
/// 5. cirrocumulus, fine rippled grains;
/// 6. cirrus, combed into fibres and hooks by the wind aloft.
///
/// Cumulonimbus towers rise through all of them.
pub const LAYERS: u32 = 7;

pub const LAYER_BASE: f32 = 2000.0;

pub const LAYER_SPACING: f32 = 10_000.0 / 6.0;

/// How far a cloud's base strays above or below its layer's, regionally
/// and from one cloud to the next, so the layers never read as flat sheets.
pub const LAYER_DRIFT: f32 = 525.0;

pub fn layer_altitude(layer: u32) -> f32 {
    LAYER_BASE + layer as f32 * LAYER_SPACING
}

/// The deepest a layer's cloud grows above its base. Cumulus grows as deep
/// as its column's convection allows, up to this.
pub const fn layer_depth(layer: u32) -> f32 {
    match layer {
        0 => 3750.0,
        1 | 3 => 625.0,
        2 => 450.0,
        4 => 750.0,
        5 => 300.0,
        _ => 500.0,
    }
}

/// The layer nearest an altitude.
pub fn nearest_layer(altitude: f32) -> u32 {
    ((altitude - LAYER_BASE) / LAYER_SPACING)
        .round()
        .clamp(0.0, (LAYERS - 1) as f32) as u32
}

/// Weather systems in flight at once.
pub const CLOUD_SLOTS: u32 = 8192;

/// Cloud map texels across one cube face. Each face carries a one texel
/// border so that filtering never reads across a face edge.
pub const CLOUD_FACE: u32 = 512;

pub const CLOUD_MAP_WIDTH: u32 = (CLOUD_FACE + 2) * 6;

pub const CLOUD_MAP_HEIGHT: u32 = CLOUD_FACE + 2;

/// Rain falls this far below the base of a deep storm, nearly to the aur.
pub const RAIN_DEPTH: f32 = 1625.0;

/// Texels across one cube face of the cloud light volume.
///
/// Over the same faces as the cloud map, it holds the cloud's shadow from the
/// pole lights and from the aur below at every altitude; its atlas is padded
/// in width for buffer copies.
pub const LIGHT_FACE: u32 = 128;

pub const LIGHT_WIDTH: u32 = 800;

pub const LIGHT_HEIGHT: u32 = LIGHT_FACE + 2;

pub const LIGHT_LAYERS: u32 = 32;

/// The light volume's lowest layer, below every cloud base.
///
/// Beneath it the clouds' shadows only travel along the light, so points
/// there look up the shadow where their ray towards the light crosses this
/// altitude.
pub const LIGHT_BOTTOM: f32 = LAYER_BASE - LAYER_DRIFT;

/// Parcel index cells across one cube face.
pub const CLOUD_GRID: u32 = 24;

pub const CLOUD_CELLS: usize = (CLOUD_GRID * CLOUD_GRID * 6) as usize;

pub const CUMULUS: u32 = 0;

pub const TOWER: u32 = 1;

pub const DECK: u32 = 2;

pub const CIRRUS: u32 = 3;

pub const CYCLONE: u32 = 4;

/// A front: a stack of layers sloping up ahead of it, from stratocumulus
/// through altostratus to the cirrus that heralds it.
pub const FRONT: u32 = 5;

/// Seconds for a typical system to reach its pole.
const JOURNEY: f32 = 64_000.0;

/// Systems have dissolved by this latitude.
const POLAR_LIMIT: f32 = 1.48;

const POLEWARD_SPEED: f32 = (POLAR_LIMIT - 0.07) * PLANET_RADIUS / (JOURNEY * 1.05);

const EASTWARD_SPEED: f32 = 1.05 * PLANET_RADIUS / (JOURNEY * 1.05);

/// How strongly systems quicken polewards: latitude goes as progress to
/// this power.
const ACCELERATION: f32 = 1.5;

#[derive(Clone, Copy, Debug)]
pub struct Climate {
    pub wind: Vec3,
    pub altitude: f32,
    pub pressure: f32,
    pub humidity: f32,
    pub light: f32,
    pub aur: f32,
    pub storm: f32,
    pub temperature: f32,
    pub precipitation: f32,
    pub ice: f32,
    pub low_cloud: f32,
    pub middle_cloud: f32,
    pub high_cloud: f32,
}

/// One weather system, born near the equator, carried towards its pole and
/// dissolved there.
///
/// It is a field of cumulus, a lone thunderhead and its anvil, a deck in one
/// of the middle layers, a cirrus sheet, a front or a hurricane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParcel {
    /// Unit direction to the centre, and the radius of everything it holds.
    pub centre: Vec4,
    /// Kind, strength, base altitude and top altitude; a deck or cirrus
    /// sheet holds its layer in place of a top.
    pub shape: Vec4,
    /// Core radius, spin phase, layer thickness and a stable seed.
    pub detail: Vec4,
}

/// Cloud cover over one column, accumulated from the parcels above it.
#[derive(Clone, Copy, Default)]
pub struct CloudCover {
    convective: f32,
    /// The share of the convective cover from thunderhead cores, the only
    /// convection dense enough to close into overcast.
    towers: f32,
    top: f32,
    /// The weight of the layers above the cumulus: 1 to 4, then 5 and 6.
    middle: Vec4,
    upper: Vec2,
}

impl Climate {
    pub fn at(position: Vec3, seconds: f32, weather: Vec4) -> Self {
        Self::from_weather(
            position,
            seconds,
            FRAC_PI_2 - position.y / PLANET_RADIUS,
            -Vec3::Y,
            weather,
        )
    }

    pub fn on_planet(point: Vec3, seconds: f32, weather: Vec4) -> Self {
        let radius = point.length();
        let up = point / radius;
        let mut climate = Self::from_polar(
            Vec3::Z * (radius - PLANET_RADIUS),
            seconds,
            up.z,
            Vec3::Y,
            weather,
        );
        let east = vec3(-up.y, up.x, 0.0);
        let north = Vec3::Z - up * up.z;
        climate.wind = east * climate.wind.x + north * climate.wind.y + up * climate.wind.z;
        climate
    }

    pub fn weather_on_planet(point: Vec3, seconds: f32) -> Vec4 {
        weather_at(
            (point.normalize() * PLANET_RADIUS - vec3(1.2, -0.7, 0.9) * seconds) * WEATHER_SCALE,
        )
    }

    pub fn sample(position: Vec3, seconds: f32, latitude: f32) -> Self {
        Self::from_weather(
            position,
            seconds,
            latitude,
            Vec3::Y,
            weather_at(((position.xy() - vec2(1.2, -0.7) * seconds) * WEATHER_SCALE).extend(0.0)),
        )
    }

    pub fn from_weather(
        position: Vec3,
        seconds: f32,
        latitude: f32,
        north: Vec3,
        weather: Vec4,
    ) -> Self {
        Self::from_polar(position, seconds, latitude.sin(), north, weather)
    }

    fn from_polar(
        position: Vec3,
        seconds: f32,
        latitude_sine: f32,
        north: Vec3,
        weather: Vec4,
    ) -> Self {
        let activity = 0.1 + 0.9 * (0.5 + 0.5 * (seconds * TAU / 2400.0).cos());
        let altitude = position.z.max(0.0);
        let polar = latitude_sine.abs();
        let sinking = polar.smoothstep(0.94, 1.0);
        let moisture = (weather.x + (1.0 - polar) * 0.025).smoothstep(0.28, 0.72);
        let convection =
            (weather.y + (1.0 - polar * polar) * 0.08 - sinking * 0.15).smoothstep(0.32, 0.74);
        let humidity =
            (moisture * (0.7 + weather.w * 0.3) * (1.0 - sinking * 0.75)).clamp(0.02, 1.0);
        let storm = moisture.smoothstep(0.55, 0.95) * convection.smoothstep(0.65, 0.95);
        let upper = altitude.smoothstep(9000.0, 42_000.0);
        let basal_lift = (8.0 + convection * 4.0) * (-altitude / 1800.0).exp();
        let vertical = (basal_lift + (1.6 + convection * 3.0) * (1.0 - upper) + storm * 5.0)
            * (0.65 + activity * 0.35)
            - sinking * (4.0 + upper * 12.0);
        let direction = north * latitude_sine;
        let wind = direction * ((4.0 + polar * 45.0) * upper)
            + vec3((weather.z - 0.5) * 5.0, 0.0, vertical);
        let temperature = 284.0 + polar * 25.0 - altitude * 0.0055;
        let ice = 1.0 - temperature.smoothstep(248.0, 273.0);
        Self {
            wind,
            altitude: position.z,
            pressure: (-altitude / 11_000.0).exp(),
            humidity,
            light: activity * (0.08 + cube(polar) * 0.92) * (1.0 - moisture * 0.35),
            aur: activity * (-altitude / 1800.0).exp(),
            storm,
            temperature,
            precipitation: (storm + moisture.smoothstep(0.75, 1.0) * 0.2) * (1.0 - ice * 0.7),
            ice,
            low_cloud: humidity.smoothstep(0.25, 0.8) * (0.35 + convection * 0.65),
            middle_cloud: moisture.smoothstep(0.5, 0.95) * (1.0 - storm * 0.5),
            high_cloud: weather.z.smoothstep(0.43, 0.7) * (0.25 + moisture * 0.75),
        }
    }
}

impl CloudParcel {
    pub fn kind(self) -> u32 {
        self.shape.x as u32
    }

    /// Altitudes its cloud can fill below the cirrus level, and in it.
    pub fn extent(self) -> Vec4 {
        let (base, top) = (self.shape.z, self.shape.w);
        let band = |layer: u32| {
            vec2(
                layer_altitude(layer) - LAYER_DRIFT,
                layer_altitude(layer) + layer_depth(layer) + LAYER_DRIFT,
            )
        };
        let none = vec2(1.0e9, -1.0e9);
        let (low, high) = match self.kind() {
            CUMULUS => (vec2(base - LAYER_DRIFT, top + LAYER_DRIFT), none),
            TOWER => (
                vec2(
                    (base - RAIN_DEPTH).max(CLOUD_BOTTOM),
                    top + self.detail.z + LAYER_DRIFT,
                ),
                none,
            ),
            DECK => (band(top as u32), none),
            CIRRUS => (none, band(top as u32)),
            FRONT => (vec2(band(1).x, band(4).y), vec2(band(5).x, band(6).y)),
            _ => (
                vec2((base - RAIN_DEPTH).max(CLOUD_BOTTOM), top + LAYER_DRIFT),
                band(6),
            ),
        };
        low.extend(high.x).extend(high.y)
    }
}

impl CloudCover {
    pub fn add(&mut self, parcel: CloudParcel, up: Vec3) {
        let centre = parcel.centre.xyz();
        let strength = parcel.shape.y;
        let distance = ((up - centre).length() * 0.5).min(1.0).asin() * 2.0 * PLANET_RADIUS;
        if strength <= 0.0 || distance >= parcel.centre.w {
            return;
        }
        let q = distance / parcel.centre.w;
        let fall = 1.0 - q.smoothstep(0.45, 1.0);
        let (base, top) = (parcel.shape.z, parcel.shape.w);
        match parcel.kind() {
            CUMULUS => {
                self.convective(
                    fall * strength * 0.6,
                    base + (top - base) * (0.35 + 0.65 * fall),
                    false,
                );
            }
            TOWER => {
                // Steep walls and a flat crown at the anvil, not a mound.
                let core = parcel.detail.x;
                let core = 1.0 - distance.smoothstep(core * 0.6, core);
                self.convective(
                    core * strength * 1.5,
                    base + (top - base) * core.smoothstep(0.0, 0.3),
                    true,
                );
            }
            DECK | CIRRUS => self.layer(top as u32, fall * strength * 0.4),
            FRONT => {
                // Cirrus runs far ahead of the front, altostratus thickens
                // towards it and stratocumulus trails behind.
                let frame = surface_frame(centre);
                let forward = if centre.z < 0.0 {
                    -frame.y_axis
                } else {
                    frame.y_axis
                };
                let ahead = (up - centre).dot(forward) * PLANET_RADIUS / parcel.centre.w;
                let across = 1.0 - q.smoothstep(0.6, 1.0);
                let band = |from: f32, to: f32| {
                    ahead.smoothstep(from - 0.25, from) * (1.0 - ahead.smoothstep(to, to + 0.25))
                };
                let weight = strength * across * 0.5;
                self.layer(6, band(-0.1, 0.95) * weight * 0.8);
                self.layer(5, band(-0.2, 0.55) * weight * 0.45);
                self.layer(4, band(-0.45, 0.25) * weight * 0.4);
                self.layer(3, band(-0.6, 0.35) * weight);
                self.layer(2, band(-0.85, -0.1) * weight * 0.6);
                self.layer(1, band(-1.0, -0.3) * weight * 0.9);
            }
            _ => {
                // Two logarithmic rain bands spiral into an eyewall of towers
                // around a clear eye, beneath a dense overcast and an outflow
                // of cirrus that fans out far beyond the storm.
                let frame = surface_frame(centre);
                let offset = up - centre;
                let spin = if centre.z < 0.0 { -1.0 } else { 1.0 };
                let angle = offset.dot(frame.y_axis).atan2(offset.dot(frame.x_axis)) * spin;
                let eye = q.smoothstep(0.03, 0.065);
                let wall = (-square((q - 0.09) / 0.035)).exp();
                let arms =
                    cube(0.5 + 0.5 * (angle * 2.0 + (q + 0.02).ln() * 5.5 - parcel.detail.y).cos());
                let bands = arms * q.smoothstep(0.1, 0.2) * (1.0 - q.smoothstep(0.35, 0.95));
                let lift = (wall + bands * 0.55).min(1.0);
                self.convective(
                    (wall * 1.6 + bands * 1.1) * eye * strength,
                    base + (top - base) * lift,
                    true,
                );
                self.layer(3, (1.0 - q.smoothstep(0.1, 0.42)) * eye * strength * 0.6);
                self.layer(
                    6,
                    q.smoothstep(0.15, 0.4)
                        * (1.0 - q.smoothstep(0.75, 1.0))
                        * (0.6 + arms * 0.4)
                        * strength,
                );
            }
        }
    }

    fn convective(&mut self, weight: f32, top: f32, tower: bool) {
        self.convective += weight;
        if tower {
            self.towers += weight;
        }
        if weight > 0.01 {
            self.top = self.top.max(top);
        }
    }

    fn layer(&mut self, layer: u32, weight: f32) {
        // Added through a mask rather than a chosen field: shaders cannot
        // pick between pointers.
        let at = |index: u32| if layer == index { weight } else { 0.0 };
        self.middle += vec4(at(1), at(2), at(3), at(4));
        self.upper += vec2(at(5), at(6));
    }

    /// The two cloud map texels: the cover of each layer, then the top of
    /// the convection. Overlapping systems thicken the cover towards
    /// overcast without a hard ceiling.
    pub fn texels(self) -> [Vec4; 2] {
        let saturate = |weight: f32| 1.0 - (-weight * 1.3).exp();
        // The layers above the cumulus close into overcast only where
        // several systems pile up.
        let layered = |weight: f32| 1.0 - (-weight).exp();
        // Fields of cumulus stay broken however many overlap.
        let convective =
            saturate(self.convective - self.towers).min(0.45) + saturate(self.towers) * 0.55;
        let (middle, upper) = (self.middle, self.upper);
        [
            vec4(
                convective,
                layered(middle.x),
                layered(middle.y),
                layered(middle.z),
            ),
            vec4(
                layered(middle.w),
                layered(upper.x),
                layered(upper.y),
                self.top,
            ),
        ]
    }
}

const fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn slot_random(slot: u32, cycle: u32) -> Vec4 {
    let a = mix(slot.wrapping_mul(0x9e37_79b9) ^ cycle.wrapping_mul(0x85eb_ca6b));
    let b = mix(a);
    let c = mix(b);
    let d = mix(c);
    vec4(
        (a >> 8) as f32,
        (b >> 8) as f32,
        (c >> 8) as f32,
        (d >> 8) as f32,
    ) / 16_777_216.0
}

/// How readily the equator raises weather at a longitude. Wet sectors feed
/// streams of cloud towards the poles; dry ones leave clear corridors.
pub fn equatorial_supply(longitude: f32, time: f32) -> f32 {
    let drift = time * 0.000_02;
    let wave = 0.5
        + 0.3 * (longitude * 3.0 + drift).sin()
        + 0.2 * (longitude * 7.0 - drift * 2.3 + 1.3).sin()
        + 0.12 * (longitude * 13.0 + drift * 4.1 + 0.4).sin();
    0.12 + 0.88 * wave.smoothstep(0.3, 0.75)
}

/// The weather system flying in a slot at a time. Its whole life is a pure
/// function of the slot and time, so any moment can be shown at once.
#[expect(
    clippy::too_many_lines,
    reason = "every kind of weather system is formed here"
)]
pub fn cloud_parcel(slot: u32, time: f32) -> CloudParcel {
    let seed = slot_random(slot, 0);
    let journey = JOURNEY * (0.8 + seed.y * 0.5);
    let cycle_time = time / journey + seed.z;
    let cycle = cycle_time.floor();
    let progress = cycle_time - cycle;
    let birth = slot_random(slot, cycle as u32 + 1);
    let form = slot_random(slot ^ 0x5bd1_e995, cycle as u32 + 1);
    let born = time - progress * journey;
    let hemisphere = if birth.y < 0.5 { -1.0 } else { 1.0 };
    // Systems drift slowly out of the tropics and quicken towards the pole.
    let latitude = hemisphere * (birth.w * 0.14).lerp(POLAR_LIMIT, progress.powf(ACCELERATION));
    let longitude = birth.x * TAU
        + (0.6 + seed.w * 0.9) * progress
        + 0.05 * (progress * 9.0 + seed.x * TAU).sin();
    let centre = vec3(
        latitude.cos() * longitude.cos(),
        latitude.cos() * longitude.sin(),
        latitude.sin(),
    );
    let active = birth.z < equatorial_supply(birth.x * TAU, born);
    let life = progress.smoothstep(0.0, 0.05) * (1.0 - progress.smoothstep(0.93, 1.0));
    // The converging flow gathers moisture: systems swell and deepen polewards.
    let humid = latitude.abs().sin();
    let strength = if active {
        life * (0.7 + humid * 0.45)
    } else {
        0.0
    };
    // The young tropical air holds little low cloud; mostly high cirrus.
    let tropical = latitude.abs().smoothstep(0.15, 0.6);
    let low = strength * (0.2 + 0.8 * tropical);
    let kind = if seed.x < 0.36 {
        CUMULUS
    } else if seed.x < 0.52 {
        TOWER
    } else if seed.x < 0.72 {
        DECK
    } else if seed.x < 0.88 {
        CIRRUS
    } else if seed.x < 0.992 {
        FRONT
    } else {
        CYCLONE
    };
    let (reach, core, base, top, thickness, strength) = match kind {
        CUMULUS => {
            let development = (form.z * 0.75 + humid * 0.45).min(1.0);
            let base = layer_altitude(0);
            (
                (40_000.0 + form.x * 60_000.0) * (0.8 + humid * 0.4),
                0.0,
                base,
                base + 375.0 + development.powf(1.5) * 3250.0,
                0.0,
                low,
            )
        }
        TOWER => {
            // Thunderheads build, mature and collapse again as they drift.
            let maturity = 0.5 + 0.5 * (progress * TAU * 5.0 + seed.w * TAU).sin();
            let core = 1750.0 + form.x * 2000.0;
            (
                // At least as far as its anvil spreads, so every cell its
                // anvil covers lists it.
                core * (1.6 + form.w * 1.6 * maturity),
                core,
                layer_altitude(0),
                7250.0 + form.z * 5000.0 * (0.55 + 0.45 * maturity),
                500.0 + form.w * 625.0,
                low * (0.6 + 0.4 * humid),
            )
        }
        DECK => {
            let layer = 1 + (form.y * 4.0).min(3.0) as u32;
            (
                60_000.0 + form.x * 90_000.0,
                0.0,
                layer_altitude(layer),
                layer as f32,
                layer_depth(layer),
                low * 0.85,
            )
        }
        CIRRUS => {
            let layer = if form.y < 0.35 { 5 } else { 6 };
            (
                35_000.0 + form.x * 70_000.0,
                0.0,
                layer_altitude(layer),
                layer as f32,
                layer_depth(layer),
                strength * 0.8,
            )
        }
        FRONT => (
            80_000.0 + form.x * 100_000.0,
            0.0,
            layer_altitude(1),
            layer_altitude(6),
            0.0,
            strength * (0.5 + 0.5 * tropical),
        ),
        _ => {
            // Hurricanes spin up over the warm polar seas and die at the pole.
            let growth = progress.smoothstep(0.3, 0.6);
            (
                (260_000.0 + form.x * 320_000.0) * (0.25 + 0.75 * growth),
                0.0,
                layer_altitude(0),
                10_000.0 + form.y * 2250.0,
                625.0,
                low * growth,
            )
        }
    };
    CloudParcel {
        centre: centre.extend(reach),
        shape: vec4(kind as f32, strength, base, top),
        detail: vec4(core, time * (0.0005 + form.w * 0.0004), thickness, seed.x),
    }
}

/// The large-scale flow carrying every weather system: polewards in each
/// hemisphere, deflected east. Metres per second, in planet coordinates.
pub fn cloud_wind(up: Vec3) -> Vec3 {
    let frame = surface_frame(up);
    let latitude = up.z.clamp(-1.0, 1.0).asin();
    // The speed of a system at this latitude, under its quickening journey.
    let quicken = ACCELERATION
        * (latitude.abs() / POLAR_LIMIT)
            .min(1.0)
            .powf(1.0 - 1.0 / ACCELERATION);
    let poleward = (latitude / 0.05).clamp(-1.0, 1.0) * POLEWARD_SPEED * quicken.max(0.3);
    frame.y_axis * poleward + frame.x_axis * (EASTWARD_SPEED * latitude.cos())
}

/// The cube face holding a direction, and its position across that face in
/// [-1, 1]², warped so that equal steps subtend nearly equal angles.
pub fn cube_face(up: Vec3) -> (u32, Vec2) {
    let a = up.abs();
    let (face, plane, depth) = if a.x >= a.y && a.x >= a.z {
        (u32::from(up.x < 0.0), vec2(up.y, up.z), a.x)
    } else if a.y >= a.z {
        (2 + u32::from(up.y < 0.0), vec2(up.x, up.z), a.y)
    } else {
        (4 + u32::from(up.z < 0.0), vec2(up.x, up.y), a.z)
    };
    let p = plane / depth;
    (face, vec2(p.x.atan(), p.y.atan()) / FRAC_PI_4)
}

pub fn cube_direction(face: u32, position: Vec2) -> Vec3 {
    let p = vec2(
        (position.x * FRAC_PI_4).tan(),
        (position.y * FRAC_PI_4).tan(),
    );
    let sign = if face & 1 == 0 { 1.0 } else { -1.0 };
    match face / 2 {
        0 => vec3(sign, p.x, p.y),
        1 => vec3(p.x, sign, p.y),
        _ => vec3(p.x, p.y, sign),
    }
    .normalize()
}

/// The parcel index cell holding a direction.
pub fn cloud_cell(up: Vec3) -> usize {
    let (face, position) = cube_face(up);
    let cell = ((position * 0.5 + 0.5) * CLOUD_GRID as f32)
        .floor()
        .clamp(Vec2::ZERO, Vec2::splat((CLOUD_GRID - 1) as f32));
    ((face * CLOUD_GRID + cell.y as u32) * CLOUD_GRID + cell.x as u32) as usize
}

/// Where a direction lies in an atlas of six bordered cube faces of `size`
/// texels, in texels.
pub fn atlas_texel(up: Vec3, size: u32) -> Vec2 {
    let (face, position) = cube_face(up);
    (position * 0.5 + 0.5) * size as f32 + vec2((face * (size + 2) + 1) as f32, 1.0)
}

/// The direction at the centre of an atlas texel, borders included.
pub fn atlas_direction(pixel: Vec2, size: u32) -> Vec3 {
    let face = ((pixel.x / (size + 2) as f32).floor() as u32).min(5);
    let local = pixel - vec2((face * (size + 2) + 1) as f32, 1.0);
    cube_direction(face, local / size as f32 * 2.0 - 1.0)
}

/// The altitude of a light volume layer.
pub fn light_altitude(layer: f32) -> f32 {
    LIGHT_BOTTOM + layer / (LIGHT_LAYERS - 1) as f32 * (CLOUD_TOP - LIGHT_BOTTOM)
}

pub fn weather_texel(cell: IVec3) -> Vec4 {
    let channel = |seed| (lattice(cell, WEATHER_SIZE, seed) * 255.0).round() / 255.0;
    vec4(
        channel(0x81),
        channel(0x81 + 1013),
        channel(0x81 + 2026),
        channel(0x81 + 3039),
    )
}

pub fn weather_at(point: Vec3) -> Vec4 {
    let cell = point.floor().as_ivec3();
    let fraction = point - point.floor();
    let mut weather = Vec4::ZERO;
    for index in 0..8 {
        let offset = ivec3(index & 1, (index >> 1) & 1, (index >> 2) & 1);
        let weight = Vec3::select(offset.cmpeq(IVec3::ZERO), Vec3::ONE - fraction, fraction);
        weather += weather_texel(cell + offset) * (weight.x * weight.y * weight.z);
    }
    weather
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basal_updraft_is_strongest_close_to_the_aur() {
        let wet = Vec4::splat(0.95);
        let surface = Climate::from_weather(Vec3::ZERO, 0.0, 0.0, Vec3::Y, wet);
        let upper = Climate::from_weather(Vec3::Z * 6000.0, 0.0, 0.0, Vec3::Y, wet);
        assert!(surface.wind.z > upper.wind.z + 8.0);
        let north = Climate::at(vec3(0.0, 800_000.0, 15_000.0), 0.0, wet);
        let south = Climate::at(
            vec3(0.0, PLANET_RADIUS * (FRAC_PI_2 * 2.0) - 800_000.0, 15_000.0),
            0.0,
            wet,
        );
        assert!(north.wind.y < 0.0 && south.wind.y > 0.0);
    }

    #[test]
    fn circulation_rises_at_equator_diverges_aloft_and_sinks_at_poles() {
        let weather = Vec4::splat(0.5);
        let at =
            |z, latitude| Climate::from_weather(vec3(0.0, 0.0, z), 0.0, latitude, Vec3::Y, weather);
        assert!(at(1000.0, 0.0).wind.z > 0.0);
        assert!(at(15_000.0, 0.8).wind.y > 0.0);
        assert!(at(15_000.0, -0.8).wind.y < 0.0);
        assert!(at(15_000.0, 1.57).wind.z < 0.0);
        assert!(at(15_000.0, 0.0).temperature < at(15_000.0, 1.57).temperature);
        let left = vec3(-2.0, 0.0, PLANET_RADIUS + 15_000.0);
        let right = vec3(2.0, 0.0, PLANET_RADIUS + 15_000.0);
        let a = Climate::weather_on_planet(left, 100.0);
        let b = Climate::weather_on_planet(right, 100.0);
        assert!(
            (a - b).abs().max_element() < 0.001,
            "weather seam at the pole"
        );
        let a = Climate::on_planet(left, 100.0, a);
        let b = Climate::on_planet(right, 100.0, b);
        assert!(
            a.wind.distance(b.wind) < 0.01,
            "wind discontinuity at the pole"
        );
    }

    #[test]
    fn moisture_and_instability_produce_distinct_vertical_cloud_regimes() {
        let at = |weather| Climate::from_weather(Vec3::Z * 2400.0, 0.0, 0.0, Vec3::Y, weather);
        let dry = at(Vec4::ZERO);
        assert!((dry.low_cloud + dry.middle_cloud + dry.high_cloud + dry.storm).abs() < 1e-6);
        let high = at(vec4(0.1, 0.1, 0.9, 0.5));
        assert!(high.high_cloud > 0.0 && high.low_cloud == 0.0);
        let wet = at(vec4(0.95, 0.95, 0.7, 0.95));
        assert!(wet.storm > 0.9);
    }
}
