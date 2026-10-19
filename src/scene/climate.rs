//! The shared climate: regional weather, and the weather systems that carry
//! clouds from the equator to the poles. All of it is analytic, so the game
//! and the renderer always agree on where the clouds are.
use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU};

use spirv_std::glam::{
    FloatExt, IVec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles, ivec3, vec2, vec3, vec4,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

use crate::scene::PLANET_RADIUS;
use crate::scene::math::{cube, lattice, square};
use crate::scene::planet::surface_frame;

pub const WEATHER_SCALE: f32 = 1.0 / 256_000.0;

pub const WEATHER_SIZE: i32 = 64;

/// The cloud shell, from the rain beneath storms up to anvils and cirrus.
pub const CLOUD_BOTTOM: f32 = 800.0;

pub const CLOUD_TOP: f32 = 50_000.0;

/// Weather systems in flight at once.
pub const CLOUD_SLOTS: u32 = 8192;

/// Cloud map texels across one cube face. Each face carries a one texel
/// border so that filtering never reads across a face edge.
pub const CLOUD_FACE: u32 = 512;

pub const CLOUD_MAP_WIDTH: u32 = (CLOUD_FACE + 2) * 6;

pub const CLOUD_MAP_HEIGHT: u32 = CLOUD_FACE + 2;

/// Rain falls this far below the base of a deep storm.
pub const RAIN_DEPTH: f32 = 6000.0;

/// Texels across one cube face of the cloud light volume.
///
/// Over the same faces as the cloud map, it holds the cloud's shadow from the
/// pole lights and from the aur below at every altitude; its atlas is padded
/// in width for buffer copies.
pub const LIGHT_FACE: u32 = 128;

pub const LIGHT_WIDTH: u32 = 800;

pub const LIGHT_HEIGHT: u32 = LIGHT_FACE + 2;

pub const LIGHT_LAYERS: u32 = 16;

/// Parcel index cells across one cube face.
pub const CLOUD_GRID: u32 = 24;

pub const CLOUD_CELLS: usize = (CLOUD_GRID * CLOUD_GRID * 6) as usize;

pub const CUMULUS: u32 = 0;

pub const TOWER: u32 = 1;

pub const DECK: u32 = 2;

pub const CIRRUS: u32 = 3;

pub const CYCLONE: u32 = 4;

const CYCLONE_OUTFLOW: f32 = 43_000.0;

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

/// One weather system: a field of cumulus, a lone thunderhead and its anvil,
/// a stratus deck, a cirrus sheet or a hurricane. Each is born near the
/// equator, carried towards its pole and dissolved there.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudParcel {
    /// Unit direction to the centre, and the radius of everything it holds.
    pub centre: Vec4,
    /// Kind, strength, base altitude, top altitude.
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
    base: f32,
    top: f32,
    deck: f32,
    deck_altitude: f32,
    thickness: f32,
    cirrus: f32,
    cirrus_altitude: f32,
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

    pub fn weather_coordinates(point: Vec3, seconds: f32) -> Vec3 {
        (point.normalize() * PLANET_RADIUS - vec3(1.2, -0.7, 0.9) * seconds) * WEATHER_SCALE
    }

    pub fn weather_on_planet(point: Vec3, seconds: f32) -> Vec4 {
        let p = Self::weather_coordinates(point, seconds);
        weather_at(p)
    }

    pub fn weather_uv(position: Vec3, seconds: f32) -> Vec2 {
        (position.xy() - vec2(1.2, -0.7) * seconds) * WEATHER_SCALE
    }

    pub fn sample(position: Vec3, seconds: f32, latitude: f32) -> Self {
        Self::from_weather(
            position,
            seconds,
            latitude,
            Vec3::Y,
            weather_at(Self::weather_uv(position, seconds).extend(0.0)),
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
        let (base, top, thickness) = (self.shape.z, self.shape.w, self.detail.z);
        match self.kind() {
            CUMULUS => vec4(base, top, 1.0e9, -1.0e9),
            TOWER => vec4(
                (base - RAIN_DEPTH).max(CLOUD_BOTTOM),
                top + thickness * 0.3,
                1.0e9,
                -1.0e9,
            ),
            DECK => vec4(
                base - thickness * 0.8,
                base + thickness * 0.8,
                1.0e9,
                -1.0e9,
            ),
            CIRRUS => vec4(1.0e9, -1.0e9, base - 1800.0, base + 1800.0),
            _ => vec4(
                (base - RAIN_DEPTH).max(CLOUD_BOTTOM),
                top,
                CYCLONE_OUTFLOW - 1800.0,
                CYCLONE_OUTFLOW + 1800.0,
            ),
        }
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
        let (base, top, thickness) = (parcel.shape.z, parcel.shape.w, parcel.detail.z);
        match parcel.kind() {
            CUMULUS => {
                self.convective(
                    fall * strength * 0.6,
                    base,
                    base + (top - base) * (0.35 + 0.65 * fall),
                    false,
                );
            }
            TOWER => {
                // Steep walls and a flat crown at the anvil, not a mound.
                let core = 1.0 - distance.smoothstep(parcel.detail.x * 0.6, parcel.detail.x);
                self.convective(
                    core * strength * 1.5,
                    base,
                    base + (top - base) * core.smoothstep(0.0, 0.3),
                    true,
                );
            }
            DECK => self.deck(fall * strength * 0.75, base, thickness),
            CIRRUS => self.cirrus(fall * strength * 0.7, base),
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
                    base,
                    base + (top - base) * lift,
                    true,
                );
                self.deck(
                    (1.0 - q.smoothstep(0.1, 0.42)) * eye * strength,
                    22_000.0,
                    thickness,
                );
                self.cirrus(
                    q.smoothstep(0.15, 0.4)
                        * (1.0 - q.smoothstep(0.75, 1.0))
                        * (0.6 + arms * 0.4)
                        * strength,
                    CYCLONE_OUTFLOW,
                );
            }
        }
    }

    fn convective(&mut self, weight: f32, base: f32, top: f32, tower: bool) {
        self.convective += weight;
        if tower {
            self.towers += weight;
        }
        self.base += weight * base;
        if weight > 0.01 {
            self.top = self.top.max(top);
        }
    }

    fn deck(&mut self, weight: f32, altitude: f32, thickness: f32) {
        self.deck += weight;
        self.deck_altitude += weight * altitude;
        self.thickness += weight * thickness;
    }

    fn cirrus(&mut self, weight: f32, altitude: f32) {
        self.cirrus += weight;
        self.cirrus_altitude += weight * altitude;
    }

    /// The two cloud map texels: coverage of the convective layer and its
    /// top, of the deck and of the cirrus; then the convective base, the
    /// deck's altitude and thickness, and the cirrus altitude. Overlapping
    /// systems thicken the cover towards overcast without a hard ceiling.
    pub fn texels(self) -> [Vec4; 2] {
        let saturate = |weight: f32| 1.0 - (-weight * 1.3).exp();
        [
            vec4(
                // Fields of cumulus stay broken however many overlap.
                saturate(self.convective - self.towers).min(0.45) + saturate(self.towers) * 0.55,
                self.top,
                saturate(self.deck),
                saturate(self.cirrus),
            ),
            vec4(
                mean(self.base, self.convective, 6000.0),
                mean(self.deck_altitude, self.deck, 20_000.0),
                mean(self.thickness, self.deck, 2000.0),
                mean(self.cirrus_altitude, self.cirrus, 40_000.0),
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

fn mean(sum: f32, weight: f32, empty: f32) -> f32 {
    if weight > 0.0001 { sum / weight } else { empty }
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
    let kind = if seed.x < 0.45 {
        CUMULUS
    } else if seed.x < 0.65 {
        TOWER
    } else if seed.x < 0.81 {
        DECK
    } else if seed.x < 0.992 {
        CIRRUS
    } else {
        CYCLONE
    };
    let (reach, core, base, top, thickness, strength) = match kind {
        CUMULUS => {
            let development = (form.z * 0.75 + humid * 0.45).min(1.0);
            let base = 4500.0 + form.y * form.y * 12_000.0;
            (
                (40_000.0 + form.x * 60_000.0) * (0.8 + humid * 0.4),
                0.0,
                base,
                base + 1500.0 + development.powf(1.5) * 15_000.0,
                0.0,
                low,
            )
        }
        TOWER => {
            // Thunderheads build, mature and collapse again as they drift.
            let maturity = 0.5 + 0.5 * (progress * TAU * 5.0 + seed.w * TAU).sin();
            let core = 9000.0 + form.x * 14_000.0;
            (
                // At least as far as its anvil spreads, so every cell its
                // anvil covers lists it.
                core * (1.6 + form.w * 1.6 * maturity),
                core,
                5000.0 + form.y * 4000.0,
                22_000.0 + form.z * 26_000.0 * (0.55 + 0.45 * maturity),
                2500.0 + form.w * 3500.0,
                low * (0.6 + 0.4 * humid),
            )
        }
        DECK => (
            60_000.0 + form.x * 90_000.0,
            0.0,
            9000.0 + form.y * 24_000.0,
            0.0,
            700.0 + form.z * 3500.0,
            low * 0.85,
        ),
        CIRRUS => (
            60_000.0 + form.x * 100_000.0,
            0.0,
            34_000.0 + form.y * 13_000.0,
            0.0,
            1200.0 + form.z * 2400.0,
            strength * 0.8,
        ),
        _ => {
            // Hurricanes spin up over the warm polar seas and die at the pole.
            let growth = progress.smoothstep(0.3, 0.6);
            (
                (260_000.0 + form.x * 320_000.0) * (0.25 + 0.75 * growth),
                0.0,
                5500.0,
                34_000.0 + form.y * 12_000.0,
                5000.0,
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
    layer / (LIGHT_LAYERS - 1) as f32 * CLOUD_TOP
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
