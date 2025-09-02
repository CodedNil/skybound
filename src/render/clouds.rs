use bevy::{
    camera::primitives::{Frustum, Sphere},
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Buffer, BufferInitDescriptor, BufferUsages, ShaderType},
        renderer::{RenderDevice, RenderQueue},
    },
};
use rand::Rng;
use std::cmp::Ordering;

const MAX_VISIBLE: usize = 1024;

/// Stores the current state of all clouds in the main world.
#[derive(Resource)]
pub struct CloudsState {
    clouds: Vec<Cloud>,
}

/// Buffer for visible cloud data, extracted to the render world and the GPU
#[derive(Resource, ExtractResource, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CloudsBufferData {
    clouds: [Cloud; MAX_VISIBLE],
    total: u32,
}

/// Represents a cloud packed into exactly 16 bytes for GPU upload
#[derive(Default, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Cloud {
    /// u0 (bits 0..31):
    ///  - `len` (14 bits, bits 0..13): cloud length in meters, 0..16383
    ///  - `x_pos` (18 bits signed, bits 14..31): X world position, two's complement, range approx
    ///    -131072..131071
    d0: u32,
    /// u1 (bits 0..31):
    ///  - `height` (14 bits, bits 0..13): cloud vertical thickness in meters, 0..16383
    ///  - `y_pos` (18 bits signed, bits 14..31): Y world position, two's complement, range approx
    ///    -131072..131071
    d1: u32,
    /// u2 (bits 0..31): packed properties (exact bit widths):
    ///  - `seed` (7 bits, bits 0..6): integer seed used by noise/variation, 0..127
    ///  - `width` (5 bits, bits 7..11): width factor quantized to 32 steps (0..31) representing
    ///    0.0..1.0
    ///  - `form` (2 bits, bits 12..13): `CloudForm` enum
    ///    (0=cumulus,1=stratus,2=cirrus,3=cumulonimbus)
    ///  - `density` (4 bits, bits 14..17): overall fill 0..1 quantized in 16 steps
    ///  - `detail` (4 bits, bits 18..21): noise/detail level 0..1 quantized in 16 steps
    ///  - `brightness` (4 bits, bits 22..25): brightness 0..1 quantized in 16 steps
    ///  - `yaw` (6 bits, bits 26..31): horizontal rotation in 64 discrete steps (0..63)
    d2: u32,
    /// u3 (bits 0..31):
    ///  - `altitude` (15 bits, bits 0..14): cloud base altitude in meters, 0..32767
    ///  - remaining bits in u3 are unused/reserved
    d3: u32,
}

enum CloudForm {
    Cumulus,
    Stratus,
    Cirrus,
    Cumulonimbus,
}

#[allow(dead_code)]
impl Cloud {
    const ALT_BITS: u32 = 15;
    // 21
    // u3 layout (altitude stored entirely in u3)
    const ALT_HIGH_SHIFT: u32 = 0;
    const BRIGHTNESS_BITS: u32 = 4;
    // 15
    const BRIGHTNESS_SHIFT: u32 = Self::DETAIL_SHIFT + Self::DETAIL_BITS;
    const DENSITY_BITS: u32 = 4;
    // 10
    const DENSITY_SHIFT: u32 = Self::FORM_SHIFT + Self::FORM_BITS;
    const DETAIL_BITS: u32 = 4;
    // 12
    const DETAIL_SHIFT: u32 = Self::DENSITY_SHIFT + Self::DENSITY_BITS;
    const FORM_BITS: u32 = 2;
    // 6
    const FORM_SHIFT: u32 = Self::WIDTH_SHIFT + Self::SIZE_WIDTH_BITS;
    // in u0

    const HEIGHT_SHIFT: u32 = 0;
    // layout shifts
    const LEN_SHIFT: u32 = 0;
    // pos bits (signed)
    const POS_BITS: u32 = 18;
    // seed uses remaining bits in u2: 32 - (5+2+4+4+4+6) = 7
    const SEED_BITS: u32 = 7;
    // in u1

    // u2 layout shifts
    const SEED_SHIFT: u32 = 0;
    const SIZE_HEIGHT_BITS: u32 = 14;
    // size bits
    const SIZE_LEN_BITS: u32 = 14;
    const SIZE_WIDTH_BITS: u32 = Self::WIDTH_BITS;
    // fixed bit widths for u2 packing (sum must be 32)
    const WIDTH_BITS: u32 = 5;
    const WIDTH_SHIFT: u32 = Self::SEED_SHIFT + Self::SEED_BITS;
    // u0 low
    const X_SHIFT: u32 = Self::LEN_SHIFT + Self::SIZE_LEN_BITS;
    const YAW_BITS: u32 = 6;
    // 18
    const YAW_SHIFT: u32 = Self::BRIGHTNESS_SHIFT + Self::BRIGHTNESS_BITS;
    // u1 low
    const Y_SHIFT: u32 = Self::HEIGHT_SHIFT + Self::SIZE_HEIGHT_BITS;

    /// Return the maximum unsigned value representable with `bits` bits.
    #[inline]
    const fn max_for_bits(bits: u32) -> u32 {
        (1u32 << bits) - 1
    }

    /// Set an unsigned bitfield inside a u32 container.
    #[inline]
    const fn set_bits_field(container: &mut u32, value: u32, bits: u32, shift: u32) {
        let mask = Self::max_for_bits(bits) << shift;
        *container = (*container & !mask) | ((value & Self::max_for_bits(bits)) << shift);
    }

    /// Read an unsigned bitfield from a u32 container.
    #[inline]
    const fn get_bits_field(container: u32, bits: u32, shift: u32) -> u32 {
        (container >> shift) & Self::max_for_bits(bits)
    }

    // signed field helpers for POS_BITS
    /// Set a signed bitfield inside a u32 container using two's complement.
    #[inline]
    const fn set_signed_field(container: &mut u32, value: i32, bits: u32, shift: u32) {
        let mask = Self::max_for_bits(bits) << shift;
        let max_mask = Self::max_for_bits(bits);
        // clamp value to signed range
        let half = 1i32 << (bits - 1);
        let min = -half;
        let max = half - 1;
        let v = if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        };
        let encoded = (v as u32) & max_mask;
        *container = (*container & !mask) | (encoded << shift);
    }

    /// Read a signed bitfield from a u32 container interpreting two's complement.
    #[inline]
    const fn get_signed_field(container: u32, bits: u32, shift: u32) -> i32 {
        let raw = Self::get_bits_field(container, bits, shift);
        let sign_bit = 1u32 << (bits - 1);
        if (raw & sign_bit) != 0 {
            // negative
            (raw as i32) - (1i32 << bits)
        } else {
            raw as i32
        }
    }

    // pos: Vec3(x, y, altitude)
    /// Pack a Vec3 position into the Cloud bitfields.
    pub const fn set_pos(mut self, pos: Vec3) -> Self {
        // x and y are signed POS_BITS
        let x = pos.x.round() as i32;
        let y = pos.y.round() as i32;
        Self::set_signed_field(&mut self.d0, x, Self::POS_BITS, Self::X_SHIFT);
        Self::set_signed_field(&mut self.d1, y, Self::POS_BITS, Self::Y_SHIFT);
        let alt_raw = pos
            .z
            .clamp(0.0, Self::max_for_bits(Self::ALT_BITS) as f32)
            .round() as u32;
        Self::set_bits_field(&mut self.d3, alt_raw, Self::ALT_BITS, Self::ALT_HIGH_SHIFT);
        self
    }

    /// Unpack a Vec3 position from the Cloud bitfields.
    pub const fn get_pos(&self) -> Vec3 {
        let x = Self::get_signed_field(self.d0, Self::POS_BITS, Self::X_SHIFT) as f32;
        let y = Self::get_signed_field(self.d1, Self::POS_BITS, Self::Y_SHIFT) as f32;
        let alt_raw = Self::get_bits_field(self.d3, Self::ALT_BITS, Self::ALT_HIGH_SHIFT);
        Vec3::new(x, y, alt_raw as f32)
    }

    // size: Vec3(length_meters, height_meters, width_factor_0to1)
    /// Pack cloud size (len, width factor, height) into the bitfields.
    pub const fn set_size(mut self, size: Vec3) -> Self {
        let len_raw = size
            .x
            .clamp(0.0, Self::max_for_bits(Self::SIZE_LEN_BITS) as f32)
            .round() as u32;
        let width_raw = (size.y.clamp(0.0, 1.0)
            * (Self::max_for_bits(Self::SIZE_WIDTH_BITS) as f32))
            .round() as u32;
        let height_raw = size
            .z
            .clamp(0.0, Self::max_for_bits(Self::SIZE_HEIGHT_BITS) as f32)
            .round() as u32;

        Self::set_bits_field(&mut self.d0, len_raw, Self::SIZE_LEN_BITS, Self::LEN_SHIFT);
        Self::set_bits_field(
            &mut self.d1,
            height_raw,
            Self::SIZE_HEIGHT_BITS,
            Self::HEIGHT_SHIFT,
        );
        Self::set_bits_field(
            &mut self.d2,
            width_raw,
            Self::SIZE_WIDTH_BITS,
            Self::WIDTH_SHIFT,
        );
        self
    }

    /// Unpack cloud size into a Vec3 (len, width factor, height).
    pub const fn get_size(&self) -> Vec3 {
        let len_raw = Self::get_bits_field(self.d0, Self::SIZE_LEN_BITS, Self::LEN_SHIFT);
        let width_raw = Self::get_bits_field(self.d2, Self::SIZE_WIDTH_BITS, Self::WIDTH_SHIFT);
        let height_raw = Self::get_bits_field(self.d1, Self::SIZE_HEIGHT_BITS, Self::HEIGHT_SHIFT);
        Vec3::new(
            len_raw as f32,
            (width_raw as f32) / (Self::max_for_bits(Self::SIZE_WIDTH_BITS) as f32),
            height_raw as f32,
        )
    }

    /// Set the cloud form enum into the packed fields.
    pub const fn set_form(mut self, form: CloudForm) -> Self {
        let raw = match form {
            CloudForm::Cumulus => 0,
            CloudForm::Stratus => 1,
            CloudForm::Cirrus => 2,
            CloudForm::Cumulonimbus => 3,
        };
        Self::set_bits_field(&mut self.d2, raw, Self::FORM_BITS, Self::FORM_SHIFT);
        self
    }

    /// Read the cloud form enum from the packed fields.
    pub const fn get_form(&self) -> CloudForm {
        match Self::get_bits_field(self.d2, Self::FORM_BITS, Self::FORM_SHIFT) {
            0 => CloudForm::Cumulus,
            1 => CloudForm::Stratus,
            2 => CloudForm::Cirrus,
            _ => CloudForm::Cumulonimbus,
        }
    }

    /// Pack a normalized density (0..1) into the fields.
    pub const fn set_density(mut self, density: f32) -> Self {
        let raw = (density.clamp(0.0, 1.0) * (Self::max_for_bits(Self::DENSITY_BITS) as f32))
            .round() as u32;
        Self::set_bits_field(&mut self.d2, raw, Self::DENSITY_BITS, Self::DENSITY_SHIFT);
        self
    }

    /// Unpack normalized density (0..1) from the fields.
    pub const fn get_density(&self) -> f32 {
        Self::get_bits_field(self.d2, Self::DENSITY_BITS, Self::DENSITY_SHIFT) as f32
            / (Self::max_for_bits(Self::DENSITY_BITS) as f32)
    }

    /// Pack a normalized detail value (0..1) into the fields.
    pub const fn set_detail(mut self, detail: f32) -> Self {
        let raw = (detail.clamp(0.0, 1.0) * (Self::max_for_bits(Self::DETAIL_BITS) as f32)).round()
            as u32;
        Self::set_bits_field(&mut self.d2, raw, Self::DETAIL_BITS, Self::DETAIL_SHIFT);
        self
    }

    /// Unpack normalized detail (0..1) from the fields.
    pub const fn get_detail(&self) -> f32 {
        Self::get_bits_field(self.d2, Self::DETAIL_BITS, Self::DETAIL_SHIFT) as f32
            / (Self::max_for_bits(Self::DETAIL_BITS) as f32)
    }

    /// Pack a normalized brightness (0..1) into the fields.
    pub const fn set_brightness(mut self, brightness: f32) -> Self {
        let raw = (brightness.clamp(0.0, 1.0) * (Self::max_for_bits(Self::BRIGHTNESS_BITS) as f32))
            .round() as u32;
        Self::set_bits_field(
            &mut self.d2,
            raw,
            Self::BRIGHTNESS_BITS,
            Self::BRIGHTNESS_SHIFT,
        );
        self
    }

    /// Unpack normalized brightness (0..1) from the fields.
    pub const fn get_brightness(&self) -> f32 {
        Self::get_bits_field(self.d2, Self::BRIGHTNESS_BITS, Self::BRIGHTNESS_SHIFT) as f32
            / (Self::max_for_bits(Self::BRIGHTNESS_BITS) as f32)
    }

    /// Set quantized yaw value for the cloud.
    pub const fn set_yaw(mut self, yaw: u32) -> Self {
        Self::set_bits_field(&mut self.d2, yaw, Self::YAW_BITS, Self::YAW_SHIFT);
        self
    }

    /// Get quantized yaw value for the cloud.
    pub const fn get_yaw(&self) -> u32 {
        Self::get_bits_field(self.d2, Self::YAW_BITS, Self::YAW_SHIFT)
    }

    /// Set integer seed used by procedural variations.
    pub const fn set_seed(mut self, seed: u32) -> Self {
        Self::set_bits_field(&mut self.d2, seed, Self::SEED_BITS, Self::SEED_SHIFT);
        self
    }

    /// Get integer seed used by procedural variations.
    pub const fn get_seed(&self) -> u32 {
        Self::get_bits_field(self.d2, Self::SEED_BITS, Self::SEED_SHIFT)
    }
}

/// Create a randomized set of clouds and insert cloud resources.
pub fn setup_clouds(mut commands: Commands) {
    let mut rng = rand::rng();

    // --- Configurable Weather Variables ---
    let stratiform_chance = 0.15; // Chance for stratus clouds
    let cloudiness = 1.0f32; // Overall cloud coverage (0.0 to 1.0)
    let turbulence = 0.5f32; // Affects detail level of clouds (0.0 = calm, 1.0 = very turbulent)
    let field_extent = 6000.0; // Distance to cover in meters for x/z plane

    // Calculate approximate number of clouds based on coverage and field size
    let num_clouds = ((field_extent * field_extent) / 50_000.0 * cloudiness).round() as usize;

    let mut clouds = Vec::with_capacity(num_clouds);
    for _ in 0..num_clouds {
        // Determine cloud form
        let form = if rng.random::<f32>() < stratiform_chance {
            CloudForm::Stratus
        } else if rng.random::<f32>() < 0.9 {
            CloudForm::Cumulus
        } else if rng.random::<f32>() < 0.95 {
            CloudForm::Cirrus
        } else {
            CloudForm::Cumulonimbus
        };

        // Determine height based on cloud form
        let altitude = match form {
            CloudForm::Cumulus => rng.random_range(200.0..=12000.0),
            CloudForm::Stratus => rng.random_range(6000.0..=26000.0),
            CloudForm::Cirrus => rng.random_range(20000.0..=30000.0),
            CloudForm::Cumulonimbus => rng.random_range(1000.0..=10000.0),
        };

        // Determine size and proportion based on cloud form
        let (length, width_factor, height) = match form {
            CloudForm::Cumulus => (
                rng.random_range(400.0..=2000.0),
                rng.random_range(0.8..=1.0),
                rng.random_range(300.0..=1200.0),
            ),
            CloudForm::Stratus => (
                rng.random_range(1500.0..=3500.0),
                rng.random_range(0.3..=0.8),
                rng.random_range(60.0..=140.0),
            ),
            CloudForm::Cirrus => (
                rng.random_range(2000.0..=4000.0),
                rng.random_range(0.1..=0.5),
                rng.random_range(40.0..=80.0),
            ),
            CloudForm::Cumulonimbus => (
                rng.random_range(3000.0..=7000.0),
                rng.random_range(0.9..=1.0),
                rng.random_range(6000.0..=10000.0),
            ),
        };

        // Determine position within the defined field extent
        let x = rng.random_range(-field_extent..=field_extent);
        let y = rng.random_range(-field_extent..=field_extent);

        // Determine density and detail based on form and turbulence
        let density = match form {
            CloudForm::Cumulus => rng.random_range(0.8..=1.0),
            CloudForm::Stratus => rng.random_range(0.5..=0.8),
            CloudForm::Cirrus => rng.random_range(0.4..=0.7),
            CloudForm::Cumulonimbus => 1.0,
        };
        let detail = match form {
            CloudForm::Cumulus => rng.random_range(0.7..=1.0),
            CloudForm::Stratus => rng.random_range(0.3..=0.6),
            CloudForm::Cirrus => rng.random_range(0.1..=0.2),
            CloudForm::Cumulonimbus => 0.5,
        } * turbulence; // Apply turbulence factor to detail

        // Create and configure the cloud, then add to the list
        clouds.push(
            Cloud::default()
                .set_pos(Vec3::new(x, y, altitude))
                .set_size(Vec3::new(length, width_factor, height))
                .set_seed(rng.random_range(0..=Cloud::max_for_bits(Cloud::SEED_BITS)))
                .set_yaw(rng.random_range(0..=Cloud::max_for_bits(Cloud::YAW_BITS)))
                .set_form(form)
                .set_density(density)
                .set_detail(detail)
                .set_brightness(1.0),
        );
    }

    // Insert resources into the main world
    commands.insert_resource(CloudsState { clouds });
    commands.insert_resource(CloudsBufferData {
        clouds: [Cloud::default(); MAX_VISIBLE], // Initialize with default clouds
        total: 0,
    });
}

/// Update cloud positions, perform frustum culling, and fill the GPU buffer.
pub fn update_clouds(
    time: Res<Time>,
    mut state: ResMut<CloudsState>,
    mut buffer: ResMut<CloudsBufferData>,
    camera_query: Query<(&GlobalTransform, &Frustum), With<Camera>>,
) {
    // Get camera data for frustum culling and sorting
    let Ok((camera_transform, camera_frustum)) = camera_query.single() else {
        buffer.total = 0;
        error!("No camera found, clearing clouds buffer");
        return;
    };

    let mut visible_cloud_count = 0;
    let cam_pos = camera_transform.translation();

    // Iterate through all clouds, update position, and check visibility
    for cloud in &mut state.clouds {
        let mut pos = cloud.get_pos();
        pos.x += time.delta_secs() * 10.0;
        cloud.set_pos(pos);

        // compute radius from size: use largest of length and height
        let size_v = cloud.get_size();
        let radius = (size_v.x.max(size_v.y)) * 0.5;

        // Check if the cloud's bounding sphere intersects the camera frustum
        if visible_cloud_count < MAX_VISIBLE
            && camera_frustum.intersects_sphere(
                &Sphere {
                    center: pos.into(),
                    radius,
                },
                false,
            )
        {
            // If visible and space available, add to the buffer
            buffer.clouds[visible_cloud_count] = *cloud;
            visible_cloud_count += 1;
        }
    }

    // Sort visible clouds by distance from the camera
    buffer.clouds[..visible_cloud_count].sort_unstable_by(|a, b| {
        let dist_a_sq = (a.get_pos() - cam_pos).length_squared();
        let dist_b_sq = (b.get_pos() - cam_pos).length_squared();
        // Compare squared distances to avoid sqrt, then unwrap or default to Equal
        dist_a_sq.partial_cmp(&dist_b_sq).unwrap_or(Ordering::Equal)
    });

    // Update the number of clouds to be rendered
    buffer.total = u32::try_from(visible_cloud_count).unwrap_or_else(|e| {
        error!("Failed to convert visible cloud count to u32: {}", e);
        0
    });
}

/// Resource holding the GPU buffer for cloud data.
#[derive(Resource)]
pub struct CloudsBuffer {
    pub buffer: Buffer,
}

/// Upload the current `CloudsBufferData` into a GPU buffer for the render world.
pub fn update_clouds_buffer(
    mut commands: Commands,
    clouds_data: Res<CloudsBufferData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    clouds_buffer: Option<Res<CloudsBuffer>>,
) {
    // Convert the CloudsBufferData resource into a byte slice for GPU upload
    let bytes = bytemuck::bytes_of(&*clouds_data);
    if let Some(clouds_buffer) = clouds_buffer {
        // If the buffer already exists, simply write the new data to it
        render_queue.write_buffer(&clouds_buffer.buffer, 0, bytes);
    } else {
        // If the buffer doesn't exist, create a new one with the data
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("clouds_buffer"),
            contents: bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        commands.insert_resource(CloudsBuffer { buffer });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test packing and unpacking of position and size fields.
    fn pack_unpack_pos_size() {
        let c = Cloud::default()
            .set_pos(Vec3::new(12345.0, -54321.0, 25000.0))
            .set_size(Vec3::new(2000.0, 0.75, 600.0));
        let pos = c.get_pos();
        assert!((pos.x - 12345.0).abs() < 0.001);
        assert!((pos.y - -54321.0).abs() < 0.001);
        assert!((pos.z - 25000.0).abs() < 0.001);
        let size = c.get_size();
        assert_eq!(size.x as i32, 2000);
        assert!((size.y - 0.75).abs() < 0.01);
        assert_eq!(size.z as i32, 600);
    }

    #[test]
    /// Test packing and unpacking of cloud property bitfields.
    fn pack_unpack_properties() {
        let mut c = Cloud::default();
        c = c
            .set_density(0.5)
            .set_detail(0.25)
            .set_brightness(0.75)
            .set_yaw(33);
        c = c
            .set_form(CloudForm::Cirrus)
            .set_seed(Cloud::max_for_bits(Cloud::SEED_BITS));

        assert!((c.get_density() - 0.5).abs() < 0.08);
        assert!((c.get_detail() - 0.25).abs() < 0.08);
        assert!((c.get_brightness() - 0.75).abs() < 0.08);
        assert_eq!(c.get_yaw(), 33);
        assert_eq!(c.get_seed(), Cloud::max_for_bits(Cloud::SEED_BITS));
        match c.get_form() {
            CloudForm::Cirrus => {}
            _ => panic!("form mismatch"),
        }
    }
}
