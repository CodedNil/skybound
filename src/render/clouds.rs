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

/// Buffer for visible cloud data, extracted to the render world and sent to the GPU.
#[derive(Resource, ExtractResource, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CloudsBufferData {
    clouds: [Cloud; MAX_VISIBLE],
    num_clouds: u32,
}

/// Represents a cloud with its properties
#[derive(Default, Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Cloud {
    // 16 bytes
    x_pos: i32, // X position (4 bytes)
    y_pos: i32, // Y position (4 bytes)
    size: u32, // 14 bits length 0..16383, 14 bits height 0..16383, 4 bits width factor 0..16 (4 bytes)
    data: u32, // Packed data (4 bytes)
               // 6 bits for seed, as a number 0..64
               // 15 bits for altitude, as a number 0..32767
               // 2 bits for form, 0..4 from CloudForm enum
               // 3 bits for density (Overall fill, 0=almost empty mist, 1=solid cloud mass), 0..1 in 8 steps
               // 3 bits for detail (Noise detail power, 0=smooth blob, 1=lots of little puffs), 0..1 in 8 steps
               // 3 bits for brightness, 0..1 in 8 steps
}

enum CloudForm {
    Cumulus,
    Stratus,
    Cirrus,
    Cumulonimbus,
}

impl Cloud {
    // --- layout (data u32) ---
    // altitude : 15 bits  (0..32767 meters)
    // seed     :  6 bits  (0..63)
    // form     :  2 bits  (0..3)
    // density  :  3 bits  (0..7)
    // detail   :  3 bits  (0..7)
    // brightness: 3 bits  (0..7)
    const ALT_BITS: u32 = 15;
    const SEED_BITS: u32 = 6;
    const FORM_BITS: u32 = 2;
    const DENSITY_BITS: u32 = 3;
    const DETAIL_BITS: u32 = 3;
    const BRIGHTNESS_BITS: u32 = 3;

    const ALT_SHIFT: u32 = 0;
    const SEED_SHIFT: u32 = Self::ALT_SHIFT + Self::ALT_BITS;
    const FORM_SHIFT: u32 = Self::SEED_SHIFT + Self::SEED_BITS;
    const DENSITY_SHIFT: u32 = Self::FORM_SHIFT + Self::FORM_BITS;
    const DETAIL_SHIFT: u32 = Self::DENSITY_SHIFT + Self::DENSITY_BITS;
    const BRIGHTNESS_SHIFT: u32 = Self::DETAIL_SHIFT + Self::DETAIL_BITS;

    // --- layout (size u32) ---
    // length : 14 bits (0..16383)
    // height : 14 bits (0..16383)
    // width  :  4 bits (0..15) -> normalized as 0..1
    const SIZE_LEN_BITS: u32 = 14;
    const SIZE_HEIGHT_BITS: u32 = 14;
    const SIZE_WIDTH_BITS: u32 = 4;

    const SIZE_LEN_SHIFT: u32 = 0;
    const SIZE_HEIGHT_SHIFT: u32 = Self::SIZE_LEN_SHIFT + Self::SIZE_LEN_BITS;
    const SIZE_WIDTH_SHIFT: u32 = Self::SIZE_HEIGHT_SHIFT + Self::SIZE_HEIGHT_BITS;

    #[inline]
    const fn max_for_bits(bits: u32) -> u32 {
        (1u32 << bits) - 1
    }

    #[inline]
    const fn set_bits_field(container: &mut u32, value: u32, bits: u32, shift: u32) {
        let mask = Self::max_for_bits(bits) << shift;
        *container = (*container & !mask) | ((value & Self::max_for_bits(bits)) << shift);
    }

    #[inline]
    const fn get_bits_field(container: u32, bits: u32, shift: u32) -> u32 {
        (container >> shift) & Self::max_for_bits(bits)
    }

    // pos: Vec3(x, altitude, z) -> x_pos, altitude packed into data, y_pos stores z
    pub const fn set_pos(mut self, pos: Vec3) -> Self {
        self.x_pos = pos.x.round() as i32;
        self.y_pos = pos.z.round() as i32; // store z into y_pos field (legacy naming)
        let alt_raw = pos
            .y
            .clamp(0.0, Self::max_for_bits(Self::ALT_BITS) as f32)
            .round() as u32;
        Self::set_bits_field(&mut self.data, alt_raw, Self::ALT_BITS, Self::ALT_SHIFT);
        self
    }

    pub const fn get_pos(&self) -> Vec3 {
        let alt_raw = Self::get_bits_field(self.data, Self::ALT_BITS, Self::ALT_SHIFT);
        Vec3::new(self.x_pos as f32, alt_raw as f32, self.y_pos as f32)
    }

    // size: Vec3(length_meters, height_meters, width_factor_0to1)
    pub const fn set_size(mut self, size: Vec3) -> Self {
        let len_raw = size
            .x
            .clamp(0.0, Self::max_for_bits(Self::SIZE_LEN_BITS) as f32)
            .round() as u32;
        let height_raw = size
            .y
            .clamp(0.0, Self::max_for_bits(Self::SIZE_HEIGHT_BITS) as f32)
            .round() as u32;
        let width_raw = (size.z.clamp(0.0, 1.0)
            * (Self::max_for_bits(Self::SIZE_WIDTH_BITS) as f32))
            .round() as u32;

        Self::set_bits_field(
            &mut self.size,
            len_raw,
            Self::SIZE_LEN_BITS,
            Self::SIZE_LEN_SHIFT,
        );
        Self::set_bits_field(
            &mut self.size,
            height_raw,
            Self::SIZE_HEIGHT_BITS,
            Self::SIZE_HEIGHT_SHIFT,
        );
        Self::set_bits_field(
            &mut self.size,
            width_raw,
            Self::SIZE_WIDTH_BITS,
            Self::SIZE_WIDTH_SHIFT,
        );
        self
    }

    pub const fn get_size(&self) -> Vec3 {
        let len_raw = Self::get_bits_field(self.size, Self::SIZE_LEN_BITS, Self::SIZE_LEN_SHIFT);
        let height_raw =
            Self::get_bits_field(self.size, Self::SIZE_HEIGHT_BITS, Self::SIZE_HEIGHT_SHIFT);
        let width_raw =
            Self::get_bits_field(self.size, Self::SIZE_WIDTH_BITS, Self::SIZE_WIDTH_SHIFT);
        Vec3::new(
            len_raw as f32,
            height_raw as f32,
            (width_raw as f32) / (Self::max_for_bits(Self::SIZE_WIDTH_BITS) as f32),
        )
    }

    // form (enum)
    pub const fn set_form(mut self, form: CloudForm) -> Self {
        let raw = match form {
            CloudForm::Cumulus => 0,
            CloudForm::Stratus => 1,
            CloudForm::Cirrus => 2,
            CloudForm::Cumulonimbus => 3,
        };
        Self::set_bits_field(&mut self.data, raw, Self::FORM_BITS, Self::FORM_SHIFT);
        self
    }

    pub const fn get_form(&self) -> CloudForm {
        match Self::get_bits_field(self.data, Self::FORM_BITS, Self::FORM_SHIFT) {
            0 => CloudForm::Cumulus,
            1 => CloudForm::Stratus,
            2 => CloudForm::Cirrus,
            _ => CloudForm::Cumulonimbus,
        }
    }

    // density/detail/brightness: mapped 0..7 -> 0.0..1.0
    pub const fn set_density(mut self, density: f32) -> Self {
        let raw = (density.clamp(0.0, 1.0) * (Self::max_for_bits(Self::DENSITY_BITS) as f32))
            .round() as u32;
        Self::set_bits_field(&mut self.data, raw, Self::DENSITY_BITS, Self::DENSITY_SHIFT);
        self
    }
    pub const fn get_density(&self) -> f32 {
        Self::get_bits_field(self.data, Self::DENSITY_BITS, Self::DENSITY_SHIFT) as f32
            / (Self::max_for_bits(Self::DENSITY_BITS) as f32)
    }

    pub const fn set_detail(mut self, detail: f32) -> Self {
        let raw = (detail.clamp(0.0, 1.0) * (Self::max_for_bits(Self::DETAIL_BITS) as f32)).round()
            as u32;
        Self::set_bits_field(&mut self.data, raw, Self::DETAIL_BITS, Self::DETAIL_SHIFT);
        self
    }
    pub const fn get_detail(&self) -> f32 {
        Self::get_bits_field(self.data, Self::DETAIL_BITS, Self::DETAIL_SHIFT) as f32
            / (Self::max_for_bits(Self::DETAIL_BITS) as f32)
    }

    pub const fn set_brightness(mut self, brightness: f32) -> Self {
        let raw = (brightness.clamp(0.0, 1.0) * (Self::max_for_bits(Self::BRIGHTNESS_BITS) as f32))
            .round() as u32;
        Self::set_bits_field(
            &mut self.data,
            raw,
            Self::BRIGHTNESS_BITS,
            Self::BRIGHTNESS_SHIFT,
        );
        self
    }
    pub const fn get_brightness(&self) -> f32 {
        Self::get_bits_field(self.data, Self::BRIGHTNESS_BITS, Self::BRIGHTNESS_SHIFT) as f32
            / (Self::max_for_bits(Self::BRIGHTNESS_BITS) as f32)
    }

    // optional: seed getter/setter (6 bits)
    pub const fn set_seed(mut self, seed: u32) -> Self {
        Self::set_bits_field(&mut self.data, seed, Self::SEED_BITS, Self::SEED_SHIFT);
        self
    }
    pub const fn get_seed(&self) -> u32 {
        Self::get_bits_field(self.data, Self::SEED_BITS, Self::SEED_SHIFT)
    }
}

/// Sets up the initial state of clouds in the world.
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
                rng.random_range(8000.0..=16000.0),
                rng.random_range(0.9..=1.0),
                rng.random_range(6000.0..=10000.0),
            ),
        };

        // Determine position within the defined field extent
        let x = rng.random_range(-field_extent..=field_extent);
        let y = rng.random_range(-field_extent..=field_extent);
        let position = Vec3::new(x, y, altitude);
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
                .set_pos(position)
                .set_size(Vec3::new(length, height, width_factor))
                .set_seed(rng.random_range(0..64))
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
        num_clouds: 0,
    });
}

/// Main world system: Updates cloud positions and identifies visible clouds for rendering.
pub fn update_clouds(
    time: Res<Time>,
    mut state: ResMut<CloudsState>,
    mut buffer: ResMut<CloudsBufferData>,
    camera_query: Query<(&GlobalTransform, &Frustum), With<Camera>>,
) {
    // Get camera data for frustum culling and sorting
    let Ok((camera_transform, camera_frustum)) = camera_query.single() else {
        buffer.num_clouds = 0;
        error!("No camera found, clearing clouds buffer");
        return;
    };

    let mut visible_cloud_count = 0;
    let cam_pos = camera_transform.translation();

    // Iterate through all clouds, update position, and check visibility
    for ref mut cloud in &mut state.clouds {
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
                    center: cloud.get_pos().into(),
                    radius,
                },
                false,
            )
        {
            // If visible and space available, add to the buffer
            buffer.clouds[visible_cloud_count] = **cloud;
            visible_cloud_count += 1;
        }
    }

    // Sort visible clouds by distance from the camera (for proper alpha blending/overdraw)
    buffer.clouds[..visible_cloud_count].sort_unstable_by(|a, b| {
        let dist_a_sq = (a.get_pos() - cam_pos).length_squared();
        let dist_b_sq = (b.get_pos() - cam_pos).length_squared();
        // Compare squared distances to avoid sqrt, then unwrap or default to Equal
        dist_a_sq.partial_cmp(&dist_b_sq).unwrap_or(Ordering::Equal)
    });

    // Update the number of clouds to be rendered
    buffer.num_clouds = u32::try_from(visible_cloud_count).unwrap_or_else(|e| {
        error!("Failed to convert visible cloud count to u32: {}", e);
        0
    });
}

/// Resource holding the GPU buffer for cloud data.
#[derive(Resource)]
pub struct CloudsBuffer {
    buffer: Buffer,
}

/// Render world system: Uploads the `CloudsBufferData` to the GPU.
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
