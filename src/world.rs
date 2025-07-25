use bevy::prelude::*;
// Use the custom components defined in main.rs. This requires them to be public.
use crate::{Mesh3d, MeshMaterial3d};

// --- Constants ---
// These constants define the dimensions of our flat-world-pretending-to-be-a-sphere.
const PLANET_RADIUS: f32 = 1000.0;
// The Z-coordinate distance from the equator (z=0) to a pole. Based on sphere circumference.
const EQUATOR_TO_POLE_DISTANCE: f32 = PLANET_RADIUS * std::f32::consts::FRAC_PI_2;
// The X-coordinate distance to wrap around the world at the equator.
const WORLD_CIRCUMFERENCE: f32 = 2.0 * std::f32::consts::PI * PLANET_RADIUS;

// --- Components ---

#[derive(Component, Default)]
pub struct WorldCoordinates {
    pub latitude: f32,
    pub longitude: f32,
}

#[derive(Component)]
struct NorthPoleMarker;

#[derive(Component)]
struct SouthPoleMarker;

// --- Plugin ---

pub struct WorldPlugin;

impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_world)
            // A single system handles all world-related updates for simplicity.
            .add_systems(Update, update_world_and_poles);
    }
}

// --- Systems ---

fn setup_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // North Pole Marker (placed at the "top" of the world on the Z axis)
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(10.0, 500.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.0, 0.5, 1.0))),
        Transform::from_xyz(0.0, 250.0, EQUATOR_TO_POLE_DISTANCE),
        NorthPoleMarker,
        Name::new("North Pole Marker"),
    ));

    // South Pole Marker (placed at the "bottom" of the world on the Z axis)
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(10.0, 500.0))),
        MeshMaterial3d(materials.add(Color::srgb(1.0, 0.5, 0.0))),
        Transform::from_xyz(0.0, 250.0, -EQUATOR_TO_POLE_DISTANCE),
        SouthPoleMarker,
        Name::new("South Pole Marker"),
    ));
}

/// This combined system handles camera wrapping, lat/lon calculation, and pole orientation.
fn update_world_and_poles(
    mut camera_query: Query<(&mut Transform, &mut WorldCoordinates), With<Camera>>,
    mut pole_query: Query<
        (&mut Transform, AnyOf<(&NorthPoleMarker, &SouthPoleMarker)>),
        Without<Camera>,
    >,
) {
    // Get the camera's transform and world coordinates.
    if let Ok((mut camera_transform, mut world_coords)) = camera_query.single_mut() {
        // --- 1. Handle Camera Position Wrapping ---

        // If the camera goes past the north pole, wrap it to the other side.
        if camera_transform.translation.z > EQUATOR_TO_POLE_DISTANCE {
            camera_transform.translation.z =
                2.0 * EQUATOR_TO_POLE_DISTANCE - camera_transform.translation.z;
            camera_transform.translation.x *= -1.0; // Invert longitude position.
        }
        // If the camera goes past the south pole, wrap it to the other side.
        if camera_transform.translation.z < -EQUATOR_TO_POLE_DISTANCE {
            camera_transform.translation.z =
                -2.0 * EQUATOR_TO_POLE_DISTANCE - camera_transform.translation.z;
            camera_transform.translation.x *= -1.0; // Invert longitude position.
        }

        let position = camera_transform.translation;

        // --- 2. Calculate Latitude and Longitude from Camera's Flat-World Position ---

        // Latitude angle is based on Z position.
        let lat_rad = (position.z / PLANET_RADIUS)
            .clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

        // Longitude angle is based on X position.
        let lon_rad = (position.x / PLANET_RADIUS).rem_euclid(2.0 * std::f32::consts::PI);

        world_coords.latitude = lat_rad.to_degrees();
        world_coords.longitude = lon_rad.to_degrees();

        // --- 3. Calculate the Geometric Rotation for the Poles ---

        // Find the camera's "up" vector on the conceptual sphere.
        // This vector points from the sphere's center to the camera's surface position.
        let camera_up_on_sphere = Vec3::new(
            lat_rad.cos() * lon_rad.sin(),
            lat_rad.sin(),
            lat_rad.cos() * lon_rad.cos(),
        )
        .normalize();

        // Calculate the rotation needed to make the camera's "up" on the sphere
        // align with the world's "up" (Vec3::Y). This simulates the world tilting beneath you.
        let world_tilt_rotation = Quat::from_rotation_arc(camera_up_on_sphere, Vec3::Y);

        for (mut pole_transform, markers) in &mut pole_query {
            let is_north_pole = markers.1.is_some();

            // Define the pole's "up" vector in the sphere's local space.
            let pole_up_on_sphere = if is_north_pole { Vec3::Y } else { -Vec3::Y };

            // Apply the world tilt to the pole's "up" vector to find its final orientation.
            let final_pole_direction = world_tilt_rotation * pole_up_on_sphere;

            // The cylinder mesh's default orientation is along the Y axis.
            // We rotate it to point in the calculated final direction.
            pole_transform.rotation = Quat::from_rotation_arc(Vec3::Y, final_pole_direction);
        }

        // --- 4. Debug Print ---
        info!(
            "Lat: {:.2}, Lon: {:.2}",
            world_coords.latitude, world_coords.longitude
        );
    }
}
