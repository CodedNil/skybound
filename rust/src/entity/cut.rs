use super::Entity;
use anyhow::{Context, Result};
use godot::{
    engine::{mesh::PrimitiveType, ImmediateMesh, Material},
    prelude::*,
};
use std::f32::consts::PI;

impl Entity {
    fn get_camera_transform(&self) -> Result<Transform3D> {
        let transform = self
            .base
            .get_viewport()
            .context("Failed to get node's viewport")?
            .get_camera_3d()
            .context("Viewport does not have a camera")?
            .get_global_transform();

        Ok(transform)
    }

    pub fn get_cut_plane(&self) -> Result<(Vector3, Vector3, Vector2)> {
        let camera = self.get_camera_transform()?;

        let plane_center = camera.origin + camera.basis * Vector3::new(0.0, 0.0, -0.5);
        let plane_euler_rotation =
            Vector3::new(0.0, 0.0, if self.plane_rotate { PI / 2.0 } else { 0.0 });
        let plane_size = Vector2::new(self.plane_size, self.plane_size);

        Ok((plane_center, plane_euler_rotation, plane_size))
    }

    pub fn cut_on_plane(&mut self) -> Result<()> {
        // Get cut plane
        let cut_plane = match self.get_cut_plane() {
            Ok(plane) => plane,
            Err(e) => {
                godot_print!("Failed to get cut plane {e}");
                return Err(e);
            }
        };

        // Find cuts
        let mut cut_connections = Vec::new();
        for (i1, particle) in self.particles.iter().enumerate() {
            for (i2, connection) in particle.connections.iter().enumerate() {
                if connection.active {
                    // Check if the connection intersects with the plane
                    let intersects = line_intersects_finite_plane(
                        particle.position,
                        self.particles[connection.target_index].position,
                        cut_plane,
                    );
                    if intersects {
                        cut_connections.push((i1, i2));
                    }
                }
            }
        }

        // Make cuts
        for (i1, i2) in cut_connections {
            self.particles[i1].connections[i2].active = false;
        }
        Ok(())
    }
}

pub fn render_cut_plane(
    render_geometry: &mut ImmediateMesh,
    material: Gd<Material>,
    cut_plane: (Vector3, Vector3, Vector2),
) {
    let (plane_center, plane_euler_rotation, plane_size) = cut_plane;
    render_geometry.call(
        StringName::from("surface_begin"),
        &[
            Variant::from(PrimitiveType::PRIMITIVE_TRIANGLE_STRIP),
            Variant::from(material),
        ],
    );

    // Convert Euler rotation to a rotation matrix (or basis)
    let plane_rotation = Basis::from_euler(EulerOrder::XYZ, plane_euler_rotation);

    // Calculate the four corners of the plane in local coordinates
    let plane_size = plane_size * 0.5;
    let top_left = Vector3::BACK * plane_size.y + Vector3::LEFT * plane_size.x;
    let top_right = Vector3::BACK * plane_size.y + Vector3::RIGHT * plane_size.x;
    let bottom_left = Vector3::FORWARD * plane_size.y + Vector3::LEFT * plane_size.x;
    let bottom_right = Vector3::FORWARD * plane_size.y + Vector3::RIGHT * plane_size.x;

    // Rotate the corners and translate them to the plane's position
    let rotated_top_left = plane_center + plane_rotation * top_left;
    let rotated_top_right = plane_center + plane_rotation * top_right;
    let rotated_bottom_left = plane_center + plane_rotation * bottom_left;
    let rotated_bottom_right = plane_center + plane_rotation * bottom_right;

    // Render the plane
    render_geometry.surface_set_color(Color::from_rgb(1.0, 0.0, 0.0));
    render_geometry.surface_set_normal(plane_rotation * Vector3::UP);
    render_geometry.surface_add_vertex(rotated_top_left);
    render_geometry.surface_add_vertex(rotated_bottom_left);
    render_geometry.surface_add_vertex(rotated_top_right);
    render_geometry.surface_add_vertex(rotated_bottom_right);

    render_geometry.surface_end();
}

pub fn line_intersects_finite_plane(
    a: Vector3,
    b: Vector3,
    cut_plane: (Vector3, Vector3, Vector2),
) -> bool {
    let (plane_center, plane_euler_rotation, plane_size) = cut_plane;
    let plane_rotation = Basis::from_euler(EulerOrder::XYZ, plane_euler_rotation);
    let plane_normal = plane_rotation * Vector3::UP;

    let line_direction = b - a;
    let denominator = line_direction.dot(plane_normal);

    // If line is parallel to plane, no intersection
    if denominator.abs() < std::f32::EPSILON {
        return false;
    }

    let t = (plane_center - a).dot(plane_normal) / denominator;

    // Check if intersection with infinite plane is outside the segment
    if !(0.0..=1.0).contains(&t) {
        return false;
    }

    let intersection = a + t * line_direction;

    // Calculate half size of the plane
    let half_size = plane_size * 0.5;

    // Transform intersection point to plane's local coordinates
    let local_intersection = plane_rotation.inverse() * (intersection - plane_center);

    // Check if the intersection point is within the finite plane bounds in local coordinates
    local_intersection.x.abs() <= half_size.x && local_intersection.z.abs() <= half_size.y
}
