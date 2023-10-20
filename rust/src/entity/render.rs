use super::cut::{line_intersects_finite_plane, render_cut_plane};
use super::{Entity, ParticleMaterial};
use anyhow::{Context, Result};
use godot::{
    engine::{mesh::PrimitiveType, ImmediateMesh, Material, MeshInstance3D, ResourceLoader},
    prelude::*,
};
use std::str::FromStr;

impl Entity {
    #[allow(clippy::too_many_lines)]
    pub fn render_particles(&mut self) {
        let mut render_geometry = match self.get_immediate_mesh() {
            Ok(mesh) => mesh,
            Err(e) => {
                godot_print!("Failed to get immediate mesh {e}");
                return;
            }
        };
        let transform = self.base.get_global_transform();
        let transform_inv = transform.affine_inverse();

        render_geometry.clear_surfaces();

        // Load material
        let material = ResourceLoader::singleton()
            .load(GodotString::from_str("res://debug_node.tres").unwrap())
            .unwrap()
            .cast::<Material>();

        // Render particles
        render_geometry.call(
            StringName::from("surface_begin"),
            &[
                Variant::from(PrimitiveType::PRIMITIVE_TRIANGLES),
                Variant::from(material.clone()),
            ],
        );
        let diamond_size = 0.05;
        let faces = vec![
            (Vector3::FORWARD, Vector3::RIGHT, Vector3::UP), // Top Front Right Face
            (Vector3::UP, Vector3::LEFT, Vector3::FORWARD),  // Top Front Left Face
            (Vector3::BACK, Vector3::LEFT, Vector3::UP),     // Top Back Left Face
            (Vector3::UP, Vector3::RIGHT, Vector3::BACK),    // Top Back Right Face
            (Vector3::FORWARD, Vector3::DOWN, Vector3::RIGHT), // Bottom Front Right Face
            (Vector3::DOWN, Vector3::FORWARD, Vector3::LEFT), // Bottom Front Left Face
            (Vector3::BACK, Vector3::DOWN, Vector3::LEFT),   // Bottom Back Left Face
            (Vector3::DOWN, Vector3::BACK, Vector3::RIGHT),  // Bottom Back Right Face
        ];
        for particle in &self.particles {
            let material_color = match particle.material {
                ParticleMaterial::Flesh => Color::from_rgb(1.0, 0.3, 0.3),
                ParticleMaterial::Bone => Color::from_rgb(1.0, 1.0, 1.0),
                ParticleMaterial::Heart => Color::from_rgb(1.0, 0.0, 0.0),
            };
            let position = transform_inv * particle.position;
            render_geometry.surface_set_color(material_color);
            for (pos1, pos2, pos3) in faces.clone() {
                render_geometry.surface_set_normal((pos1 + pos2 + pos3).normalized());
                render_geometry.surface_add_vertex(position + pos1 * diamond_size);
                render_geometry.surface_add_vertex(position + pos2 * diamond_size);
                render_geometry.surface_add_vertex(position + pos3 * diamond_size);
            }
        }
        render_geometry.surface_end();

        // Render cut plane on camera
        let cut_plane = match self.get_cut_plane() {
            Ok(plane) => plane,
            Err(e) => {
                godot_print!("Failed to get cut plane {e}");
                return;
            }
        };
        render_cut_plane(&mut render_geometry, material.clone(), cut_plane);

        // Render connection lines
        render_geometry.call(
            StringName::from("surface_begin"),
            &[
                Variant::from(PrimitiveType::PRIMITIVE_LINES),
                Variant::from(material),
            ],
        );
        for particle in &self.particles {
            for connection in &particle.connections {
                if connection.active {
                    let a = transform_inv * particle.position;
                    let b = transform_inv * self.particles[connection.target_index].position;
                    let dist = a.distance_to(b);
                    let strain = f32::abs(connection.distance - dist);

                    let color = if line_intersects_finite_plane(a, b, cut_plane) {
                        Color::from_rgb(1.0, 0.5, 0.0)
                    } else {
                        // Lerp between green and red based on strain
                        let lerp = (strain / 0.4).clamp(0.0, 1.0);
                        Color::from_rgb(lerp, 1.0 - lerp, 0.0)
                    };
                    render_geometry.surface_set_color(color);
                    render_geometry.surface_add_vertex(a);
                    render_geometry.surface_add_vertex(b);
                }
            }
        }
        render_geometry.surface_end();
    }

    fn get_immediate_mesh(&self) -> Result<Gd<ImmediateMesh>> {
        self.base
            .get_child(self.base.get_child_count() - 1)
            .context("No child at index 0")?
            .try_cast::<MeshInstance3D>()
            .context("Failed to cast node to MeshInstance3D")?
            .get_mesh()
            .context("MeshInstance3D does not have a mesh")?
            .try_cast::<ImmediateMesh>()
            .context("Failed to cast mesh to ImmediateMesh")
    }
}
