use anyhow::{Context, Result};
use godot::{
    engine::{
        global::{Key, MouseButton},
        mesh::PrimitiveType,
        ImmediateMesh, InputEvent, InputEventKey, InputEventMouseButton, Material, Mesh,
        MeshInstance3D, ResourceLoader, RigidBody3D, RigidBody3DVirtual,
    },
    prelude::*,
};
use rayon::prelude::*;
use std::{f32::consts::PI, str::FromStr};

const PARTICLE_DISTANCE: f32 = 0.25;
const PARTICLE_MASS: f32 = PARTICLE_DISTANCE * PARTICLE_DISTANCE * PARTICLE_DISTANCE * 0.5;

const GRAVITY_MULTIPLIER: f32 = 20.0;
const SPRING_CONSTANT: f32 = 30.0;
const REPULSION_STRENGTH: f32 = 40.0;
const DAMPING: f32 = 0.95;

#[derive(PartialEq, Clone)]
struct Connection {
    target_index: usize,
    direction: Vector3,
    active: bool,
}

impl Connection {
    fn new(origin_particle: &Particle, target_particle: &Particle, target_index: usize) -> Self {
        Self {
            target_index,
            direction: origin_particle.position - target_particle.position,
            active: true,
        }
    }
}

#[derive(PartialEq, Clone)]
struct Particle {
    position: Vector3,
    old_position: Vector3,
    connections: Vec<Connection>,
    anchored: bool,
}

impl Particle {
    const fn new(pos: Vector3) -> Self {
        Self {
            position: pos,
            old_position: pos,
            connections: Vec::new(),
            anchored: false,
        }
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Entity {
    #[base]
    base: Base<RigidBody3D>,
    particles: Vec<Particle>,

    plane_rotate: bool,
    plane_size: f32,
}

#[godot_api]
impl RigidBody3DVirtual for Entity {
    fn init(base: Base<RigidBody3D>) -> Self {
        let mut instance = Self {
            base,
            particles: Vec::new(),
            plane_rotate: false,
            plane_size: 0.2,
        };
        instance.base.set_gravity_scale(0.0);

        // Add render mesh to it
        let mut render_mesh = MeshInstance3D::new_alloc();
        render_mesh.set_mesh(ImmediateMesh::new().upcast::<Mesh>());
        instance.base.add_child(render_mesh.upcast::<Node>());

        instance
    }

    fn ready(&mut self) {
        let grid_size: i32 = 4;

        // Create grid of particles
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                for z in -grid_size..=grid_size {
                    // Create new particle
                    let mut particle = Particle::new(
                        self.base.get_global_position()
                            + Vector3::new(x as f32, y as f32, z as f32) * PARTICLE_DISTANCE,
                    );
                    // Anchor a strip of particles in the center
                    let error_margin = 0.05;
                    if (particle.position.x - self.base.get_global_position().x).abs()
                        < error_margin
                        && (particle.position.y - self.base.get_global_position().y).abs()
                            < error_margin
                    {
                        particle.anchored = true;
                    }
                    // Add particle to entity
                    self.particles.push(particle);
                }
            }
        }

        // Connect adjacent particles based on distance threshold
        let connections: Vec<_> = self
            .particles
            .par_iter()
            .enumerate()
            .flat_map(|(idx1, particle1)| {
                let mut local_connections = Vec::new();
                for (idx2, particle2) in self.particles.iter().enumerate() {
                    if idx1 != idx2
                        && particle1.position.distance_to(particle2.position)
                            <= PARTICLE_DISTANCE * 1.5
                    {
                        // Connect the particles
                        local_connections.push((idx1, Connection::new(particle1, particle2, idx2)));
                    }
                }
                local_connections
            })
            .collect();

        // Add connections to particles
        for (index, connection) in connections {
            self.particles[index].connections.push(connection);
        }
    }

    fn process(&mut self, delta: f64) {
        self.process_physics(delta);
        self.render_particles();
        // If c is pressed, cut the connections that intersect with the plane
        if Input::singleton().is_key_pressed(Key::KEY_C) {
            self.cut_on_plane().unwrap();
        }
    }

    fn input(&mut self, event: Gd<InputEvent>) {
        if let Some(event_key) = event.try_cast::<InputEventKey>() {
            if event_key.is_pressed() && event_key.get_keycode() == Key::KEY_SPACE {
                self.plane_rotate = !self.plane_rotate;
            } else if event_key.is_pressed() && event_key.get_keycode() == Key::KEY_R {
                self.plane_size = f32::min(self.plane_size + 0.05, 1.0);
            } else if event_key.is_pressed() && event_key.get_keycode() == Key::KEY_F {
                self.plane_size = f32::max(self.plane_size - 0.05, 0.1);
            }
        }
    }
}

impl Entity {
    fn process_physics(&mut self, delta: f64) {
        let delta = delta as f32;
        let gravity = Vector3::new(0.0, -GRAVITY_MULTIPLIER, 0.0) * PARTICLE_MASS;

        let new_positions: Vec<(usize, Vector3)> = self
            .particles
            .par_iter()
            .enumerate()
            .map(|(i, particle)| {
                if particle.anchored {
                    (i, particle.position)
                } else {
                    // Calculate velocity based on the difference of old and current positions
                    let velocity = (particle.position - particle.old_position) * DAMPING;

                    // Initialize the net force to gravity
                    let mut net_force = gravity;

                    // Connection Forces
                    for connection in &particle.connections {
                        if connection.active {
                            let target = &self.particles[connection.target_index];
                            let target_position = target.position + connection.direction;
                            let direction = target_position - particle.position;
                            let spring_force = direction * SPRING_CONSTANT;
                            net_force += spring_force;
                        }
                    }

                    // Repulsion Forces
                    for j in i + 1..self.particles.len() {
                        let other = &self.particles[j];
                        if other.position != particle.position {
                            let difference = particle.position - other.position;
                            let distance = difference.length();
                            if distance < PARTICLE_DISTANCE {
                                let repulsion_force = difference.normalized()
                                    * (PARTICLE_DISTANCE - distance)
                                    * REPULSION_STRENGTH;
                                net_force += repulsion_force;
                            }
                        }
                    }

                    // Verlet integration step
                    (i, particle.position + velocity + net_force * delta * delta)
                }
            })
            .collect();

        // Apply the new positions and store old positions
        for (i, new_position) in new_positions {
            self.particles[i].old_position = self.particles[i].position;
            self.particles[i].position = new_position;
        }
    }

    fn render_particles(&mut self) {
        let mut render_geometry = match self.get_immediate_mesh() {
            Ok(mesh) => mesh,
            Err(e) => {
                godot_print!("Failed to get immediate mesh {e}");
                return;
            }
        };

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
            render_geometry.surface_set_color(Color::from_rgb(0.2, 0.2, 1.0));
            for (pos1, pos2, pos3) in faces.clone() {
                render_geometry.surface_set_normal((pos1 + pos2 + pos3).normalized());
                render_geometry.surface_add_vertex(particle.position + pos1 * diamond_size);
                render_geometry.surface_add_vertex(particle.position + pos2 * diamond_size);
                render_geometry.surface_add_vertex(particle.position + pos3 * diamond_size);
            }
        }
        render_geometry.surface_end();

        // Render cut plane on camera
        let (plane_center, plane_euler_rotation, plane_size) = match self.get_cut_plane() {
            Ok(plane) => plane,
            Err(e) => {
                godot_print!("Failed to get cut plane {e}");
                return;
            }
        };
        render_cut_plane(
            &mut render_geometry,
            material.clone(),
            plane_center,
            plane_euler_rotation,
            plane_size,
        );

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
                    // Check if the connection intersects with the plane
                    let color = if line_intersects_finite_plane(
                        particle.position,
                        self.particles[connection.target_index].position,
                        plane_center,
                        plane_euler_rotation,
                        plane_size,
                    ) {
                        Color::from_rgb(1.0, 0.5, 0.0)
                    } else {
                        Color::from_rgb(0.0, 1.0, 0.0)
                    };
                    render_geometry.surface_set_color(color);
                    render_geometry.surface_add_vertex(particle.position);
                    render_geometry
                        .surface_add_vertex(self.particles[connection.target_index].position);
                }
            }
        }
        render_geometry.surface_end();
    }

    fn get_immediate_mesh(&self) -> Result<Gd<ImmediateMesh>> {
        self.base
            .get_child(0)
            .context("No child at index 0")?
            .try_cast::<MeshInstance3D>()
            .context("Failed to cast node to MeshInstance3D")?
            .get_mesh()
            .context("MeshInstance3D does not have a mesh")?
            .try_cast::<ImmediateMesh>()
            .context("Failed to cast mesh to ImmediateMesh")
    }

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

    fn get_cut_plane(&self) -> Result<(Vector3, Vector3, Vector2)> {
        let camera = self.get_camera_transform()?;

        let plane_center = camera.origin + camera.basis * Vector3::new(0.0, 0.0, -0.5);
        let plane_euler_rotation =
            Vector3::new(0.0, 0.0, if self.plane_rotate { PI / 2.0 } else { 0.0 });
        let plane_size = Vector2::new(self.plane_size, self.plane_size);

        Ok((plane_center, plane_euler_rotation, plane_size))
    }

    fn cut_on_plane(&mut self) -> Result<()> {
        // Get cut plane
        let (plane_center, plane_euler_rotation, plane_size) = match self.get_cut_plane() {
            Ok(plane) => plane,
            Err(e) => {
                godot_print!("Failed to get cut plane {e}");
                return Err(e);
            }
        };

        // Find cuts
        let mut cuts = Vec::new();
        for (i1, particle) in self.particles.iter().enumerate() {
            for (i2, connection) in particle.connections.iter().enumerate() {
                if connection.active {
                    // Check if the connection intersects with the plane
                    let intersects = line_intersects_finite_plane(
                        particle.position,
                        self.particles[connection.target_index].position,
                        plane_center,
                        plane_euler_rotation,
                        plane_size,
                    );
                    if intersects {
                        cuts.push((i1, i2));
                    }
                }
            }
        }

        // Make cuts
        for (i1, i2) in cuts {
            self.particles[i1].connections[i2].active = false;
        }
        Ok(())
    }
}

fn render_cut_plane(
    render_geometry: &mut ImmediateMesh,
    material: Gd<Material>,
    plane_center: Vector3,
    plane_euler_rotation: Vector3,
    plane_size: Vector2,
) {
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

fn line_intersects_finite_plane(
    a: Vector3,
    b: Vector3,
    plane_center: Vector3,
    plane_euler_rotation: Vector3,
    plane_size: Vector2,
) -> bool {
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
