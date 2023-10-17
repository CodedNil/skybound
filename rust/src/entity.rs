use std::str::FromStr;

use godot::{
    engine::{
        mesh::PrimitiveType, ImmediateMesh, Material, Mesh, MeshInstance3D, ResourceLoader,
        RigidBody3D, RigidBody3DVirtual,
    },
    prelude::*,
};
use rayon::prelude::*;

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
}

#[godot_api]
impl RigidBody3DVirtual for Entity {
    fn init(base: Base<RigidBody3D>) -> Self {
        let mut instance = Self {
            base,
            particles: Vec::new(),
        };
        instance.base.set_gravity_scale(0.0);

        // Add render mesh to it
        let mut render_mesh = MeshInstance3D::new_alloc();
        let mut render_geometry = ImmediateMesh::new();
        render_mesh.set_mesh(render_geometry.upcast::<Mesh>());
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
        if let Some(render_mesh_node) = self.base.get_child(0) {
            if let Some(render_mesh) = render_mesh_node.try_cast::<MeshInstance3D>() {
                let mut render_geometry = render_mesh.get_mesh().unwrap().cast::<ImmediateMesh>();
                render_geometry.clear_surfaces();

                // Load material
                let material = ResourceLoader::singleton()
                    .load(GodotString::from_str("res://debug_node.tres").unwrap())
                    .unwrap()
                    .cast::<Material>();

                // Render out particles
                render_geometry.call(
                    StringName::from("surface_begin"),
                    &[
                        Variant::from(PrimitiveType::PRIMITIVE_TRIANGLE_STRIP),
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

                // Render out connection lines
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
                            render_geometry.surface_set_color(Color::from_rgb(0.0, 1.0, 0.0));
                            render_geometry.surface_add_vertex(particle.position);
                            render_geometry.surface_add_vertex(
                                self.particles[connection.target_index].position,
                            );
                        }
                    }
                }
                render_geometry.surface_end();
            }
        }
    }
}
