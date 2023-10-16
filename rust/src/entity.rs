use godot::{
    engine::{Mesh, MeshInstance3D, RigidBody3D, RigidBody3DVirtual, SphereMesh},
    prelude::*,
};

const PARTICLE_DISTANCE: f32 = 0.25;
const PARTICLE_MASS: f32 = PARTICLE_DISTANCE * PARTICLE_DISTANCE * PARTICLE_DISTANCE * 0.5;

const GRAVITY_MULTIPLIER: f32 = 5.0;
const SPRING_CONSTANT: f32 = 30.0;
const REPULSION_STRENGTH: f32 = 40.0;
const DAMPING: f32 = 0.95;

#[derive(PartialEq, Clone)]
struct Connection {
    target: Particle,
    direction: Vector3,
    distance: f32,
    active: bool,
}

impl Connection {
    fn new(origin_particle: &Particle, target_particle: &Particle) -> Self {
        let direction = (origin_particle.position - target_particle.position).normalized();
        let distance = origin_particle
            .position
            .distance_to(target_particle.position);

        Self {
            target: target_particle.clone(),
            direction,
            distance,
            active: true,
        }
    }
}

#[derive(PartialEq, Clone)]
struct Particle {
    position: Vector3,
    velocity: Vector3,
    connections: Vec<Connection>,
    anchored: bool,
}

impl Particle {
    const fn new(pos: Vector3) -> Self {
        Self {
            position: pos,
            velocity: Vector3::new(0.0, 0.0, 0.0),
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
        Self {
            base,
            particles: Vec::new(),
        }
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

                    // Instantiate the sphere for each particle
                    let mut sphere = MeshInstance3D::new_alloc();
                    let mesh = SphereMesh::new().upcast::<Mesh>();
                    sphere.set_mesh(mesh);
                    sphere.set_scale(Vector3::new(
                        PARTICLE_DISTANCE * 0.5,
                        PARTICLE_DISTANCE * 0.5,
                        PARTICLE_DISTANCE * 0.5,
                    ));
                    self.base.add_child(sphere.upcast::<Node>());
                }
            }
        }

        // Connect adjacent particles based on distance threshold
        let mut connections = Vec::new();
        let len = self.particles.len();
        for i in 0..len {
            let p1 = &self.particles[i];
            for j in i + 1..len {
                let p2 = &self.particles[j];
                if p1.position.distance_to(p2.position) <= PARTICLE_DISTANCE * 2.0 {
                    // Store the connections temporarily
                    let connection = Connection::new(p1, p2);
                    connections.push((i, connection));
                }
            }
        }

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

        let mut new_velocities = Vec::with_capacity(self.particles.len());
        let mut new_positions = Vec::with_capacity(self.particles.len());

        let len = self.particles.len();
        for i in 0..len {
            let particle = &self.particles[i];
            if particle.anchored {
                continue;
            }

            // Initialize the net force to gravity
            let mut net_force = gravity;

            // // Connection Forces
            // for connection in &particle.connections {
            //     if connection.active {
            //         let target_position =
            //             connection.target.position + connection.direction * connection.distance;
            //         let direction = target_position - particle.position;
            //         let spring_force = direction * SPRING_CONSTANT;
            //         net_force += spring_force;
            //     }
            // }

            // // Repulsion Forces
            // for j in i + 1..len {
            //     let other = &self.particles[j];
            //     if other.position != particle.position {
            //         let difference = particle.position - other.position;
            //         let distance = difference.length();
            //         if distance < PARTICLE_DISTANCE {
            //             let repulsion_force = difference.normalized()
            //                 * (PARTICLE_DISTANCE - distance)
            //                 * REPULSION_STRENGTH;
            //             net_force += repulsion_force;
            //         }
            //     }
            // }

            // Calculate new velocity and position
            let new_velocity = particle.velocity + net_force * delta;
            let new_position = particle.position + new_velocity * delta * DAMPING;

            new_velocities.push(new_velocity);
            new_positions.push(new_position);
        }

        // Apply the new velocities and positions
        // for (i, particle) in self.particles.iter_mut().enumerate() {
        //     particle.velocity = new_velocities[i];
        //     particle.position = new_positions[i];
        // }
    }

    fn render_particles(&mut self) {
        for (i, particle) in self.particles.iter().enumerate() {
            if let Some(sphere) = self.base.get_child(i as i32) {
                if let Some(mut sphere_mesh) = sphere.try_cast::<MeshInstance3D>() {
                    sphere_mesh.set_global_position(particle.position);
                }
            }
        }
    }
}
