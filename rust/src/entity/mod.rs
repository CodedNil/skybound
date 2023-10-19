use godot::{
    engine::{
        global::Key, ImmediateMesh, InputEvent, InputEventKey, Mesh, MeshInstance3D, RigidBody3D,
        RigidBody3DVirtual,
    },
    prelude::*,
};

mod cut;
mod physics;
mod render;

const PARTICLE_DISTANCE: f32 = 0.25;

#[derive(PartialEq, Clone)]
pub struct Connection {
    target_index: usize,
    direction: Vector3,
    distance: f32,
    active: bool,
}

impl Connection {
    const fn new(target_index: usize, direction: Vector3, distance: f32) -> Self {
        Self {
            target_index,
            direction,
            distance,
            active: true,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct Particle {
    position: Vector3,
    old_position: Vector3,
    connections: Vec<Connection>,
}

impl Particle {
    const fn new(position: Vector3) -> Self {
        Self {
            position,
            old_position: position,
            connections: Vec::new(),
        }
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Entity {
    #[base]
    base: Base<RigidBody3D>,
    particles: Vec<Particle>,
    central_particle: usize,

    accumulator: f32,

    plane_rotate: bool,
    plane_size: f32,
}

#[godot_api]
impl RigidBody3DVirtual for Entity {
    fn init(base: Base<RigidBody3D>) -> Self {
        let mut instance = Self {
            base,
            particles: Vec::new(),
            central_particle: 0,

            accumulator: 0.0,

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
        let grid_size: i32 = 3;
        let mut local_particles = Vec::new();

        // Create grid of particles
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                for z in -grid_size..=grid_size {
                    #[allow(clippy::cast_precision_loss)]
                    let position =
                        Vector3::new(x as f32, y as f32 + 5.0, z as f32) * PARTICLE_DISTANCE;
                    local_particles.push(Particle::new(position));
                    // Set the central particle
                    if x == 0 && y == 0 && z == 0 {
                        self.central_particle = local_particles.len() - 1;
                    }
                }
            }
        }
        godot_print!("Created {} particles", local_particles.len());

        // Connect particles, starting from central particle expanding outwards
        let mut total_connections = 0;
        let mut processed_particles = Vec::new();
        let mut connected_particles_next = vec![self.central_particle];
        while !connected_particles_next.is_empty() {
            let mut next_next = Vec::new();
            let mut new_connections = Vec::new();

            for particle_index in connected_particles_next {
                // If the particle has already been processed, skip
                if processed_particles.contains(&particle_index) {
                    continue;
                }
                // Add the particle to the connected list
                processed_particles.push(particle_index);

                // Connect the particle to nearby particles
                let particle = &local_particles[particle_index];
                let particle_pos = particle.position;
                for (other_index, other_particle) in local_particles.iter().enumerate() {
                    if particle_index == other_index || processed_particles.contains(&other_index) {
                        continue;
                    }
                    let other_particle_pos = other_particle.position;

                    let distance = particle_pos.distance_to(other_particle_pos);
                    if distance > PARTICLE_DISTANCE * 1.8 {
                        continue;
                    }

                    // Check if that particle already has a connection to the current particle
                    let has_connection = local_particles[other_index]
                        .connections
                        .iter()
                        .any(|connection| connection.target_index == particle_index);

                    if !has_connection {
                        // Connect the particles
                        let direction = (other_particle_pos - particle_pos).normalized();

                        new_connections.push((
                            particle_index,
                            Connection::new(other_index, direction, distance),
                        ));

                        total_connections += 1;

                        // Add the other particle to the next list
                        next_next.push(other_index);
                    }
                }
            }

            // Set the next list to the next next list
            connected_particles_next = next_next;

            // Add connections to particles
            for (index, connection) in new_connections {
                local_particles[index].connections.push(connection);
            }
        }
        godot_print!("Connected particles: {}", total_connections);

        // Replace the particles with the local version
        self.particles = local_particles;
    }

    #[allow(clippy::cast_possible_truncation)]
    fn process(&mut self, delta: f64) {
        // If c is pressed, cut the connections that intersect with the plane
        if Input::singleton().is_key_pressed(Key::KEY_C) {
            self.cut_on_plane().unwrap();
        }

        let time_step = 1.0 / 60.0;
        self.accumulator += delta as f32;
        if self.accumulator >= time_step {
            // Physics step
            let new_positions = physics::process_step(&self.particles, time_step);
            for (index, position) in new_positions {
                self.particles[index].old_position = self.particles[index].position;
                self.particles[index].position = position;
            }
            self.accumulator -= time_step;
        }

        self.render_particles();
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
