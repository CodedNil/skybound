use godot::{
    engine::{
        global::Key, ImmediateMesh, InputEvent, InputEventKey, Mesh, MeshInstance3D, RigidBody3D,
        RigidBody3DVirtual,
    },
    prelude::*,
};
use rayon::prelude::*;
use std::{
    sync::{
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
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
    const fn new(pos: Vector3) -> Self {
        Self {
            position: pos,
            old_position: pos,
            connections: Vec::new(),
        }
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Entity {
    #[base]
    base: Base<RigidBody3D>,
    particles: Arc<Mutex<Vec<Particle>>>,
    central_particle: usize,
    physics_sender: mpsc::Sender<Vec<(usize, Vector3)>>,
    physics_receiver: Receiver<Vec<(usize, Vector3)>>,
    time_since_last_tick: f32,
    physics_update_rate: f32,

    plane_rotate: bool,
    plane_size: f32,
}

#[godot_api]
impl RigidBody3DVirtual for Entity {
    fn init(base: Base<RigidBody3D>) -> Self {
        // Create a channel for physics communication
        let (sender, main_receiver) = mpsc::channel();
        let shared_particles = Arc::new(Mutex::new(Vec::new()));

        let mut instance = Self {
            base,
            particles: shared_particles,
            central_particle: 0,
            physics_sender: sender,
            physics_receiver: main_receiver,
            time_since_last_tick: 0.0,
            physics_update_rate: 0.05,

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
                    local_particles.push(Particle::new(
                        Vector3::new(x as f32, y as f32 + 4.0, z as f32) * PARTICLE_DISTANCE,
                    ));
                    // Set the central particle
                    if x == 0 && y == 0 && z == 0 {
                        self.central_particle = local_particles.len() - 1;
                    }
                }
            }
        }

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
                for (other_index, other_particle) in local_particles.iter().enumerate() {
                    if particle_index == other_index || processed_particles.contains(&other_index) {
                        continue;
                    }

                    if particle.position.distance_to(other_particle.position)
                        > PARTICLE_DISTANCE * 1.8
                    {
                        continue;
                    }

                    // Check if that particle already has a connection to the current particle
                    let has_connection = local_particles[other_index]
                        .connections
                        .iter()
                        .any(|connection| connection.target_index == particle_index);

                    if !has_connection {
                        // Connect the particles
                        let direction = (other_particle.position - particle.position).normalized();
                        let distance = particle.position.distance_to(other_particle.position);
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

        // Lock the mutex and replace the shared particles with the local version
        let mut shared_particles = self.particles.lock().unwrap();
        *shared_particles = local_particles;

        // Start the physics thread
        let physics_particles = Arc::clone(&self.particles);
        let physics_update_rate = 0.05;
        let sender = self.physics_sender.clone();
        thread::spawn(move || {
            let mut last_run = Instant::now();
            loop {
                if last_run.elapsed().as_secs_f32() > physics_update_rate {
                    let particles = physics_particles.lock().unwrap().clone();
                    let new_positions = physics::process_step(&particles, physics_update_rate);
                    sender.send(new_positions).unwrap();
                    last_run = Instant::now();
                }
                thread::sleep(Duration::from_millis(1));
            }
        });
    }

    #[allow(clippy::cast_possible_truncation)]
    fn process(&mut self, delta: f64) {
        self.render_particles(self.time_since_last_tick / self.physics_update_rate);
        // If c is pressed, cut the connections that intersect with the plane
        if Input::singleton().is_key_pressed(Key::KEY_C) {
            self.cut_on_plane().unwrap();
        }

        // Receive and apply the new positions
        self.time_since_last_tick += delta as f32;
        if let Ok(new_positions) = self.physics_receiver.try_recv() {
            let mut particles = self.particles.lock().unwrap();
            for (i, new_position) in new_positions {
                particles[i].old_position = particles[i].position;
                particles[i].position = new_position;
            }
            self.time_since_last_tick = 0.0;
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
