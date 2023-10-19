use godot::{
    engine::{
        global::Key, ImmediateMesh, InputEvent, InputEventKey, Mesh, MeshInstance3D, RigidBody3D,
        RigidBody3DVirtual,
    },
    prelude::*,
};
use rapier3d::prelude::*;

mod cut;
// mod physics;
mod render;

const PARTICLE_DISTANCE: f32 = 0.25;

#[derive(PartialEq, Clone)]
pub struct Connection {
    target_index: usize,
    direction: Vector3,
    distance: f32,
    joint_handle: ImpulseJointHandle,
    active: bool,
}

impl Connection {
    const fn new(
        target_index: usize,
        direction: Vector3,
        distance: f32,
        joint_handle: ImpulseJointHandle,
    ) -> Self {
        Self {
            target_index,
            direction,
            distance,
            joint_handle,
            active: true,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct Particle {
    body_handle: RigidBodyHandle,
    connections: Vec<Connection>,
}

impl Particle {
    const fn new(body_handle: RigidBodyHandle) -> Self {
        Self {
            body_handle,
            connections: Vec::new(),
        }
    }

    fn get_position(&self, rigid_body_set: &RigidBodySet) -> Vector3 {
        match rigid_body_set.get(self.body_handle) {
            Some(rigid_body) => {
                let vec = rigid_body.translation();
                Vector3::new(vec.x, vec.y, vec.z)
            }
            None => Vector3::ZERO,
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

    physics_pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    physics_hooks: (),
    event_handler: (),
    accumulator: f32,

    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,

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

            physics_pipeline: PhysicsPipeline::new(),
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            physics_hooks: (),
            event_handler: (),
            accumulator: 0.0,

            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),

            plane_rotate: false,
            plane_size: 0.2,
        };
        instance.integration_parameters.dt = 1.0 / 60.0;
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

        /* Create the ground. */
        let collider = ColliderBuilder::cuboid(100.0, 0.5, 100.0)
            .translation(vector![0.0, -0.5, 0.0])
            .restitution(0.7)
            .build();
        self.collider_set.insert(collider);

        // Create grid of particles
        for x in -grid_size..=grid_size {
            for y in -grid_size..=grid_size {
                for z in -grid_size..=grid_size {
                    #[allow(clippy::cast_precision_loss)]
                    let body = RigidBodyBuilder::dynamic()
                        .translation(
                            vector![x as f32, y as f32, z as f32] * PARTICLE_DISTANCE
                                + vector![0.0, 2.0, 0.0],
                        )
                        .linear_damping(0.5)
                        .angular_damping(0.5)
                        .build();
                    let collider = ColliderBuilder::ball(0.05).restitution(0.7).build();
                    let handle = self.rigid_body_set.insert(body);
                    self.collider_set.insert_with_parent(
                        collider,
                        handle,
                        &mut self.rigid_body_set,
                    );

                    #[allow(clippy::cast_precision_loss)]
                    local_particles.push(Particle::new(handle));
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
                let particle_pos = particle.get_position(&self.rigid_body_set);
                for (other_index, other_particle) in local_particles.iter().enumerate() {
                    if particle_index == other_index || processed_particles.contains(&other_index) {
                        continue;
                    }
                    let other_particle_pos = other_particle.get_position(&self.rigid_body_set);

                    let distance = particle_pos.distance_to(other_particle_pos);
                    if distance > PARTICLE_DISTANCE * 1.5 {
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

                        let joint = FixedJointBuilder::new()
                            .local_anchor1(point![particle_pos.x, particle_pos.y, particle_pos.z])
                            .local_anchor2(point![
                                other_particle_pos.x,
                                other_particle_pos.y,
                                other_particle_pos.z
                            ]);
                        let joint_handle = self.impulse_joint_set.insert(
                            particle.body_handle,
                            other_particle.body_handle,
                            joint,
                            true,
                        );
                        new_connections.push((
                            particle_index,
                            Connection::new(other_index, direction, distance, joint_handle),
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
        self.accumulator += delta as f32;
        // If c is pressed, cut the connections that intersect with the plane
        if Input::singleton().is_key_pressed(Key::KEY_C) {
            self.cut_on_plane().unwrap();
        }

        while self.accumulator >= self.integration_parameters.dt {
            let gravity: Vector<Real> = vector![0.0, -9.81, 0.0];
            self.physics_pipeline.step(
                &gravity,
                &self.integration_parameters,
                &mut self.island_manager,
                &mut self.broad_phase,
                &mut self.narrow_phase,
                &mut self.rigid_body_set,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                &mut self.ccd_solver,
                None,
                &self.physics_hooks,
                &self.event_handler,
            );
            self.accumulator -= self.integration_parameters.dt;
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
