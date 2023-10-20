use godot::{
    engine::{
        global::Key, ImmediateMesh, InputEvent, InputEventKey, Mesh, MeshInstance3D, RigidBody3D,
        RigidBody3DVirtual,
    },
    prelude::*,
};
use rapier3d::prelude::*;

mod cut;
mod render;

const PARTICLE_DISTANCE: f32 = 0.2;

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

enum ShapeType {
    Box,
    Ellipsoid,
}

struct Shape {
    name: String,
    material: String,
    transform: Transform3D,
    shape_type: ShapeType,
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

        instance
    }

    fn ready(&mut self) {
        let mut local_particles = Vec::new();

        // // Create the ground
        // let collider = ColliderBuilder::cuboid(100.0, 0.5, 100.0)
        //     .translation(vector![0.0, -0.5, 0.0])
        //     .restitution(0.7)
        //     .build();
        // self.collider_set.insert(collider);

        let (shapes, grid_min, grid_max) = self.sort_shapes();

        for x in grid_min.0..=grid_max.0 {
            for y in grid_min.1..=grid_max.1 {
                for z in grid_min.2..=grid_max.2 {
                    #[allow(clippy::cast_precision_loss)]
                    let position = Vector3::new(x as f32, y as f32, z as f32) * PARTICLE_DISTANCE;

                    // Check if the position is inside a shape
                    let mut inside_shape = false;
                    for shape in &shapes {
                        let shape_size = shape.transform.basis.scale();
                        let local_position = shape.transform.affine_inverse() * position;
                        let inside = match shape.shape_type {
                            ShapeType::Box => {
                                local_position.x.abs() < shape_size.x * 0.5
                                    && local_position.y.abs() < shape_size.y * 0.5
                                    && local_position.z.abs() < shape_size.z * 0.5
                            }
                            ShapeType::Ellipsoid => {
                                local_position.x.powi(2)
                                    + local_position.y.powi(2)
                                    + local_position.z.powi(2)
                                    < 0.25
                            }
                        };
                        if inside {
                            inside_shape = true;
                            break;
                        }
                    }
                    if !inside_shape {
                        continue;
                    }

                    let body = RigidBodyBuilder::dynamic()
                        .translation(vector![position.x, position.y, position.z])
                        .linear_damping(0.5)
                        .angular_damping(0.5)
                        .build();
                    let collider = ColliderBuilder::ball(PARTICLE_DISTANCE * 0.5)
                        .restitution(0.7)
                        .build();
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

        self.connect_particles(&mut local_particles);

        // Replace the particles with the local version
        self.particles = local_particles;

        // Add render mesh
        let mut render_mesh = MeshInstance3D::new_alloc();
        render_mesh.set_mesh(ImmediateMesh::new().upcast::<Mesh>());
        self.base.add_child(render_mesh.upcast::<Node>());
    }

    #[allow(clippy::cast_possible_truncation)]
    fn process(&mut self, delta: f64) {
        self.accumulator += delta as f32;
        // If c is pressed, cut the connections that intersect with the plane
        if Input::singleton().is_key_pressed(Key::KEY_C) {
            self.cut_on_plane().unwrap();
        }

        // if self.accumulator >= self.integration_parameters.dt {
        //     let gravity: Vector<Real> = vector![0.0, -9.81, 0.0];
        //     self.physics_pipeline.step(
        //         &gravity,
        //         &self.integration_parameters,
        //         &mut self.island_manager,
        //         &mut self.broad_phase,
        //         &mut self.narrow_phase,
        //         &mut self.rigid_body_set,
        //         &mut self.collider_set,
        //         &mut self.impulse_joint_set,
        //         &mut self.multibody_joint_set,
        //         &mut self.ccd_solver,
        //         None,
        //         &self.physics_hooks,
        //         &self.event_handler,
        //     );
        //     self.accumulator -= self.integration_parameters.dt;
        // }

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

type IntVec3 = (i32, i32, i32);

impl Entity {
    fn sort_shapes(&mut self) -> (Vec<Shape>, IntVec3, IntVec3) {
        // Get all meshes in entity
        let mut shapes = Vec::new();
        let mut bounds_min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut bounds_max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
        for mesh in self.base.get_children().iter_shared() {
            if let Some(mesh_instance) = mesh.try_cast::<MeshInstance3D>() {
                godot_print!("Found mesh: {}", mesh_instance.get_name());

                // Get the global transform of the mesh
                let transform = mesh_instance.get_global_transform();

                // Transform these local corners to world space
                let corner_1 = transform * Vector3::new(-0.5, -0.5, -0.5);
                let corner_2 = transform * Vector3::new(0.5, 0.5, 0.5);

                // Update the global bounds.
                bounds_min.x = f32::min(bounds_min.x, corner_1.x);
                bounds_min.y = f32::min(bounds_min.y, corner_1.y);
                bounds_min.z = f32::min(bounds_min.z, corner_1.z);
                bounds_min.x = f32::min(bounds_min.x, corner_2.x);
                bounds_min.y = f32::min(bounds_min.y, corner_2.y);
                bounds_min.z = f32::min(bounds_min.z, corner_2.z);

                bounds_max.x = f32::max(bounds_max.x, corner_1.x);
                bounds_max.y = f32::max(bounds_max.y, corner_1.y);
                bounds_max.z = f32::max(bounds_max.z, corner_1.z);
                bounds_max.x = f32::max(bounds_max.x, corner_2.x);
                bounds_max.y = f32::max(bounds_max.y, corner_2.y);
                bounds_max.z = f32::max(bounds_max.z, corner_2.z);

                let shape_type = match mesh_instance
                    .get_mesh()
                    .unwrap()
                    .get_class()
                    .to_string()
                    .as_str()
                {
                    "BoxMesh" => ShapeType::Box,
                    "SphereMesh" => ShapeType::Ellipsoid,
                    _ => {
                        godot_error!(
                            "Unexpected mesh type for mesh_instance: {}",
                            mesh_instance.get_name()
                        );
                        continue;
                    }
                };

                // Add shape
                let name = mesh_instance.get_name().to_string();
                let name_parts: Vec<&str> = name.split('_').collect();

                if name_parts.len() == 2 {
                    let name = name_parts[0].to_string();
                    let material = name_parts[1].to_string();

                    shapes.push(Shape {
                        name,
                        material,
                        transform,
                        shape_type,
                    });
                } else {
                    godot_error!(
                        "Unexpected name format for mesh_instance: {}",
                        mesh_instance.get_name()
                    );
                }
            }
        }
        // Round the bounds to the nearest particle distance
        bounds_min.x = (bounds_min.x / PARTICLE_DISTANCE).floor() * PARTICLE_DISTANCE;
        bounds_min.y = (bounds_min.y / PARTICLE_DISTANCE).floor() * PARTICLE_DISTANCE;
        bounds_min.z = (bounds_min.z / PARTICLE_DISTANCE).floor() * PARTICLE_DISTANCE;

        bounds_max.x = (bounds_max.x / PARTICLE_DISTANCE).ceil() * PARTICLE_DISTANCE;
        bounds_max.y = (bounds_max.y / PARTICLE_DISTANCE).ceil() * PARTICLE_DISTANCE;
        bounds_max.z = (bounds_max.z / PARTICLE_DISTANCE).ceil() * PARTICLE_DISTANCE;

        // Remove the meshes
        // for mesh in self.base.get_children().iter_shared() {
        //     self.base.remove_child(mesh);
        // }
        godot_print!("Bounds: {} {}", bounds_min, bounds_max);

        // Create grid of particles
        #[allow(clippy::cast_possible_truncation)]
        let grid_min = (
            (bounds_min.x / PARTICLE_DISTANCE) as i32,
            (bounds_min.y / PARTICLE_DISTANCE) as i32,
            (bounds_min.z / PARTICLE_DISTANCE) as i32,
        );
        #[allow(clippy::cast_possible_truncation)]
        let grid_max = (
            (bounds_max.x / PARTICLE_DISTANCE) as i32,
            (bounds_max.y / PARTICLE_DISTANCE) as i32,
            (bounds_max.z / PARTICLE_DISTANCE) as i32,
        );

        (shapes, grid_min, grid_max)
    }

    fn connect_particles(&mut self, local_particles: &mut [Particle]) {
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
    }
}
