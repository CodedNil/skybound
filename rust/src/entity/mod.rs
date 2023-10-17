use godot::{
    engine::{
        global::Key, ImmediateMesh, InputEvent, InputEventKey, Mesh, MeshInstance3D, RigidBody3D,
        RigidBody3DVirtual,
    },
    prelude::*,
};
use rayon::prelude::*;

mod cut;
mod physics;
mod render;

const PARTICLE_DISTANCE: f32 = 0.25;

#[derive(PartialEq, Clone)]
struct Connection {
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
                    #[allow(clippy::cast_precision_loss)]
                    let mut particle = Particle::new(
                        self.base.get_global_position()
                            + Vector3::new(x as f32, y as f32, z as f32) * PARTICLE_DISTANCE,
                    );
                    // Anchor a the top corner particles
                    if (x == -grid_size || x == grid_size)
                        && y == grid_size
                        && (z == -grid_size || z == grid_size)
                    {
                        // if y == grid_size {
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
                            <= PARTICLE_DISTANCE * 1.8
                    {
                        // Connect the particles
                        let direction = (particle2.position - particle1.position).normalized();
                        let dist = particle1.position.distance_to(particle2.position);
                        local_connections.push((idx1, Connection::new(idx2, direction, dist)));
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
