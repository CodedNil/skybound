use godot::{
    engine::{RigidBody3D, RigidBody3DVirtual},
    prelude::*,
};

const PARTICLE_DISTANCE: f32 = 0.25;

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
                    let mut particle = Particle::new(
                        self.base.get_global_position()
                            + Vector3::new(x as f32, y as f32, z as f32) * PARTICLE_DISTANCE,
                    );
                    let error_margin = f32::EPSILON;
                    if (particle.position.x - self.base.get_global_position().x).abs()
                        < error_margin
                        && (particle.position.y - self.base.get_global_position().y).abs()
                            < error_margin
                    {
                        particle.anchored = true;
                    }
                    self.particles.push(particle);
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

    // fn process(&mut self, delta: f64) {}
}
