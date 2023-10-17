use godot::prelude::*;
use rayon::prelude::*;

use super::{Entity, PARTICLE_DISTANCE};

impl Entity {
    pub fn process_physics(&mut self, delta: f64) {
        #[allow(clippy::cast_possible_truncation)]
        let delta = delta as f32;

        let particle_mass = PARTICLE_DISTANCE * PARTICLE_DISTANCE;
        let gravity = Vector3::new(0.0, -15.0, 0.0) * particle_mass;

        let new_positions: Vec<(usize, Vector3)> = self
            .particles
            .par_iter()
            .enumerate()
            .map(|(i, particle)| {
                if particle.anchored {
                    (i, particle.position)
                } else {
                    // Initialize the net force to gravity
                    let mut net_force = gravity;

                    // Connection forces
                    let spring_constant = 10.0;
                    for connection in &particle.connections {
                        if connection.active {
                            let target = &self.particles[connection.target_index];

                            let direction = target.position - particle.position;
                            let current_distance = direction.length();

                            let delta_distance = current_distance - connection.distance;
                            let spring_force_magnitude = spring_constant * delta_distance;
                            let force_vector = direction.normalized() * spring_force_magnitude;

                            net_force += force_vector;
                        }
                    }

                    // Repulsion forces
                    let target_distance = PARTICLE_DISTANCE;
                    let repulsion_strength: f32 = 10.0;
                    for (j, other_particle) in self.particles.iter().enumerate() {
                        if i != j {
                            let direction = particle.position - other_particle.position;
                            let distance = direction.length();

                            if distance < target_distance {
                                let delta_distance = target_distance - distance;
                                let repulsion_force_magnitude = repulsion_strength * delta_distance
                                    / distance.mul_add(distance, 1.0); // The +1.0 helps in preventing division by very small values
                                let repulsion_vector =
                                    direction.normalized() * repulsion_force_magnitude;
                                net_force += repulsion_vector;
                            }
                        }
                    }

                    // Clamp max force
                    let max_force = 100.0;
                    if net_force.length() > max_force {
                        net_force = net_force.normalized() * max_force;
                    }

                    // Verlet integration step
                    let damping = 0.98;
                    let velocity = (particle.position - particle.old_position) * damping;
                    let new_pos = particle.position + velocity + net_force * delta * delta;

                    (i, new_pos)
                }
            })
            .collect();

        // Apply the new positions and store old positions
        for (i, new_position) in new_positions {
            self.particles[i].old_position = self.particles[i].position;
            self.particles[i].position = new_position;
        }
    }
}