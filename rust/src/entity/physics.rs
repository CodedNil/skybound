use super::{Particle, PARTICLE_DISTANCE};
use godot::prelude::*;
use rand::seq::SliceRandom;

pub fn process_step(particles: &Vec<Particle>, delta: f32) -> Vec<(usize, Vector3)> {
    let start_time = std::time::Instant::now();
    let particle_mass = PARTICLE_DISTANCE * PARTICLE_DISTANCE;
    let gravity = Vector3::new(0.0, -15.0, 0.0) * particle_mass;

    let new_positions = particles
        .par_iter()
        .enumerate()
        .map(|(i, particle)| {
            // Initialize the net force to gravity
            let mut net_force = gravity;

            // Connection forces
            let spring_constant = 10.0;
            for connection in &particle.connections {
                if connection.active {
                    let target = &particles[connection.target_index];

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
            for (j, other_particle) in particles.iter().enumerate() {
                if i != j {
                    let direction = particle.position - other_particle.position;
                    let distance = direction.length();

                    if distance < target_distance {
                        let delta_distance = target_distance - distance;
                        let repulsion_force_magnitude =
                            repulsion_strength * delta_distance / distance.mul_add(distance, 1.0); // The +1.0 helps in preventing division by very small values
                        let repulsion_vector = direction.normalized() * repulsion_force_magnitude;
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
            let new_position = particle.position + velocity + net_force * delta * delta;

            (i, new_position)
        })
        .collect();

    godot_print!("Physics step took {}ms", start_time.elapsed().as_millis());
    new_positions
}
