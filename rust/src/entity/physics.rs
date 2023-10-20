use std::collections::HashMap;

use super::{Particle, PARTICLE_DISTANCE};
use godot::prelude::*;
use rayon::prelude::*;

pub fn process_step(particles: &Vec<Particle>, delta: f32) -> Vec<(usize, Vector3)> {
    let start_time = std::time::Instant::now();
    let particle_mass = PARTICLE_DISTANCE * PARTICLE_DISTANCE;
    let gravity = Vector3::new(0.0, -50.0, 0.0) * particle_mass;

    // Create grid for lookup
    let cell_size = PARTICLE_DISTANCE * 1.5;
    let mut grid: HashMap<(isize, isize), Vec<usize>> = HashMap::new();
    for (index, particle) in particles.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let (cell_x, cell_y) = (
            (particle.position.x / cell_size) as isize,
            (particle.position.y / cell_size) as isize,
        );
        grid.entry((cell_x, cell_y)).or_default().push(index);
    }

    let new_positions = particles
        .par_iter()
        .enumerate()
        .map(|(particle_index, particle)| {
            // Initialize the net force to gravity
            let mut net_force = gravity;

            // Connection forces
            let spring_constant = 50.0;
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
            let repulsion_strength: f32 = 200.0;
            #[allow(clippy::cast_possible_truncation)]
            let (cell_x, cell_y) = (
                (particle.position.x / cell_size) as isize,
                (particle.position.y / cell_size) as isize,
            );
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(neighbors) = grid.get(&(cell_x + dx, cell_y + dy)) {
                        for &other_index in neighbors {
                            if particle_index == other_index {
                                continue;
                            }

                            let direction = particle.position - particles[other_index].position;
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
            let mut new_position = particle.position + velocity + net_force * delta * delta;

            // Check for floor collision
            let floor_y = 0.0;
            let restitution = 0.7;
            if new_position.y < floor_y {
                // Reflect the implicit velocity based on restitution (simple bounce)
                let implicit_velocity_y = new_position.y - particle.old_position.y;
                let reflected_velocity_y = -implicit_velocity_y * restitution;

                new_position.y = floor_y + reflected_velocity_y;
            }

            (particle_index, new_position)
        })
        .collect();

    godot_print!("Physics step took {}ms", start_time.elapsed().as_millis());
    new_positions
}
