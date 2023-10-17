use super::{Particle, PARTICLE_DISTANCE};
use godot::prelude::*;
use rand::seq::SliceRandom;

pub fn process_step(particles: &Vec<Particle>, delta: f32) -> Vec<(usize, Vector3)> {
    // Position-Based Dynamics

    let start_time = std::time::Instant::now();
    let particle_mass = PARTICLE_DISTANCE * PARTICLE_DISTANCE;
    let gravity = Vector3::new(0.0, -15.0, 0.0) * particle_mass;

    // 1. Prediction Phase
    let damping = 0.98;
    let mut predicted_positions: Vec<Vector3> = Vec::with_capacity(particles.len());
    for particle in particles {
        let velocity = (particle.position - particle.old_position) * damping;
        let new_position = particle.position + velocity + gravity * delta * delta;
        predicted_positions.push(new_position);
    }

    // Shuffle indices for randomized processing
    let mut indices: Vec<usize> = (0..particles.len()).collect();
    indices.shuffle(&mut rand::thread_rng());

    // 2. Constraint Phase
    let solver_iterations: i32 = 8;
    let mut corrected_positions = predicted_positions.clone();
    for _ in 0..solver_iterations {
        // Collision constraints
        let restitution = 0.5; // 1 is perfectly elastic, 0 is no bounce
        for &i in &indices {
            if corrected_positions[i].y < 0.0 {
                let penetration = 0.0 - corrected_positions[i].y;

                let approach_velocity =
                    (predicted_positions[i] - particles[i].old_position).y / delta;
                let bounce_velocity = if approach_velocity < 0.0 {
                    -approach_velocity * restitution
                } else {
                    0.0
                };

                let total_correction = penetration + bounce_velocity * delta;
                corrected_positions[i].y += total_correction;
            }
        }

        // Self-collision constraints
        let minimum_distance = PARTICLE_DISTANCE;
        let stiffness = 0.2;
        for &i in &indices {
            for &j in &indices {
                if i >= j {
                    continue; // Skip self-check and duplicate checks
                }
                let dir = corrected_positions[j] - corrected_positions[i];
                let current_distance = dir.length();
                if current_distance < minimum_distance {
                    let correction = (dir / current_distance)
                        * ((minimum_distance - current_distance) * 0.5)
                        * stiffness;
                    corrected_positions[i] -= correction;
                    corrected_positions[j] += correction;
                }
            }
        }

        // Spring constraints
        let stiffness = 0.2;
        for &i in &indices {
            let particle = &particles[i];
            for connection in &particle.connections {
                if connection.active {
                    let j = connection.target_index;
                    let direction = (corrected_positions[j] - corrected_positions[i]).normalized();
                    let current_distance =
                        (corrected_positions[j] - corrected_positions[i]).length();
                    let difference = current_distance - connection.distance;
                    let correction = direction * (difference * 0.5) * stiffness;

                    // Adjust the positions of both particles
                    corrected_positions[i] += correction;
                    corrected_positions[j] -= correction;
                }
            }
        }
    }

    godot_print!("Physics step took {}ms", start_time.elapsed().as_millis());

    // 3. Update Phase
    corrected_positions
        .iter()
        .enumerate()
        .map(|(i, &pos)| (i, pos))
        .collect()
}

// particles
//     .par_iter()
//     .enumerate()
//     .map(|(i, particle)| {
//         if particle.anchored {
//             (i, particle.position)
//         } else {
//             // Initialize the net force to gravity
//             let mut net_force = gravity;

//             // Connection forces
//             let spring_constant = 10.0;
//             for connection in &particle.connections {
//                 if connection.active {
//                     let target = &particles[connection.target_index];

//                     let direction = target.position - particle.position;
//                     let current_distance = direction.length();

//                     let delta_distance = current_distance - connection.distance;
//                     let spring_force_magnitude = spring_constant * delta_distance;
//                     let force_vector = direction.normalized() * spring_force_magnitude;

//                     net_force += force_vector;
//                 }
//             }

//             // Clamp max force
//             let max_force = 100.0;
//             if net_force.length() > max_force {
//                 net_force = net_force.normalized() * max_force;
//             }

//             // Verlet integration step
//             let damping = 0.98;
//             let velocity = (particle.position - particle.old_position) * damping;
//             let new_position = particle.position + velocity + net_force * delta * delta;

//             (i, new_position)
//         }
//     })
//     .collect()

//             // Repulsion forces
//             let target_distance = PARTICLE_DISTANCE;
//             let repulsion_strength: f32 = 10.0;
//             for (j, other_particle) in particles.iter().enumerate() {
//                 if i != j {
//                     let direction = particle.position - other_particle.position;
//                     let distance = direction.length();

//                     if distance < target_distance {
//                         let delta_distance = target_distance - distance;
//                         let repulsion_force_magnitude = repulsion_strength * delta_distance
//                             / distance.mul_add(distance, 1.0); // The +1.0 helps in preventing division by very small values
//                         let repulsion_vector =
//                             direction.normalized() * repulsion_force_magnitude;
//                         net_force += repulsion_vector;
//                     }
//                 }
//             }
