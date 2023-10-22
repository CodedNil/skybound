use super::{Particle, PARTICLE_DISTANCE};
use crate::entity::ParticleMaterial;
use godot::prelude::*;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Mutex};

#[allow(clippy::too_many_lines)]
pub fn process_step(particles: &Vec<Particle>, delta: f32) -> Vec<(usize, Vector3)> {
    let start_time = std::time::Instant::now();
    let gravity = Vector3::new(0.0, -10.0, 0.0);

    // Create grid for lookup
    let cell_size = PARTICLE_DISTANCE;
    let mut grid: HashMap<(isize, isize), Vec<usize>> = HashMap::new();
    for (index, particle) in particles.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let (cell_x, cell_y) = (
            (particle.position.x / cell_size) as isize,
            (particle.position.y / cell_size) as isize,
        );
        grid.entry((cell_x, cell_y)).or_default().push(index);
    }

    // 1. Prediction Phase
    let damping = 0.98;
    let mut predicted_positions: Vec<Vector3> = Vec::with_capacity(particles.len());
    for particle in particles {
        let velocity = (particle.position - particle.old_position) * damping;

        let mut acceleration = gravity;

        // Wing uplift
        if particle.material == ParticleMaterial::Wing {
            let wing_uplift_strength = 50.0;
            acceleration += Vector3::new(0.0, wing_uplift_strength, 0.0);
        }

        let mut new_position = particle.position + velocity + acceleration * delta * delta;
        if particle.material == ParticleMaterial::Bone {
            new_position = particle.position;
        }
        predicted_positions.push(new_position);
    }

    // Shuffle indices for randomized processing
    let mut indices: Vec<usize> = (0..particles.len()).collect();
    indices.shuffle(&mut rand::thread_rng());

    // 2. Constraint Phase
    let solver_iterations: i32 = 4;
    let mut corrected_positions = predicted_positions.clone();
    for _ in 0..solver_iterations {
        // Collision constraints
        let restitution = 0.7; // 1 is perfectly elastic, 0 is no bounce
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
        let corrections: Mutex<Vec<(usize, Vector3)>> = Mutex::new(Vec::new());

        indices.par_iter().for_each(|&particle_index| {
            let position = corrected_positions[particle_index];
            let mut local_corrections = Vec::new();

            #[allow(clippy::cast_possible_truncation)]
            let (cell_x, cell_y) = (
                (position.x / cell_size) as isize,
                (position.y / cell_size) as isize,
            );
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(neighbors) = grid.get(&(cell_x + dx, cell_y + dy)) {
                        for &other_index in neighbors {
                            if particle_index == other_index {
                                continue;
                            }

                            let dir = corrected_positions[other_index] - position;
                            let current_distance = dir.length();
                            if current_distance < minimum_distance {
                                let correction = (dir / current_distance)
                                    * ((minimum_distance - current_distance) * 0.5)
                                    * stiffness;
                                local_corrections.push((particle_index, -correction));
                                local_corrections.push((other_index, correction));
                            }
                        }
                    }
                }
            }
            // Extend the shared corrections with local results
            corrections
                .lock()
                .unwrap()
                .extend(local_corrections.into_iter());
        });
        // Apply corrections
        for (index, correction) in corrections.lock().unwrap().iter() {
            corrected_positions[*index] += *correction;
        }

        // Spring constraints
        let stiffness = 0.1;
        for &i in &indices {
            let particle = &particles[i];

            for connection in &particle.connections {
                if connection.active {
                    // let j = connection.target_index;
                    // let offset = corrected_positions[j] - corrected_positions[i];
                    // let direction = offset.normalized();
                    // let current_distance = offset.length();
                    // let difference = current_distance - connection.distance;
                    // let correction = direction * (difference * 0.5) * stiffness;

                    // // Adjust the positions of both particles
                    // corrected_positions[i] += correction;
                    // corrected_positions[j] -= correction;

                    let j = connection.target_index;

                    // Compute the offset and its direction
                    let offset = corrected_positions[j] - corrected_positions[i];
                    let direction = offset.normalized();

                    // Compute how much the current direction deviates from the desired direction
                    let direction_difference = direction - connection.direction.normalized();

                    // Calculate the correction based on the direction difference
                    let directional_correction = direction_difference * stiffness;

                    // Calculate the stretching or compressing of the spring
                    let current_distance = offset.length();
                    let distance_difference = current_distance - connection.distance;
                    let stretch_correction = direction * (distance_difference * 0.5) * stiffness;

                    // Combine the corrections
                    let total_correction = stretch_correction + directional_correction;

                    // Adjust the positions of both particles
                    corrected_positions[i] += total_correction;
                    corrected_positions[j] -= total_correction;
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
