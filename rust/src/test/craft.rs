use crate::gpt::{query_openai, Func, FuncParam, MessageResponse};
use anyhow::{anyhow, Context, Result};
use godot::{
    engine::{RigidBody3D, RigidBody3DVirtual},
    prelude::*,
};
use smol::channel::{unbounded, Receiver};

// The crafts mind, it's goals and it's outputs, things it alters each thought cycle
#[derive(Default, Debug)]
struct Mind {
    goal_thoughts: String,
    goal_long: String,
    goal_medium: String,
    goal_short: String,

    target_velocity: Vector3,
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Craft {
    gpt_result_rx: Option<Receiver<Result<MessageResponse>>>,
    mind: Mind,

    #[base]
    base: Base<RigidBody3D>,
}

#[godot_api]
impl RigidBody3DVirtual for Craft {
    fn init(base: Base<RigidBody3D>) -> Self {
        let mut instance = Self {
            base,
            mind: Mind::default(),
            gpt_result_rx: None,
        };
        instance.base.add_to_group("Craft".into());
        instance.base.set_contact_monitor(true);
        instance.base.set_max_contacts_reported(10);
        instance
    }

    fn process(&mut self, delta: f64) {
        // self.seek_food();
        self.eat_food();

        self.brain_think();
        if let Err(e) = self.brain_process() {
            godot_error!("Error processing brain: {:?}", e);
        }

        let velocity: Vector3 = self.mind.target_velocity.into();
        self.base.set_linear_velocity(velocity.normalized() * 2.0);
    }
}

impl Craft {
    fn get_all_food(&mut self) -> Vec<Gd<RigidBody3D>> {
        let mut foods = vec![];
        if let Some(mut scene_tree) = self.base.get_tree() {
            let food = scene_tree.get_nodes_in_group("Food".into());
            for node in food.iter_shared() {
                if let Some(food) = node.try_cast::<RigidBody3D>() {
                    foods.push(food);
                }
            }
        }
        foods
    }

    fn seek_food(&mut self) {
        // Get food objects
        let foods = self.get_all_food();

        // Get closet food object
        let mut closest_food = None;
        let mut closest_distance = 0.0;
        for food in &foods {
            let distance = self
                .base
                .get_global_position()
                .distance_to(food.get_global_position());
            if closest_food.is_none() || distance < closest_distance {
                closest_food = Some(food);
                closest_distance = distance;
            }
        }

        // Move towards closest food object
        if let Some(food) = closest_food {
            let direction = food.get_global_position() - self.base.get_global_position();
            let direction = direction.normalized();
            self.base.set_linear_velocity(direction * 2.0);
        }
    }

    fn eat_food(&mut self) {
        let colliders = self.base.get_colliding_bodies();
        for collider in colliders.iter_shared() {
            if let Some(mut food) = collider.try_cast::<RigidBody3D>() {
                if food.is_in_group("Food".into()) {
                    food.queue_free();
                }
            }
        }
    }

    fn brain_think(&mut self) {
        if self.gpt_result_rx.is_none() {
            // Initialise the system_message
            let mut system_message: String =
                "You are a game npc aircraft, you like collecting food\n\n".into();

            // Get food positions
            let foods = self.get_all_food();
            let mut food_pos: Vec<String> = vec![];
            for food in &foods {
                food_pos.push(vector3_to_string(food.get_global_position()));
            }

            // Add perception to system_message
            let perception_data = [
                format!(
                    "Position: {}",
                    vector3_to_string(self.base.get_global_position())
                ),
                format!(
                    "Velocity: {}",
                    vector3_to_string(self.base.get_linear_velocity())
                ),
                format!("Food Positions: {}", food_pos.join(", ")),
            ]
            .join("\n");
            system_message.push_str(&format!("Your perception data:\n{perception_data}\n\n",));

            // Add mind to system_message
            let mind_data = [
                format!("goal_thoughts: {}", self.mind.goal_thoughts),
                format!("goal_long: {}", self.mind.goal_long),
                format!("goal_medium: {}", self.mind.goal_medium),
                format!("goal_short: {}", self.mind.goal_short),
                format!(
                    "target_velocity: {}",
                    vector3_to_string(self.mind.target_velocity)
                ),
            ]
            .join("\n");
            system_message.push_str(&format!("Your mind state as toml data is\n{mind_data}"));

            // Create function
            let function = Func::new(
                "set_mind_state",
                "Update the mind state based on perception and current mind state",
                vec![
                    FuncParam::new("goal_thoughts", "Thoughts surrounding the crafts situation and information that decides the goals"),
                    FuncParam::new("goal_long", "Long term goal, what the craft is trying to achieve over the next few hours"),
                    FuncParam::new("goal_medium", "Medium term goal, what the craft is trying to achieve over the next 30 minutes"),
                    FuncParam::new("goal_short", "Short term goal, what the craft is trying to achieve in the next few minutes"),
                    FuncParam::new("target_velocity", "The velocity the craft should be aiming for, normalised so that 1.0 is max speed, in Vec3 format 'x0.0 y0.0 z0.0'"),
                ],
            );

            // Send data to GPT
            let (tx, rx) = unbounded();
            smol::spawn(async move {
                let result = query_openai(
                    &format!("s:{system_message}|u:Based on perception and mind state, run set_mind_state function"),
                    256,
                    &Some(vec![function]),
                );
                tx.send(result).await.unwrap();
            })
            .detach();
            self.gpt_result_rx = Some(rx);
        }
    }

    fn brain_process(&mut self) -> Result<()> {
        if self.gpt_result_rx.is_some() {
            if let Ok(result) = self
                .gpt_result_rx
                .as_mut()
                .context("Failed to get GPT result")?
                .try_recv()
            {
                let result2 = match result {
                    Ok(reply) => self.brain_act(&reply),
                    Err(e) => Err(anyhow::anyhow!("Error: {}", e)),
                };
                self.gpt_result_rx = None;
                result2?;
            }
        }
        Ok(())
    }

    fn brain_act(&mut self, reply: &MessageResponse) -> Result<()> {
        let reply = format!("goal_thoughts = {reply:?}");
        godot_print!("GPT reply: {}", reply);
        // // Parse mind and update from it
        // for line in reply.lines() {
        //     let parts: Vec<&str> = line.split(" = ").collect();
        //     if parts.len() != 2 {
        //         continue;
        //     }

        //     let key = parts[0].trim();
        //     let mut value = parts[1].trim();

        //     // Strip quotes from string values
        //     if value.starts_with('"') && value.ends_with('"') {
        //         value = &value[1..value.len() - 1];
        //     }

        //     match key {
        //         "goal_thoughts" => self.mind.goal_thoughts = value.to_string(),
        //         "goal_long" => self.mind.goal_long = value.to_string(),
        //         "goal_medium" => self.mind.goal_medium = value.to_string(),
        //         "goal_short" => self.mind.goal_short = value.to_string(),
        //         "target_velocity" => self.mind.target_velocity = Vec3::from_str(value)?,
        //         _ => continue,
        //     }
        // }

        // godot_print!("Parsed mind: {:#?}", self.mind);
        Ok(())
    }
}

fn vector3_to_string(vector: Vector3) -> String {
    format!("x{:.1} y{:.1} z{:.1}", vector.x, vector.y, vector.z)
}

fn vector3_from_string(string: &str) -> Result<Vector3> {
    let parts: Vec<&str> = string.split_whitespace().collect();

    if parts.len() == 3 {
        Ok(Vector3::new(
            parse_coordinate(parts[0], 'x')?,
            parse_coordinate(parts[1], 'y')?,
            parse_coordinate(parts[2], 'z')?,
        ))
    } else {
        Err(anyhow::anyhow!(
            "Expected 3 parts in the string representation of Vec3"
        ))
    }
}

fn parse_coordinate(coord_str: &str, prefix: char) -> Result<f32> {
    coord_str
        .strip_prefix(prefix)
        .ok_or_else(|| anyhow!("Expected '{}' prefix for coordinate", prefix))?
        .parse()
        .map_err(|e| anyhow!("Failed to parse {} coordinate: {}", prefix, e))
}
