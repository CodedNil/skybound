use crate::gpt::query_openai;
use crate::vec3::Vec3;
use anyhow::{Context, Result};
use godot::{
    engine::{RigidBody3D, RigidBody3DVirtual},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use smol::channel::{unbounded, Receiver};

// The crafts perception, what is fed into the mind each thought cycle
#[derive(Deserialize, Serialize)]
struct Perception {
    position: Vec3,
    velocity: Vec3,
    food_pos: Vec<Vec3>,
}

impl Perception {
    fn to_toml(&self) -> String {
        let mut toml_string = String::new();

        toml_string.push_str(&format!("position = {}\n", self.position));

        toml_string.push_str(&format!("velocity = {}\n", self.velocity));

        let food_positions: Vec<String> = self
            .food_pos
            .iter()
            .map(|vec3| format!("\"{vec3}\""))
            .collect();
        toml_string.push_str(&format!(
            "food_pos = [{}] # The positions of the food near\n",
            food_positions.join(", ")
        ));

        toml_string
    }
}

// The crafts mind, it's goals and it's outputs, things it alters each thought cycle
#[derive(Default, Deserialize, Serialize, Debug)]
struct Mind {
    goal_thoughts: String,
    goal_long: String,
    goal_medium: String,
    goal_short: String,

    target_velocity: Vec3,
}

impl Mind {
    fn to_toml(&self) -> String {
        let mut toml_string = String::new();
        toml_string.push_str(&format!(
            "goal_thoughts = \"{}\" # Thoughts behind the goals, descriptive\n",
            self.goal_thoughts
        ));
        toml_string.push_str(&format!("goal_long = \"{}\" # concise\n", self.goal_long));
        toml_string.push_str(&format!(
            "goal_medium = \"{}\" # concise\n",
            self.goal_medium
        ));
        toml_string.push_str(&format!("goal_short = \"{}\" # concise\n", self.goal_short));
        toml_string.push_str(&format!(
            "target_velocity = {} # Velocity for craft to aim for, concise\n",
            self.target_velocity
        ));
        toml_string
    }
}

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Craft {
    gpt_result_rx: Option<Receiver<Result<String>>>,
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
            let mut food_pos = vec![];
            for food in &foods {
                food_pos.push(food.get_global_position().into());
            }

            // Add perception to system_message
            let perception = Perception {
                position: self.base.get_global_position().into(),
                velocity: self.base.get_linear_velocity().into(),
                food_pos,
            };
            let perception_toml = perception.to_toml();
            system_message.push_str(&format!(
                "Your perception as toml data is\n{perception_toml}\n\n"
            ));

            // Add mind to system_message
            let mind_toml = self.mind.to_toml();
            system_message.push_str(&format!("Your mind state as toml data is\n{mind_toml}\n\n"));

            godot_print!("system_message: {}", system_message);

            // Send data to GPT
            let (tx, rx) = unbounded();
            smol::spawn(async move {
                let result = query_openai(&format!("s:{system_message}|u:Based on perception and mind state, set your goals, give commands to your body to reach this goals, output should be purely key: values in the exact same format as mind input data"));
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
                    Ok(reply) => self.brain_act(reply),
                    Err(e) => Err(anyhow::anyhow!("Error: {}", e)),
                };
                self.gpt_result_rx = None;
                result2?;
            }
        }
        Ok(())
    }

    fn brain_act(&mut self, reply: String) -> Result<()> {
        // Try to parse reply as toml into mind struct
        let mind: Mind = match toml::from_str(&reply) {
            Ok(mind) => mind,
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to parse GPT reply as toml: {reply}. Error: {:#?}",
                    e
                ));
            }
        };
        godot_print!("Parsed mind: {:#?}", mind);
        Ok(())
    }
}
