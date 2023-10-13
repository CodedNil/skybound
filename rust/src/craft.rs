use godot::{
    engine::{RigidBody3D, RigidBody3DVirtual},
    prelude::*,
};

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Craft {
    #[base]
    base: Base<RigidBody3D>,
}

#[godot_api]
impl RigidBody3DVirtual for Craft {
    fn init(base: Base<RigidBody3D>) -> Self {
        let mut instance = Self { base };
        instance.base.add_to_group("Craft".into());
        instance.base.set_contact_monitor(true);
        instance.base.set_max_contacts_reported(10);
        instance
    }

    fn process(&mut self, delta: f64) {
        // Get food objects
        let mut foods = vec![];
        if let Some(mut scene_tree) = self.base.get_tree() {
            let food = scene_tree.get_nodes_in_group("Food".into());
            for node in food.iter_shared() {
                if let Some(food) = node.try_cast::<RigidBody3D>() {
                    foods.push(food);
                }
            }
        }
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

        // Eat food
        let colliders = self.base.get_colliding_bodies();
        for collider in colliders.iter_shared() {
            if let Some(mut food) = collider.try_cast::<RigidBody3D>() {
                if food.is_in_group("Food".into()) {
                    food.queue_free();
                }
            }
        }
    }
}
