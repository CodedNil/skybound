use godot::{
    engine::{RigidBody3D, RigidBody3DVirtual},
    prelude::*,
};

#[derive(GodotClass)]
#[class(base=RigidBody3D)]
struct Craft {
    velocity: Vector3,

    #[base]
    body: Base<RigidBody3D>,
}

#[godot_api]
impl RigidBody3DVirtual for Craft {
    fn init(body: Base<RigidBody3D>) -> Self {
        godot_print!("Hello, world!"); // Prints to the Godot console

        Self {
            velocity: Vector3::new(0.0, 0.0, 0.0),
            body,
        }
    }
}
