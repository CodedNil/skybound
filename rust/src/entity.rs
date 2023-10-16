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
        instance
    }

    fn process(&mut self, delta: f64) {}
}
