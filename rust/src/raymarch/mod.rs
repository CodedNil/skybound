use godot::{
    engine::{
        control::LayoutPreset, ColorRect, Control, ControlVirtual, Material, ResourceLoader,
        ShaderMaterial,
    },
    prelude::*,
};
use std::{str::FromStr, time::Instant};

#[derive(GodotClass)]
#[class(base=Control)]
struct Raymarch {
    #[base]
    base: Base<Control>,
    start_time: Instant,
}

#[godot_api]
impl ControlVirtual for Raymarch {
    fn init(base: Base<Control>) -> Self {
        let mut instance = Self {
            base,
            start_time: Instant::now(),
        };

        // Add ColorRect as child
        let mut color_rect = ColorRect::new_alloc();
        let material = ResourceLoader::singleton()
            .load(GodotString::from_str("res://RaymarchMaterial.tres").unwrap())
            .unwrap()
            .cast::<ShaderMaterial>();
        color_rect.set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
        color_rect.set_material(material.upcast::<Material>());
        instance.base.add_child(color_rect.upcast::<Node>());

        instance
    }

    fn process(&mut self, _delta: f64) {
        let time = self.start_time.elapsed().as_secs_f32();
        let light_pos = Vector3::new(2.0 * time.cos(), 5.0, 2.0 * time.sin());

        let viewport = self.base.get_viewport().unwrap();
        let camera = viewport.get_camera_3d().unwrap();
        let camera_transform = camera.get_global_transform();

        if let Some(first_child) = self.base.get_child(0) {
            if let Some(color_rect) = first_child.try_cast::<ColorRect>() {
                if let Some(material) = color_rect.get_material() {
                    if let Some(mut shader) = material.try_cast::<ShaderMaterial>() {
                        shader.set_shader_parameter(
                            StringName::from_str("lightPos").unwrap(),
                            light_pos.to_variant(),
                        );
                        shader.set_shader_parameter(
                            StringName::from_str("cameraPos").unwrap(),
                            camera_transform.origin.to_variant(),
                        );
                        shader.set_shader_parameter(
                            StringName::from_str("front").unwrap(),
                            (-camera_transform.basis.col_c()).to_variant(),
                        );
                        shader.set_shader_parameter(
                            StringName::from_str("fov").unwrap(),
                            camera.get_fov().to_variant(),
                        );
                    }
                }
            }
        }
    }
}
