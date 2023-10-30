use crate::entity::Entity;
use godot::{
    engine::{
        control::LayoutPreset, ColorRect, Control, ControlVirtual, Material, ResourceLoader,
        ShaderMaterial,
    },
    prelude::*,
};
use std::time::Instant;

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
            .load("res://RaymarchMaterial.tres".into())
            .unwrap()
            .cast::<ShaderMaterial>();
        color_rect.set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
        color_rect.set_material(material.upcast::<Material>());
        instance.base.add_child(color_rect.upcast::<Node>());

        instance
    }

    #[allow(clippy::too_many_lines)]
    fn process(&mut self, _delta: f64) {
        let time = self.start_time.elapsed().as_secs_f32();
        let light_pos = Vector3::new(2.0 * time.cos(), 5.0, 2.0 * time.sin());

        let viewport = self.base.get_viewport().unwrap();
        let camera = viewport.get_camera_3d().unwrap();
        let camera_transform = camera.get_global_transform();

        let mut entity_particle_locations = VariantArray::new();
        // Get all entities
        for entity_node in self
            .base
            .get_tree()
            .unwrap()
            .get_nodes_in_group("Entity".into())
            .iter_shared()
        {
            if let Some(entity) = entity_node.try_cast::<Entity>() {
                for particle in &entity.bind().particles {
                    if particle.interior {
                        continue;
                    }
                    entity_particle_locations.push(particle.position.to_variant());
                }
            }
        }
        godot_print!(
            "Entity particle locations: {}",
            entity_particle_locations.len()
        );

        if let Some(first_child) = self.base.get_child(0) {
            if let Some(color_rect) = first_child.try_cast::<ColorRect>() {
                if let Some(material) = color_rect.get_material() {
                    if let Some(mut shader) = material.try_cast::<ShaderMaterial>() {
                        shader.set_shader_parameter("lightPos".into(), light_pos.to_variant());
                        shader.set_shader_parameter(
                            "cameraPos".into(),
                            camera_transform.origin.to_variant(),
                        );
                        shader.set_shader_parameter(
                            "cameraFront".into(),
                            (-camera_transform.basis.col_c()).to_variant(),
                        );
                        shader.set_shader_parameter(
                            "cameraUp".into(),
                            camera_transform.basis.col_b().to_variant(),
                        );
                        shader.set_shader_parameter("fov".into(), camera.get_fov().to_variant());

                        // Set sphere data
                        shader.set_shader_parameter(
                            "sphereN".into(),
                            (entity_particle_locations.len() as u64).to_variant(),
                        );
                        shader.set_shader_parameter(
                            "sphereCenters".into(),
                            entity_particle_locations.to_variant(),
                        );
                    }
                }
            }
        }
    }
}
