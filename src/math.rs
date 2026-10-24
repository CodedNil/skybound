use bevy::math as engine;
use spirv_std::glam as scene;

pub trait ToScene {
    type Output;
    fn to_scene(self) -> Self::Output;
}

pub trait ToEngine {
    type Output;
    fn to_engine(self) -> Self::Output;
}

macro_rules! vectors {
    ($($type:ident),* $(,)?) => {$(
        impl ToScene for engine::$type {
            type Output = scene::$type;
            fn to_scene(self) -> Self::Output {
                scene::$type::from_array(self.to_array())
            }
        }
        impl ToEngine for scene::$type {
            type Output = engine::$type;
            fn to_engine(self) -> Self::Output {
                engine::$type::from_array(self.to_array())
            }
        }
    )*};
}
vectors!(Vec2, Vec3, Vec4, DVec3, Quat);

impl ToScene for engine::Mat4 {
    type Output = scene::Mat4;
    fn to_scene(self) -> Self::Output {
        scene::Mat4::from_cols_array(&self.to_cols_array())
    }
}

impl ToEngine for scene::Mat4 {
    type Output = engine::Mat4;
    fn to_engine(self) -> Self::Output {
        engine::Mat4::from_cols_array(&self.to_cols_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_conversion_preserves_transforms_and_planet_coordinates() {
        let transform = engine::Mat4::from_rotation_translation(
            engine::Quat::from_rotation_y(0.7),
            engine::Vec3::new(100.0, -20.0, 3.0),
        );
        let point = engine::Vec3::new(1.0, 2.0, 3.0);
        let actual = transform
            .to_scene()
            .transform_point3(point.to_scene())
            .to_engine();
        assert!(actual.distance(transform.transform_point3(point)) < 0.0001);
        let position = engine::DVec3::new(1_500_000.0, 0.000_001, -33.5);
        assert_eq!(position, position.to_scene().to_engine());
        assert_eq!(transform, transform.to_scene().to_engine());
    }
}
