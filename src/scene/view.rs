use spirv_std::glam::{Mat4, Quat, Vec2, Vec3, Vec4, Vec4Swizzles};

use crate::scene::atmosphere::{get_pole_light_color, render_sky};
use crate::scene::geometry::get_pole_position;
use crate::scene::life::CreatureInstance;

pub const PLANET_RADIUS: f32 = 1_500_000.0;

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
    pub clip_from_world: Mat4,
    pub world_from_clip: Mat4,
    pub prev_clip_from_world: Mat4,
    pub direction_from_clip_unjittered: Mat4,
    pub direction_from_clip: Mat4,
    pub world_position: Vec4,
    pub camera_position: Vec4,
    pub planet_center_value: Vec4,
    pub planet_rotation: Quat,
    pub times: Vec4,
    pub weather: Vec4,
    pub pole_position: Vec4,
    pub pole_color: Vec4,
    pub ambient_color: Vec4,
    pub sky_horizon: Vec4,
    pub cache_origin: Vec4,
    pub cache_east: Vec4,
    pub cache_north: Vec4,
    pub cloud_time: Vec4,
    pub aur_flashes: [[Vec4; 2]; 32],
    pub aur_bolts: [[Vec4; 2]; 32],
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniform {
    pub view: ViewUniform,
    pub players: [CreatureInstance; 2],
    pub viewport: Vec4,
}

impl ViewUniform {
    /// Reconstruct a ray without subtracting large world positions near the camera.
    pub fn ray_direction(&self, uv: Vec2, jittered: bool) -> Vec3 {
        let matrix = if jittered {
            self.direction_from_clip
        } else {
            self.direction_from_clip_unjittered
        };
        let clip = (uv * Vec2::new(2.0, -2.0) + Vec2::new(-1.0, 1.0))
            .extend(0.01)
            .extend(1.0);
        (matrix * clip).xyz().normalize()
    }

    pub fn up(&self, local: Vec3) -> Vec3 {
        (local - self.planet_center()).normalize()
    }

    pub fn down(&self, local: Vec3) -> Vec3 {
        -self.up(local)
    }

    pub fn prepare_atmosphere(&mut self) {
        let pole = get_pole_position(
            self.planet_center(),
            self.planet_rotation,
            self.ro_relative(),
        );
        let up = (self.world_position.xyz() - self.planet_center()).normalize();
        let source = pole - self.planet_center();
        let ro = self.ro_relative();
        self.pole_position = pole.extend(0.0);
        self.pole_color = get_pole_light_color(ro, source).extend(0.0);
        self.ambient_color =
            (render_sky(up, ro, source) * 0.7 + render_sky(-up, ro, source) * 0.15).extend(0.0);
        let position = self.surface_position(self.world_position.xyz());
        self.ambient_color += super::aur::ambient(position, self.time()).extend(0.0);
    }

    #[cfg(not(target_arch = "spirv"))]
    pub fn prepare_flashes(&mut self) {
        use super::aur::lightning::events;
        self.aur_flashes = [[Vec4::ZERO; 2]; 32];
        self.aur_bolts = [[Vec4::ZERO; 2]; 32];
        let camera = self.to_planet(self.world_position.xyz());
        let mut visible: Vec<_> = events(self.time())
            .filter_map(|event| {
                let offset = event.position - camera;
                let distance = offset.length();
                // Cull the solid planet, including the upper part of a tall strike.
                if event.up.dot(camera) < PLANET_RADIUS - event.reach {
                    return None;
                }
                let local = self.from_planet(event.position);
                let clip = self.clip_from_world * local.extend(1.0);
                let margin = (event.reach + event.radius) * 3.0 / distance.max(1.0);
                if distance > event.reach + event.radius
                    && (clip.w <= 0.0
                        || (clip.truncate().truncate() / clip.w).abs().max_element() > 1.0 + margin)
                {
                    return None;
                }
                let importance = event.strength
                    * (0.5 + 0.5 * (event.radius / distance.max(1.0)).sqrt().min(1.0));
                Some((importance, event))
            })
            .collect();
        visible.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.id.cmp(&b.1.id)));
        let mut bolt_count = 0;
        for (index, (_, event)) in visible.iter().take(32).enumerate() {
            let root = event.up * (PLANET_RADIUS - 900.0);
            self.aur_flashes[index] = [
                root.extend(event.radius.recip()),
                (super::aur::energy_color(event.random, 0.45 + event.random * 0.2)
                    * event.strength
                    * 9.0)
                    .extend(event.strength),
            ];
            if bolt_count + 8 > self.aur_bolts.len() {
                continue;
            }
            for segment in event.segments() {
                self.aur_bolts[bolt_count] = [
                    self.from_planet(segment[0].truncate()).extend(segment[0].w),
                    self.from_planet(segment[1].truncate()).extend(segment[1].w),
                ];
                bolt_count += 1;
            }
        }
    }

    pub fn planet_center(&self) -> Vec3 {
        self.planet_center_value.xyz()
    }

    pub fn ro_relative(&self) -> Vec3 {
        self.world_position.xyz() - self.planet_center()
    }

    pub fn to_planet(&self, point: Vec3) -> Vec3 {
        self.planet_rotation.conjugate() * (point - self.planet_center())
    }

    pub fn from_planet(&self, point: Vec3) -> Vec3 {
        self.planet_center() + self.planet_rotation * point
    }

    pub fn surface_position(&self, point: Vec3) -> Vec3 {
        super::planet::surface_coordinates(self.to_planet(point))
    }

    pub fn altitude(&self) -> f32 {
        self.camera_position.z
    }

    pub fn time(&self) -> f32 {
        self.times.x
    }

    pub fn pole_direction(&self) -> Vec3 {
        let offset = self.pole_position.truncate() - self.world_position.truncate();
        if offset.length_squared() < 0.0001 {
            Vec3::Z
        } else {
            offset.normalize()
        }
    }

    pub fn frame_count(&self) -> f32 {
        self.times.y
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::polar::REACTION_RADIUS;
    use core::f32::consts::FRAC_PI_2;
    use spirv_std::glam::{camera::rh::proj::directx::perspective_infinite_reverse, vec3};

    use super::*;

    #[test]
    fn adjacent_pixel_rays_remain_distinct_after_large_world_translations() {
        let matrix = Mat4::from_quat(Quat::from_rotation_x(0.7))
            * perspective_infinite_reverse(1.0, 16.0 / 9.0, 0.1).inverse();
        let mut view = ViewUniform {
            direction_from_clip: matrix,
            direction_from_clip_unjittered: matrix,
            ..Default::default()
        };
        let uv = Vec2::new(0.7, 0.8);
        let expected = view.ray_direction(uv, false);
        for translation in [
            Vec3::ZERO,
            Vec3::splat(1_500_000.0),
            Vec3::splat(100_000_000.0),
        ] {
            view.world_position = translation.extend(0.0);
            let direction = view.ray_direction(uv, false);
            let adjacent = view.ray_direction(uv + Vec2::X / 3840.0, false);
            assert!(direction.distance(expected) < 1e-7);
            assert!(direction.distance(adjacent) > 0.0001);
        }
    }

    #[test]
    fn lightning_remains_planet_fixed_when_local_coordinates_change() {
        let mut first = ViewUniform {
            planet_center_value: (-Vec3::Z * (PLANET_RADIUS + 476_000.0)).extend(0.0),
            clip_from_world: perspective_infinite_reverse(1.2, 1.7, 0.1),
            times: Vec4::new(17.5, 0.0, 0.0, 0.0),
            ..ViewUniform::default()
        };
        first.prepare_flashes();
        assert!(first.aur_bolts[0][0].w > 0.0);
        let offset = vec3(10_000.0, 20_000.0, -5000.0);
        let mut shifted = first;
        shifted.world_position = offset.extend(0.0);
        shifted.planet_center_value += offset.extend(0.0);
        shifted.clip_from_world *= Mat4::from_translation(-offset);
        shifted.prepare_flashes();
        for (a, b) in first.aur_bolts.iter().zip(shifted.aur_bolts.iter()) {
            if a[0].w <= 0.0 {
                break;
            }
            for endpoint in 0..2 {
                assert!((a[endpoint].truncate() + offset).distance(b[endpoint].truncate()) < 1.0);
                assert!(
                    (first.to_planet(a[endpoint].truncate()).length() - PLANET_RADIUS).abs()
                        < 30_000.0
                );
            }
        }
    }

    #[test]
    fn polar_light_rises_symmetrically_from_dark_equatorial_conditions() {
        let light = |latitude: f32| {
            let mut view = ViewUniform {
                world_position: vec3(0.0, 0.0, 2400.0).extend(0.0),
                planet_center_value: vec3(0.0, 0.0, -PLANET_RADIUS).extend(0.0),
                planet_rotation: Quat::from_rotation_x(FRAC_PI_2 - latitude),
                ..ViewUniform::default()
            };
            view.prepare_atmosphere();
            view.pole_color.truncate().length()
        };
        let sunrise = (PLANET_RADIUS / REACTION_RADIUS).asin() + 0.05;
        assert!(light(0.0) < 0.001);
        assert!(light(sunrise) > light(0.0) + 0.1);
        assert!(light(1.4) > light(sunrise));
        assert!((light(sunrise) - light(-sunrise)).abs() < 0.01);
    }

    #[test]
    fn atmosphere_stays_finite_across_altitudes_and_latitudes() {
        for altitude in [-1000.0, 0.0, 20_000.0, 400_000.0] {
            for latitude in [-1.0, 0.0, 1.0] {
                let mut view = ViewUniform {
                    world_position: vec3(0.0, 0.0, altitude).extend(0.0),
                    planet_center_value: vec3(0.0, 0.0, -PLANET_RADIUS).extend(0.0),
                    camera_position: Vec4::new(latitude, 0.0, 0.0, 0.0),
                    planet_rotation: Quat::from_rotation_x(latitude),
                    ..ViewUniform::default()
                };
                view.prepare_atmosphere();
                assert!(view.pole_position.is_finite(), "invalid pole position");
                assert!(view.pole_color.is_finite(), "invalid pole color");
                assert!(view.ambient_color.is_finite(), "invalid ambient color");
            }
        }
    }
}
