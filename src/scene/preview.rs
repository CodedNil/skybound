//! Creature previews traced on the GPU.
//!
//! One creature to a tile against a plain sky, through the same tracer the world draws them with,
//! so the previews can be animated at full size in the time a single still once took.

use crate::scene::life::surface::Ray;
#[cfg(all(target_arch = "spirv", feature = "preview"))]
use crate::scene::life::{self, CreatureInstance, surface::TraceResult};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};
#[cfg(all(target_arch = "spirv", feature = "preview"))]
use spirv_std::{glam::UVec3, spirv};

/// One tile of a preview: where the eye is and which way it looks, how wide it sees and where the
/// view is centred, its pixels across, and the sky behind it.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PreviewView {
    /// The eye, and how wide the view is.
    pub origin: Vec4,
    /// Which way it looks, and how far across the view is centred.
    pub forward: Vec4,
    /// Its right, and how far up the view is centred.
    pub right: Vec4,
    /// Its up, and how wide each sample is.
    pub up: Vec4,
    /// Samples across the tile, where its pixels start in the output, whether the sky is a
    /// gradient (0) or one even tone (1), and spare.
    pub tile: Vec4,
}

/// The sky behind a preview, from the bottom of the view to its top.
pub fn backdrop(height: f32, even: bool) -> Vec3 {
    let height = if even { 0.5 } else { height };
    vec3(0.025, 0.12, 0.32).lerp(vec3(0.10, 0.22, 0.34), height)
}

impl PreviewView {
    /// The ray through a point of the view, `uv` from its bottom left to its top right.
    pub fn ray(&self, uv: Vec2) -> Ray {
        let field = self.origin.w;
        let p = vec2(self.forward.w, self.right.w) + (uv - 0.5) * field;
        Ray {
            origin: self.origin.truncate(),
            direction: (self.forward.truncate()
                + self.right.truncate() * p.x
                + self.up.truncate() * p.y)
                .normalize(),
            light: vec3(-0.4, 0.8, -0.6).normalize(),
            max_distance: 1e6,
            footprint: self.up.w,
            ambient: vec3(0.50, 0.65, 0.85),
            radiance: vec3(3.60, 3.15, 2.40),
        }
    }
}

/// One sample of one tile, `id.z` the tile: a ray through it composited over the sky, and whether
/// it struck the creature, in linear colour. Every tile of a dispatch is of one species, fixed as
/// the pipeline is built, so the driver compiles in that species' body alone: the tracer with
/// every species' body in it needs so many registers that it runs a hundred times slower.
/// Compiled only into the preview module, so the game's shader is no larger for it.
#[cfg(all(target_arch = "spirv", feature = "preview"))]
#[spirv(compute(threads(8, 8, 1), entry_point_name = "preview"))]
pub fn preview(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] views: &[PreviewView],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] creatures: &[CreatureInstance],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pixels: &mut [Vec4],
    #[spirv(spec_constant(id = 0, default = 0))] species: u32,
) {
    let view = &views[id.z as usize];
    let size = view.tile.x as u32;
    if id.x >= size || id.y >= size {
        return;
    }
    let creature = &creatures[id.z as usize];
    let uv = (vec2(id.x as f32, id.y as f32) + 0.5) / size as f32;
    let uv = vec2(uv.x, 1.0 - uv.y);
    // Traced as the game draws it, through the one tracer every species shares: its solid body,
    // then the thin membranes in front of it.
    let ray = view.ray(uv);
    let solid = life::render_opaque_of(species, ray, creature);
    let mut hit = TraceResult::opaque(solid);
    let thin = Ray {
        max_distance: if solid.is_hit() {
            solid.distance()
        } else {
            ray.max_distance
        },
        ..ray
    };
    life::visit_thin_of(species, thin, creature, |layer| hit.add(layer));
    let colour = hit.composite(backdrop(1.0 - uv.y, view.tile.z > 0.5));
    let covered = if hit.is_hit() { 1.0 } else { 0.0 };
    let at = view.tile.y as u32 + id.y * size + id.x;
    pixels[at as usize] = colour.extend(covered);
}
