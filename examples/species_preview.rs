use image::RgbImage;
use rayon::prelude::*;
use skybound::scene::{
    life::{self, CreatureInstance, SPECIES, Taxon},
    surface::Ray,
};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};
use std::{env, f32::consts::PI, fs, path::Path};

struct View {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    center: Vec2,
    field: f32,
}
impl View {
    fn ray(&self, uv: Vec2) -> Ray {
        let p = self.center + (uv - 0.5) * self.field;
        Ray {
            origin: self.origin,
            direction: (self.forward + self.right * p.x + self.up * p.y).normalize(),
            light: vec3(-0.4, 0.8, -0.6).normalize(),
            max_distance: 1e6,
        }
    }
    fn fit(index: usize, c: &CreatureInstance) -> Self {
        let angle = index as f32 * (2.0 * PI / 9.0) + 0.15;
        let elevation = [0.2f32, 0.55, -0.2][index / 3];
        let origin = vec3(
            angle.sin() * elevation.cos(),
            elevation.sin(),
            -angle.cos() * elevation.cos(),
        ) * c.position.w
            * 1.7;
        let forward = -origin.normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let mut view = Self {
            origin,
            forward,
            right,
            up: right.cross(forward),
            center: Vec2::ZERO,
            field: 1.35,
        };
        let mut min = Vec2::ONE;
        let mut max = Vec2::ZERO;
        for x in 0..96 {
            for y in 0..96 {
                let uv = vec2(x as f32 + 0.5, y as f32 + 0.5) / 96.0;
                if life::render(view.ray(uv), c).is_hit() {
                    min = min.min(uv);
                    max = max.max(uv);
                }
            }
        }
        if max.cmpge(min).all() {
            view.center = ((min + max) * 0.5 - 0.5) * view.field;
            view.field *= ((max - min).max_element() + 2.0 / 96.0) * 1.15;
        }
        view
    }
}

fn slug(name: &str) -> String {
    name.to_lowercase()
        .replace('æ', "ae")
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}
fn filename(mut taxon: &Taxon) -> String {
    let mut branches = Vec::new();
    while let Some(parent) = taxon.parent {
        let sibling = sibling_index(taxon, parent);
        branches.push(format!("{sibling:02}_{}", slug(taxon.name)));
        taxon = parent;
    }
    branches.reverse();
    branches.join("__")
}

fn sibling_index(taxon: &Taxon, parent: &Taxon) -> usize {
    let mut names = Vec::new();
    for species in SPECIES {
        let mut node = *species;
        while let Some(ancestor) = node.parent {
            if ancestor.name == parent.name && !names.contains(&node.name) {
                names.push(node.name);
            }
            node = ancestor;
        }
    }
    names.sort_unstable();
    names
        .iter()
        .position(|&name| name == taxon.name)
        .expect("registered taxon")
        + 1
}

fn main() {
    let tile: usize = env::args().nth(1).map_or(384, |value| {
        value.parse().expect("tile size must be an integer")
    });
    assert!(
        (32..=1024).contains(&tile),
        "tile size must be between 32 and 1024"
    );
    fs::create_dir_all("renders").expect("create renders folder");
    let width = tile * 3;
    for (kind, taxon) in SPECIES.iter().enumerate() {
        let c = CreatureInstance::new(
            Vec3::ZERO.extend(taxon.radius),
            Vec4::W,
            Vec4::new(0.05, 0.0, 1.0, 48.0),
            kind as u32,
        );
        let views: Vec<View> = (0..9).map(|index| View::fit(index, &c)).collect();
        let mut pixels = vec![0u8; width * width * 3];
        pixels.par_chunks_mut(3).enumerate().for_each(|(i, pixel)| {
            let column = i % width;
            let row = i / width;
            let uv = vec2((column % tile) as f32 + 0.5, (row % tile) as f32 + 0.5) / tile as f32;
            let hit = life::render(
                views[(row / tile) * 3 + column / tile].ray(vec2(uv.x, 1.0 - uv.y)),
                &c,
            );
            let color = if hit.is_hit() {
                hit.color()
            } else {
                vec3(0.075, 0.12, 0.22).lerp(vec3(0.28, 0.4, 0.55), uv.y)
            };
            pixel.copy_from_slice(&[
                (color.x.clamp(0.0, 1.0) * 255.0) as u8,
                (color.y.clamp(0.0, 1.0) * 255.0) as u8,
                (color.z.clamp(0.0, 1.0) * 255.0) as u8,
            ]);
        });
        let path = Path::new("renders")
            .join(filename(taxon))
            .with_extension("png");
        let legacy = Path::new("renders")
            .join(slug(taxon.name))
            .with_extension("png");
        if legacy.exists() && !path.exists() {
            fs::rename(legacy, &path).expect("rename existing preview");
        }
        RgbImage::from_raw(width as u32, width as u32, pixels)
            .expect("valid pixel dimensions")
            .save(&path)
            .expect("save species preview");
        println!("{}: {}", taxon.name, path.display());
    }
}
