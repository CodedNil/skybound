use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::{env, fs};

use image::RgbImage;
use rayon::prelude::*;
use skybound::scene::life::{self, CreatureInstance, SPECIES, SPECIES_MODULES, Taxon};
use skybound::scene::surface::{Ray, tonemap};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};

fn adjacent_preview(kind: usize) -> PathBuf {
    let module = SPECIES_MODULES[kind]
        .split_whitespace()
        .collect::<String>()
        .replace("::", "/");
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/scene/life")
        .join(module);
    let folder = if source.is_dir() {
        source.as_path()
    } else {
        source.parent().expect("species folder")
    };
    let source_stem = source
        .file_stem()
        .expect("species module")
        .to_string_lossy();
    let mut references: Vec<_> = fs::read_dir(folder)
        .expect("read species folder")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "png")
                && path.file_stem().is_some_and(|stem| {
                    let stem = stem.to_string_lossy();
                    stem == source_stem || stem == source_stem.trim_end_matches('s')
                })
        })
        .collect();
    references.sort();
    let stem = references
        .first()
        .unwrap_or(&source)
        .file_stem()
        .expect("species filename")
        .to_string_lossy();
    folder.join(format!("{stem}_render.png"))
}

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
            footprint: self.field / 768.0,
            ambient: vec3(0.50, 0.65, 0.85),
            radiance: vec3(3.60, 3.15, 2.40),
        }
    }
    fn fit(index: usize, c: &CreatureInstance) -> Self {
        let angle = [0.08, -0.25, 0.25, -0.5, 1.0, 0.5, 0.75, 0.0, 0.0][index] * PI;
        let elevation = [0.1f32, 0.2, 0.2, 0.0, 0.1, 0.0, 0.25, 1.25, -0.95][index];
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
            view.field *= ((max - min).max_element() + 2.0 / 96.0) * 1.08;
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

fn option(name: &str) -> Option<String> {
    let mut args = env::args();
    args.find(|arg| arg == name).and_then(|_| args.next())
}
fn background(uv: Vec2) -> Vec3 {
    vec3(0.025, 0.12, 0.32).lerp(vec3(0.10, 0.22, 0.34), uv.y)
}

#[derive(Clone, Copy)]
struct Sample {
    color: Vec3,
    distance: f32,
}
fn sample(view: &View, uv: Vec2, c: &CreatureInstance, tile: usize, anatomy: bool) -> Sample {
    let mut ray = view.ray(vec2(uv.x, 1.0 - uv.y));
    ray.footprint = view.field / tile as f32;
    let hit = if anatomy {
        life::anatomy::render(ray, c)
    } else {
        life::render(ray, c)
    };
    Sample {
        color: hit.composite(background(uv)),
        distance: hit.distance(),
    }
}

fn render_grid(c: &CreatureInstance, tile: usize, anatomy: bool) -> Vec<u8> {
    let width = tile * 3;
    let mut pixels = vec![0u8; width * width * 3];
    for index in 0..9 {
        let view = View::fit(index, c);
        let samples: Vec<Sample> = (0..tile * tile)
            .into_par_iter()
            .map(|i| {
                sample(
                    &view,
                    vec2((i % tile) as f32 + 0.5, (i / tile) as f32 + 0.5) / tile as f32,
                    c,
                    tile,
                    anatomy,
                )
            })
            .collect();
        let refined: Vec<[u8; 3]> = samples
            .par_iter()
            .enumerate()
            .map(|(i, center)| {
                let x = i % tile;
                let y = i / tile;
                let edge = [
                    (x.saturating_sub(1), y),
                    ((x + 1).min(tile - 1), y),
                    (x, y.saturating_sub(1)),
                    (x, (y + 1).min(tile - 1)),
                ]
                .iter()
                .any(|&(x, y)| {
                    let other = samples[y * tile + x];
                    (other.distance - center.distance).abs() > center.distance.max(1.0) * 0.02
                        || other.color.distance(center.color) > 0.12
                });
                let mut color = center.color;
                if edge {
                    color = Vec3::ZERO;
                    for offset in [
                        vec2(0.25, 0.25),
                        vec2(0.75, 0.25),
                        vec2(0.25, 0.75),
                        vec2(0.75, 0.75),
                    ] {
                        color += sample(
                            &view,
                            (vec2(x as f32, y as f32) + offset) / tile as f32,
                            c,
                            tile,
                            anatomy,
                        )
                        .color
                            * 0.25;
                    }
                }
                let color = tonemap(color);
                [
                    (color.x * 255.0).round() as u8,
                    (color.y * 255.0).round() as u8,
                    (color.z * 255.0).round() as u8,
                ]
            })
            .collect();
        for (i, color) in refined.iter().enumerate() {
            let x = index % 3 * tile + i % tile;
            let y = index / 3 * tile + i / tile;
            pixels[(y * width + x) * 3..(y * width + x) * 3 + 3].copy_from_slice(color);
        }
    }
    pixels
}

fn main() {
    let tile: usize = option("--size")
        .or_else(|| env::args().nth(1).filter(|v| !v.starts_with('-')))
        .map_or(768, |value| {
            value.parse().expect("tile size must be an integer")
        });
    assert!(
        (32..=2048).contains(&tile),
        "tile size must be between 32 and 2048"
    );
    let output = option("--out");
    let selected = option("--species");
    let seconds =
        option("--time").map_or(1.0, |value| value.parse().expect("time must be a number"));
    if let Some(output) = &output {
        fs::create_dir_all(output).expect("create renders folder");
    }
    let width = tile * 3;
    for (kind, taxon) in SPECIES.iter().enumerate() {
        if selected.as_ref().is_some_and(|filter| {
            !filter.split(',').any(|name| {
                slug(taxon.name).contains(name) || slug(taxon.english_name).contains(name)
            })
        }) {
            continue;
        }
        let c = CreatureInstance::new(
            Vec3::ZERO.extend(taxon.radius),
            Vec4::W,
            Vec4::new(0.05, 0.0, seconds, 0.0),
            kind as u32,
        );
        let anatomy = env::args().any(|arg| arg == "--anatomy");
        let pixels = render_grid(&c, tile, anatomy);
        let path = output.as_ref().map_or_else(
            || adjacent_preview(kind),
            |output| {
                Path::new(output)
                    .join(filename(taxon))
                    .with_extension("png")
            },
        );
        let path = if anatomy {
            path.with_file_name(format!(
                "{}_anatomy.png",
                path.file_stem()
                    .expect("preview stem")
                    .to_string_lossy()
                    .trim_end_matches("_render")
            ))
        } else {
            path
        };
        RgbImage::from_raw(width as u32, width as u32, pixels)
            .expect("valid pixel dimensions")
            .save(&path)
            .expect("save species preview");
        println!("{}: {}", taxon.name, path.display());
    }
}
