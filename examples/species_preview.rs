use std::f32::consts::{PI, TAU};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::{env, fs};

use image::codecs::gif::{GifEncoder, Repeat};
use image::imageops;
use image::{Delay, Frame, RgbImage, RgbaImage};
use rayon::prelude::*;
use skybound::scene::life::{self, CreatureInstance, SPECIES, SPECIES_MODULES, Taxon};
use skybound::scene::surface::{Ray, tonemap};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};

fn adjacent_preview(kind: usize) -> PathBuf {
    let module = SPECIES_MODULES[kind]
        .split_whitespace()
        .collect::<String>()
        .replace("::", "/");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/scene/life")
        .join(module)
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
        branches.push(slug(taxon.name));
        taxon = parent;
    }
    branches.reverse();
    branches.join("__")
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

fn render_tile(c: &CreatureInstance, view: &View, tile: usize, anatomy: bool) -> Vec<u8> {
    let samples: Vec<Sample> = (0..tile * tile)
        .into_par_iter()
        .map(|i| {
            sample(
                view,
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
                        view,
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
    refined.into_iter().flatten().collect()
}

const SEEDS: [u32; 9] = [11, 23, 37, 53, 71, 89, 107, 131, 157];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    Render,
    Anatomy,
    Lifecycle,
    Variation,
    Idle,
}
impl Output {
    const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Anatomy => "anatomy",
            Self::Lifecycle => "lifecycle",
            Self::Variation => "variation",
            Self::Idle => "idle",
        }
    }
}

fn specimen(kind: usize, days: f32, seed: u32, seconds: f32) -> CreatureInstance {
    CreatureInstance::specimen(
        Vec3::ZERO.extend(life::KINDS[kind].render_radius()),
        Vec4::W,
        Vec4::new((seconds * TAU / 4.0).sin() * 0.12, 0.0, seconds, 0.0),
        kind as u32,
        days,
        seed,
    )
}

fn label(pixels: &mut [u8], size: usize, text: &str) {
    let scale = (size / 128).max(1);
    for (column, ch) in text.chars().enumerate() {
        let glyph: [u8; 5] = match ch {
            '0' => [62, 65, 65, 65, 62],
            '1' => [0, 66, 127, 64, 0],
            '2' => [98, 81, 73, 73, 70],
            '3' => [34, 65, 73, 73, 54],
            '4' => [24, 20, 18, 127, 16],
            '5' => [39, 69, 69, 69, 57],
            '6' => [62, 73, 73, 73, 48],
            '7' => [1, 113, 9, 5, 3],
            '8' => [54, 73, 73, 73, 54],
            '9' => [6, 73, 73, 73, 62],
            '.' => [0, 96, 96, 0, 0],
            'D' => [127, 65, 65, 65, 62],
            'Y' => [7, 8, 112, 8, 7],
            'S' => [38, 73, 73, 73, 50],
            _ => [0; 5],
        };
        for (x, bits) in glyph.into_iter().enumerate() {
            for y in 0..7 {
                for dx in 0..scale {
                    for dy in 0..scale {
                        let px = 6 * scale + (column * 6 + x) * scale + dx;
                        let py = size - 10 * scale + y * scale + dy;
                        if px < size {
                            let start = (py * size + px) * 3;
                            pixels[start..start + 3].copy_from_slice(if bits & (1 << y) != 0 {
                                &[225, 235, 245]
                            } else {
                                &[12, 25, 40]
                            });
                        }
                    }
                }
            }
        }
    }
}

fn render_grid(kind: usize, size: usize, output: Output, seconds: f32) -> RgbImage {
    let biology = life::KINDS[kind].biology();
    let adult = specimen(kind, biology.maturity_days * 1.25, 0, seconds);
    let mut fixed = View::fit(1, &adult);
    fixed.field *= 1.2;
    let mut grid = RgbImage::new((size * 3) as u32, (size * 3) as u32);
    for (panel, &seed) in SEEDS.iter().enumerate() {
        let age = if output == Output::Lifecycle {
            life::development::grid_age(biology, panel)
        } else {
            biology.maturity_days * 1.25
        };
        let seed = if output == Output::Variation { seed } else { 0 };
        let c = specimen(kind, age, seed, seconds);
        let angled =
            matches!(output, Output::Render | Output::Anatomy).then(|| View::fit(panel, &c));
        let view = angled.as_ref().unwrap_or(&fixed);
        let mut pixels = render_tile(&c, view, size, output == Output::Anatomy);
        if output == Output::Lifecycle {
            let title = if age >= 365.0 {
                format!("{:.1}Y", age / 365.0)
            } else {
                format!("{age:.1}D")
            };
            label(&mut pixels, size, &title);
        } else if output == Output::Variation {
            label(&mut pixels, size, &format!("S{seed}"));
        }
        let tile = RgbImage::from_raw(size as u32, size as u32, pixels).expect("tile pixels");
        imageops::replace(
            &mut grid,
            &tile,
            i64::try_from((panel % 3) * size).expect("grid width"),
            i64::try_from((panel / 3) * size).expect("grid height"),
        );
    }
    grid
}

fn idle(path: &Path, kind: usize, size: usize, seconds: f32) {
    let age = life::KINDS[kind].biology().maturity_days * 1.25;
    let c = specimen(kind, age, 0, seconds);
    let mut view = View::fit(1, &c);
    view.field *= 1.2;
    let mut encoder =
        GifEncoder::new_with_speed(BufWriter::new(File::create(path).expect("create GIF")), 10);
    encoder.set_repeat(Repeat::Infinite).expect("GIF loop");
    for frame in 0..48 {
        let c = specimen(kind, age, 0, seconds + frame as f32 / 12.0);
        let pixels = render_tile(&c, &view, size, false);
        let rgba: Vec<u8> = pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        encoder
            .encode_frame(Frame::from_parts(
                RgbaImage::from_raw(size as u32, size as u32, rgba).expect("GIF pixels"),
                0,
                0,
                Delay::from_numer_denom_ms(1000, 12),
            ))
            .expect("encode GIF frame");
    }
}

fn main() {
    if env::args().any(|arg| arg == "--help") {
        println!(
            "species_preview [--size 768] [--species comma,separated,names] [--out directory] [--time seconds] [--outputs render,anatomy,lifecycle,variation,idle]\nBy default, writes all five assets beside each species source file. --anatomy selects anatomy only."
        );
        return;
    }
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
    let requested = option("--outputs").unwrap_or_else(|| {
        if env::args().any(|arg| arg == "--anatomy") {
            "anatomy".into()
        } else {
            "render,anatomy,lifecycle,variation,idle".into()
        }
    });
    let outputs: Vec<_> = requested
        .split(',')
        .map(|name| match name {
            "render" => Output::Render,
            "anatomy" => Output::Anatomy,
            "lifecycle" => Output::Lifecycle,
            "variation" => Output::Variation,
            "idle" => Output::Idle,
            _ => panic!("unknown output: {name}"),
        })
        .collect();
    let mut count = 0;
    for (kind, taxon) in SPECIES.iter().enumerate() {
        if selected.as_ref().is_some_and(|filter| {
            !filter.split(',').any(|name| {
                slug(taxon.name).contains(name) || slug(taxon.english_name).contains(name)
            })
        }) {
            continue;
        }
        count += 1;
        let base = output.as_ref().map_or_else(
            || adjacent_preview(kind),
            |output| Path::new(output).join(filename(taxon)),
        );
        for &output in &outputs {
            let extension = if matches!(output, Output::Idle) {
                "gif"
            } else {
                "png"
            };
            let path = base.with_file_name(format!(
                "{}_{}.{extension}",
                base.file_stem()
                    .expect("preview stem")
                    .to_string_lossy()
                    .trim_end_matches("_render"),
                output.name()
            ));
            if matches!(output, Output::Idle) {
                idle(&path, kind, tile, seconds);
            } else {
                render_grid(kind, tile, output, seconds)
                    .save(&path)
                    .expect("save species preview");
            }
            println!("{}: {}", taxon.name, path.display());
        }
    }
    assert!(count > 0, "no species matched the filter");
}
