use image::{
    Delay, Frame, RgbImage, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
    imageops,
};
use rayon::prelude::*;
use skybound::scene::{
    life,
    life::{
        CreatureInstance, SPECIES, SPECIES_MODULES, Taxon,
        animalia::physiology::Brood,
        surface::{Ray, tonemap},
    },
};
use spirv_std::glam::{Vec2, Vec3, Vec4, vec2, vec3};
use std::{
    cmp::Reverse,
    env,
    f32::consts::{PI, TAU},
    fs,
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

const SEEDS: [u32; 9] = [11, 23, 37, 53, 71, 89, 107, 131, 157];

struct View {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    center: Vec2,
    field: f32,
    /// The sky behind the creature, across the view from zero to one.
    backdrop: fn(Vec2) -> Vec3,
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
            backdrop: background,
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

#[derive(Clone, Copy)]
struct Sample {
    color: Vec3,
    distance: f32,
}

/// The sheets the preview can render for a species.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Output {
    /// The adult from nine angles.
    Render,
    /// The adult cut away to show its organs, from nine angles.
    Anatomy,
    /// One individual from egg or birth to old age.
    Lifecycle,
    /// Nine individuals, to show how they differ.
    Variation,
    /// A looping animation of the adult.
    Idle,
    /// At every fork in the tree, every species below it side by side at their true sizes.
    Lineup,
}

impl Output {
    const fn name(self) -> &'static str {
        match self {
            Self::Render => "render",
            Self::Anatomy => "anatomy",
            Self::Lifecycle => "lifecycle",
            Self::Variation => "variation",
            Self::Idle => "idle",
            Self::Lineup => "lineup",
        }
    }
}

fn adjacent_preview(kind: usize) -> PathBuf {
    let module = SPECIES_MODULES[kind]
        .split_whitespace()
        .collect::<String>()
        .replace("::", "/");
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/scene/life")
        .join(module)
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

/// One even sky behind a whole lineup, so its creatures stand together rather than in boxes.
fn even_sky(_: Vec2) -> Vec3 {
    background(Vec2::splat(0.5))
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
        color: hit.composite((view.backdrop)(uv)),
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
            '0' | 'O' => [62, 65, 65, 65, 62],
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
            'B' => [127, 73, 73, 73, 54],
            'R' => [127, 9, 25, 41, 70],
            'U' => [63, 64, 64, 64, 63],
            'H' => [127, 8, 8, 8, 127],
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
    let tiles: Vec<(usize, Vec<u8>)> = SEEDS
        .par_iter()
        .enumerate()
        .map(|(panel, &seed)| {
            let age = if output == Output::Lifecycle {
                life::development::grid_age(biology, panel)
            } else {
                biology.maturity_days * 1.25
            };
            let seed = if output == Output::Variation { seed } else { 0 };
            let mut c = specimen(kind, age, seed, seconds);
            // The first grown panel of a lifecycle shows a parent with its brood, where its lineage
            // carries one.
            let brooding = output == Output::Lifecycle && panel == 6;
            if brooding {
                match life::KINDS[kind].animal().map(|animal| animal.brood) {
                    Some(Brood::Carried) => c.kind.z = life::CLUTCH,
                    Some(Brood::Pouch) => c.kind.z = life::FULL_POUCH,
                    _ => {}
                }
            }
            // A lifecycle keeps the adult's view, but closes in on the young so their forms show,
            // up to three times nearer.
            let angled = match output {
                Output::Render | Output::Anatomy => Some(View::fit(panel, &c)),
                Output::Lifecycle => {
                    let own = View::fit(1, &c);
                    (own.field < fixed.field).then(|| View {
                        field: (own.field * 1.2).max(fixed.field / 3.0),
                        ..own
                    })
                }
                Output::Variation | Output::Idle | Output::Lineup => None,
            };
            let view = angled.as_ref().unwrap_or(&fixed);
            let mut pixels = render_tile(&c, view, size, output == Output::Anatomy);
            if output == Output::Lifecycle {
                let title = if age >= 365.0 {
                    format!("{:.1}Y", age / 365.0)
                } else {
                    format!("{age:.1}D")
                };
                let title = if brooding && c.kind.z != 0 {
                    format!("{title} BROOD")
                } else {
                    title
                };
                label(&mut pixels, size, &title);
            } else if output == Output::Variation {
                label(&mut pixels, size, &format!("S{seed}"));
            }
            (panel, pixels)
        })
        .collect();
    for (panel, pixels) in tiles {
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

/// The idle animation, at half the stills' size: a GIF's palette discards more detail than the
/// halved resolution does. Frames render all at once, so the cores stay busy through each frame's
/// ragged end.
fn idle(path: &Path, kind: usize, size: usize, seconds: f32) {
    let size = (size / 2).max(64);
    let age = life::KINDS[kind].biology().maturity_days * 1.25;
    let c = specimen(kind, age, 0, seconds);
    let mut view = View::fit(1, &c);
    view.field *= 1.2;
    let mut encoder =
        GifEncoder::new_with_speed(BufWriter::new(File::create(path).expect("create GIF")), 10);
    encoder.set_repeat(Repeat::Infinite).expect("GIF loop");
    let frames: Vec<Vec<u8>> = (0..48)
        .into_par_iter()
        .map(|frame| {
            let c = specimen(kind, age, 0, seconds + frame as f32 / 12.0);
            render_tile(&c, &view, size, false)
        })
        .collect();
    for pixels in frames {
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

/// The taxa where the tree forks: those with more than one branch that has species below it.
fn forks() -> Vec<&'static Taxon> {
    let mut branches: Vec<(&'static Taxon, Vec<&'static str>)> = Vec::new();
    for &species in SPECIES {
        let mut child: &'static Taxon = species;
        while let Some(parent) = child.parent {
            match branches
                .iter_mut()
                .find(|(taxon, _)| taxon.name == parent.name)
            {
                Some((_, children)) if !children.contains(&child.name) => children.push(child.name),
                Some(_) => {}
                None => branches.push((parent, vec![child.name])),
            }
            child = parent;
        }
    }
    branches
        .into_iter()
        .filter(|(_, children)| children.len() > 1)
        .map(|(taxon, _)| taxon)
        .collect()
}

/// Where a taxon's own sheets live: its module's folder, beside its `mod.rs`.
fn taxon_folder(taxon: &Taxon) -> PathBuf {
    let mut depth = 0;
    let mut ancestor = taxon;
    while let Some(parent) = ancestor.parent {
        depth += 1;
        ancestor = parent;
    }
    let kind = SPECIES
        .iter()
        .position(|species| species.descends_from(taxon))
        .expect("a fork has species below it");
    let module = SPECIES_MODULES[kind].split_whitespace().collect::<String>();
    module.split("::").take(depth).fold(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scene/life"),
        |path, part| path.join(part),
    )
}

/// Every species below a taxon, grown, seen from the same side and drawn to one scale, packed in
/// rows into a sheet near to square. The largest is `size` pixels across; none is drawn smaller
/// than a few pixels, so the smallest can still be found.
fn render_lineup(taxon: &Taxon, size: usize, seconds: f32) -> RgbImage {
    let members: Vec<(usize, CreatureInstance, View, f32)> = SPECIES
        .iter()
        .enumerate()
        .filter(|(_, species)| species.descends_from(taxon))
        .map(|(kind, _)| {
            let days = life::KINDS[kind].biology().maturity_days * 1.25;
            let c = specimen(kind, days, 0, seconds);
            let mut view = View::fit(1, &c);
            // Room for wing tips and trailing tails the fit's coarse look can miss.
            view.field *= 1.2;
            view.backdrop = even_sky;
            // How wide the view is in metres at the creature, from how far off the camera sits.
            let across = view.field * view.origin.length();
            (kind, c, view, across)
        })
        .collect();
    let widest = members.iter().map(|m| m.3).fold(0.0f32, f32::max).max(1e-3);
    let mut tiles: Vec<(usize, usize)> = members
        .iter()
        .enumerate()
        .map(|(index, member)| (index, ((member.3 / widest * size as f32) as usize).max(12)))
        .collect();
    tiles.sort_by_key(|&(_, side)| Reverse(side));
    let gap = 4;
    let area: usize = tiles
        .iter()
        .map(|(_, side)| (side + gap) * (side + gap))
        .sum();
    let limit = (((area as f32).sqrt() * 1.15) as usize).max(tiles[0].1 + gap);
    // Shelves, each as tall as its first and largest tile, filled up to the limit.
    let mut rows: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut filled = 0;
    for &tile in &tiles {
        match rows.last_mut() {
            Some(row) if filled + tile.1 + gap <= limit => row.push(tile),
            _ => {
                rows.push(vec![tile]);
                filled = 0;
            }
        }
        filled += tile.1 + gap;
    }
    let span = |row: &[(usize, usize)]| row.iter().map(|(_, side)| side + gap).sum::<usize>();
    let width = rows.iter().map(|row| span(row)).max().unwrap_or(0) + gap;
    // Each row centred across the sheet, and each tile within its row's height.
    let mut placed = Vec::new();
    let mut y = gap;
    for row in &rows {
        let height = row[0].1;
        let mut x = gap + (width - gap - span(row)) / 2;
        for &(index, side) in row {
            placed.push((index, side, x, y + (height - side) / 2));
            x += side + gap;
        }
        y += height + gap;
    }
    let height = y;
    let rendered: Vec<(usize, usize, Vec<u8>)> = placed
        .par_iter()
        .map(|&(index, side, x, y)| {
            let (_, c, view, _) = &members[index];
            (
                x,
                y,
                RgbImage::from_raw(side as u32, side as u32, render_tile(c, view, side, false))
                    .expect("lineup tile")
                    .into_raw(),
            )
        })
        .collect();
    let mut sheet = RgbImage::from_fn(width as u32, height as u32, |_, _| {
        let shade = tonemap(even_sky(Vec2::ZERO));
        image::Rgb([
            (shade.x * 255.0) as u8,
            (shade.y * 255.0) as u8,
            (shade.z * 255.0) as u8,
        ])
    });
    for ((x, y, pixels), &(_, side, _, _)) in rendered.into_iter().zip(&placed) {
        let tile = RgbImage::from_raw(side as u32, side as u32, pixels).expect("lineup tile");
        imageops::replace(
            &mut sheet,
            &tile,
            i64::try_from(x).expect("sheet width"),
            i64::try_from(y).expect("sheet height"),
        );
    }
    sheet
}

/// A lineup beside every fork in the tree, or in `output`, for the forks `selected` names.
fn lineups(output: Option<&str>, selected: Option<&str>, tile: usize, seconds: f32) {
    for taxon in forks() {
        if selected.is_some_and(|filter| {
            !filter
                .split(',')
                .any(|name| slug(taxon.name).contains(name))
        }) {
            continue;
        }
        let path = output.map_or_else(
            || taxon_folder(taxon).join(format!("{}.png", slug(taxon.name))),
            |output| {
                let name = if taxon.parent.is_some() {
                    filename(taxon)
                } else {
                    slug(taxon.name)
                };
                Path::new(output).join(format!("{name}.png"))
            },
        );
        render_lineup(taxon, tile, seconds)
            .save(&path)
            .expect("save lineup");
        println!("{}: {}", taxon.name, path.display());
    }
}

fn main() {
    if env::args().any(|arg| arg == "--help") {
        println!(
            "species_preview [--size 768] [--species comma,separated,names] [--out directory] [--time seconds] [--outputs render,anatomy,lifecycle,variation,idle,lineup]\nBy default, writes all five assets beside each species source file, and at each fork in the tree a lineup of every species below it, named for the taxon, beside its mod.rs. --anatomy selects anatomy only."
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
            "render,anatomy,lifecycle,variation,idle,lineup".into()
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
            "lineup" => Output::Lineup,
            _ => panic!("unknown output: {name}"),
        })
        .collect();
    if outputs.contains(&Output::Lineup) {
        lineups(output.as_deref(), selected.as_deref(), tile, seconds);
    }
    let outputs: Vec<_> = outputs
        .into_iter()
        .filter(|&output| output != Output::Lineup)
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
    assert!(
        count > 0 || outputs.is_empty(),
        "no species matched the filter"
    );
}
