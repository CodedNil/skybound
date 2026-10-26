use image::{
    Delay, Frame, RgbImage, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
    imageops,
};
use rayon::prelude::*;
use skybound::scene::{
    life,
    life::{
        CreatureInstance, Rank, SPECIES, SPECIES_MODULES, Taxon,
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
    time::SystemTime,
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
    /// How much of the view the creature fills across and up, once fitted.
    filled: Vec2,
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
            filled: Vec2::ONE,
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
            let span = max - min + 2.0 / 96.0;
            view.field *= span.max_element() * 1.08;
            view.filled = span / (span.max_element() * 1.08);
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

/// When anything a species' previews are drawn from last changed: its own file, every module
/// file of the taxa above it, the shared life code, the planet and this previewer.
fn sources(kind: usize) -> SystemTime {
    let species = adjacent_preview(kind).with_extension("rs");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let life = root.join("src/scene/life");
    let mut files = vec![
        species.clone(),
        root.join("src/scene/planet.rs"),
        root.join("examples/species_preview.rs"),
    ];
    for folder in species.ancestors().skip(1) {
        if let Ok(entries) = fs::read_dir(folder) {
            files.extend(
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "rs")),
            );
        }
        if folder == life {
            break;
        }
    }
    files
        .iter()
        .filter_map(|file| fs::metadata(file).and_then(|meta| meta.modified()).ok())
        .max()
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// Whether a preview already shows what it would be drawn from now.
fn fresh(path: &Path, since: SystemTime) -> bool {
    fs::metadata(path)
        .and_then(|meta| meta.modified())
        .is_ok_and(|drawn| drawn >= since)
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

/// The taxa where the tree forks, those with more than one branch that has species below it, and
/// every phylum, each a body plan of its own worth a sheet even if only one branch of it lives.
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
        .filter(|(taxon, children)| children.len() > 1 || taxon.rank == Rank::Phylum)
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

/// Every species below a taxon on a clear square sheet `SHEET` pixels across, drawn as large as
/// will fit: measured once at a trial size, then drawn again at the size that fills the square.
fn render_lineup(taxon: &Taxon, seconds: f32) -> RgbaImage {
    const TRIAL: usize = 400;
    let trial = render_lineup_at(taxon, TRIAL, seconds);
    let fill = SHEET as f32 / trial.width().max(trial.height()) as f32;
    let mut sheet = render_lineup_at(taxon, (TRIAL as f32 * fill * 0.98) as usize, seconds);
    // Packing does not scale exactly, so the last few pixels are taken up or given back.
    let over = sheet.width().max(sheet.height());
    if over > SHEET {
        let shrink = SHEET as f32 / over as f32;
        sheet = imageops::resize(
            &sheet,
            ((sheet.width() as f32 * shrink) as u32).max(1),
            ((sheet.height() as f32 * shrink) as u32).max(1),
            imageops::FilterType::Lanczos3,
        );
    }
    let mut square = RgbaImage::new(SHEET, SHEET);
    imageops::overlay(
        &mut square,
        &sheet,
        i64::from((SHEET - sheet.width()) / 2),
        i64::from((SHEET - sheet.height()) / 2),
    );
    square
}

/// The side of a lineup sheet.
const SHEET: u32 = 1024;

/// Every species below a taxon, grown, seen from the same side and drawn to one scale on a clear
/// sheet. The largest is `size` pixels across, and none is drawn smaller than a few pixels so the
/// smallest can still be found.
///
/// Each creature is cut out by its own silhouette, and they are nested together largest first:
/// each goes in the first place, scanning down the sheet and across, where its silhouette clears
/// all those already placed by a small margin. Nothing is turned or rescaled to fit.
fn render_lineup_at(taxon: &Taxon, size: usize, seconds: f32) -> RgbaImage {
    let members: Vec<(CreatureInstance, View, f32)> = SPECIES
        .iter()
        .enumerate()
        .filter(|(_, species)| species.descends_from(taxon))
        .map(|(kind, _)| {
            let days = life::KINDS[kind].biology().maturity_days * 1.25;
            let c = specimen(kind, days, 0, seconds);
            let mut view = View::fit(1, &c);
            // Room for wing tips and trailing tails the fit's coarse look can miss.
            view.field *= 1.2;
            view.filled /= 1.2;
            view.backdrop = even_sky;
            // How wide the view is in metres at the creature, from how far off the camera sits.
            let across = view.field * view.origin.length();
            (c, view, across)
        })
        .collect();
    let widest = members.iter().map(|m| m.2).fold(0.0f32, f32::max).max(1e-3);
    let sprites: Vec<RgbaImage> = members
        .par_iter()
        .map(|(c, view, metres)| {
            let edge = ((metres / widest * size as f32) as usize).max(12);
            let pixels = render_tile(c, view, edge, false);
            // The sky as it was drawn, read from the corner the creature never reaches.
            let sky = Vec3::new(
                f32::from(pixels[0]),
                f32::from(pixels[1]),
                f32::from(pixels[2]),
            );
            // Clear where the sky shows through, and partly clear where a membrane lets it.
            let sprite = RgbaImage::from_fn(edge as u32, edge as u32, |x, y| {
                let at = (y as usize * edge + x as usize) * 3;
                let rgb = [pixels[at], pixels[at + 1], pixels[at + 2]];
                let difference = (f32::from(rgb[0]) - sky.x)
                    .abs()
                    .max((f32::from(rgb[1]) - sky.y).abs())
                    .max((f32::from(rgb[2]) - sky.z).abs());
                let alpha = ((difference - 3.0) / 14.0).clamp(0.0, 1.0);
                image::Rgba([rgb[0], rgb[1], rgb[2], (alpha * 255.0) as u8])
            });
            trim(&sprite)
        })
        .collect();
    nest(&sprites)
}

/// A sprite cut down to the box around what is not clear in it.
fn trim(sprite: &RgbaImage) -> RgbaImage {
    let (mut low, mut high) = ((u32::MAX, u32::MAX), (0, 0));
    for (x, y, pixel) in sprite.enumerate_pixels() {
        if pixel[3] > 8 {
            low = (low.0.min(x), low.1.min(y));
            high = (high.0.max(x), high.1.max(y));
        }
    }
    if low.0 > high.0 {
        return RgbaImage::new(1, 1);
    }
    imageops::crop_imm(sprite, low.0, low.1, high.0 - low.0 + 1, high.1 - low.1 + 1).to_image()
}

/// Sprites nested into one clear sheet by their silhouettes, largest first, each in the first
/// place down and across the sheet where it overlaps none already placed. The sheet's width is the
/// one, of many tried, that leaves it nearest to square.
fn nest(sprites: &[RgbaImage]) -> RgbaImage {
    let shapes: Vec<Silhouette> = sprites.iter().map(Silhouette::of).collect();
    let mut order: Vec<usize> = (0..sprites.len()).collect();
    order.sort_by_key(|&index| Reverse(shapes[index].area()));
    let filled: u32 = shapes.iter().map(Silhouette::area).sum();
    let narrowest = shapes.iter().map(|shape| shape.width).max().unwrap_or(1);
    let side = (filled as f32).sqrt();
    let (placed, width, _) = (0..24)
        .map(|step| narrowest.max((side * (0.9 + step as f32 * 0.05)) as u32))
        .map(|width| {
            let (placed, height) = pack(&shapes, &order, width);
            (placed, width, height)
        })
        .min_by_key(|&(_, width, height)| width.max(height))
        .expect("widths were tried");
    let height = sprites
        .iter()
        .zip(&placed)
        .map(|(sprite, &(_, y))| y * SILHOUETTE_CELL + sprite.height())
        .max()
        .unwrap_or(1);
    let mut sheet = RgbaImage::new(
        width * SILHOUETTE_CELL + SILHOUETTE_CELL,
        height + SILHOUETTE_CELL,
    );
    for (sprite, &(x, y)) in sprites.iter().zip(&placed) {
        imageops::overlay(
            &mut sheet,
            sprite,
            i64::from(x * SILHOUETTE_CELL),
            i64::from(y * SILHOUETTE_CELL),
        );
    }
    sheet
}

/// Silhouettes are matched on a coarse grid of cells this many pixels across, each grown by a cell
/// so neighbours keep a margin.
const SILHOUETTE_CELL: u32 = 3;

/// The cells a sprite covers.
struct Silhouette {
    width: u32,
    height: u32,
    cells: Vec<bool>,
}

impl Silhouette {
    fn of(sprite: &RgbaImage) -> Self {
        let width = sprite.width().div_ceil(SILHOUETTE_CELL);
        let height = sprite.height().div_ceil(SILHOUETTE_CELL);
        let mut cells = vec![false; (width * height) as usize];
        for (x, y, pixel) in sprite.enumerate_pixels() {
            if pixel[3] > 24 {
                let (cx, cy) = (x / SILHOUETTE_CELL, y / SILHOUETTE_CELL);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (gx, gy) = (cx + dx, cy + dy);
                    if gx < width && gy < height {
                        cells[(gy * width + gx) as usize] = true;
                    }
                }
            }
        }
        Self {
            width,
            height,
            cells,
        }
    }

    fn area(&self) -> u32 {
        self.cells.iter().filter(|&&filled| filled).count() as u32
    }

    fn covers(&self, x: u32, y: u32) -> bool {
        self.cells[(y * self.width + x) as usize]
    }
}

/// Where each silhouette goes, in cells, on a sheet `width` cells wide, and how many cells tall
/// the sheet comes out.
fn pack(shapes: &[Silhouette], order: &[usize], width: u32) -> (Vec<(u32, u32)>, u32) {
    let mut taken: Vec<bool> = Vec::new();
    let mut placed = vec![(0, 0); shapes.len()];
    let mut height = 0;
    for &index in order {
        let shape = &shapes[index];
        let fits = |taken: &[bool], x: u32, y: u32| {
            (0..shape.height).all(|sy| {
                (0..shape.width).all(|sx| {
                    !shape.covers(sx, sy)
                        || !taken
                            .get(((y + sy) * width + x + sx) as usize)
                            .copied()
                            .unwrap_or(false)
                })
            })
        };
        // Below everything placed so far there is always room.
        let clear = taken.len() as u32 / width + 1;
        let (x, y) = (0..=clear)
            .flat_map(|y| (0..=width - shape.width).map(move |x| (x, y)))
            .find(|&(x, y)| fits(&taken, x, y))
            .expect("a sheet without end has room");
        let rows = ((y + shape.height) * width) as usize;
        if taken.len() < rows {
            taken.resize(rows, false);
        }
        for sy in 0..shape.height {
            for sx in 0..shape.width {
                if shape.covers(sx, sy) {
                    taken[((y + sy) * width + x + sx) as usize] = true;
                }
            }
        }
        placed[index] = (x, y);
        height = height.max(y + shape.height);
    }
    (placed, height)
}

/// A lineup beside every fork in the tree, or in `output`, for the forks `selected` names.
fn lineups(output: Option<&str>, selected: Option<&str>, seconds: f32, all: bool) {
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
        let since = SPECIES
            .iter()
            .enumerate()
            .filter(|(_, species)| species.descends_from(taxon))
            .map(|(kind, _)| sources(kind))
            .max()
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if !all && fresh(&path, since) {
            continue;
        }
        render_lineup(taxon, seconds)
            .save(&path)
            .expect("save lineup");
        println!("{}: {}", taxon.name, path.display());
    }
}

fn main() {
    if env::args().any(|arg| arg == "--help") {
        println!(
            "species_preview [--size 768] [--species comma,separated,names] [--out directory] [--time seconds] [--outputs render,anatomy,lifecycle,variation,idle,lineup]\nBy default, writes all five assets beside each species source file, redrawing only those older than the code they are drawn from (--all redraws everything), and at each fork in the tree a lineup of every species below it, named for the taxon, beside its mod.rs. --anatomy selects anatomy only."
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
    // Only previews older than what they are drawn from are drawn again, unless all are asked for.
    let all = env::args().any(|arg| arg == "--all");
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
        lineups(output.as_deref(), selected.as_deref(), seconds, all);
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
        let since = sources(kind);
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
            if !all && fresh(&path, since) {
                continue;
            }
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
