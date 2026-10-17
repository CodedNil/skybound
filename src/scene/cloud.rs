//! Compact cloud records and the single planet-wide spatial index.
use spirv_std::glam::{IVec3, Vec2, Vec3, Vec4};

pub const CELL_SIZE: f32 = 64_000.0;
pub const GRID_WIDTH: u32 = 52;
pub const GRID_CELLS: usize = (GRID_WIDTH * GRID_WIDTH * GRID_WIDTH) as usize;
pub const GRID_ORIGIN: f32 = -(GRID_WIDTH as f32 * CELL_SIZE * 0.5);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cloud {
    pub center: Vec4,     // planet position, east radius
    pub up: Vec4,         // radial axis, vertical radius
    pub east: Vec4,       // horizontal axis, north radius
    pub appearance: Vec4, // extinction, erosion, noise scale, convection
    pub motion: Vec4,     // velocity, stable seed
}
impl Cloud {
    pub fn ray_interval(self, origin: Vec3, direction: Vec3, elapsed: f32) -> Vec2 {
        let local = self.coordinates(origin, elapsed);
        let north = self.up.truncate().cross(self.east.truncate());
        let ray = Vec3::new(
            direction.dot(self.east.truncate()) / self.center.w,
            direction.dot(north) / self.east.w,
            direction.dot(self.up.truncate()) / self.up.w,
        );
        super::geometry::intersect_sphere(local, ray, 1.0)
    }
    pub fn coordinates(self, point: Vec3, elapsed: f32) -> Vec3 {
        let p = point - self.center.truncate() - self.motion.truncate() * elapsed;
        Vec3::new(
            p.dot(self.east.truncate()) / self.center.w,
            p.dot(self.up.truncate().cross(self.east.truncate())) / self.east.w,
            p.dot(self.up.truncate()) / self.up.w,
        )
    }
    pub fn extent(self) -> Vec3 {
        self.east.truncate().abs() * self.center.w
            + self.up.truncate().cross(self.east.truncate()).abs() * self.east.w
            + self.up.truncate().abs() * self.up.w
    }
}

pub fn cell(point: Vec3) -> IVec3 {
    ((point - Vec3::splat(GRID_ORIGIN)) / CELL_SIZE)
        .floor()
        .as_ivec3()
}
pub fn cell_index(cell: IVec3) -> Option<usize> {
    if cell.cmplt(IVec3::ZERO).any() || cell.cmpge(IVec3::splat(52)).any() {
        return None;
    }
    Some(((cell.z as u32 * GRID_WIDTH + cell.y as u32) * GRID_WIDTH + cell.x as u32) as usize)
}

pub struct RayEvent {
    pub ids: [u32; 8],
    pub count: u32,
    pub end: f32,
    pub step: f32,
    pub overflow: bool,
}

/// Front-to-back ellipsoid events within one spatial cell. Overlaps contribute
/// together; dense cells fall back to the complete list instead of dropping clouds.
pub fn ray_event(
    origin: Vec3,
    direction: Vec3,
    start: f32,
    limit: f32,
    elapsed: f32,
    records: &[Cloud],
    grid: &[u32],
) -> RayEvent {
    let position = origin + direction * (start + 0.5);
    let coordinate = cell(position);
    let lower = coordinate.as_vec3() * CELL_SIZE + Vec3::splat(GRID_ORIGIN);
    let face = Vec3::select(
        direction.cmpgt(Vec3::ZERO),
        lower + Vec3::splat(CELL_SIZE),
        lower,
    );
    let distances = Vec3::select(
        direction.abs().cmpgt(Vec3::splat(0.000_000_1)),
        (face - position) / direction,
        Vec3::splat(f32::MAX),
    );
    let exit = distances.min_element().max(0.5) + start + 0.5;
    let mut event = RayEvent {
        ids: [0; 8],
        count: 0,
        end: exit.min(limit),
        step: 4000.0,
        overflow: false,
    };
    let Some(index) = cell_index(coordinate) else {
        return event;
    };
    for offset in grid[index]..grid[index + 1] {
        let id = grid[GRID_CELLS + 1 + offset as usize];
        let cloud = records[id as usize];
        if cloud.appearance.x <= 0.000_000_1 {
            continue;
        }
        let interval = cloud.ray_interval(origin, direction, elapsed);
        if interval.x >= interval.y || interval.y <= start + 0.25 {
            continue;
        }
        if interval.x > start + 0.5 {
            event.end = event.end.min(interval.x);
        } else {
            event.end = event.end.min(interval.y);
            if event.count < 8 {
                event.ids[event.count as usize] = id;
                event.count += 1;
            } else {
                event.overflow = true;
            }
            let local_direction = Vec3::new(
                direction.dot(cloud.east.truncate()) / cloud.center.w,
                direction.dot(cloud.up.truncate().cross(cloud.east.truncate())) / cloud.east.w,
                direction.dot(cloud.up.truncate()) / cloud.up.w,
            );
            event.step = event.step.min(0.08 / local_direction.length());
        }
    }
    event.end = event.end.max(start + 0.5).min(limit);
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ellipsoid_intervals_preserve_world_distance_and_motion() {
        let cloud = Cloud {
            center: Vec3::ZERO.extend(100.0),
            up: Vec3::Z.extend(20.0),
            east: Vec3::X.extend(60.0),
            motion: Vec3::X.extend(0.0),
            ..Default::default()
        };
        let bounds = cloud.ray_interval(Vec3::new(-200.0, 0.0, 0.0), Vec3::X, 10.0);
        assert!((bounds.x - 110.0).abs() < 0.001);
        assert!((bounds.y - 310.0).abs() < 0.001);
        let inside = cloud.ray_interval(Vec3::ZERO, Vec3::Z, 0.0);
        assert!((inside.x + 20.0).abs() < 0.001 && (inside.y - 20.0).abs() < 0.001);
        let miss = cloud.ray_interval(Vec3::new(-200.0, 0.0, 21.0), Vec3::X, 0.0);
        assert!(miss.x > miss.y);
    }

    #[test]
    fn traversal_orders_entries_exits_and_keeps_overlapping_clouds() {
        let origin = Vec3::new(100.0, 1000.0, 1000.0);
        let cloud = |x| Cloud {
            center: Vec3::new(x, 1000.0, 1000.0).extend(100.0),
            up: Vec3::Z.extend(100.0),
            east: Vec3::X.extend(100.0),
            appearance: Vec4::splat(0.001),
            ..Default::default()
        };
        let records = [cloud(1000.0), cloud(1150.0)];
        let index = cell_index(cell(origin)).unwrap();
        let mut grid = vec![0; GRID_CELLS + 1];
        grid[index + 1..].fill(2);
        grid.extend([0, 1]);
        let first = ray_event(origin, Vec3::X, 0.0, 100_000.0, 0.0, &records, &grid);
        assert_eq!(first.count, 0);
        assert!((first.end - 800.0).abs() < 0.01);
        let entry = ray_event(origin, Vec3::X, first.end, 100_000.0, 0.0, &records, &grid);
        assert_eq!(entry.count, 1);
        assert!((entry.end - 950.0).abs() < 0.01);
        let overlap = ray_event(origin, Vec3::X, entry.end, 100_000.0, 0.0, &records, &grid);
        assert_eq!(overlap.count, 2);
        assert!((overlap.end - 1000.0).abs() < 0.01);
        let empty = ray_event(origin, Vec3::X, 2000.0, 100_000.0, 0.0, &records, &grid);
        assert!(
            empty.end > 60_000.0,
            "parallel cell axes must not restrict traversal"
        );
        let dense = [cloud(1000.0); 9];
        grid[index + 1..=GRID_CELLS].fill(9);
        grid.truncate(GRID_CELLS + 1);
        grid.extend(0..9);
        let overlap = ray_event(origin, Vec3::X, 900.0, 100_000.0, 0.0, &dense, &grid);
        assert!(overlap.overflow);
        assert_eq!(overlap.count, 8);
    }
}
