use anyhow::{anyhow, Error, Result};
use godot::prelude::*;
use std::str::FromStr;

#[derive(Default, Debug, Clone, Copy)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl From<Vector3> for Vec3 {
    fn from(godot_vec: Vector3) -> Self {
        Self {
            x: godot_vec.x,
            y: godot_vec.y,
            z: godot_vec.z,
        }
    }
}

impl From<Vec3> for Vector3 {
    fn from(vec: Vec3) -> Self {
        Self::new(vec.x, vec.y, vec.z)
    }
}

impl ToString for Vec3 {
    fn to_string(&self) -> String {
        format!("x{:.1} y{:.1} z{:.1}", self.x, self.y, self.z)
    }
}

impl FromStr for Vec3 {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.len() == 3 {
            Ok(Self {
                x: parse_coordinate(parts[0], 'x')?,
                y: parse_coordinate(parts[1], 'y')?,
                z: parse_coordinate(parts[2], 'z')?,
            })
        } else {
            Err(anyhow!(
                "Expected 3 parts in the string representation of Vec3"
            ))
        }
    }
}

fn parse_coordinate(coord_str: &str, prefix: char) -> Result<f32> {
    coord_str
        .strip_prefix(prefix)
        .ok_or_else(|| anyhow!("Expected '{}' prefix for coordinate", prefix))?
        .parse()
        .map_err(|e| anyhow!("Failed to parse {} coordinate: {}", prefix, e))
}
