use godot::prelude::*;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

#[derive(Default, Debug)]
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

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x{:.1} y{:.1} z{:.1}", self.x, self.y, self.z)
    }
}

impl Serialize for Vec3 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("x{:.1} y{:.1} z{:.1}", self.x, self.y, self.z))
    }
}

impl<'de> Deserialize<'de> for Vec3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.len() == 6 {
            Ok(Self {
                x: parts[1].parse().map_err(Error::custom)?,
                y: parts[3].parse().map_err(Error::custom)?,
                z: parts[5].parse().map_err(Error::custom)?,
            })
        } else {
            Err(D::Error::custom("Expected format 'x0.0 y0.0 z0.0'"))
        }
    }
}
