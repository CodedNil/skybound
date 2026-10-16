pub fn square(value: f32) -> f32 {
    value * value
}

pub fn cube(value: f32) -> f32 {
    value * value * value
}

pub fn fourth(value: f32) -> f32 {
    square(square(value))
}
