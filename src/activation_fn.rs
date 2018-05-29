pub fn sigmoid(x: f64) -> f64 {
    1f64 / (1f64 + (-x).exp())
}
