pub struct Sigmoid;

impl Sigmoid {
    pub fn eval_at(x: f64) -> f64 {
        1f64 / (1f64 + (-x).exp())
    }

    pub fn gradient_at(x: f64) -> f64 {
        let out = Self::eval_at(x);
        Self::gradient_from_output(out)
    }

    // This evaluates the gradient exactly like `Sigmoid::gradient_at`,
    // but takes not the paremeter `x`, but the result of `Sigmoid::eval_at(x)`.
    // This is useful for gradient descent implementation, where the output is given,
    // but not the `x`. See `NN::train()` for a usage example.
    pub fn gradient_from_output(output: f64) -> f64 {
        output * (1f64 - output)
    }
}
