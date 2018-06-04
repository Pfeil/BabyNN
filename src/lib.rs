//! This is a crate for educational purposes.
//! It's inspired by the book "Make your own Neural Network" by Tariq Rahshid
//! (actually, it was the german version of the book, "Neuronale Netze selbst programmieren",
//! published by O'reilly).

extern crate rand;
extern crate rulinalg;

mod activation_fn;

use rand::distributions::{Distribution, Normal};
pub use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::Matrix;
pub use rulinalg::vector::Vector;

use activation_fn::Sigmoid;

/// This represents a minimal neural net.
/// # Examples
/// ```
/// let input_len = 2;
/// let hidden_len = 3;
/// let output_len = 5;
/// let learn_rate = 0.3;
/// let nn = baby_nn::NN::new(input_len, hidden_len, output_len, learn_rate);
/// // TODO Train your net before querying, otherwise it will output something more or less random.
/// let input = baby_nn::Vector::new(vec![5., 0.]);
/// let output = nn.query(&input);
/// assert_eq!(output.size(), output_len);
/// ```
#[allow(dead_code)]
#[derive(Debug)]
pub struct NN {
    learn_rate: f64,
    weights_ih: Matrix<f64>,
    weights_ho: Matrix<f64>,
}

#[allow(dead_code)]
impl NN {
    /// construct a simpe 1-Layer feed forward net, using the given
    /// topology (the number of neurons for each layer).
    pub fn new(num_input: usize, num_hidden: usize, num_output: usize, learn_rate: f64) -> Self {
        // prepare random number generator.
        let sqrt_num_input = (num_input as f64).powf(-0.5);
        let sqrt_num_hidden = (num_hidden as f64).powf(-0.5);
        let random_in = Normal::new(0.0, sqrt_num_input);
        let random_hid = Normal::new(0.0, sqrt_num_hidden);
        let rng = &mut rand::thread_rng();
        // prepare vectors with random numbers in a range that is supposed to be not that bad.
        let val_ih: Vec<f64> = (0..num_hidden * num_input)
            .map(|_| random_in.sample(rng))
            .collect();
        let val_ho: Vec<f64> = (0..num_output * num_hidden)
            .map(|_| random_hid.sample(rng))
            .collect();
        // create Matrices using the random numbers from the vectors
        let weights_ih = Matrix::new(num_hidden, num_input, val_ih);
        let weights_ho = Matrix::new(num_output, num_hidden, val_ho);
        NN {
            learn_rate,
            weights_ih,
            weights_ho,
        }
    }

    pub fn train(&mut self, sample: &Sample) {
        assert_eq!(self.num_input_neurons(), sample.input.size());
        assert_eq!(self.num_output_neurons(), sample.target.size());
        let (hidden_out, output) = self.query_all(&sample.input);
        // Backpropagation
        let output_error = &sample.target - &output;
        let hidden_error = self.weights_ho.transpose() * &output_error;
        // Gradient Descent
        let sigmoid_grads: Vector<f64> = output
            .into_iter()
            .map(|out| Sigmoid::gradient_from_output(out))
            .collect();
        self.weights_ho += self.learn_rate * (output_error.elemul(&sigmoid_grads)).dot(&hidden_out);
        let sigmoid_grads: Vector<f64> = hidden_out
            .into_iter()
            .map(|out| Sigmoid::gradient_from_output(out))
            .collect();
        self.weights_ih +=
            self.learn_rate * (hidden_error.elemul(&sigmoid_grads)).dot(&sample.input);
    }

    /// Process the net to get an output vector to the given sample.
    /// It will only return the output vector. To get the vector of the hidden layer too,
    /// see `query_all`.
    pub fn query(&self, input: &Vector<f64>) -> Vector<f64> {
        let (_, output) = self.query_all(input);
        output
    }

    /// A helper function that does the query of the net,
    /// but also returns the output of the hidden layer.
    /// It is used in the training routine only and is therefore private.
    fn query_all(&self, input: &Vector<f64>) -> (Vector<f64>, Vector<f64>) {
        let hidden_summed: Vector<f64> = &self.weights_ih * input;
        assert_eq!(&hidden_summed.size(), &self.weights_ih.rows());
        let hidden_activated: Vector<f64> = hidden_summed
            .into_iter()
            .map(|x| Sigmoid::eval_at(x))
            .collect();
        assert_eq!(&hidden_activated.size(), &self.weights_ih.rows());
        let output_summed: Vector<f64> = &self.weights_ho * &hidden_activated;
        assert_eq!(&output_summed.size(), &self.weights_ho.rows());
        let output_activated: Vector<f64> = output_summed
            .into_iter()
            .map(|x| Sigmoid::eval_at(x))
            .collect();
        assert_eq!(&output_activated.size(), &self.weights_ho.rows());
        (hidden_activated, output_activated)
    }

    fn num_input_neurons(&self) -> usize {
        self.weights_ih.cols()
    }

    fn num_hidden_neurons(&self) -> usize {
        self.weights_ih.rows()
    }

    fn num_output_neurons(&self) -> usize {
        self.weights_ho.rows()
    }
}

/// `Sample` Represents an input vector with the desired output vector.
/// This way, a Neural Net can learn what to return for a given data.
#[derive(Debug)]
pub struct Sample {
    pub input: Vector<f64>,
    pub target: Vector<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_can_tell_amount_of_neurons() {
        // TODO do this for random generated amounts or so
        let inp = 3;
        let hid = 2;
        let out = 1;
        let learn_rate = 0.3;
        let nn = NN::new(inp, hid, out, learn_rate);
        assert_eq!(inp, nn.num_input_neurons());
        assert_eq!(hid, nn.num_hidden_neurons());
        assert_eq!(out, nn.num_output_neurons());
    }

    #[test]
    fn create_and_query_simple_sanity_check() {
        // TODO do this for random generated amounts or so
        let inp = 3;
        let hid = 2;
        let out = 1;
        let learn_rate = 0.3;
        let nn = NN::new(inp, hid, out, learn_rate);
        let input = Vector::new(vec![5.; inp]);
        let output = nn.query(&input);
        assert_eq!(out, output.size());
        //println!("{}", output);
    }

    #[test]
    fn init_weight_sanity_check() {
        let inp = 784;
        let hid = 100;
        let out = 10;
        let learn_rate = 0.3;
        let nn = NN::new(inp, hid, out, learn_rate);
        // check boundaries
        nn.weights_ih.clone().into_vec().into_iter().for_each(|x| {
            if x.abs() >= 1.0 {
                panic!()
            }
        });
        nn.weights_ho.clone().into_vec().into_iter().for_each(|x| {
            if x.abs() >= 1.0 {
                panic!()
            }
        });
        // check absolute maximum: max(x.abs()) for each x.
        let max: f64 = nn.weights_ih
            .clone()
            .into_vec()
            .into_iter()
            .fold(0f64, |res, x| if x.abs() > res { x } else { res });
        println!("Max absolute in weights in->hid: {}", max);
        assert!(max < 1f64);
        let max: f64 = nn.weights_ho
            .clone()
            .into_vec()
            .into_iter()
            .fold(0f64, |res, x| if x.abs() > res { x } else { res });
        println!("Max absolute in weights hid->out: {}", max);
        assert!(max < 1f64);
    }
}
