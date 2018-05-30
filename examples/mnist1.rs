extern crate csv;
extern crate baby_nn;

use csv::ReaderBuilder;
use baby_nn::{Sample, Vector, NN};
use std::f64;

const INP: usize = 784;
const OUT: usize = 10;

/// This struct is supposed to represent the given data, not adjusted to the neural net.
/// Due to the implemented From-Trait, it is easily to create a Sample from it.
#[derive(Debug)]
struct LabeledImage {
    label: u8,
    image: Vec<u8>,
}

impl From<LabeledImage> for Sample {
    fn from(img: LabeledImage) -> Self {
        // Note: The return value of the sigmoid function (and therefore the output)
        // will not reach 1.0 or 0.0, but only values inbetween.
        // So the ideal label [0,0,0,1,0,0,0,0,0,0] will be
        // [.01,.01,.01,.99,.01,.01,.01,.01,.01,.01] to work around this.
        let mut label = vec![0.01; 10]; // most values are (nearly) zero, so we begin like this.
        label[img.label as usize] = 0.99; // set the label.
        let label = label; // remove mutability just because we do not need it anymore.
        let image: Vec<f64> = img.image
            .into_iter()
            .map(|byte| byte_into_nn_compatible(byte))
            .collect();
        // make sure that the input fits into our NN topology (input/output neurons amount).
        assert_eq!(image.len(), INP);
        assert_eq!(label.len(), OUT);
        Sample {
            input: Vector::new(image),
            target: Vector::new(label),
        }
    }
}

/// Since our neural network uses the sigmoid function, we need to adjust our byte data to fit into the interval [0,1]. More percisely, it needs to exclude 0.0 and 1.1 specifically, since the sigmoid function used will never return 0 or 1. This means our outputs should be in this interval, and a general rule says that the input should be based on the same interval as the output.
fn byte_into_nn_compatible(num: u8) -> f64 {
    ((num as f64 / 255.0) * 0.98) + 0.01 // NOTE reference uses 0.99 instead of 0.98
}

fn read_labeled_images(path: &str) -> Vec<LabeledImage> {
    //let path = "examples/mnist_train_100.csv";
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect(&format!("File {} not found.", path));
    let mut samples: Vec<LabeledImage> = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Result unwrap did not work.");
        let stuff: Vec<u8> = record // record contains csv strings (bytes as strings) values
            .into_iter()
            .map(|s| s.parse().expect("could not parse into u8")) // str --> u8 (byte)
            .collect(); // u8 iterator --> Vec<u8>
        let (head, tail) = stuff
            .split_first()
            .expect("fail: tried to split an empty list.");
        samples.push(LabeledImage {
            label: head.clone(),
            image: tail.to_vec(),
        });
    }
    //println!("{:?}", samples);
    samples
}

fn main() {
    // load training data
    let training_samples: Vec<Sample> = read_labeled_images("examples/mnist_train_100.csv")
        .into_iter() // for each sample... (iterating over them)
        .map(|img| img.into()) // into() here equals to Sample::from(img). Rust is magic :)
        .collect(); // collect all samples to match the given type of `training_samples`.
    // set up network topology and learning rate
    let hid = 100;
    let learn_rate = 0.3;
    let mut nn = NN::new(INP, hid, OUT, learn_rate);
    // train the network
    for _epoch in 0..1 {
        for sample in &training_samples {
            println!("training sample...");
            nn.train(&sample);
        }
    }
    // test the network
    let test_samples: Vec<Sample> = read_labeled_images("examples/mnist_test_10.csv")
        .into_iter()
        .map(|img| img.into())
        .collect();
    let mut scorecard: (usize, usize) = (0,0);
    for sample in test_samples {
        let output = nn.query(&sample.input);
        // println!("{}", &sample.target);
        let result = &output.argmax();
        let target = &sample.target.argmax().0;
        println!("{} ({}) -- {}", result.0, target, result.1);
        if result.0 == *target {
            scorecard.0 += 1;
        } else {
            scorecard.1 += 1;
        }
    }
    let precision = scorecard.0 as f64 / (scorecard.0 as f64 + scorecard.1 as f64);
    println!("precision: {}", precision);
}
