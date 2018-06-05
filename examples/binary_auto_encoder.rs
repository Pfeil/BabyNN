extern crate baby_nn;

use baby_nn::{Sample, Vector, NN};

fn main() {
    let samples: Vec<Sample> = get_samples();
    let mut nn = NN::new(4, 2, 4, 0.3);

    test("before", &nn, &samples);

    for _ in 0..1 {
        for sample in samples.clone() {
            nn.train(&sample);
        }
    }

    test("after", &nn, &samples);
}

fn get_samples() -> Vec<Sample> {
    vec![
        Vector::new(vec![0.99, 0., 0., 0.]),
        Vector::new(vec![0., 0.99, 0., 0.]),
        Vector::new(vec![0., 0., 0.99, 0.]),
        Vector::new(vec![0., 0., 0., 0.99]),
    ].iter()
        .map(|x| Sample {
            input: x.clone(),
            target: x.clone(),
        })
        .collect::<Vec<Sample>>()
}

fn test(context: &str, nn: &NN, samples: &Vec<Sample>) {
    let mut correct = 0;
    for sample in samples.clone() {
        let result = nn.query(&sample.input);
        println!("result: {:?}", result);
        if result.argmax().0 == sample.target.argmax().0 {
            correct += 1;
        }
    }
    println!("{}: {}/4", context, correct);
}
