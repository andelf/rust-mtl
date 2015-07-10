extern crate mtl;

use mtl::Dense2D;
use std::str::FromStr;

fn main() {
    let m: Dense2D<f64> = FromStr::from_str("1 2; 3 4").unwrap();
    println!("{}", m);
}
