extern crate mtl;

use mtl::matrix::Matrix;
use std::str::FromStr;
use std::mem;

fn main() {
    let m: Matrix<f64> = FromStr::from_str("1 2; 3 4").unwrap();
    println!("{}", m);

    // 40: 24 for Vec<T>
    //     16 for (usize, usize)
    println!("size => {}", mem::size_of_val(&m));


    let mut m: Matrix<f64> = FromStr::from_str("1 2 3; 4 5 6").unwrap();
    println!("{}", m);
    print!("after reshape:");
    m.reshape((3, 2));
    println!("{}", m);

    println!("{}", &m + &m);
}
