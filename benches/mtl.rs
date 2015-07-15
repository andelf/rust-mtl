#![feature(test)]

extern crate mtl;

extern crate test;
extern crate rand;


use test::Bencher;
use rand::{thread_rng, Rng};

use mtl::Dense2D;
use mtl::Matrix;


const SIZE: usize = 20;

#[bench]
fn access_via_double_index(b: &mut Bencher) {
    let mut mv = Vec::new();
    for _ in 0 .. SIZE {
        mv.push(thread_rng().gen_iter().take(SIZE).collect::<Vec<f64>>());
    }
    let m = Dense2D::from_vec(mv);
    let mut v = 0f64;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = m[i][j];
            }
        }
    });
}

 #[bench]
fn access_via_tuple_index(b: &mut Bencher) {
    let mut mv = Vec::new();
    for _ in 0 .. SIZE {
        mv.push(thread_rng().gen_iter().take(SIZE).collect::<Vec<f64>>());
    }
    let m = Dense2D::from_vec(mv);
    let mut v = 0f64;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = m[(i,j)];
            }
        }
    });
}

#[bench]
fn access_via_matrix_get(b: &mut Bencher) {
    let mut mv = Vec::new();
    for _ in 0 .. SIZE {
        mv.push(thread_rng().gen_iter().take(SIZE).collect::<Vec<f64>>());
    }
    let m = Dense2D::from_vec(mv);
    let mut v = 0f64;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = *m.get(i,j).unwrap();
            }
        }
    });
}

#[bench]
fn access_via_matrix_get_unchecked(b: &mut Bencher) {
    let mut mv = Vec::new();
    for _ in 0 .. SIZE {
        mv.push(thread_rng().gen_iter().take(SIZE).collect::<Vec<f64>>());
    }
    let m = Dense2D::from_vec(mv);
    let mut v = 0f64;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = unsafe { *m.get_unchecked(i, j) };
            }
        }
    });
}
