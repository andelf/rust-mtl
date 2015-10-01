#![feature(test)]

extern crate mtl;

extern crate test;
extern crate rand;


use test::Bencher;
use rand::{thread_rng, Rng};

// use mtl::Dense2D;
use mtl::matrix::Matrix;
use mtl::matrix::sparse::SparseYale;


const SIZE: usize = 500;
const SPARSY: u32 = 20;

#[bench]
fn access_via_double_index(b: &mut Bencher) {
    let mut mv = Vec::new();
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    // for _ in 0 .. SIZE {
    //     mv.push(thread_rng().gen_iter().take(SIZE).collect::<Vec<i32>>());
    // }
    let m = Matrix::from_vec(mv);
    let mut v = 0i32;
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
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = Matrix::from_vec(mv);
    let mut v = 0i32;
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
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = Matrix::from_vec(mv);
    let mut v = 0i32;
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
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = Matrix::from_vec(mv);
    let mut v = 0i32;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = unsafe { *m.get_unchecked(i, j) };
            }
        }
    });
}


#[bench]
fn sparse_matrix_access_via_tuple_index(b: &mut Bencher) {
    let mut mv = Vec::new();
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = SparseYale::from_vec(mv);
    let mut v = 0i32;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = m[(i,j)];
            }
        }
    });
}


#[bench]
fn sparse_matrix_access_via_get(b: &mut Bencher) {
    let mut mv = Vec::new();
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = SparseYale::from_vec(mv);
    let mut v = 0i32;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                v = *m.get((i,j)).unwrap();
            }
        }
    });
}


#[bench]
fn sparse_matrix_access_via_get_unchecked(b: &mut Bencher) {
    let mut mv = Vec::new();
    let mut rng = thread_rng();
    for _ in 0 .. SIZE {
        let mut row = Vec::new();
        for _ in 0 .. SIZE {
            if rng.gen_weighted_bool(SPARSY) {
                row.push(rng.gen::<i32>());
            } else {
                row.push(0);
            }
        }
        mv.push(row);
    }
    let m = SparseYale::from_vec(mv);
    let mut v = 0i32;
    b.iter(|| {
        for i in 0 .. m.shape().0 {
            for j in 0 .. m.shape().1 {
                unsafe {
                    v = *m.get_unchecked((i,j));
                }
            }
        }
    });
}
