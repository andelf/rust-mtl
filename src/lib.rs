#![feature(heap_api, associated_consts, unique, unboxed_closures, core)]

extern crate num;
extern crate rand;


use rand::{thread_rng, Rng};
use std::fmt;
use std::rt::heap::{allocate, deallocate, reallocate, reallocate_inplace, stats_print};
use std::rt::heap::EMPTY;
use std::ops;
use std::marker::PhantomData;
use std::mem;

use std::ptr::Unique;
use num::traits::{Num, One};


pub trait Matrix<T> {
    fn new(usize, usize) -> Self;
    fn transpose(&mut self);
}


// pub struct DenseVector<T> {
//     data: Vec<T>
// }

// impl DenseVector {

// }

pub trait Arranging {
    #[inline]
    fn index_of_dim((usize,usize), row: usize, col: usize) -> usize;
}

pub struct RowMajor;
pub struct ColMajor;

impl Arranging for RowMajor {
    #[inline]
    fn index_of_dim((_nrow, ncol): (usize,usize), row: usize, col: usize) -> usize {
        row * ncol + col
    }
}

impl Arranging for ColMajor {
    #[inline]
    fn index_of_dim((nrow, _ncol): (usize,usize), row: usize, col: usize) -> usize {
        col * nrow + row
    }
}



pub struct Dense2D<T, A=RowMajor> {
    data: Vec<T>,
    // row x col
    dim: (usize, usize),
    _marker: PhantomData<A>
}


impl<T, A: Arranging> Dense2D<T, A> {
    pub fn iter_indices<'a>(&'a self) -> ::std::vec::IntoIter<(usize, usize)> {
        let mut res = Vec::new();
        for j in 0 .. self.dim.0 {
            for i in 0 .. self.dim.1 {
                res.push((j,i));
            }
        }
        res.into_iter()
    }
}

impl<T, A: Arranging> Matrix<T> for Dense2D<T, A> {
    fn new(num_rows: usize, num_cols: usize) -> Dense2D<T, A> {
        let mut v = Vec::with_capacity(num_rows * num_cols);
        unsafe { v.set_len(num_rows * num_cols) };
        Dense2D {
            data: v,
            dim: (num_rows, num_cols),
            _marker: PhantomData
        }
    }

    fn transpose(&mut self) {
        assert_eq!(self.dim.0, self.dim.1);
        for j in 0 .. self.dim.0 {
            for i in 0 .. j + 1 {
                let a = A::index_of_dim(self.dim, j, i);
                let b = A::index_of_dim(self.dim, i, j);
                self.data.swap(a, b);
            }
        }
    }
}

impl<T: One, A: Arranging> Dense2D<T, A> {
    pub fn eye(n: usize) -> Dense2D<T, A> {
        let mut m = Dense2D::new(n, n);
        for i in 0 .. n {
            m[(i, i)] = One::one();
        }
        m
    }
}


impl<T, A: Arranging> ops::Index<(usize, usize)> for Dense2D<T, A> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[A::index_of_dim(self.dim, row, col)]
    }
}

impl<T, A: Arranging> ops::IndexMut<(usize, usize)> for Dense2D<T, A> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        &mut self.data[A::index_of_dim(self.dim, row, col)]
    }
}


macro_rules! impl_ops_for_dense2d {
    ($op:ident, $func:ident) => (
        impl<T: ops::$op + Copy, A: Arranging> ops::$op for Dense2D<T, A> {
            type Output = Dense2D<T::Output, A>;

            fn $func(self, rhs: Dense2D<T, A>) -> Dense2D<T::Output, A> {
                let mut result = Dense2D::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[(j,i)] = self[(j,i)].$func(rhs[(j,i)]);
                }
                result
            }
        }

    )
}

impl_ops_for_dense2d!(Add, add);
impl_ops_for_dense2d!(Sub, sub);
impl_ops_for_dense2d!(BitAnd, bitand);

// impl<T: ops::Add + Copy, A: Arranging> ops::Add for Dense2D<T, A> {
//     type Output = Dense2D<T::Output, A>;

//     fn add(self, rhs: Dense2D<T, A>) -> Dense2D<T::Output, A> {
//         let mut result = Dense2D::new(self.dim.0, self.dim.1);
//         for (j, i) in self.iter_indices() {
//             result[(j,i)] = self[(j,i)] + rhs[(j,i)];
//         }
//         result
//     }
// }




// impl<T: ops::Add<R> + Copy, A: Arranging, R> ops::Add<R> for Dense2D<T, A> {
//     type Output = Dense2D<T::Output, A>;

//     fn add(self, rhs: R) -> Dense2D<T::Output, A> {
//         let mut result = Dense2D::new(self.dim.0, self.dim.1);
//         for (j, i) in self.iter_indices() {
//             result[(j,i)] = self[(j,i)] + rhs
//         }
//         result
//     }
// }


// impl<T, A: Arranging> FnOnce<(usize, usize)> for Dense2D<T, A> {
//     type Output = &T;

//     extern "rust-call" fn call_once(self, (row, col): (usize, usize)) -> &T {
//         self.data.into_iter().nth(A::index_of_dim(self.dim, row, col)).unwrap()
//     }
// }


// impl<T, A: Arranging> FnMut<(usize, usize)> for Dense2D<T, A> {

//     extern "rust-call" fn call_mut(&mut self, (row, col): (usize, usize)) -> T {
//         self.data.into_iter().nth(A::index_of_dim(self.dim, row, col)).unwrap()
//     }
// }


// impl<T, A: Arranging> Fn<(usize, usize)> for Dense2D<T, A> {

//     extern "rust-call" fn call(&self, (row, col): (usize, usize)) -> T {
//         self.data.into_iter().nth(A::index_of_dim(self.dim, row, col)).unwrap()
//     }
// }



// impl<'a, T> ops::Index<usize> for Dense2D<T> {
//     type Output = RowView<'a, T>;

//     #[inline]
//     fn index(&'a self, row: usize) -> &RowView<'a, T> {
//         let mut ret = Vec::with_capacity(self.ncol);
//         for i in 0 .. self.ncol {
//             ret.push(&self.data[row * self.ncol + i])
//         }
//         &ret
//     }
// }


impl<T: fmt::Debug, A: Arranging> fmt::Debug for Dense2D<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<Matrix dim={:?}, {:?}>", self.dim, self.data);
        Ok(())
    }
}



impl<T: fmt::Display, A: Arranging> fmt::Display for Dense2D<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "");
        for j in 0 .. self.dim.0 {
            if j == 0 {
                write!(f, "[[");
            } else {
                write!(f, " [");
            }
            for i in 0 .. self.dim.1 {
                write!(f, "{:-4.2}", self[(j,i)]);
                if i == self.dim.1 - 1 {
                    write!(f, "]");
                } else {
                    write!(f, ", ");
                }
            }
            if j == self.dim.0 - 1 {
                writeln!(f, "]");
            } else {
                writeln!(f, "");
            }
        }
        Ok(())
    }
}





// }


// pub struct Dense2D<T> { }

// impl<T> Matrix for Dense2D<T> {
// }

#[test]
fn test_add() {
    let mut m1 = Dense2D::<f64>::new(4, 4);
    let m2 = Dense2D::<f64>::eye(4);

    let mut rng = thread_rng();

    for i in 0 .. 10 {
        m1[(rng.gen_range(0, 4), rng.gen_range(0, 4))] = rng.gen();
    }

    println!("m1 => {}", m1);
    println!("m2 => {}", m2);
    let m3 = m1 + m2;
    println!("plus => {}", m3);
    println!("sub => {}", m3 - Dense2D::eye(4) - Dense2D::eye(4) - Dense2D::eye(4) - Dense2D::eye(4));
}


#[test]
fn it_works() {
    let mut m = Dense2D::<f64>::new(5, 5);

    println!("Matrix => {}", m);

    println!("idx => {}", m[(1,2)]);
    m[(1,2)] = 100.0;
    m[(3,4)] = 10.0;
    m[(0,2)] = 1.0;
    println!("idx => {}", m[(1,2)]);
    println!("Matrix => {}", m);
    m.transpose();
    println!("Matrix => {}", m);
    println!("Matrix => {:?}", m);

    let mut m = Dense2D::<f64, ColMajor>::new(3, 3);
    m[(1,2)] = 100.0;
    m[(0,1)] = 10.0;
    println!("Matrix => {}", m);
    println!("Matrix => {:?}", m);


    let m = Dense2D::<f32>::eye(4);
    println!("Matrix => {}", m);
}
