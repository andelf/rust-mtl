#![feature(unboxed_closures, core, fixed_size_array)]

extern crate num;
extern crate rand;
extern crate core;

use rand::{thread_rng, Rng};
use core::array::FixedSizeArray;
use std::fmt;
use std::ops;
use std::marker::PhantomData;
use std::str::FromStr;
use num::traits::One;


pub trait Matrix<T> {
    fn new(usize, usize) -> Self;
    fn transpose(&mut self);
}


// pub struct DenseVector<T> {
//     data: Vec<T>
// }

// impl DenseVector {

// }

// memory arranging
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


// TODO: fit this into Matrix definition
pub trait MatrixStorage<T> {
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

impl<T> MatrixStorage<T> for FixedSizeArray<T> {
    fn as_slice(&self) -> &[T] {
        FixedSizeArray::<T>::as_slice(self)
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        FixedSizeArray::<T>::as_mut_slice(self)
    }
}

impl<T> MatrixStorage<T> for Vec<T> {
    fn as_slice(&self) -> &[T] {
        &self[..]
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self[..]
    }
}


// Dense 2D matrix
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

impl<T> Dense2D<T> {
    pub fn from_vec_and_dim(vec: Vec<T>, (nrow, ncol): (usize, usize)) -> Dense2D<T> {
        assert_eq!(vec.len(), nrow * ncol);
        Dense2D {
            data: vec,
            dim: (nrow, ncol),
            _marker: PhantomData
        }
    }

    pub fn reshape(&mut self, (nrow, ncol): (usize, usize)) {
        assert_eq!(self.dim.0 * self.dim.1, nrow * ncol);
        self.dim = (nrow, ncol);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseMatrixError {
    _priv: ()
}

impl fmt::Display for ParseMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "parse matrix error".fmt(f)
    }
}

impl<T> FromStr for Dense2D<T>
    where T: FromStr, <T as FromStr>::Err: fmt::Debug {

    type Err = ParseMatrixError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut v: Vec<T> = Vec::new();
        let nrow = s.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).count();

        s.split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .map(|seg| v.push(seg.parse().unwrap())).count();

        let ncol = v.len() / nrow;
        assert_eq!(v.len(), ncol * nrow);
        Ok(Dense2D::from_vec_and_dim(v, (nrow, ncol)))
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

// element index

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

// binary operation

macro_rules! impl_binary_ops_for_dense2d {
    ($op:ident, $func:ident) => (
        // FIXME: introducing RHS here will cause some other function's type inferring fail
        //        So `T: Op<T>`
        // For e.g.:
        //        m + Matrix::eye(4) will fail to guess type.
        impl<T: ops::$op<T> + Copy, A: Arranging> ops::$op<Dense2D<T, A>> for Dense2D<T, A> {
            type Output = Dense2D<T::Output, A>;

            fn $func(self, rhs: Dense2D<T, A>) -> Dense2D<T::Output, A> {
                let mut result = Dense2D::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[(j,i)] = self[(j,i)].$func(rhs[(j,i)]);
                }
                result
            }
        }

        impl<RHS: Copy, T: ops::$op<RHS> + Copy, A: Arranging> ops::$op<RHS> for Dense2D<T, A> {
            type Output = Dense2D<T::Output, A>;

            fn $func(self, rhs: RHS) -> Dense2D<T::Output, A> {
                let mut result = Dense2D::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[(j,i)] = self[(j,i)].$func(rhs);
                }
                result
            }
        }

    )
}

impl_binary_ops_for_dense2d!(Add, add);
impl_binary_ops_for_dense2d!(BitAnd, bitand);
impl_binary_ops_for_dense2d!(BitOr, bitor);
impl_binary_ops_for_dense2d!(BitXor, bitxor);
impl_binary_ops_for_dense2d!(Div, div);
impl_binary_ops_for_dense2d!(Rem, rem);
impl_binary_ops_for_dense2d!(Shl, shl);
impl_binary_ops_for_dense2d!(Shr, shr);
impl_binary_ops_for_dense2d!(Sub, sub);

// special handling for multiply operation
impl<T: ops::Mul + Copy, A: Arranging> Dense2D<T, A> {
    /// element-wise multiply
    pub fn x(self, rhs: Dense2D<T, A>) -> Dense2D<T::Output, A> {
        let mut result = Dense2D::new(self.dim.0, self.dim.1);
        for (j, i) in self.iter_indices() {
            result[(j,i)] = self[(j,i)] * rhs[(j,i)];
        }
        result
    }
}

// for m * 2
impl<RHS: Copy, T: ops::Mul<RHS> + Copy, A: Arranging> ops::Mul<RHS> for Dense2D<T, A> {
    type Output = Dense2D<T::Output, A>;

    fn mul(self, rhs: RHS) -> Dense2D<T::Output, A> {
        let mut result = Dense2D::new(self.dim.0, self.dim.1);
        for (j, i) in self.iter_indices() {
            result[(j,i)] = self[(j,i)] * rhs;
        }
        result
    }
}

impl<T: ops::Mul<T> + Copy, A: Arranging> ops::Mul<Dense2D<T, A>> for Dense2D<T, A> {
    type Output = Dense2D<T::Output, A>;

    // matrix multiply
    fn mul(self, rhs: Dense2D<T, A>) -> Dense2D<T::Output, A> {
        let mut result = Dense2D::new(self.dim.0, self.dim.1);
        for (j, i) in self.iter_indices() {
            result[(j,i)] = self[(j,i)] * (rhs[(j,i)]);
        }
        result
    }
}

// unnary operation

macro_rules! impl_unnary_ops_for_dense2d {
    ($op:ident, $func:ident) => (
        impl<T: ops::$op + Copy, A: Arranging> ops::$op for Dense2D<T, A> {
            type Output = Dense2D<T::Output, A>;

            fn $func(self) -> Dense2D<T::Output, A> {
                let mut result = Dense2D::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[(j,i)] = self[(j,i)].$func()
                }
                result
            }
        }
    )
}

impl_unnary_ops_for_dense2d!(Neg, neg);
impl_unnary_ops_for_dense2d!(Not, not);


impl<T: Clone, A> Clone for Dense2D<T, A> {
    fn clone(&self) -> Self {
        Dense2D {
            data: self.data.clone(),
            dim: self.dim,
            _marker: PhantomData
        }
    }
}

impl<T: PartialEq, A: Arranging> PartialEq for Dense2D<T, A> {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.dim, other.dim);
        for j in 0 .. self.dim.0 {
            for i in 0 .. self.dim.1 {
                if self[(j,i)] != other[(j,i)] {
                    return false;
                }
            }
        }
        true
    }
}

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


// column & row view

pub struct Dense1DView<'a, T: 'a> {
    inner: &'a [T],
    index_map: Vec<usize>,
    size: usize,
}

pub struct Dense2Dview<'a, T: 'a> {
    inner: &'a [T],
    indices_map: Vec<usize>,
    dim: (usize, usize)
}


// issue #1
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


impl<T: fmt::Debug, A> fmt::Debug for Dense2D<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "<Matrix dim={:?}, {:?}>", self.dim, self.data));
        Ok(())
    }
}


#[allow(unused_must_use)]
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
                write!(f, "{:-4.}", self[(j,i)]);
                if i == self.dim.1 - 1 {
                    write!(f, "]");
                } else {
                    write!(f, ", ");
                }
            }
            if j == self.dim.0 - 1 {
                write!(f, "]");
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
fn test_equal_from_str() {
    let m1 = Dense2D::from_vec_and_dim(vec![1i32, 2, 3, 4], (2, 2));
    let m2 = Dense2D::from_vec_and_dim(vec![1i32, 2, 3, 4], (2, 2));
    assert_eq!(m1, m2);
    assert_eq!(m1, Dense2D::from_str("1 2; 3 4").unwrap());
}


#[test]
fn test_from_vec_and_dim(){
    let m1 = Dense2D::from_vec_and_dim(vec![1f32, 2f32, 3f32, 4f32], (2, 2));
    let m2 = Dense2D::from_vec_and_dim(vec![5f32, 6., 7., 8.], (2, 2));
    println!("m1 * m2 = {}", m1.x(m2));

}

#[test]
fn test_add_sub() {
    let m1: Dense2D<i32> = FromStr::from_str("1 2; 3 4").unwrap();
    let m2: Dense2D<i32> = FromStr::from_str("5 6; 7 8").unwrap();

    let m3: Dense2D<i32> = FromStr::from_str("6 8; 10 12").unwrap();
    let m4: Dense2D<i32> = FromStr::from_str("4 4; 4 4").unwrap();

    assert_eq!(m1.clone() + m2.clone(), m3);
    assert_eq!(- (m1.clone() - m2.clone()), m4);
}



#[test]
fn test_add() {
    let mut m1 = Dense2D::<f64>::new(4, 4);
    let m2 = Dense2D::<f64>::eye(4);

    let mut rng = thread_rng();

    for _ in 0 .. 10 {
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
