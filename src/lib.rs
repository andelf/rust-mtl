#![feature(unboxed_closures, core, fixed_size_array, zero_one, iter_arith)]
#![warn(bad_style, unused, unused_import_braces,
        unused_qualifications, unused_results)]


extern crate num;
extern crate rand;
extern crate core;

use core::array::FixedSizeArray;
use std::fmt;
use std::ops;
use std::str::FromStr;
use num::traits::{One, Zero};


pub mod array;

use self::view::{MatrixSubView, MatrixMutSubView};

pub mod view;

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
pub struct Matrix<T> {
    data: Vec<T>,
    // row x col
    dim: (usize, usize),
}


impl<T> Matrix<T> {
    pub fn iter_indices<'a>(&'a self) -> ::std::vec::IntoIter<(usize, usize)> {
        let mut res = Vec::new();
        for j in 0 .. self.dim.0 {
            for i in 0 .. self.dim.1 {
                res.push((j,i));
            }
        }
        res.into_iter()
    }

    #[inline]
    fn index_of_dim(&self, row: usize, col: usize) -> usize {
        row * self.dim.1 + col
    }

    pub fn from_vec_and_dim(vec: Vec<T>, (nrow, ncol): (usize, usize)) -> Matrix<T> {
        assert_eq!(vec.len(), nrow * ncol);
        Matrix {
            data: vec,
            dim: (nrow, ncol),
        }
    }

    pub fn from_vec(vec: Vec<Vec<T>>) -> Matrix<T> {
        let nrow = vec.len();
        let ncol = vec[0].len();

        let data = vec.into_iter().flat_map(|v| v.into_iter()).collect::<Vec<T>>();
        // let mut data = Vec::with_capacity(nrow * ncol);
        // for row in vec {
        //     for item in row {
        //         data.push(item);
        //     }
        // }
        Matrix {
            data: data,
            dim: (nrow, ncol),
        }
    }

    pub fn reshape(&mut self, (nrow, ncol): (usize, usize)) {
        assert_eq!(self.dim.0 * self.dim.1, nrow * ncol);
        self.dim = (nrow, ncol);
    }

    pub fn sub_mat<'a>(&'a self, rows: ops::Range<usize>, cols: ops::Range<usize>) -> MatrixSubView<'a, T> {
        let mut indices = Vec::with_capacity(self.dim.0 * self.dim.1);
        for r in rows.clone() {
            for c in cols.clone() {
                indices.push(r * self.dim.1 + c);
            }
        }
        let dim = (rows.end - rows.start, cols.end - cols.start);
        MatrixSubView {
            inner: &self.data[..],
            orig_dim: self.dim,
            offset: (rows.start, cols.start),
            dim: dim
        }
    }

    pub fn sub_mat_mut<'a>(&'a mut self, rows: ops::Range<usize>, cols: ops::Range<usize>) -> MatrixMutSubView<'a, T> {
        let mut indices = Vec::with_capacity(self.dim.0 * self.dim.1);
        for r in rows.clone() {
            for c in cols.clone() {
                indices.push(r * self.dim.1 + c);
            }
        }
        let dim = (rows.end - rows.start, cols.end - cols.start);
        MatrixMutSubView {
            inner: &mut self.data[..],
            orig_dim: self.dim,
            offset: (rows.start, cols.start),
            dim: dim
        }
    }
}

#[test]
fn test_sub_matrix() {
    let m = Matrix::<f32>::from_vec(
        vec![
            vec![ 4.303,  9.689,  5.349,  7.353,  8.228],
            vec![ 6.591,  5.806,  1.838,  8.379,  2.097],
            vec![ 4.081,  5.902,  9.858,  6.635,  9.396],
            vec![ 4.935,  8.029,  0.033,  6.624,  5.538],
            vec![ 8.774,  2.399,  2.132,  2.793,  3.005]]);

    println!("got => {}", m);
    let m1 = m.sub_mat(1..4, 2..4);
    println!("sub of -> {:?}", m1.offset);
    println!("sub => {}", m1);
    println!("sub[1] => {:?}", &m1[1]);
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

impl<T> FromStr for Matrix<T>
    where T: FromStr, <T as FromStr>::Err: fmt::Debug {

    type Err = ParseMatrixError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut v: Vec<T> = Vec::new();
        let nrow = s.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).count();

        let _ = s.split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .map(|seg| v.push(seg.parse().unwrap())).count();

        let ncol = v.len() / nrow;
        assert_eq!(v.len(), ncol * nrow);
        Ok(Matrix::from_vec_and_dim(v, (nrow, ncol)))
    }
}

impl<T> Matrix<T> {
    pub fn new(num_rows: usize, num_cols: usize) -> Matrix<T> {
        let mut v = Vec::with_capacity(num_rows * num_cols);
        unsafe { v.set_len(num_rows * num_cols) };
        Matrix {
            data: v,
            dim: (num_rows, num_cols),
        }
    }

    pub fn transpose(&mut self) {
        assert_eq!(self.dim.0, self.dim.1);
        for j in 0 .. self.dim.0 {
            for i in 0 .. j + 1 {
                let a = self.index_of_dim(j, i);
                let b = self.index_of_dim(i, j);
                self.data.swap(a, b);
            }
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.dim
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.dim.0 || col >= self.dim.0 {
            None
        } else {
            Some(&self[row][col])
        }
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.dim.0 || col >= self.dim.0 {
            None
        } else {
            Some(&mut self[row][col])
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        &self[row][col]
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self[row][col]
    }
}

impl<T: One> Matrix<T> {
    pub fn eye(n: usize) -> Matrix<T> {
        let mut m = Matrix::new(n, n);
        for i in 0 .. n {
            m[i][i] = One::one();
        }
        m
    }
}

impl<T: Zero> Matrix<T> {
    pub fn zeros(nrow: usize, ncol: usize) -> Matrix<T> {
        let mut m = Matrix::new(nrow, ncol);
        for i in 0 .. nrow {
            for j in 0 .. ncol {
                m[i][j] = Zero::zero();
            }
        }
        m
    }
}

// element index
// index by row: faster
impl<T> ops::Index<usize> for Matrix<T> {
    type Output = [T];

    #[inline]
    fn index<'a>(&'a self, row: usize) -> &'a [T] {
        assert!(row < self.dim.0);
        &self.data[row * self.dim.1 .. (row + 1) * self.dim.1]
    }
}

impl<T> ops::IndexMut<usize> for Matrix<T> {

    #[inline]
    fn index_mut<'a>(&'a mut self, row: usize) -> &'a mut [T] {
        assert!(row < self.dim.0);
        &mut self.data[row * self.dim.1 .. (row + 1) * self.dim.1]
    }
}

// index by tuple: slower
impl<T> ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.data[self.index_of_dim(row, col)]
    }
}

impl<T> ops::IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        let idx = self.index_of_dim(row, col);
        &mut self.data[idx]
    }
}

// binary operation

macro_rules! impl_binary_ops_for_dense2d {
    ($op:ident, $func:ident) => (
        // FIXME: introducing RHS here will cause some other function's type inferring fail
        //        So `T: Op<T>`
        // For e.g.:
        //        m + Matrix::eye(4) will fail to guess type.
        impl<T: ops::$op<T, Output=T> + Copy> ops::$op for Matrix<T> {
            type Output = Matrix<T>;

            fn $func(self, rhs: Matrix<T>) -> Matrix<T> {
                let mut result = Matrix::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[j][i] = self[j][i].$func(rhs[j][i]);
                }
                result
            }
        }

        // &M op M
        impl<'a, T: ops::$op<T> + Copy> ops::$op<Matrix<T>> for &'a Matrix<T> {
            type Output = Matrix<T::Output>;

            fn $func(self, rhs: Matrix<T>) -> Matrix<T::Output> {
                self.$func(&rhs)
            }
        }
        // M op &M
        impl<'a, T: ops::$op<T> + Copy> ops::$op<&'a Matrix<T>> for Matrix<T> {
            type Output = Matrix<T::Output>;

            fn $func(self, rhs: &'a Matrix<T>) -> Matrix<T::Output> {
                (&self).$func(rhs)
            }
        }
        // &M op &M
        impl<'a, 'b, T: ops::$op<T> + Copy> ops::$op<&'a Matrix<T>> for &'b Matrix<T> {
            type Output = Matrix<T::Output>;

            fn $func(self, rhs: &'a Matrix<T>) -> Matrix<T::Output> {
                let mut result = Matrix::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[j][i] = self[j][i].$func(rhs[j][i]);
                }
                result
            }
        }

        // M + s
        // impl<RHS: Copy, T: ops::$op<RHS> + Copy> ops::$op<RHS> for Matrix<T> {
        //     type Output = Matrix<T::Output>;

        //     fn $func(self, rhs: RHS) -> Matrix<T::Output> {
        //         let mut result = Matrix::new(self.dim.0, self.dim.1);
        //         for (j, i) in self.iter_indices() {
        //             result[j][i] = self[j][i].$func(rhs);
        //         }
        //         result
        //     }
        // }

    )
}

impl_binary_ops_for_dense2d!(Add, add);
impl_binary_ops_for_dense2d!(BitAnd, bitand);
impl_binary_ops_for_dense2d!(BitOr, bitor);
impl_binary_ops_for_dense2d!(BitXor, bitxor);
impl_binary_ops_for_dense2d!(Div, div);
impl_binary_ops_for_dense2d!(Rem, rem);
impl_binary_ops_for_dense2d!(Sub, sub);

// special handling for multiply operation
impl<T: ops::Mul + Copy> Matrix<T> {
    /// element-wise multiply
    pub fn x(self, rhs: &Matrix<T>) -> Matrix<T::Output> {
        let mut result = Matrix::new(self.dim.0, self.dim.1);
        for (j, i) in self.iter_indices() {
            result[j][i] = self[j][i] * rhs[j][i];
        }
        result
    }
}

// // for m * 2 element-wise
// impl<RHS: Copy, T: ops::Mul<RHS> + Copy> ops::Mul<RHS> for Matrix<T> {
//     type Output = Matrix<T::Output>;

//     fn mul(self, rhs: RHS) -> Matrix<T::Output> {
//         let mut result = Matrix::new(self.dim.0, self.dim.1);
//         for (j, i) in self.iter_indices() {
//             result[j][i] = self[j][i] * rhs;
//         }
//         result
//     }
// }

// matrix multiply
impl<T: ops::Mul<T, Output=T> + ops::Add<T, Output=T> + ::core::num::Zero + Copy> ops::Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        (&self).mul(&rhs)
    }
}

impl<'a, T: ops::Mul<T, Output=T> + ops::Add<T, Output=T> + ::core::num::Zero + Copy> ops::Mul<Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        self.mul(&rhs)
    }
}

impl<'a, T: ops::Mul<T, Output=T> + ops::Add<T, Output=T> + ::core::num::Zero + Copy> ops::Mul<&'a Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'a Matrix<T>) -> Matrix<T> {
        (&self).mul(rhs)
    }
}

impl<'a, 'b, T: ops::Mul<T, Output=T> + ops::Add<T, Output=T> + ::core::num::Zero + Copy> ops::Mul<&'a Matrix<T>> for &'b Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'a Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dim.1, rhs.dim.0);
        let mut result = Matrix::new(self.dim.0, rhs.dim.1);
        for j in 0 .. self.dim.0 {
            for i in 0 .. rhs.dim.1 {
                result[j][i] = (0..self.dim.1).map(|k| self[j][k] * rhs[k][i]).sum()
            }
        }
        result
    }
}


// unnary operation

macro_rules! impl_unnary_ops_for_dense2d {
    ($op:ident, $func:ident) => (
        impl<T: ops::$op + Copy> ops::$op for Matrix<T> {
            type Output = Matrix<T::Output>;

            fn $func(self) -> Matrix<T::Output> {
                (&self).$func()
            }
        }

        impl<'a, T: ops::$op + Copy> ops::$op for &'a Matrix<T> {
            type Output = Matrix<T::Output>;

            fn $func(self) -> Matrix<T::Output> {
                let mut result = Matrix::new(self.dim.0, self.dim.1);
                for (j, i) in self.iter_indices() {
                    result[j][i] = self[j][i].$func()
                }
                result
            }
        }
    )
}

impl_unnary_ops_for_dense2d!(Neg, neg);
impl_unnary_ops_for_dense2d!(Not, not);


impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Matrix {
            data: self.data.clone(),
            dim: self.dim,
        }
    }
}

impl<T: PartialEq> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        assert_eq!(self.dim, other.dim);
        for j in 0 .. self.dim.0 {
            for i in 0 .. self.dim.1 {
                if self[j][i] != other[j][i] {
                    return false;
                }
            }
        }
        true
    }
}

// debug show
impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "<Matrix dim={:?}, {:?}>", self.dim, self.data));
        Ok(())
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, ""));
        for j in 0 .. self.dim.0 {
            if j == 0 {
                try!(write!(f, "[["));
            } else {
                try!(write!(f, " ["));
            }
            for i in 0 .. self.dim.1 {
                try!(write!(f, "{:-4.}", self[j][i]));
                if i == self.dim.1 - 1 {
                    try!(write!(f, "]"));
                } else {
                    try!(write!(f, ", "));
                }
            }
            if j == self.dim.0 - 1 {
                try!(write!(f, "]"));
            } else {
                try!(writeln!(f, ""));
            }
        }
        Ok(())
    }
}


#[test]
fn test_index_by_row() {
    let mut m1 = Matrix::from_vec_and_dim(vec![1i32, 2, 3, 4], (2, 2));;
    assert_eq!(&m1[0], &[1, 2]);
    assert_eq!(&m1[1], &[3, 4]);

    m1[1][1] = 233;
    assert_eq!(m1[1][1], 233);
}

#[test]
fn test_equal_from_str() {
    let m1 = Matrix::from_vec_and_dim(vec![1i32, 2, 3, 4], (2, 2));
    let m2 = Matrix::from_vec_and_dim(vec![1i32, 2, 3, 4], (2, 2));
    assert_eq!(m1, m2);
    assert_eq!(m1, Matrix::from_str("1 2; 3 4").unwrap());
}


#[test]
fn test_from_vec_and_dim(){
    let m1 = Matrix::from_vec_and_dim(vec![1f32, 2f32, 3f32, 4f32], (2, 2));
    let m2 = Matrix::from_vec_and_dim(vec![5f32, 6., 7., 8.], (2, 2));
    println!("m1 * m2 = {}", m1.x(&m2));

}

#[test]
fn test_add_sub() {
    let m1: Matrix<i32> = FromStr::from_str("1 2; 3 4").unwrap();
    let m2: Matrix<i32> = FromStr::from_str("5 6; 7 8").unwrap();

    let m3: Matrix<i32> = FromStr::from_str("6 8; 10 12").unwrap();
    let m4: Matrix<i32> = FromStr::from_str("4 4; 4 4").unwrap();

    assert_eq!(m1.clone() + m2.clone(), m3);
    assert_eq!(- (m1.clone() - m2.clone()), m4);
}

#[test]
fn test_matrix_multiply() {
    let m0: Matrix<i32> = FromStr::from_str("1 0; 0 1").unwrap();
    let m1: Matrix<i32> = FromStr::from_str("1 2; 3 4").unwrap();
    let m2: Matrix<i32> = FromStr::from_str("5 6; 7 8").unwrap();

    let res: Matrix<i32> = FromStr::from_str("19 22; 43 50").unwrap();
    assert_eq!(res, (&m1 * &m2));
    assert_eq!(m1, (&m1 * &m0));
}


#[test]
fn test_add() {
    use rand::{thread_rng, Rng};

    let mut m1 = Matrix::<i16>::new(4, 4);
    let m2 = Matrix::<i16>::eye(4);

    let mut rng = thread_rng();

    for _ in 0 .. 10 {
        m1[rng.gen_range(0, 4)][rng.gen_range(0, 4)] = rng.gen();
    }

    println!("m1 => {}", m1);
    println!("m2 => {}", m2);
    let m3 = m1 + m2;
    println!("plus => {}", m3);
    println!("sub => {}", m3 - Matrix::eye(4) - Matrix::eye(4) - Matrix::eye(4) - Matrix::eye(4));
}


#[test]
fn it_works() {
    let mut m = Matrix::<f64>::new(5, 5);

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

    m[(1,2)] = 100.0;
    m[(0,1)] = 10.0;
    println!("Matrix => {}", m);
    println!("Matrix => {:?}", m);


    let m = Matrix::<f32>::eye(4);
    println!("Matrix => {}", &m);
    println!("Matrix => {}", - &m);
}
