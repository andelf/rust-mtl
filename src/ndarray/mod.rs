use std::fmt;
use std::ops;
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::iter;
use std::iter::FromIterator;

use num::traits::{One, Zero};
use rand::{thread_rng, Rng};

mod subref;

pub use super::traits::{ArrayType, ArrayShape};
pub use self::subref::RefArray;
pub use self::subref::RefMutArray;

#[derive(Clone, Debug)]
pub struct ArrayIndexIter {
    current: Vec<usize>,
    shape: Vec<usize>
}

impl Iterator for ArrayIndexIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let ndim = self.shape.len();
        if self.current[0] >= self.shape[0] {
            return None;
        }
        let old_current = self.current.clone();
        let _ = self.current.last_mut().map(|k| *k += 1);
        for i in (0 .. ndim).rev() {
            if self.current[i] == self.shape[i] && i > 0 {
                self.current[i] = 0;
                self.current[i-1] += 1;
            }
        }
        Some(old_current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut nth = 0;
        for (i, &offs) in self.current.iter().enumerate().rev() {
            if i + 1 < self.shape.len() {
                nth += self.shape[i+1 .. ].iter().product::<usize>() * offs;
            } else {
                nth += offs
            }
        }
        let sz = self.shape.nelem() - nth;
        (sz, Some(sz))
    }
}

impl ExactSizeIterator for ArrayIndexIter {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}


// Array Index
pub trait ArrayIx {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize>;
    fn size(&self, ax: usize, dims: &[usize]) -> usize {
        self.to_idx_vec(ax, dims).len()
    }
}

impl ArrayIx for RangeFull {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
        assert!(dims.len() > ax, "ax must be in range");
        (0 .. dims[ax]).collect()
    }
}

macro_rules! impl_array_ix_for_range {
    ($typ:ty) => (
        impl ArrayIx for Range<$typ> {
            fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
                assert!(dims.len() > ax, "ax must be in range");
                assert!((self.start as usize) < dims[ax], "range start overflow");
                assert!((self.end as usize) < dims[ax], "range start overflow");
                (self.start as usize .. self.end as usize).map(|i| i as usize).collect()
            }
        }

        impl ArrayIx for RangeFrom<$typ> {
            fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
                assert!(dims.len() > ax, "ax must be in range");
                assert!((self.start as usize) < dims[ax], "range start overflow");
                (self.start as usize .. dims[ax]).collect()
            }
        }

        impl ArrayIx for RangeTo<$typ> {
            fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
                assert!(dims.len() > ax, "ax must be in range");
                assert!((self.end as usize) < dims[ax], "range start overflow");
                (0 .. self.end as usize).collect()
            }
        }
    )
}

impl_array_ix_for_range!(i32);
impl_array_ix_for_range!(usize);

impl ArrayIx for [usize] {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
        assert!(self.iter().filter(|&&i| i >= dims[ax]).count() == 0, "ax must be in range");
        self.iter().map(|&i| i).collect()
    }
}

// i32 allows negative index
impl ArrayIx for [i32] {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
        self.iter().map(|&i| if i < 0 { dims[ax] - (-i) as usize } else { i as usize }).collect()
    }
}


impl ArrayIx for usize {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
        assert!(*self < dims[ax], "ax must be in range");
        vec![*self]
    }
}

impl ArrayIx for i32 {
    fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
        let idx = if *self < 0 {
            dims[ax] - (- *self) as usize
        } else {
            *self as usize
        };
        assert!(idx < dims[ax], "ax must be in range");
        vec![idx]
    }
}

macro_rules! impl_array_ix_for_fixed_size_array {
    ($typ:ty, $size:expr) => (
        impl ArrayIx for [$typ; $size] {
            fn to_idx_vec(&self, ax: usize, dims: &[usize]) -> Vec<usize> {
                self.as_ref().to_idx_vec(ax, dims)
            }
        }
    )
}

impl_array_ix_for_fixed_size_array!(i32, 1);
impl_array_ix_for_fixed_size_array!(i32, 2);
impl_array_ix_for_fixed_size_array!(i32, 3);
impl_array_ix_for_fixed_size_array!(i32, 4);

macro_rules! ix {
    ($($arg:expr),*) => (
        vec![$( Box::new($arg) as Box<ArrayIx> ),*]
    )
}

#[test]
fn test_index_ranges() {
    let dims = vec![4, 5, 6];

    assert_eq!((1..3).to_idx_vec(0, &dims), vec![1, 2]);
    assert_eq!((1..).to_idx_vec(0, &dims), vec![1, 2, 3]);
    assert_eq!((..2).to_idx_vec(0, &dims), vec![0, 1]);
    assert_eq!((..).to_idx_vec(0, &dims), vec![0, 1, 2, 3]);
}


/// n-d Array
#[derive(Clone, PartialEq, Debug)]
pub struct Array<T> {
    data: Vec<T>,
    shape: Vec<usize>
}

impl<T: Copy> ArrayType<T> for Array<T> {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    // FIXME: handle index overflow
    fn get<D: AsRef<[usize]>>(&self, index: D) -> Option<&T> {
        Some(&self.data[self.offset_of(index.as_ref())])
    }

    unsafe fn get_unchecked<D: AsRef<[usize]>>(&self, index: D) -> &T {
        &self.data[self.offset_of_unchecked(index.as_ref())]
    }

    fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> Option<&mut T> {
        let idx = self.offset_of(index.as_ref());
        Some(&mut self.data[idx])
    }

    unsafe fn get_unchecked_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T {
        let idx = self.offset_of_unchecked(index.as_ref());
        &mut self.data[idx]
    }
}

impl<T: Copy> Array<T> {
    pub fn new<S: ArrayShape>(shape: S) -> Array<T> {
        let shape = shape.to_shape_vec();
        let nelem = shape.iter().product();
        let mut v = Vec::with_capacity(nelem);
        unsafe { v.set_len(nelem) };
        Array {
            data: v,
            shape: shape
        }
    }

    pub fn iter_indices(&self) -> ArrayIndexIter {
        let shape = self.shape();
        let start_idx = iter::repeat(0).take(shape.len()).collect();
        ArrayIndexIter {
            current: start_idx,
            shape: shape
        }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn from_vec(v: Vec<T>) -> Array<T> {
        let nelem = v.len();
        Array {
            data: v,
            shape: vec![nelem]
        }
    }

    pub fn new_with_shape(v: Vec<T>, shape: Vec<usize>) -> Array<T> {
        assert!(shape.clone().iter().product::<usize>() == v.len());
        Array {
            data: v,
            shape: shape
        }
    }

    fn offset_of_unchecked(&self, index: &[usize]) -> usize {
        index.iter().enumerate()
            .map(|(i, &ax)| {
                self.shape.iter().skip(i+1).product::<usize>() * ax
            })
            .sum()
    }

    fn offset_of(&self, index: &[usize]) -> usize {
        assert!(index.len() == self.ndim());
        index.iter().enumerate()
            .map(|(i, &ax)| {
                assert!(ax < self.shape[i], "ax {} overflow, should be lower than {}", ax, self.shape[i]);
                self.shape.iter().skip(i+1).product::<usize>() * ax
            })
            .sum()
    }

    pub fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T {
        let offset = self.offset_of(index.as_ref());
        &mut self.data[offset]
    }

    pub fn reshape<S: ArrayShape>(mut self, shape: S) -> Array<T> {
        assert_eq!(self.data.len(), shape.nelem());
        self.shape = shape.to_shape_vec();
        self
    }

    pub fn flatten(&mut self) {
        self.shape = vec![self.shape.nelem()];
    }

    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        rng.shuffle(&mut self.data);
    }

    pub fn all<F>(&mut self, mut f: F) -> bool where Self: Sized, F: FnMut(T) -> bool {
        for &x in self.data.iter() {
            if !f(x) {
                return false;
            }
        }
        true
    }

    pub fn any<F>(&mut self, mut f: F) -> bool where Self: Sized, F: FnMut(T) -> bool {
        for &x in self.data.iter() {
            if f(x) {
                return true;
            }
        }
        false
    }

    pub fn enumerate<'t>(&'t self) -> Enumerate<'t, T> {
        Enumerate {
            iter: self.iter_indices(),
            arr: self,
            count: self.size()
        }
    }
}


pub struct Enumerate<'t, T:'t> {
    iter: ArrayIndexIter,
    arr: &'t Array<T>,
    count: usize
}

impl<'t, T: Copy> Iterator for Enumerate<'t, T> {
    type Item = (Vec<usize>, &'t T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.iter.next() {
            let val = &self.arr[&idx];
            self.count -= 1;
            Some((idx, val))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }
}

impl<'t, T: Copy> ExactSizeIterator for Enumerate<'t, T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}



impl<A: Copy> FromIterator<A> for Array<A> {
    fn from_iter<T: IntoIterator<Item=A>>(iterator: T) -> Self {
        Array::from_vec(iterator.into_iter().collect())
    }
}

// arr[idx]
impl<T: Copy, D: AsRef<[usize]>> ops::Index<D> for Array<T> {
    type Output = T;

    #[inline]
    fn index<'a>(&'a self, index: D) -> &'a T {
        self.get(index).unwrap()
    }
}

// arr[idx] = val
impl<T: Copy, D: AsRef<[usize]>> ops::IndexMut<D> for Array<T> {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: D) -> &'a mut T {
        self.get_mut(index)
    }
}

impl<T: Copy + Zero> Array<T> {
    pub fn zeros<S: ArrayShape>(s: S) -> Array<T> {
        let shape = s.to_shape_vec();
        let v = iter::repeat(Zero::zero()).take(shape.nelem()).collect::<Vec<T>>();
        Array { data: v, shape: shape }
    }
}

impl<T: Copy + One> Array<T> {
    pub fn ones<S: ArrayShape>(s: S) -> Array<T> {
        let shape = s.to_shape_vec();
        let v = iter::repeat(One::one()).take(shape.nelem()).collect::<Vec<T>>();
        Array { data: v, shape: shape }
    }
}

impl<T: Copy + One + Zero> Array<T> {
    pub fn eye(n: usize) -> Array<T> {
        let mut arr = Array::zeros([n,n]);
        for i in 0 .. n {
            arr[[i,i]] = One::one();
        }
        arr
    }
}


// use PartialOrd, this returen unravel_index
impl<T: Copy + PartialOrd> Array<T> {
    pub fn argmax(&self) -> usize {
        let mut maxi = 0;
        for i in 0 .. self.shape().nelem() {
            if self.data[maxi] < self.data[i] {
                maxi = i;
            }
        }
        maxi
    }

    pub fn argmin(&self) -> usize {
        let mut mini = 0;
        for i in 0 .. self.shape().nelem() {
            if self.data[mini] > self.data[i] {
                mini = i;
            }
        }
        mini
    }
}

macro_rules! impl_binary_ops_for_array {
    ($op:ident, $func:ident) => (
        // M op M
        impl<T: ops::$op<T, Output=T> + Copy> ops::$op for Array<T> {
            type Output = Array<T>;

            fn $func(self, rhs: Array<T>) -> Array<T> {
                assert!(self.shape() == rhs.shape());
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }

        // &M op M
        impl<'a, T: ops::$op<T, Output=T> + Copy> ops::$op<Array<T>> for &'a Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: Array<T>) -> Array<T::Output> {
                self.$func(&rhs)
            }
        }

        // M op &M
        impl<'a, T: ops::$op<T, Output=T> + Copy> ops::$op<&'a Array<T>> for Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: &'a Array<T>) -> Array<T::Output> {
                (&self).$func(rhs)
            }
        }

        // &M op &M
        impl<'a, 'b, T: ops::$op<T, Output=T> + Copy> ops::$op<&'a Array<T>> for &'b Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: &'a Array<T>) -> Array<T::Output> {
                assert!(self.shape() == rhs.shape());
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }

        // M + s
        impl<T: ops::$op<T, Output=T> + Copy> ops::$op<T> for Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: T) -> Array<T::Output> {
                (&self).$func(rhs)
            }
        }

        // &M + s
        impl<'a, T: ops::$op<T, Output=T> + Copy> ops::$op<T> for &'a Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: T) -> Array<T::Output> {
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs);
                }
                result
            }
        }

        // &A op slice
        impl<'a, 'b, T: ops::$op<T, Output=T> + Copy> ops::$op<RefArray<'b, T>> for &'a Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: RefArray<'b, T>) -> Array<T::Output> {
                assert!(self.shape() == rhs.shape());
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }

        // A op slice
        impl<'b, T: ops::$op<T, Output=T> + Copy> ops::$op<RefArray<'b, T>> for Array<T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: RefArray<'b, T>) -> Array<T::Output> {
                (&self).$func(rhs)
            }
        }

        // slice + s
        impl<'a, T: ops::$op<T, Output=T> + Copy> ops::$op<T> for RefArray<'a, T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: T) -> Array<T::Output> {
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs);
                }
                result
            }
        }

        // slice + A
        impl<'a, T: ops::$op<T, Output=T> + Copy> ops::$op<Array<T>> for RefArray<'a, T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: Array<T>) -> Array<T::Output> {
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }

        // slice + A
        impl<'a, 'b, T: ops::$op<T, Output=T> + Copy> ops::$op<&'b Array<T>> for RefArray<'a, T> {
            type Output = Array<T::Output>;

            fn $func(self, rhs: &'b Array<T>) -> Array<T::Output> {
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }

        // slice op slice
        impl<'a, 'b, T: ops::$op<T, Output=T> + Copy> ops::$op<RefArray<'b, T>> for RefArray<'a, T> {
            type Output = Array<T>;

            fn $func(self, rhs: RefArray<'b, T>) -> Array<T> {
                assert!(self.shape() == rhs.shape());
                let mut result = Array::new(self.shape());
                for ref idx in self.iter_indices() {
                    result[idx] = self[idx].$func(rhs[idx]);
                }
                result
            }
        }
    )
}

impl_binary_ops_for_array!(Add, add);
impl_binary_ops_for_array!(Div, div);
impl_binary_ops_for_array!(Rem, rem);
impl_binary_ops_for_array!(Sub, sub);
impl_binary_ops_for_array!(Mul, mul);



pub fn concatenate<T: Copy, A: AsRef<[Array<T>]>>(arrs: A, axis: usize) -> Array<T> {
    let arrs = arrs.as_ref();
    assert!(arrs.len() > 1, "concatenation at least 2 array");
    assert!(arrs.iter().map(Array::shape)
            .zip(arrs.iter().skip(1).map(Array::shape))
            .all(|(as1, as2)| {
                as1[.. axis] == as2[.. axis] && as1[axis+1 ..] == as2[axis+1 ..]
            }),
            "all the input array dimensions except for the concatenation axis must match exactly"
            );
    let mut shape = arrs[0].shape();
    shape[axis] = arrs.iter().map(|a| a.shape()[axis]).sum();

    let mut ret: Array<T> = Array::new(&shape);

    let mut ix_offset = 0;
    for arr in arrs.iter() {
        for idx in arr.iter_indices() {
            let mut new_idx = idx.clone();
            new_idx[axis] += ix_offset;

            *ret.get_mut(new_idx) = arr[idx];
        }
        ix_offset += arr.shape()[axis];
    }

    ret
}

// impl<T: Copy> ops::Index<(Range<usize>,)> for Array<T> {
//     type Output = T;

//     #[inline]
//     fn index<'a>(&'a self, index: (Range<usize,)) -> &'a T {

//         println!("start => {}", index.0.start());
//         println!("end => {}", index.0.start());
//         &self.data[0]
//     }
// }




// helper function for print an array
fn dump_data<T: Copy + fmt::Display>(a: &Array<T>, dims: &[usize], nd: usize, index: &[usize]) -> String {
    let mut ret = String::new();
    if nd == 0 {
        ret.push_str(&format!("{:4}", a[index]));
    } else {
        ret.push('[');
        for i in 0 .. dims[0] {
            let index = index.iter().map(|&i| i).chain(iter::once(i)).collect::<Vec<usize>>();
            ret.push_str(&dump_data(a, &dims[1..], nd-1, &index));
            if i < dims[0] - 1 {
                ret.push_str(", ");
            }
        }
        ret.push(']');
    }
    ret
}

impl<T: Copy + fmt::Display> fmt::Display for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ndim = self.shape.len();
        let dims = self.shape.clone();

        let ret = dump_data(self, &dims, ndim, &vec![]);
        // FIXME: adding line break's buggy format
        write!(f, "{}", ret.replace("],", "],\n"))
    }
}

pub trait ToArray<T> {
    fn to_array(&self) -> Array<T>;
}

impl<T: Copy> ToArray<T> for Vec<T> {
    fn to_array(&self) -> Array<T> {
        Array::from_vec(self.clone()).reshape([self.len()])
    }
}

impl<T: Copy> ToArray<T> for [T] {
    fn to_array(&self) -> Array<T> {
        Array::from_vec(self.to_vec()).reshape([self.len()])
    }
}

#[test]
fn test_array() {
    let mut v = Array::from_vec(vec![ 0,  1,  2,  3,  4,  5,  6,  7,
                                      8,  9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23]).reshape([2, 3, 4]);

    assert_eq!(v[[1, 1, 2]], 18);

    v[[1,2,3]] = 100;
    println!("fuck {:?}", v[[1,1,2]]);

    let shape = vec![2, 3, 4];
    let s = dump_data(&v, &shape, 3, &vec![]);

    println!("DEBUG print => \n{}", v);
    println!("DEBUG print => \n{:?}", s);

    let v = v.reshape([6, 4]);
    println!("DEBUG print => \n{}", v);

    let v2 = ix!([1, 2, 3], [2, 0]);
    let v3 = v.slice(v2);
    println!("debug => {:?}", v3[[0,0]]);

    println!("SUB[1,2,3; 2,0] => \n{}", v3);

    println!("SUB => \n{}", v.slice(ix!(3.., [2,0])));
}

#[test]
fn test_array_eye_zeros_ones() {
    let arr = Array::<f64>::eye(3);
    for i in 0 .. 3 {
        for j in 0 .. 3 {
            if i == j {
                assert_eq!(arr[[i,j]], 1.);
            } else {
                assert_eq!(arr[[i,j]], 0.);
            }
        }
    }

    let arr = Array::<f32>::zeros([3,4]);
    for ref idx in arr.iter_indices() {
        assert_eq!(arr[idx], 0.);
    }

    let arr = Array::<f32>::ones([3,4]);
    for ref idx in arr.iter_indices() {
        assert_eq!(arr[idx], 1.);
    }
}


#[test]
fn test_array_concat() {
    let v1 = Array::new_with_shape(vec![0, 1,
                                        2, 3], vec![2, 2]);

    assert_eq!(v1.iter_indices().len(), 4);
    let v2 = Array::new_with_shape(vec![8, 9, 10,
                                        11, 9, 20], vec![2, 3]);

    assert_eq!(v2.iter_indices().len(), 6);
    let mut it = v2.iter_indices();
    assert!(it.next().is_some());
    assert_eq!(it.len(), 5);

    let res = concatenate([v1, v2], 1);
    assert_eq!(res.shape(), vec![2, 5]);
    assert_eq!(res[[1, 1]], 3);
    assert_eq!(res[[0, 0]], 0);
    assert_eq!(res[[1, 2]], 11);
    assert_eq!(res[[0, 4]], 10);
}


#[test]
fn test_array_binary_op() {
    let v1 = Array::<f64>::eye(4);
    let v2 = Array::<f64>::ones(4);

    println!("plus => \n{}", &v1 + &v2);
    println!("mul => \n{}", (&v1 - 2.) * (&v2 + 3.));

}

#[test]
fn test_array_shuffle() {
    let mut arr = (0..12).collect::<Array<usize>>().reshape([3,4]);
    let arr2 = arr.clone();
    arr.shuffle();
    assert!(arr != arr2, "shuffed array");
}



#[test]
fn test_test_any_all() {
    let mut arr = (0..12).collect::<Array<usize>>().reshape([3,4]);
    assert!(arr.any(|i| i >= 10));
    assert!(arr.any(|i| i <= 13));
}

#[test]
fn test_flatten() {
    let mut arr = (0..12).collect::<Array<i32>>().reshape([3,4]);
    arr.flatten();
    assert!(arr.shape() == vec![12]);
}

#[test]
fn test_to_array_trait() {
    let v = vec![2, 3, 4, 5];
    let arr: Array<i32> = v.to_array();
    assert!(arr.shape() == vec![4], "ToArray should perserve shape");
}

#[test]
fn test_enumerate() {
    let v = vec![2, 3, 4, 5];
    let arr: Array<i32> = v.to_array().reshape([2,2]);
    let mut it = arr.enumerate();
    assert_eq!(it.next(), Some((vec![0,0], &2)));
    for _ in 0..2 {
        let _  = it.next().unwrap();
    }
    assert_eq!(it.next(), Some((vec![1,1], &5)));
    assert_eq!(it.next(), None);
}
