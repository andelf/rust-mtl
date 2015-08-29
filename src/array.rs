use std::fmt;
use std::ops;
use std::ops::{Range, RangeFrom, RangeTo, RangeFull};
use std::iter;
use std::iter::FromIterator;

use num::traits::{One, Zero};

//
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

impl ArrayIx for [i32] {
    fn to_idx_vec(&self, _ax: usize, _dims: &[usize]) -> Vec<usize> {
        self.iter().map(|&i| i as usize).collect()
    }
}


macro_rules! impl_array_ix_for_fixed_size_array {
    ($typ:ty, $size:expr) => (
        impl ArrayIx for [$typ; $size] {
            fn to_idx_vec(&self, _ax: usize, _dims: &[usize]) -> Vec<usize> {
                self.iter().map(|&i| i as usize).collect()
            }
        }
    )
}

impl_array_ix_for_fixed_size_array!(i32, 1);
impl_array_ix_for_fixed_size_array!(i32, 2);
impl_array_ix_for_fixed_size_array!(i32, 3);
impl_array_ix_for_fixed_size_array!(i32, 4);
impl_array_ix_for_fixed_size_array!(i32, 5);
impl_array_ix_for_fixed_size_array!(i32, 6);
impl_array_ix_for_fixed_size_array!(i32, 7);
impl_array_ix_for_fixed_size_array!(i32, 8);

#[test]
fn test_index_ranges() {
    let dims = vec![4, 5, 6];

    assert_eq!((1..3).to_idx_vec(0, &dims), vec![1, 2]);
    assert_eq!((1..).to_idx_vec(0, &dims), vec![1, 2, 3]);
    assert_eq!((..2).to_idx_vec(0, &dims), vec![0, 1]);
    assert_eq!((..).to_idx_vec(0, &dims), vec![0, 1, 2, 3]);
}




// nd Array
#[derive(Clone, PartialEq, Debug)]
pub struct Array<T> {
    data: Vec<T>,
    shape: Vec<usize>
}

macro_rules! simple_impl_for_array {
    ($op:ty, $func:ident) => (
        impl
    )
}

pub struct ArrayIndexIter<'a, T: 'a> {
    arr: &'a Array<T>,
    current: Vec<usize>,
    shape: Vec<usize>
}

impl<'a, T> Iterator for ArrayIndexIter<'a, T> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let ndim = self.shape.len();
        if self.current[0] >= self.shape[0] {
            return None;
        }
        let old_current = self.current.clone();
        self.current.last_mut().map(|k| *k += 1);
        for i in (0 .. ndim).rev() {
            if self.current[i] == self.shape[i] && i > 0 {
                self.current[i] = 0;
                self.current[i-1] += 1;
            }
        }
        Some(old_current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let sz = self.shape.iter()
            .zip(self.current.iter())
            .map(|(&sz, &i)| (sz - i -1) * sz).product();
        (sz, Some(sz))
    }
}


impl<T: Copy> Array<T> {
    pub fn new<S: AsRef<[usize]>>(shape: S) -> Array<T> {
        let nelem = shape.as_ref().iter().product();
        let mut v = Vec::with_capacity(nelem);
        unsafe { v.set_len(nelem) };
        Array {
            data: v,
            shape: shape.as_ref().to_vec()
        }
    }

    pub fn iter_indices<'a>(&'a self) -> ArrayIndexIter<'a, T> {
        let shape = self.shape();
        let zeros = iter::repeat(0).take(shape.len()).collect();
        ArrayIndexIter {
            arr: self,
            current: zeros,
            shape: shape
        }
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

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn offset_of(&self, index: &[usize]) -> usize {
        index.iter().enumerate()
            .map(|(i, &ax)| {
                assert!(ax < self.shape[i]);
                self.shape.iter().skip(i+1).product::<usize>() * ax
            })
            .sum()
    }

    pub fn get<D: AsRef<[usize]>>(&self, index: D) -> T {
        self.data[self.offset_of(index.as_ref())]
    }

    pub fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T {
        let offset = self.offset_of(index.as_ref());
        &mut self.data[offset]
    }

    pub fn reshape<D: AsRef<[usize]>>(&mut self, shape: D) {
        assert_eq!(self.data.len(), shape.as_ref().iter().product());
        self.shape = shape.as_ref().into()
    }

    pub fn slice<'a>(&'a self, ix: Vec<Box<ArrayIx>>) -> RefArray<'a, T> {
        RefArray {
            arr: self,
            open_mesh: ix
        }
    }
}

impl<A: Copy> FromIterator<A> for Array<A> {
    fn from_iter<T: IntoIterator<Item=A>>(iterator: T) -> Self {
        Array::from_vec(iterator.into_iter().collect())
    }
}

impl<T: Copy, D: AsRef<[usize]>> ops::Index<D> for Array<T> {
    type Output = T;

    #[inline]
    fn index<'a>(&'a self, index: D) -> &'a T {
        &self.data[self.offset_of(index.as_ref())]
    }
}

impl<T: Copy, D: AsRef<[usize]>> ops::IndexMut<D> for Array<T> {
    #[inline]
    fn index_mut<'a>(&'a mut self, index: D) -> &'a mut T {
        let offset = self.offset_of(index.as_ref());
        &mut self.data[offset]
    }
}

impl<T: Copy + Zero> Array<T> {
    pub fn zeros(n: usize) -> Array<T> {
        let v = iter::repeat(Zero::zero()).take(n*n).collect::<Vec<T>>();
        Array::new_with_shape(v, vec![n, n])
    }
}


impl<T: Copy + One + Zero> Array<T> {
    pub fn eye(n: usize) -> Array<T> {
        let mut arr = Array::zeros(n);
        for i in 0 .. n {
            arr[[i,i]] = One::one();
        }
        arr
    }
}


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
    let dims = shape.len();

    shape[axis] = arrs.iter().map(|a| a.shape()[axis]).sum();

    let mut ret: Array<T> = Array::new(&shape);

    let mut ix_offset = 0;
    for arr in arrs.iter() {
        let mut idx: Vec<usize> = iter::repeat(0).take(dims).collect();

        for idx in arr.iter_indices() {
            let mut new_idx = idx.clone();
            new_idx[axis] += ix_offset;

            *ret.get_mut(new_idx) = arr.get(idx);
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

// RefArray
pub struct RefArray<'a, T: 'a> {
    arr: &'a Array<T>,
    // simulation of np.ix_ function?
    open_mesh: Vec<Box<ArrayIx + 'a>>
}

impl<'a, T: Copy> RefArray<'a, T> {
    pub fn shape(&self) -> Vec<usize> {
        self.open_mesh.iter().enumerate().map(|(i,v)| (*v).size(i, &self.arr.shape())).collect()
    }

    fn offset_translate(&self, index: &[usize]) -> usize {
        let mut ix = Vec::<usize>::new();
        for (i, &k) in index.iter().enumerate() {
            ix.push(self.open_mesh[i].to_idx_vec(i, &self.arr.shape())[k]);
        }
        self.arr.offset_of(&ix)
    }

    pub fn get<D: AsRef<[usize]>>(&self, index: D) -> T {
        self.arr.data[self.offset_translate(index.as_ref())]
    }

    // pub fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T {
    //     let offset = self.offset_of(index.as_ref());
    //     &mut self.data[offset]
    // }
}

macro_rules! ix {
    ($($arg:expr),*) => (
        vec![$( Box::new($arg) as Box<ArrayIx> ),*]
    )
}

impl<'a, T: Copy + fmt::Display> fmt::Display for RefArray<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn dump_ref_array_data<'b, T: Copy + fmt::Display>(a: &'b RefArray<T>, dims: &[usize], nd: usize, index: &[usize]) -> String {
            let mut ret = String::new();
            if nd == 0 {
                ret.push_str(&format!("{:4}", a.get(index)));
            } else {
                ret.push('[');
                for i in 0 .. dims[0] {
                    let index = index.iter().map(|&i| i).chain(iter::once(i)).collect::<Vec<usize>>();
                    ret.push_str(&dump_ref_array_data(a, &dims[1..], nd-1, &index));
                    if i < dims[0] - 1 {
                        ret.push_str(", ");
                    }
                }
                ret.push(']');
            }
            ret
        }

        let ndim = self.shape().len();
        let dims = self.shape();

        let ret = dump_ref_array_data(self, &dims, ndim, &vec![]);
        // FIXME: adding line break's buggy format
        write!(f, "{}", ret.replace("],", "],\n"))
    }
}

#[test]
fn test_array() {
    let mut v = Array::from_vec(vec![ 0,  1,  2,  3,  4,  5,  6,  7,
                                      8,  9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23]);

    v.reshape([2, 3, 4]);
    assert_eq!(v.get([1, 1, 2]), 18);

    v[[1,2,3]] = 100;
    println!("fuck {:?}", v[[1,1,2]]);

    let shape = vec![2, 3, 4];
    let s = dump_data(&v, &shape, 3, &vec![]);

    println!("DEBUG print => \n{}", v);
    println!("DEBUG print => \n{:?}", s);

    v.reshape([6, 4]);
    println!("DEBUG print => \n{}", v);

    let v2 = ix!([1, 2, 3], [2, 0]);
    let v3 = v.slice(v2);
    println!("debug => {:?}", v3.get([0,0]));

    println!("SUB[1,2,3; 2,0] => \n{}", v3);

    println!("SUB => \n{}", v.slice(ix!(3.., [2,0])));
}

#[test]
fn test_array_eye() {
    println!("");
    let i = Array::<f64>::eye(5);
    println!("I => \n{}", i);
}


#[test]
fn test_array_concat() {
    let v1 = Array::new_with_shape(vec![0, 1,
                                        2, 3], vec![2, 2]);

    assert_eq!(v1.iter_indices().len(), 4);

    let v2 = Array::new_with_shape(vec![8, 9, 10,
                                        11, 9, 20], vec![2, 3]);

    let res = concatenate([v1, v2], 1);

    println!("Concat => \n{}", res);
}
