use std::iter;
use std::fmt;
use std::ops;

use super::{Array, ArrayIx, ToArray, ArrayIndexIter};
use super::super::traits::{ArrayType, ArrayShape};


// RefArray
pub struct RefArray<'a, T: 'a> {
    arr: &'a Array<T>,
    // simulation of np.ix_ function?
    open_mesh: Vec<Box<ArrayIx + 'a>>
}

impl<'a, T: Copy> RefArray<'a, T> {
    fn new<'b>(arr: &'b Array<T>, open_mesh: Vec<Box<ArrayIx + 'b>>) -> RefArray<'b, T> {
        RefArray {
            arr: arr,
            open_mesh: open_mesh
        }
    }

    fn offset_translate(&self, index: &[usize]) -> usize {
        let mut ix = Vec::<usize>::new();
        for (i, &k) in index.iter().enumerate() {
            ix.push(self.open_mesh[i].to_idx_vec(i, &self.arr.shape())[k]);
        }
        self.arr.offset_of(&ix)
    }

    pub fn iter_indices(&self) -> ArrayIndexIter {
        ArrayIndexIter {
            current: iter::repeat(0).take(self.ndim()).collect(),
            shape: self.shape()
        }
    }
}

impl<'a, T: Copy> ToArray<T> for RefArray<'a, T> {
    fn to_array(&self) -> Array<T> {
        let mut ret = Array::new(self.shape());
        for ref idx in self.iter_indices() {
            ret[idx] = self[idx];
        }
        ret
    }
}

impl<'a, T: Copy> ArrayType<T> for RefArray<'a, T> {
    fn shape(&self) -> Vec<usize> {
        self.open_mesh.iter().enumerate().map(|(i,v)| (*v).size(i, &self.arr.shape())).collect()
    }

    fn get<D: AsRef<[usize]>>(&self, index: D) -> Option<&T> {
        Some(&self.arr.data[self.offset_translate(index.as_ref())])
    }

    unsafe fn get_unchecked<D: AsRef<[usize]>>(&self, index: D) -> &T {
        &self.arr.data[self.offset_translate(index.as_ref())]
    }

    fn get_mut<D: AsRef<[usize]>>(&mut self, _index: D) -> Option<&mut T> {
        None
    }

    unsafe fn get_unchecked_mut<D: AsRef<[usize]>>(&mut self, _index: D) -> &mut T {
        panic!("can't get mut ref of a Unmutable RefArray")
    }
}

// arr[idx]
impl<'a, T: Copy, D: AsRef<[usize]>> ops::Index<D> for RefArray<'a, T> {
    type Output = T;

    #[inline]
    fn index<'b>(&'b self, index: D) -> &'b T {
        self.get(index).unwrap()
    }
}



impl<'a, T: Copy + fmt::Display> fmt::Display for RefArray<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn dump_ref_array_data<'b, T: Copy + fmt::Display>(a: &'b RefArray<T>, dims: &[usize], nd: usize, index: &[usize]) -> String {
            let mut ret = String::new();
            if nd == 0 {
                ret.push_str(&format!("{:4}", a[index]));
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


// RefMutArray
pub struct RefMutArray<'a, T: 'a> {
    arr: &'a mut Array<T>,
    // simulation of np.ix_ function?
    open_mesh: Vec<Box<ArrayIx + 'a>>
}

impl<'a, T: Copy> RefMutArray<'a, T> {
    fn new<'b>(arr: &'b mut Array<T>, open_mesh: Vec<Box<ArrayIx + 'b>>) -> RefMutArray<'b, T> {
        RefMutArray {
            arr: arr,
            open_mesh: open_mesh
        }
    }

    fn offset_translate(&self, index: &[usize]) -> usize {
        let mut ix = Vec::<usize>::new();
        for (i, &k) in index.iter().enumerate() {
            ix.push(self.open_mesh[i].to_idx_vec(i, &self.arr.shape())[k]);
        }
        self.arr.offset_of(&ix)
    }

    fn iter_indices(&self) -> ArrayIndexIter {
        ArrayIndexIter {
            current: iter::repeat(0).take(self.ndim()).collect(),
            shape: self.shape()
        }
    }
    pub fn move_from(&mut self, src: Array<T>) {
        assert!(self.shape() == src.shape(), "move_from() must be called among arrays of same shape");
        for ref idx in self.iter_indices() {
            let _ = self.get_mut(idx).map(|v| *v = src[idx]).expect("assignment");
        }
    }
}

impl<'a, T: Copy> ArrayType<T> for RefMutArray<'a, T> {
    fn shape(&self) -> Vec<usize> {
        self.open_mesh.iter().enumerate().map(|(i,v)| (*v).size(i, &self.arr.shape())).collect()
    }

    fn get<D: AsRef<[usize]>>(&self, index: D) -> Option<&T> {
        Some(&self.arr.data[self.offset_translate(index.as_ref())])
    }

    fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> Option<&mut T> {
        let offset = self.offset_translate(index.as_ref());
        Some(&mut self.arr.data[offset])
    }

    unsafe fn get_unchecked<D: AsRef<[usize]>>(&self, index: D) -> &T {
        &self.arr.data[self.offset_translate(index.as_ref())]
    }

    unsafe fn get_unchecked_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T {
        let offset = self.offset_translate(index.as_ref());
        &mut self.arr.data[offset]
    }
}

impl<'a, T: Copy, D: AsRef<[usize]>> ops::Index<D> for RefMutArray<'a, T> {
    type Output = T;

    #[inline]
    fn index<'b>(&'b self, index: D) -> &'b T {
        self.get(index).unwrap()
    }
}

// arr[idx] = val
impl<'a, T: Copy, D: AsRef<[usize]>> ops::IndexMut<D> for RefMutArray<'a, T> {
    #[inline]
    fn index_mut<'b>(&'b mut self, index: D) -> &'b mut T {
        self.get_mut(index).unwrap()
    }
}

impl<T: Copy> Array<T> {
    pub fn slice<'a>(&'a self, ix: Vec<Box<ArrayIx>>) -> RefArray<'a, T> {
        RefArray::new(self, ix)
    }

    pub fn slice_mut<'a>(&'a mut self, ix: Vec<Box<ArrayIx>>) -> RefMutArray<'a, T> {
        RefMutArray::new(self, ix)
    }
}
