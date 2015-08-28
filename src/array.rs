use std::fmt;
use std::ops;
use std::iter;
use std::iter::FromIterator;


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


impl<T: Copy> Array<T> {
    pub fn new(v: Vec<T>) -> Array<T> {
        let nelem = v.len();
        Array {
            data: v,
            shape: vec![nelem]
        }
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
}

impl<A: Copy> FromIterator<A> for Array<A> {
    fn from_iter<T: IntoIterator<Item=A>>(iterator: T) -> Self {
        Array::new(iterator.into_iter().collect())
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
        let mut ndim = self.shape.len();
        let mut dims = self.shape.clone();

        let ret = dump_data(self, &dims, ndim, &vec![]);
        // FIXME: adding line break's buggy format
        write!(f, "{}", ret.replace("],", "],\n"))
    }
}


#[test]
fn test_array() {
    let mut v = Array::new(vec![ 0,  1,  2,  3,  4,  5,  6,  7,
                             8,  9, 10, 11, 12, 13, 14, 15,
                             16, 17, 18, 19, 20, 21, 22, 23]);

    v.reshape([2, 3, 4]);
    assert_eq!(v.get([1, 1, 2]), 18);

    v[[1,2,3]] = 100;
    println!("fuck {:?}", v[[1,1,2]]);

    let shape = vec![2, 3, 4];
    let mut offs = 0;
    let s = dump_data(&v, &shape, 3, &vec![]);

    println!("DEBUG print => \n{}", v);
    println!("DEBUG print => \n{:?}", s);

    v.reshape([6, 4]);
    println!("DEBUG print => \n{}", v);
}
