use std::iter;
use std::fmt;

use super::Array;
use super::ArrayIx;

// RefArray
pub struct RefArray<'a, T: 'a> {
    arr: &'a Array<T>,
    // simulation of np.ix_ function?
    open_mesh: Vec<Box<ArrayIx + 'a>>
}

impl<'a, T: Copy> RefArray<'a, T> {
    pub fn new<'b>(arr: &'b Array<T>, open_mesh: Vec<Box<ArrayIx + 'b>>) -> RefArray<'b, T> {
        RefArray {
            arr: arr,
            open_mesh: open_mesh
        }
    }

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
