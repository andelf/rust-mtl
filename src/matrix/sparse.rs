//! sparse matrix

use std::iter;
use std::fmt;
use std::usize;
use std::ops;

use num::traits::Zero;


/// Sparse matrix, Yale repersentation
pub struct SparseYale<T> {
    dim: (usize, usize),
    vals: Vec<T>,
    col_starts: Vec<usize>,
    row_pos: Vec<usize>,
    zero: T,
}

impl<T: Copy + Zero> SparseYale<T> {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        SparseYale {
            dim: (nrow, ncol),
            vals: vec![],
            col_starts: iter::repeat(0).take(ncol).collect(),
            row_pos: vec![],
            zero: Zero::zero()
        }
    }

    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let nrow = vec.len();
        let ncol = vec[0].len();

        let mut mat = SparseYale::zeros(nrow, ncol);
        for (i, row) in vec.into_iter().enumerate() {
            for (j, item) in row.into_iter().enumerate() {
                mat.insert((i,j), item);
            }
        }
        mat
    }

    pub fn shape(&self) -> (usize, usize) {
        self.dim
    }

    #[inline]
    fn nrow(&self) -> usize {
        self.dim.0
    }

    #[inline]
    fn ncol(&self) -> usize {
        self.dim.1
    }

    /// Returns the element of the given index, or None if the index is out of bounds.
    pub fn get(&self, index: (usize, usize)) -> Option<&T> {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.vals.len()
        } else {
            self.col_starts[col+1]
        };

        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                return self.vals.get(i);
            }
        }
        Some(&self.zero)
    }

    /// Returns the element mutable ref of the given index, None if not exists
    pub fn get_mut(&mut self, index: (usize, usize)) -> Option<&mut T> {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.vals.len()
        } else {
            self.col_starts[col+1]
        };

        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                return self.vals.get_mut(i);
            }
        }
        None
    }

    pub fn insert(&mut self, index: (usize, usize), it: T) {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.vals.len()
        } else {
            self.col_starts[col+1]
        };

        let mut vals_insert_pos = col_start_idx;
        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                self.vals[i] = it;
                return;
            } else if self.row_pos[i] > row {
                vals_insert_pos = i;
                break;
            }
        }

        self.vals.insert(vals_insert_pos, it);
        for (i, idx) in self.col_starts.iter_mut().enumerate() {
            if i > col {
                *idx += 1;
            }
        }
        self.row_pos.insert(vals_insert_pos, row);
    }

    pub fn remove(&mut self, index: (usize, usize)) {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.vals.len()
        } else {
            self.col_starts[col+1]
        };

        let mut vals_remove_pos = usize::MAX;
        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                vals_remove_pos = i;
                break;
            }
        }

        if vals_remove_pos == usize::MAX {
            return;
        }
        let _ = self.vals.remove(vals_remove_pos);
        let _ = self.row_pos.remove(vals_remove_pos);
        for i in self.col_starts.iter_mut() {
            if *i > vals_remove_pos {
                *i -= 1;
            }
        }
    }
}

// index by tuple: slower
impl<T: Zero + Copy> ops::Index<(usize, usize)> for SparseYale<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.get((row,col)).unwrap()
    }
}

impl<T: Zero + Copy> ops::IndexMut<(usize, usize)> for SparseYale<T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        self.get_mut((row,col)).unwrap()
    }
}


impl<T: Copy + Zero + fmt::Debug> fmt::Debug for SparseYale<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<SparseYale vals={:?} col_starts={:?} row_pos={:?}>", self.vals, self.col_starts, self.row_pos)
    }
}


impl<T: Copy + Zero + fmt::Display> fmt::Display for SparseYale<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, ""));
        for j in 0 .. self.dim.0 {
            if j == 0 {
                try!(write!(f, "[["));
            } else {
                try!(write!(f, " ["));
            }
            for i in 0 .. self.dim.1 {
                try!(write!(f, "{:-4.}", self.get((j,i)).unwrap()));
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
fn test_sparse_yale() {
    let mut m = SparseYale {
        dim: (6, 6),
        vals: vec![10, 45, 40, 2, 4, 3, 3, 9, 19, 7],
        col_starts: vec![0, 3, 5, 8, 8, 8],
        row_pos: vec![0, 1, 3, 0, 2, 0, 1, 2, 0, 5],
        zero: 0,
    };

    let _ = m.get_mut((3,0)).map(|i| { *i = 233; });
    println!("got mat =>\n{}", m);
}

#[test]
fn test_sparse_matrix_from_vec() {
    let mat = SparseYale::from_vec(
        vec![vec![12, 23, 43, 0, 0, 0],
             vec![0,  40, 0,  0, 1, 0],
             vec![0,  0,  0,  0, 0, 2]]);

    assert_eq!(mat.shape(), (3,6));
    assert_eq!(mat.get((1,1)), Some(&40));
    assert_eq!(mat.get((2,3)), Some(&0));
}



#[test]
fn test_sparse_matrix_build() {
    let mut m: SparseYale<i32> = SparseYale::zeros(10, 4);
    m.insert((1,1), 23);
    m.insert((2,3), 22);
    m.insert((5,3), 233);
    m.insert((2,2), -3);

    assert_eq!(m.get((5,3)), Some(&233));

    assert_eq!(m[(2,3)], 22);
    assert_eq!(m[(2,2)], -3);
    m.remove((2,2));
    assert_eq!(m[(2,2)], 0);
}
