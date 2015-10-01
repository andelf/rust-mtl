//! sparse matrix

use std::iter;
use std::fmt;
use std::usize;
use std::ops;
use std::str::FromStr;
use std::collections::BTreeMap;

use num::traits::Zero;

use super::ParseMatrixError;
// use super::super::array::Array;

use self::SparseMatrix::*;


pub enum SparseMatrix<T> {
    /// Block Sparse Row matrix
    Bsr {
        shape: (usize, usize),
        data: Vec<T>,           // CSR format data array of the matrix
        indices: Vec<usize>,    // CSR format index array
        indptr: Vec<usize>,     // CSR format index pointer array
        block_size: usize
    },
    /// A sparse matrix in COOrdinate format
    Coo {
        shape: (usize, usize),
        data: Vec<T>,
        row: Vec<usize>,        // COO format row index array of the matrix
        col: Vec<usize>         // COO format column index array of the matrix
    },
    /// Compressed Sparse Column matrix
    Csc {
        shape: (usize, usize),
        // nnz: usize,
        data: Vec<T>,           // Data array of the matrix
        indices: Vec<usize>,    // CSC format index array
        indptr: Vec<usize>      // CSC format index pointer array
    },
    /// Compressed Sparse Row matrix
    Csr {
        shape: (usize, usize),
        // nnz: usize,
        data: Vec<T>,           // CSR format data array of the matrix
        indices: Vec<usize>,    // CSR format index array
        indptr: Vec<usize>      // CSR format index pointer array
    },
    /// Sparse matrix with DIAgonal storage
    Dia {
        shape: (usize, usize),
        data: Vec<Vec<T>>,           // DIA format data array of the matrix
        offsets: Vec<isize>          // DIA format offset array of the matrix
    },
    /// Dictionary Of Keys based sparse matrix
    Dok {
        shape: (usize, usize),
        data: BTreeMap<(usize,usize), T>
    },
    /// Row-based linked list sparse matrix
    Lil {
        shape: (usize, usize),
        data: Vec<Vec<T>>,
        rows: Vec<Vec<usize>>
    }
}


impl<T: Zero + Copy> SparseMatrix<T> {
    /// Shape of the matrix
    pub fn shape(&self) -> Vec<usize> {
        match *self {
            Csc { shape, .. } => vec![shape.0, shape.1],
            Csr { shape, .. } => vec![shape.0, shape.1],
            _                 => unimplemented!()
        }
    }


    /// Number of dimensions (this is always 2)
    pub fn ndim(&self) -> usize { 2 }


    /// Number of nonzero elements
    pub fn nnz(&self) -> usize {
        unimplemented!()
    }

}






/// Sparse matrix, Yale repersentation
pub struct SparseYale<T> {
    zero: T,
    shape: (usize, usize),
    data: Vec<T>,
    col_starts: Vec<usize>,
    row_pos: Vec<usize>,
}

impl<T: Copy + Zero + PartialEq> SparseYale<T> {
    pub fn from_vec(vec: Vec<Vec<T>>) -> Self {
        let nrow = vec.len();
        let ncol = vec[0].len();

        let mut mat = SparseYale::zeros(nrow, ncol);
        for (i, row) in vec.into_iter().enumerate() {
            for (j, item) in row.into_iter().enumerate() {
                if item != mat.zero {
                    mat.insert((i,j), item);
                }
            }
        }
        mat
    }
}

impl<T: Copy + Zero> SparseYale<T> {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        SparseYale {
            shape: (nrow, ncol),
            data: vec![],
            col_starts: iter::repeat(0).take(ncol).collect(),
            row_pos: vec![],
            zero: Zero::zero()
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    #[inline]
    fn nrow(&self) -> usize {
        self.shape.0
    }

    #[inline]
    fn ncol(&self) -> usize {
        self.shape.1
    }

    /// Returns the element of the given index, or None if the index is out of bounds.
    pub fn get(&self, index: (usize, usize)) -> Option<&T> {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.data.len()
        } else {
            self.col_starts[col+1]
        };

        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                return self.data.get(i);
            }
        }
        Some(&self.zero)
    }

    pub unsafe fn get_unchecked(&self, index: (usize, usize)) -> &T {
        let (row, col) = index;
        let ndata = self.data.len();

        let mut col_start_idx = self.col_starts[col];
        loop {
            if self.row_pos[col_start_idx] > row {
                return &self.zero;
            } else if self.row_pos[col_start_idx] == row {
                return &self.data[col_start_idx];
            }
            col_start_idx += 1;
            if col_start_idx >= ndata || (col < self.ncol()-1 && col_start_idx >= self.col_starts[col+1]) {
                return &self.zero;
            }
        }
    }

    /// Returns the element mutable ref of the given index, None if not exists
    pub fn get_mut(&mut self, index: (usize, usize)) -> Option<&mut T> {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.data.len()
        } else {
            self.col_starts[col+1]
        };

        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                return self.data.get_mut(i);
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
            self.data.len()
        } else {
            self.col_starts[col+1]
        };

        let mut data_insert_pos = col_start_idx;
        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                self.data[i] = it;
                return;
            } else if self.row_pos[i] > row {
                data_insert_pos = i;
                break;
            }
        }

        self.data.insert(data_insert_pos, it);
        for (i, idx) in self.col_starts.iter_mut().enumerate() {
            if i > col {
                *idx += 1;
            }
        }
        self.row_pos.insert(data_insert_pos, row);
    }

    pub fn remove(&mut self, index: (usize, usize)) {
        let (row, col) = index;
        assert!(row < self.nrow());
        assert!(col < self.ncol());

        let col_start_idx = self.col_starts[col];
        let col_end_idx = if col == self.ncol() - 1 {
            self.data.len()
        } else {
            self.col_starts[col+1]
        };

        let mut data_remove_pos = usize::MAX;
        for i in col_start_idx .. col_end_idx {
            if self.row_pos[i] == row {
                data_remove_pos = i;
                break;
            }
        }

        if data_remove_pos == usize::MAX {
            return;
        }
        let _ = self.data.remove(data_remove_pos);
        let _ = self.row_pos.remove(data_remove_pos);
        for i in self.col_starts.iter_mut() {
            if *i > data_remove_pos {
                *i -= 1;
            }
        }
    }

    pub fn transpose(&self) -> Self {
        let (nrow, ncol) = self.shape;
        SparseYale {
            shape: (ncol, nrow),
            data: vec![],
            col_starts: iter::repeat(0).take(ncol).collect(),
            row_pos: vec![],
            zero: Zero::zero()
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

impl<T: PartialEq + Zero + Copy + fmt::Debug> FromStr for SparseYale<T>
    where T: FromStr, <T as FromStr>::Err: fmt::Debug {

    type Err = ParseMatrixError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let nrow = s.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).count();
        let ncol = s.split(';').next().unwrap()
            .split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .count();

        let mut idx = 0;

        let mut mat = SparseYale::zeros(nrow, ncol);
        let _ = s.split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .map(|seg| {
                let v = seg.parse().unwrap();
                if v != mat.zero {
                    mat.insert((idx / ncol, idx % ncol), v);
                }
                idx += 1;
            }).count();

        Ok(mat)
    }
}

impl<T: Copy + Zero + fmt::Debug> fmt::Debug for SparseYale<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<SparseYale data={:?} col_starts={:?} row_pos={:?}>", self.data, self.col_starts, self.row_pos)
    }
}


/* Display a sparse matrix:
  (0, 0)        3
  (0, 2)        1
  (1, 1)        2
  (3, 3)        1
*/
impl<T: Copy + Zero + fmt::Display> fmt::Display for SparseYale<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for j in 0 .. self.shape.0 {
            if j == 0 {
                try!(write!(f, "[["));
            } else {
                try!(write!(f, " ["));
            }
            for i in 0 .. self.shape.1 {
                try!(write!(f, "{:-4.}", self.get((j,i)).unwrap()));
                if i == self.shape.1 - 1 {
                    try!(write!(f, "]"));
                } else {
                    try!(write!(f, ", "));
                }
            }
            if j == self.shape.0 - 1 {
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
        shape: (6, 6),
        data: vec![10, 45, 40, 2, 4, 3, 3, 9, 19, 7],
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
fn test_sparse_matrix_from_str() {
    let mat: SparseYale<i32> = FromStr::from_str("3 0 1 0; 0 2 0 0; 0 0 0 0; 0 0 0 1").unwrap();
    assert_eq!(mat.data.len(), 4); // assert sparse
    println!("Debug => {:?}", mat);

    assert_eq!(mat[(3,3)], 1);
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
