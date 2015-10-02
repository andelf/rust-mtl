//! sparse matrix

use std::iter;
use std::fmt;
use std::ops;
use std::str::FromStr;
use std::collections::BTreeMap;

use num::traits::Zero;

use super::ParseMatrixError;
// use super::super::array::Array;

use self::SparseMatrix::*;


#[derive(Clone, Debug)]
pub enum SparseMatrix<T> {
    /// Block Sparse Row matrix
    Bsr {
        shape: (usize, usize),
        data: Vec<T>,           // CSR format data array of the matrix
        indices: Vec<usize>,    // CSR format index array
        indptr: Vec<usize>,     // CSR format index pointer array
        block_size: usize
    },
    /// A sparse matrix in COOrdinate format.
    /// Also known as the 'ijv' or 'triplet' format.
    Coo {
        shape: (usize, usize),
        data: Vec<T>,
        rows: Vec<usize>,        // COO format row index array of the matrix
        cols: Vec<usize>         // COO format column index array of the matrix
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
    pub fn shape(&self) -> (usize, usize) {
        match *self {
            Bsr { shape, .. } | Coo { shape, .. } | Csc { shape, .. } | Csr { shape, .. } |
            Dia { shape, .. } | Dok { shape, .. } | Lil { shape, .. } => shape
        }
    }

    /// Number of dimensions (this is always 2)
    pub fn ndim(&self) -> usize { 2 }


    /// Number of nonzero elements
    pub fn nnz(&self) -> usize {
        match *self {
            Coo { ref data, .. } => data.len(),
            _ => unimplemented!()
        }
    }

    // creation
    pub fn new_coo(shape: (usize,usize), data: Vec<T>, ij: Vec<(usize,usize)>) -> Self {
        Coo {
            shape: shape,
            data: data,
            rows: ij.iter().map(|&(row,_col)| row).collect(),
            cols: ij.iter().map(|&(_row,col)| col).collect()
        }
    }

    pub fn empty_coo(shape: (usize, usize)) -> Self {
        Coo {
            shape: shape,
            data: vec![],
            rows: vec![],
            cols: vec![]
        }
    }

    pub fn empty_csc(shape: (usize, usize)) -> Self {
        Csc {
            shape: shape,
            data: vec![],
            indices: vec![],
            indptr: iter::repeat(0).take(shape.1 + 1).collect()
        }
    }

    // get/set
    /// Returns the element of the given index, or None if not exists
    pub fn get(&self, index: (usize, usize)) -> Option<&T> {
        let (row, col) = index;
        assert!(row < self.shape().0);
        assert!(col < self.shape().1);

        match *self {
            Csc { ref data, ref indptr, ref indices, .. } => {
                let col_start_idx = indptr[col];
                let col_end_idx = indptr[col+1];

                for i in col_start_idx .. col_end_idx {
                    if indices[i] == row {
                        return data.get(i);
                    }
                }
                None
            },
            Coo { ref data, ref rows, ref cols, .. } => {
                for i in 0 .. data.len() {
                    if rows[i] == row && cols[i] == col {
                        return data.get(i);
                    }
                }
                None
            },
            _ => unimplemented!()
        }
    }

    /// Returns mutable ref of the element of the given index, None if not exists
    pub fn get_mut(&mut self, index: (usize, usize)) -> Option<&mut T> {
        let (row, col) = index;
        assert!(row < self.shape().0);
        assert!(col < self.shape().1);

        match *self {
            Csc { ref mut data, ref indptr, ref indices, .. } => {
                let col_start_idx = indptr[col];
                let col_end_idx = indptr[col+1];

                for i in col_start_idx .. col_end_idx {
                    if indices[i] == row {
                        return data.get_mut(i);
                    }
                }
            },
            Coo { ref mut data, ref rows, ref cols, .. } => {
                for i in 0 .. data.len() {
                    if rows[i] == row && cols[i] == col {
                        return data.get_mut(i);
                    }
                }
            },
            _ => unimplemented!()
        }
        None
    }

    pub fn set(&mut self, index: (usize, usize), it: T) {
        let (row, col) = index;
        assert!(row < self.shape().0);
        assert!(col < self.shape().1);

        match *self {
            Csc { ref mut data, ref mut indptr, ref mut indices, .. } => {
                let col_start_idx = indptr[col];
                let col_end_idx = indptr[col+1];

                let mut data_insert_pos = col_start_idx;
                for i in col_start_idx .. col_end_idx {
                    if indices[i] == row {
                        data[i] = it;
                        return;
                    } else if indices[i] > row {
                        data_insert_pos = i;
                        break;
                    }
                }

                println!("WARNING: Changing the sparsity structure of a CSC matrix is expensive.");
                data.insert(data_insert_pos, it);
                for (i, idx) in indptr.iter_mut().enumerate() {
                    if i > col {
                        *idx += 1;
                    }
                }
                indices.insert(data_insert_pos, row);
            },
            _ => unimplemented!()
        }
    }


    // convertion
    pub fn to_csc(&self) -> Self {
        if self.nnz() == 0 {
            return Self::empty_csc(self.shape());
        }
        match *self {
            ref this @ Coo { .. } => {
                // let (m, n) = shape;
                // let indptr = iter::repeat(0usize).take(n+1).collect();
                // let indices = iter::repeat.take(this.nnz()).collect();
                // let data = iter::repeat(Zero::zero()).take(this.nnz()).collect();
                unimplemented!()
            },
            ref this @ Csc { .. } => this.clone(),
            _ => unimplemented!()
        }
    }

    pub fn to_coo(&self) -> Self {
        match *self {
            Csc { shape, ref data, ref indptr, ref indices } => {
                let mut vals = vec![];
                let mut rows = vec![];
                let mut cols = vec![];
                for (j, &ptr) in indptr.iter().take(shape.1).enumerate() {
                    for (off, val) in data[ptr .. indptr[j+1]].iter().enumerate() {
                        let i = indices[ptr + off];
                        vals.push(*val);
                        rows.push(i);
                        cols.push(j);
                    }
                }
                Coo {
                    shape: shape,
                    data: vals,
                    rows: rows,
                    cols: cols
                }
            },
            ref this @ Coo { .. } => this.clone(),
            _ => unimplemented!()
        }
    }
}


impl<T: fmt::Display> fmt::Display for SparseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Coo { ref data, ref rows, ref cols, .. } => {
                for i in 0 .. data.len() {
                    try!(writeln!(f, "  {:?}\t{}", (rows[i], cols[i]), data[i]));
                }
            },
            Csc { ref shape, ref data, ref indptr, ref indices } => {
                for (j, &ptr) in indptr.iter().take(shape.1).enumerate() {
                    for (off, val) in data[ptr .. indptr[j+1]].iter().enumerate() {
                        let i = indices[ptr + off];
                        try!(writeln!(f, "  {:?}\t{}", (i, j), val));
                    }
                }
            },
            _ => unimplemented!()
        }
        Ok(())
    }
}




/*
impl<T: PartialEq + Zero + Copy + fmt::Debug> FromStr for SparseMatrix<T>
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
}*/


#[test]
fn test_sparse_matrix_build() {
    let mut mat: SparseMatrix<f32> = SparseMatrix::empty_csc((4,3));

    mat.set((0,0), 1.0);

    mat.set((1,2), 2.0);

    println!("mat => \n{}", mat);
    println!("mat => {:?}", mat);
}
