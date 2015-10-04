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

mod tools;

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
            Coo { ref data, .. } | Csr { ref data, .. } | Csc { ref data, .. } => data.len(),
            Lil { ref data, .. } => data.iter().map(|rs| rs.len()).sum(),
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

    pub fn empty_csr(shape: (usize, usize)) -> Self {
        Csr {
            shape: shape,
            data: vec![],
            indices: vec![],
            indptr: iter::repeat(0).take(shape.0 + 1).collect()
        }
    }

    pub fn empty_lil(shape: (usize, usize)) -> Self {
        let nrow = shape.0;
        Lil {
            shape: shape,
            data: iter::repeat(vec![]).take(nrow).collect(),
            rows: iter::repeat(vec![]).take(nrow).collect()
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
            Csr { ref data, ref indptr, ref indices, .. } => {
                let row_start_idx = indptr[row];
                let row_end_idx = indptr[row+1];

                for i in row_start_idx .. row_end_idx {
                    if indices[i] == col {
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
            Lil { shape, ref data, ref rows } => {
                tools::lil::get1(shape.0, shape.1, rows, data, row, col)
            }
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
            Csr { ref mut data, ref indptr, ref indices, .. } => {
                let row_start_idx = indptr[row];
                let row_end_idx = indptr[row+1];

                for i in row_start_idx .. row_end_idx {
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
            Lil { shape, ref mut data, ref rows } => {
                return tools::lil::get1_mut(shape.0, shape.1, rows, data, row, col)
            }
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
            Csr { ref mut data, ref mut indptr, ref mut indices, .. } => {
                let row_start_idx = indptr[row];
                let row_end_idx = indptr[row+1];

                let mut data_insert_pos = row_start_idx;
                for i in row_start_idx .. row_end_idx {
                    if indices[i] == col {
                        data[i] = it;
                        return;
                    } else if indices[i] > col {
                        data_insert_pos = i;
                        break;
                    }
                }

                println!("WARNING: Changing the sparsity structure of a CSR matrix is expensive.");
                data.insert(data_insert_pos, it);
                for (i, idx) in indptr.iter_mut().enumerate() {
                    if i > row {
                        *idx += 1;
                    }
                }
                indices.insert(data_insert_pos, col);
            },
            Coo { ref mut data, ref mut rows, ref mut cols, .. } => {
                data.push(it);
                rows.push(row);
                cols.push(col);
            }
            Lil { shape, ref mut data, ref mut rows, .. } => {
                tools::lil::insert(shape.0, shape.1, rows, data, row, col, it);
            }
            _ => unimplemented!()
        }
    }

    // convertion
    pub fn to_csc(&self) -> Self {
        let nnz = self.nnz();
        if nnz == 0 {
            return Self::empty_csc(self.shape());
        }
        match *self {
            Coo { shape, ref data, ref rows, ref cols } => {
                let (colptr, row_indices, vals) = tools::coo::to_csc(shape.0, shape.1, nnz, rows, cols, data);
                Csc {
                    shape: shape,
                    indptr: colptr,
                    indices: row_indices,
                    data: vals
                }
            },
            Csr { shape, ref data, ref indptr, ref indices } => {
                let (colptr, row_indices, vals) = tools::csr::to_csc(shape.0, shape.1, indptr, indices, data);
                Csc {
                    shape: shape,
                    indptr: colptr,
                    indices: row_indices,
                    data: vals
                }
            },
            ref this @ Csc { .. } => this.clone(),
            _ => unimplemented!()
        }
    }

    pub fn to_csr(&self) -> Self {
        let nnz = self.nnz();
        if nnz == 0 {
            return Self::empty_csr(self.shape());
        }
        match *self {
            Coo { shape, ref data, ref rows, ref cols } => {
                let (rowptr, col_indices, vals) = tools::coo::to_csr(shape.0, shape.1, nnz, rows, cols, data);
                Csr {
                    shape: shape,
                    indptr: rowptr,
                    indices: col_indices,
                    data: vals
                }
            },
            Csc { shape, ref data, ref indptr, ref indices } => {
                let (rowptr, col_indices, vals) = tools::csc::to_csr(shape.0, shape.1, indptr, indices, data);
                Csr { shape: shape, indptr: rowptr, indices: col_indices, data: vals }
            },
            ref this @ Csr { .. } => this.clone(),
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
                Coo { shape: shape, data: vals, rows: rows, cols: cols }
            },
            Csr { shape, ref data, ref indptr, ref indices } => {
                let mut vals = vec![];
                let mut rows = vec![];
                let mut cols = vec![];
                for (i, &ptr) in indptr.iter().take(shape.0).enumerate() {
                    for (off, val) in data[ptr .. indptr[i+1]].iter().enumerate() {
                        let j = indices[ptr + off];
                        vals.push(*val);
                        rows.push(i);
                        cols.push(j);
                    }
                }
                Coo { shape: shape, data: vals, rows: rows, cols: cols }
            },
            ref this @ Coo { .. } => this.clone(),
            _ => unimplemented!()
        }
    }

    pub fn to_lil(&self) -> Self {
        match *self {
            Csr { shape, ref data, ref indptr, ref indices } => {
                let nrow = shape.0;
                let mut dat = data.clone();
                let mut ind = indices.clone();

                tools::csr::sort_indices(nrow, indptr, &mut ind, &mut dat);

                let mut data: Vec<Vec<T>> = Vec::with_capacity(nrow);
                let mut rows: Vec<Vec<usize>> = Vec::with_capacity(nrow);

                for n in 0 .. nrow {
                    let start = indptr[n];
                    let end = indptr[n+1];

                    // push() means insert at n
                    rows.push(ind[start..end].to_vec());
                    data.push(dat[start..end].to_vec());
                }

                Lil {
                    shape: shape,
                    data: data,
                    rows: rows
                }
            },
            ref this @ Csc { .. } => this.to_csr().to_lil(),
            ref this @ Coo { .. } => this.to_csr().to_lil(),
            ref this @ Lil { .. } => this.clone(),
            _ => unimplemented!()
        }
    }

    // operation
    pub fn transpose(&self) -> Self {
        let new_shape = (self.shape().1, self.shape().0);
        match *self {
            Csc { ref data, ref indptr, ref indices, .. } => {
                Csr {
                    shape: new_shape,
                    data: data.clone(),
                    indptr: indptr.clone(),
                    indices: indices.clone()
                }
            },
            Csr { ref data, ref indptr, ref indices, .. } => {
                Csc {
                    shape: new_shape,
                    data: data.clone(),
                    indptr: indptr.clone(),
                    indices: indices.clone()
                }
            },
            Coo { ref data, ref rows, ref cols, .. } => {
                Coo {
                    shape: new_shape,
                    data: data.clone(),
                    rows: cols.clone(),
                    cols: rows.clone()
                }
            }
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
            Csr { ref shape, ref data, ref indptr, ref indices } => {
                for (i, &ptr) in indptr.iter().take(shape.0).enumerate() {
                    for (off, val) in data[ptr .. indptr[i+1]].iter().enumerate() {
                        let j = indices[ptr + off];
                        try!(writeln!(f, "  {:?}\t{}", (i, j), val));
                    }
                }
            },
            Lil { ref data, ref rows, .. } => {
                for (i, row) in rows.iter().enumerate() {
                    for (pos, &j) in row.iter().enumerate() {
                        try!(writeln!(f, "  {:?}\t{}", (i, j), data[i][pos]));
                    }
                }
            },
            _ => unimplemented!()
        }
        Ok(())
    }
}




impl<T: PartialEq + Zero + Copy + fmt::Debug> FromStr for SparseMatrix<T>
    where T: FromStr, <T as FromStr>::Err: fmt::Debug {

    type Err = ParseMatrixError;

    /// construct a matrx of COO format
    fn from_str(s: &str) -> Result<Self, Self::Err> {

        let mut vals = vec![];
        let mut rows = vec![];
        let mut cols = vec![];

        let nrow = s.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).count();
        let ncol = s.split(';').next().unwrap()
            .split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .count();

        let mut idx = 0;

        let _ = s.split(|c| c == ';' || c == ',' || c == ' ')
            .map(|seg| seg.trim())
            .filter(|seg| !seg.is_empty())
            .map(|seg| {
                if seg != "0" {
                    let v = seg.parse().unwrap();
                    vals.push(v);
                    rows.push(idx / ncol);
                    cols.push(idx % ncol);
                }
                idx += 1;
            }).count();
        Ok(Coo {
            shape: (nrow, ncol),
            data: vals,
            rows: rows,
            cols: cols
        })
    }
}


#[test]
fn test_sparse_matrix_build() {
    let mut mat: SparseMatrix<f32> = SparseMatrix::empty_csr((4,3));

    mat.set((0,0), 1.0);
    mat.set((1,2), 2.0);

    assert_eq!(mat.get((0,0)), Some(&1.));
    assert_eq!(mat.get((1,2)), Some(&2.));

    println!("mat.T => \n{}", mat.transpose());
    println!("mat.T => {:?}", mat.transpose());

    let mut mat: SparseMatrix<f32> = SparseMatrix::empty_coo((6,5));
    mat.set((0,0), 10.0);
    mat.set((1,0), 45.0);
    mat.set((3,0), 40.0);
    mat.set((0,1), 2.0);
    mat.set((2,1), 4.0);
    mat.set((0,2), 3.0);
    mat.set((1,2), 3.0);
    mat.set((2,2), 9.0);
    mat.set((0,4), 19.0);
    mat.set((5,4), 7.0);
    println!("mat => \n{}", mat);
    println!("mat => {:?}", mat);

    // LIL matrix
    let mut mat: SparseMatrix<i32> = SparseMatrix::Lil {
        shape: (4,4),
        data: vec![vec![3,1], vec![2,3], vec![], vec![1]],
        rows: vec![vec![0,2], vec![1,3], vec![], vec![3]]
    };
    println!("mat => \n{}", mat);

    assert_eq!(mat.get((1,3)), Some(&3));
    mat.set((1,2), 5);
    assert_eq!(mat.get((1,2)), Some(&5));
}


#[test]
fn test_parse_sparse_matrix() {
    let mat: SparseMatrix<i32> =
        r#"11 22 0 0 0 0 ;
           0 33 44 0 0 0 ;
           0 0 55 66 0 0 ;
           0 0 0 77 88 0 ;
           0 0 0 0 0 99  "#.parse().unwrap();

    assert!(mat.nnz() == 9);
    let m2 = mat.to_csr();
    let m3 = mat.to_csc();

    let m4 = m2.to_csc();
    let m5 = m4.to_csr();

    let m6 = m5.to_coo();

    let m7 = m5.to_lil();
    let m8 = m4.to_lil();

    for i in 0 .. 5 {
        for j in 0 .. 6 {
            assert!(mat.get((i,j)) == m2.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m3.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m4.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m5.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m6.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m7.get((i,j)), "mat[{:?}] equeals", (i,j));
            assert!(mat.get((i,j)) == m8.get((i,j)), "mat[{:?}] equeals", (i,j));
        }
    }
}
