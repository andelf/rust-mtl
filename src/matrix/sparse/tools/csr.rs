use std::iter;
use std::usize;

use num::traits::Zero;
use num::zero;

/*
 * Compute B = A for CSR matrix A, CSC matrix B
 *
 * Also, with the appropriate arguments can also be used to:
 *   - compute B = A^t for CSR matrix A, CSR matrix B
 *   - compute B = A^t for CSC matrix A, CSC matrix B
 *   - convert CSC->CSR
 *
 * Input Arguments:
 *   I  n_row         - number of rows in A
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - row pointer
 *   I  Aj[nnz(A)]    - column indices
 *   T  Ax[nnz(A)]    - nonzeros
 *
 * Output Arguments:
 *   I  Bp[n_col+1] - column pointer
 *   I  Bj[nnz(A)]  - row indices
 *   T  Bx[nnz(A)]  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, Bx must be preallocated
 *
 * Note:
 *   Input:  column indices *are not* assumed to be in sorted order
 *   Output: row indices *will be* in sorted order
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 *
 */
pub fn to_csc<T: Copy>(nrow: usize, ncol: usize, ap: &[usize], aj: &[usize], ax: &[T]) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    let nnz = ap[nrow];

    let mut bp: Vec<usize> = iter::repeat(0).take(ncol+1).collect();
    let mut bi: Vec<usize> = iter::repeat(0).take(nnz).collect();
    let mut bx: Vec<T> = Vec::with_capacity(nnz);

    unsafe {
        bx.set_len(nnz);
    }

    for n in 0 .. nnz {
        bp[aj[n]] += 1;
    }

    // cumsum the nnz per column to get Bp[]
    let mut cumsum = 0;
    for col in 0 .. ncol {
        let temp = bp[col];
        bp[col] = cumsum;
        cumsum += temp;
    }
    bp[ncol] = nnz;

    for row in 0 .. nrow {
        for jj in ap[row] .. ap[row+1] {
            let col = aj[jj];
            let dest = bp[col];

            bi[dest] = row;
            bx[dest] = ax[jj];

            bp[col] += 1;
        }
    }

    let mut last = 0;
    for col in 0 .. ncol+1 {
        let temp = bp[col];
        bp[col] = last;
        last = temp;
    }

    (bp, bi, bx)
}


/*
 * Sort CSR column indices inplace
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  Ap[n_row+1]     - row pointer
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros
 *
 */
pub fn sort_indices<T: Copy>(nrow: usize, ap: &[usize], aj: &mut [usize], ax: &mut [T]) {
    let mut temp: Vec<(usize,T)> = vec![];

    for i in 0 .. nrow {
        let row_start = ap[i];
        let row_end = ap[i+1];

        temp.clear();

        for jj in row_start .. row_end {
            temp.push((aj[jj], ax[jj]));
        }

        temp.sort_by(|a,b| a.0.cmp(&b.0));

        for (n, jj) in (row_start .. row_end).enumerate() {
            aj[jj] = temp[n].0;
            ax[jj] = temp[n].1;
        }
    }
}


/*
 * Determine whether the matrix structure is canonical CSR.
 * Canonical CSR implies that column indices within each row
 * are (1) sorted and (2) unique.  Matrices that meet these
 * conditions facilitate faster matrix computations.
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  Ap[n_row+1]     - row pointer
 *   I  Aj[nnz(A)]      - column indices
 *
 */
pub fn has_canonical_format(nrow: usize, ap: &[usize], aj: &[usize]) -> bool {
    for i in 0 .. nrow {
        if ap[i] > ap[i+1] {
            return false;
        }
        for jj in ap[i]+1 .. ap[i+1] {
            if !(aj[jj-1] < aj[jj]) {
                return false;
            }
        }
    }
    true
}


/*
 * Compute C = A (binary_op) B for CSR matrices that are not
 * necessarily canonical CSR format.  Specifically, this method
 * works even when the input matrices have duplicate and/or
 * unsorted column indices within a given row.
 *
 * Refer to csr_binop_csr() for additional information
 *
 * Note:
 *   Output arrays Cp, Cj, and Cx must be preallocated
 *   If nnz(C) is not known a priori, a conservative bound is:
 *          nnz(C) <= nnz(A) + nnz(B)
 *
 * Note:
 *   Input:  A and B column indices are not assumed to be in sorted order
 *   Output: C column indices are not generally in sorted order
 *           C will not contain any duplicate entries or explicit zeros.
 *
 */
pub fn binop_csr_general<T: Zero + PartialOrd + Copy, U: Zero + PartialOrd + Copy, F>(
    nrow: usize, ncol: usize,
    ap: &[usize], aj: &[usize], ax: &[T],
    bp: &[usize], bj: &[usize], bx: &[T],
    mut f: F) -> (Vec<usize>, Vec<usize>, Vec<U>) where F: FnMut(T,T) -> U {

    let nnz = ap[nrow] + bp[nrow];

    let mut cp: Vec<usize> = iter::repeat(0).take(ncol+1).collect();
    let mut cj: Vec<usize> = iter::repeat(0).take(nnz).collect();
    let mut cx: Vec<U> = Vec::with_capacity(nnz);

    let mut next: Vec<usize> = iter::repeat(usize::MAX).take(ncol).collect();
    let mut a_row: Vec<T> = iter::repeat(Zero::zero()).take(ncol).collect();
    let mut b_row: Vec<T> = iter::repeat(Zero::zero()).take(ncol).collect();

    let mut nnz = 0;

    cp[0] = 0;

    for i in 0 .. nrow {
        let mut head = usize::MAX - 1;
        let mut length = 0;

        // add a row of A to A_row
        let i_start = ap[i];
        let i_end = ap[i+1];

        for jj in i_start .. i_end {
            let j = aj[jj];

            a_row[j] = ax[jj];

            if next[j] == usize::MAX {
                next[j] = head;
                head = j;
                length += 1;
            }
        }

        // add a row of B to B_row
        let i_start = bp[i];
        let i_end = bp[i+1];

        for jj in i_start .. i_end {
            let j = bj[jj];

            b_row[j] = bx[jj];

            if next[j] == usize::MAX {
                next[j] = head;
                head = j;
                length += 1;
            }
        }

        // scan through columns where A or B has
        // contributed a non-zero entry
        for _ in 0 .. length {
            let result = f(a_row[head], b_row[head]);

            if result != Zero::zero() {
                cj[nnz] = head;
                cx[nnz] = result;
                nnz += 1;
            }

            let temp = head;
            head = next[head];

            next[temp] = usize::MAX;
            a_row[temp] = zero();
            b_row[temp] = zero();
        }

        cp[i+1] = nnz;
    }
    (cp, cj, cx)
}
