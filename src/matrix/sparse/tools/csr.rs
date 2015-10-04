use std::iter;

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
