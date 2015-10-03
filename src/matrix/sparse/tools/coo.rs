use std::iter;

/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_row      - number of rows in A
 *   I  n_col      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note:
 *   Input:  row and column indices *are not* assumed to be ordered
 *
 *   Note: duplicate entries are carried over to the CSR represention
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 *
 */
pub fn to_csr<T: Copy + Sized>(nrow: usize, _ncol: usize, nnz: usize,
                                   ai: &[usize], aj: &[usize], ax: &[T]) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    //bp: &mut [usize], bj: &mut [usize], bx: &[T]) {

    let mut bp: Vec<usize> = iter::repeat(0).take(nrow+1).collect();
    let mut bj: Vec<usize> = iter::repeat(0).take(nnz).collect();
    let mut bx: Vec<T> = Vec::with_capacity(nnz);

    unsafe {
        bx.set_len(nnz);
    }

    for n in 0 .. nnz {
        bp[ai[n]] += 1;
    }

    let mut cumsum = 0;
    for i in 0 .. nrow {
        let temp = bp[i];
        bp[i] = cumsum;
        cumsum += temp;
    }

    bp[nrow] = nnz;

    // write Aj,Ax into Bj,Bx
    for n in 0 .. nnz {
        let row = ai[n];
        let dest = bp[row];

        bj[dest] = aj[n];
        bx[dest] = ax[n];

        bp[row] += 1;
    }

    let mut last = 0;
    for i in 0 .. nrow+1 {
        let temp = bp[i];
        bp[i] = last;
        last = temp;
    }
    (bp, bj, bx)
}


pub fn to_csc<T: Copy + Sized>(nrow: usize, ncol: usize, nnz: usize, ai: &[usize], aj: &[usize], ax: &[T]) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    to_csr(ncol, nrow, nnz, aj, ai, ax)
}
