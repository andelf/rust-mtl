use super::csr;


pub fn to_csr<T: Copy>(nrow: usize, ncol: usize, ap: &[usize], ai: &[usize], ax: &[T]) -> (Vec<usize>, Vec<usize>, Vec<T>) {
    csr::to_csc(ncol, nrow, ap, ai, ax)
}
