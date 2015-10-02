/// A sparse matrix in COOrdinate format.
/// Also known as the 'ijv' or 'triplet' format.
pub struct CooMatrix<T> {
    shape: (usize, usize),
    data: Vec<T>,
    row: Vec<usize>,        // COO format row index array of the matrix
    col: Vec<usize>         // COO format column index array of the matrix
}

impl CooMatrix<T> {

}
