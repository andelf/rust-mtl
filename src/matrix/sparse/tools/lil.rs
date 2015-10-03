use num::traits::Zero;

/*
 Get a single item from LIL matrix.
    Doesn't do output type conversion. Checks for bounds errors.
    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    Returns
    -------
    x
        Value at indices.
 */
pub fn get1<'t, T: Copy>(m: usize, n: usize, rows: &[Vec<usize>], datas: &'t [Vec<T>], i: usize, j: usize) -> Option<&'t T> {
    assert!(i < m, "row index {} out of bounds", i);
    assert!(j < n, "column index {} out of bounds", j);

    let row = &rows[i];
    let data = &datas[i];

    let pos = bisect_left(row, &j);

    if pos != data.len() && row[pos] == j {
        Some(&data[pos])
    } else {
        None
    }
}

pub fn get1_mut<'t, T: Copy>(m: usize, n: usize, rows: &[Vec<usize>], datas: &'t mut [Vec<T>], i: usize, j: usize) -> Option<&'t mut T> {
    assert!(i < m, "row index {} out of bounds", i);
    assert!(j < n, "column index {} out of bounds", j);

    let row = &rows[i];
    let data = &mut datas[i];

    let pos = bisect_left(row, &j);

    if pos != data.len() && row[pos] == j {
        Some(&mut data[pos])
    } else {
        None
    }
}


fn bisect_left<T: Copy + PartialOrd>(a: &[T], x: &T) -> usize {
    let mut hi = a.len();
    let mut lo = 0;
    let mut mid: usize;

    while lo < hi {
        mid = lo + (hi - lo) / 2;
        if &a[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}


/*
  Insert a single item to LIL matrix.
    Checks for bounds errors and deletes item if x is zero.
    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.
 */
// TODO add PartialEq
pub fn insert<T: Zero + Copy>(m: usize, n: usize, rows: &mut [Vec<usize>], datas: &mut [Vec<T>], i: usize, j: usize, x: T) {
    assert!(i < m, "row index {} out of bounds", i);
    assert!(j < n, "column index {} out of bounds", j);

    let row = &mut rows[i];
    let data = &mut datas[i];

    let pos = bisect_left(row, &j);
    // if x == Zero::zero() {
    //     if pos < row.len() && row[pos] == j {
    //         row.remove(pos);
    //         data.remove(pos);
    //     }
    // } else {
        if pos == row.len() {
            row.push(j);
            data.push(x);
        } else if row[pos] != j {
            row.insert(pos, j);
            data.insert(pos, x);
        } else {
            data[pos] = x;
        }
//    }
}
