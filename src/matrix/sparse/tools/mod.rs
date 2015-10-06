pub mod coo;
pub mod csr;
pub mod csc;
pub mod lil;

/*
pub fn max<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a >= b {
        a
    } else {
        b
    }
}

*/

pub fn min<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a <= b {
        a
    } else {
        b
    }
}

pub fn bisect_left<T: Copy + PartialOrd>(a: &[T], x: &T) -> usize {
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
