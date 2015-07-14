use std::fmt;
use std::ops;

// sub matrix view
pub struct Dense2DView<'a, T: 'a> {
    pub inner: &'a [T],
    pub indices_map: Vec<usize>,
    pub dim: (usize, usize)
}


pub struct Dense2DMutView<'a, T: 'a> {
    pub inner: &'a mut [T],
    pub indices_map: Vec<usize>,
    pub dim: (usize, usize)
}

impl<'a, T> ops::Index<(usize, usize)> for Dense2DView<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        &self.inner[self.indices_map[row * self.dim.1 + col]]
    }
}


impl<'a, T: fmt::Debug + Copy> fmt::Debug for Dense2DView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let data = self.indices_map.iter().map(|&i| self.inner[i]).collect::<Vec<T>>();
        try!(write!(f, "<MatrixView dim={:?}, {:?}>", self.dim, data));
        Ok(())
    }
}

impl<'a, T: fmt::Display> fmt::Display for Dense2DView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "");
        for j in 0 .. self.dim.0 {
            if j == 0 {
                write!(f, "[[");
            } else {
                write!(f, " [");
            }
            for i in 0 .. self.dim.1 {
                write!(f, "{:-4.}", self[(j,i)]);
                if i == self.dim.1 - 1 {
                    write!(f, "]");
                } else {
                    write!(f, ", ");
                }
            }
            if j == self.dim.0 - 1 {
                write!(f, "]");
            } else {
                writeln!(f, "");
            }
        }
        Ok(())
    }
}
