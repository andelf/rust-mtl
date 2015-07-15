use std::fmt;
use std::ops;
use num::traits::Zero;

// sub matrix view
pub struct Dense2DView<'a, T: 'a> {
    pub inner: &'a [T],
    pub orig_dim: (usize, usize),
    pub offset: (usize, usize),
    pub dim: (usize, usize)
}
pub struct Dense2DMutView<'a, T: 'a> {
    pub inner: &'a mut [T],
    pub orig_dim: (usize, usize),
    pub offset: (usize, usize),
    pub dim: (usize, usize)
}


impl<'a, T> Dense2DView<'a, T> {
    #[inline]
    fn map_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.dim.0 {
            None
        } else if col >= self.dim.1 {
            None
        } else {
            Some((self.offset.0 + row) * self.orig_dim.1 + self.offset.1 + col)
        }
    }
}

impl<'a, T> ops::Index<(usize, usize)> for Dense2DView<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        match self.map_index(row, col) {
            Some(i) => &self.inner[i],
            None    => panic!("index over range")
        }
    }
}


impl<'a, T> ops::Index<usize> for Dense2DView<'a, T> {
    type Output = [T];

    #[inline]
    fn index<'b>(&'b self, row: usize) -> &'b [T] {
        assert!(row < self.dim.0);
        let start = self.map_index(row, 0).unwrap();
        let end  = self.map_index(row, self.dim.1 - 1).unwrap();
        &self.inner[start .. end + 1]
    }
}

// impl<'a, T> ops::IndexMut<usize> for Dense2DView<'a, T> {
//     #[inline]
//     fn index_mut<'a>(&'a mut self, row: usize) -> &'a mut [T] {
//         assert!(row < self.dim.0);
//         &mut self.data[
//     }
// }



impl<'a, T: fmt::Debug + Copy> fmt::Debug for Dense2DView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut view = Vec::new();
        for j in 0 .. self.dim.0 {
            for i in 0 .. self.dim.1 {
                view.push(self[(j,i)]);
            }
        }
        try!(write!(f, "<MatrixView dim={:?}, {:?}>", self.dim, view));
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
