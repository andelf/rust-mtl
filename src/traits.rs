pub trait ArrayType<T> {
    fn shape(&self) -> Vec<usize>;
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn get<D: AsRef<[usize]>>(&self, index: D) -> Option<&T>;
    fn get_mut<D: AsRef<[usize]>>(&mut self, index: D) -> Option<&mut T>;

    unsafe fn get_unchecked<D: AsRef<[usize]>>(&self, index: D) -> &T;
    unsafe fn get_unchecked_mut<D: AsRef<[usize]>>(&mut self, index: D) -> &mut T;

    fn size(&self) -> usize {
        self.shape().nelem()
    }
}


/// array shape for creating Array
pub trait ArrayShape {
    fn to_shape_vec(&self) -> Vec<usize>;
    fn ndim(&self) -> usize {
        self.to_shape_vec().len()
    }
    fn nelem(&self) -> usize {
        self.to_shape_vec().iter().product()
    }
}

impl<'a, S: ArrayShape + ?Sized> ArrayShape for &'a S {
    fn to_shape_vec(&self) -> Vec<usize> {
        // must deref or leads to recusive
        (*self).to_shape_vec()
    }
}

impl ArrayShape for Vec<usize> {
    fn to_shape_vec(&self) -> Vec<usize> {
        self.clone()
    }
}

impl ArrayShape for [usize] {
    fn to_shape_vec(&self) -> Vec<usize> {
        self.to_vec()
    }
}

// usize is for MxM matrix
impl ArrayShape for usize {
    fn to_shape_vec(&self) -> Vec<usize> {
        vec![*self, *self]
    }
}

impl ArrayShape for (usize,) {
    fn to_shape_vec(&self) -> Vec<usize> {
        vec![self.0]
    }
}

impl ArrayShape for (usize, usize) {
    fn to_shape_vec(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl ArrayShape for (usize, usize, usize) {
    fn to_shape_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

macro_rules! impl_array_shape_for_fixed_size_array {
    ($typ:ty, $size:expr) => (
        impl ArrayShape for [$typ; $size] {
            fn to_shape_vec(&self) -> Vec<usize> {
                self.iter().map(|&i| i as usize).collect()
            }
        }
    )
}

impl_array_shape_for_fixed_size_array!(usize, 1);
impl_array_shape_for_fixed_size_array!(usize, 2);
impl_array_shape_for_fixed_size_array!(usize, 3);
// impl_array_shape_for_fixed_size_array!(i32, 1);
