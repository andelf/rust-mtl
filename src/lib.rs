#![feature(unboxed_closures, core, fixed_size_array, zero_one, iter_arith)]
#![warn(bad_style, unused, unused_import_braces,
        unused_qualifications, unused_results)]


extern crate num;
extern crate rand;
extern crate core;


#[macro_export]
macro_rules! ix {
    ($($arg:expr),*) => (
        vec![$( Box::new($arg) as Box<::mtl::array::ArrayIx> ),*]
    )
}

pub mod array;
pub mod matrix;
