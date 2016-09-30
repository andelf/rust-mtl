#![feature(unboxed_closures, fixed_size_array)]
#![warn(bad_style, unused, unused_import_braces,
        unused_qualifications, unused_results)]
#![allow(unknown_lints)]
#![allow(many_single_char_names)]


extern crate num;
extern crate rand;
extern crate core;


#[macro_export]
macro_rules! ix {
    ($($arg:expr),*) => (
        vec![$( Box::new($arg) as Box<::mtl::ndarray::ArrayIx> ),*]
    )
}

pub mod traits;
pub mod ndarray;
pub mod matrix;
