#![feature(heap_api)]
extern crate mtl;


use std::rt::heap::stats_print;


fn main() {
    stats_print();
}
