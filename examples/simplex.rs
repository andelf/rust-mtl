#[macro_use]
extern crate mtl;

use mtl::array::Array;

use mtl::array::concatenate;


/*
maximize:  13 * A + 23 * B
sbject to:
            5 * A + 15 * B <=  480
            4 * A +  4 * B <=  160
           35 * A + 20 * B <= 1190
*/

#[allow(non_snake_case)]
fn main() {
    let A = Array::from_vec(vec![5.0, 15.0, 4.0,
                                 4.0, 35.0, 20.0]).reshape([3, 2]);

    let b = Array::from_vec(vec![480.0, 160.0, 1190.0]).reshape([3,1]);
    let c = Array::from_vec(vec![13.0, 23.0]).reshape([1,2]);

    let mut M = concatenate([concatenate([A, Array::eye(3),    b], 1),
                             concatenate([c, Array::zeros([1,4])], 1)], 0);

    println!("M => \n{}", M);

    // fixme: to_array :(
    let mut col = M.slice(ix![3, ..]).to_array().argmax();

    let mut row = (M.slice(ix![..3, -1]) / M.slice(ix![..3, col])).argmin();

    println!("pivot M[{},{}] = {}", row, col, M[[row,col]]);

    // round 1
    let temp_row = M.slice(ix![row, ..]) / M[[row,col]];
    M.slice_mut(ix![row, ..]).move_from(temp_row);

    for r in 0 .. 4 {
        if r != row {
            let temp_row = M.slice(ix![r,..]) - M.slice(ix![row,..]) * M[[r,col]];
            M.slice_mut(ix![r, ..]).move_from(temp_row);
        }
    }

    println!("M => \n{}", M);

    col = 0;
    row = (M.slice(ix![..3,-1]) / M.slice(ix![..3,col])).argmin();

    println!("pivot M[{},{}] = {}", row, col, M[[row,col]]);

    // round 2
    let temp_row = M.slice(ix![row, ..]) / M[[row,col]];
    M.slice_mut(ix![row, ..]).move_from(temp_row);
    for r in 0 .. 4 {
        if r != row {
            let temp_row = M.slice(ix![r,..]) - M.slice(ix![row,..]) * M[[r,col]];
            M.slice_mut(ix![r, ..]).move_from(temp_row);
        }
    }

    println!("M => \n{}", M);
}
