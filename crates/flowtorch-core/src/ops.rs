#![allow(dead_code)]

use crate::Tensor;

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone)]
pub(crate) enum Op {
    Binary(Tensor, Tensor, BinaryOp),
}

pub(crate) trait BinaryOpT {
    const NAME: &'static str;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
    fn u8(v1: u8, v2: u8) -> u8;
    fn u32(v1: u32, v2: u32) -> u32;
    fn i32(v1: i32, v2: i32) -> i32;
    fn i64(v1: i64, v2: i64) -> i64;
}

pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;
            fn f32(v1: f32, v2: f32) -> f32 {
                $e(v1, v2)
            }
            fn f64(v1: f64, v2: f64) -> f64 {
                $e(v1, v2)
            }

            fn u8(v1: u8, v2: u8) -> u8 {
                $e(v1, v2)
            }

            fn u32(v1: u32, v2: u32) -> u32 {
                $e(v1, v2)
            }

            fn i32(v1: i32, v2: i32) -> i32 {
                $e(v1, v2)
            }

            fn i64(v1: i64, v2: i64) -> i64 {
                $e(v1, v2)
            }
        }
    };
}

//Add Binry Op for add
// This works like below, we will pass the function expr and call it in macro
// let add = |v1, v2| v1 + v2;
// let result = add(5, 3);
bin_op!(Add, "add", |v1, v2| v1 + v2);
bin_op!(Mul, "mul", |v1, v2| v1 * v2);
bin_op!(Sub, "sub", |v1, v2| v1 - v2);
bin_op!(Div, "div", |v1, v2| v1 / v2);
