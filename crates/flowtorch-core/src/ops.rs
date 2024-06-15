#![allow(dead_code, unused_variables)]

use crate::Tensor;

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

#[derive(Debug)]
pub enum UnaryOp {
    Sqr,
    Sqrt,
    Neg,
    Abs,
    Log,
    Sin,
    Cos,
    Tan,
    Exp,
    Ceil,
    Floor,
}

#[derive(Debug)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

#[derive(Debug)]
pub(crate) enum Op {
    Binary(Tensor, Tensor, BinaryOp),
    Unary(Tensor, UnaryOp),
    Cmp(Tensor, CmpOp),
}

pub(crate) trait BinaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32, v2: f32) -> f32 {
        unimplemented!("{} op not implemented for f32", Self::NAME)
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        unimplemented!("{} op not implemented for f64", Self::NAME)
    }
    fn u8(v1: u8, v2: u8) -> u8 {
        unimplemented!("{} op not implemented for u8", Self::NAME)
    }
    fn u32(v1: u32, v2: u32) -> u32 {
        unimplemented!("{} op not implemented for u32", Self::NAME)
    }
    fn i32(v1: i32, v2: i32) -> i32 {
        unimplemented!("{} op not implemented for i32", Self::NAME)
    }
    fn i64(v1: i64, v2: i64) -> i64 {
        unimplemented!("{} op not implemented for i64", Self::NAME)
    }
}

pub(crate) trait UnaryOpT {
    const NAME: &'static str;
    const KERNEL: &'static str;
    const V: Self;
    fn f32(v1: f32) -> f32 {
        unimplemented!("{} op not implemented for f32", Self::NAME)
    }
    fn f64(v1: f64) -> f64 {
        unimplemented!("{} op not implemented for f64", Self::NAME)
    }
    fn u8(v1: u8) -> u8 {
        unimplemented!("{} op not implemented for u8", Self::NAME)
    }
    fn u32(v1: u32) -> u32 {
        unimplemented!("{} op not implemented for u32", Self::NAME)
    }
    fn i32(v1: i32) -> i32 {
        unimplemented!("{} op not implemented for i32", Self::NAME)
    }
    fn i64(v1: i64) -> i64 {
        unimplemented!("{} op not implemented for i64", Self::NAME)
    }
}

// Binary Ops
pub(crate) struct Add;
pub(crate) struct Div;
pub(crate) struct Mul;
pub(crate) struct Sub;
pub(crate) struct Maximum;
pub(crate) struct Minimum;

// Unary Op
pub(crate) struct Sqr;
pub(crate) struct Sqrt;
pub(crate) struct Neg;
pub(crate) struct Log;
pub(crate) struct Sin;
pub(crate) struct Cos;
pub(crate) struct Tan;
pub(crate) struct Abs;
pub(crate) struct Ceil;
pub(crate) struct Floor;
pub(crate) struct Exp;

macro_rules! bin_op {
    ($op:ident, $name: literal, $e: expr) => {
        impl BinaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("b", $name);
            const V: Self = $op;
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
bin_op!(Maximum, "max", |v1, v2| if v1 > v2 { v1 } else { v2 });
bin_op!(Minimum, "min", |v1, v2| if v1 > v2 { v2 } else { v1 });

macro_rules! unary_op {
    ($op:ident, $name: literal, $a: ident, $e: expr) => {
        impl UnaryOpT for $op {
            const NAME: &'static str = $name;
            const KERNEL: &'static str = concat!("u", $name);
            const V: Self = $op;
            fn f32($a: f32) -> f32 {
                $e
            }

            fn f64($a: f64) -> f64 {
                $e
            }
        }
    };
}

unary_op!(Sqr, "sqr", v, v * v);
unary_op!(Sqrt, "sqrt", v, v.sqrt());
unary_op!(Log, "log", v, v.ln());
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tan, "tan", v, v.tan());
unary_op!(Exp, "exp", v, v.exp());

// Adding other Ops which support additional types
impl UnaryOpT for Neg {
    const NAME: &'static str = "neg";
    const KERNEL: &'static str = "uneg";
    const V: Self = Neg;

    fn f32(v1: f32) -> f32 {
        -v1
    }

    fn f64(v1: f64) -> f64 {
        -v1
    }

    fn i32(v1: i32) -> i32 {
        -v1
    }

    fn i64(v1: i64) -> i64 {
        -v1
    }
}

impl UnaryOpT for Abs {
    const NAME: &'static str = "abs";
    const KERNEL: &'static str = "uabs";
    const V: Self = Abs;

    fn f32(v1: f32) -> f32 {
        v1.abs()
    }

    fn f64(v1: f64) -> f64 {
        v1.abs()
    }

    fn u8(v1: u8) -> u8 {
        v1
    }

    fn u32(v1: u32) -> u32 {
        v1
    }

    fn i32(v1: i32) -> i32 {
        v1.abs()
    }

    fn i64(v1: i64) -> i64 {
        v1.abs()
    }
}

impl UnaryOpT for Ceil {
    const NAME: &'static str = "ceil";
    const KERNEL: &'static str = "uceil";
    const V: Self = Ceil;

    fn f32(v1: f32) -> f32 {
        v1.ceil()
    }

    fn f64(v1: f64) -> f64 {
        v1.ceil()
    }

    fn u8(v1: u8) -> u8 {
        v1
    }

    fn u32(v1: u32) -> u32 {
        v1
    }

    fn i32(v1: i32) -> i32 {
        v1
    }

    fn i64(v1: i64) -> i64 {
        v1
    }
}

impl UnaryOpT for Floor {
    const NAME: &'static str = "floor";
    const KERNEL: &'static str = "ufloor";
    const V: Self = Floor;

    fn f32(v1: f32) -> f32 {
        v1.floor()
    }

    fn f64(v1: f64) -> f64 {
        v1.floor()
    }

    fn u8(v1: u8) -> u8 {
        v1
    }

    fn u32(v1: u32) -> u32 {
        v1
    }

    fn i32(v1: i32) -> i32 {
        v1
    }

    fn i64(v1: i64) -> i64 {
        v1
    }
}
