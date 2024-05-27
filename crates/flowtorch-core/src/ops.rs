#![allow(dead_code)]

use crate::Tensor;

pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
}
