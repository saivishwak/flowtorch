#![allow(dead_code)]

use std::ops::{Add, Mul};

use crate::{Error, Tensor};

#[derive(Debug, Clone)]
pub(crate) enum Op {
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
}

impl Add for Tensor {
    type Output = Result<Self, Error>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add_(&rhs)
    }
}

impl Mul for Tensor {
    type Output = Result<Self, Error>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_(&rhs)
    }
}
