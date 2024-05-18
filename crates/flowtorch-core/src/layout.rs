#![allow(dead_code)]
use crate::shape::Shape;

pub type Strides = Vec<usize>;

#[derive(Debug)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, strides: Vec<usize>, offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    pub fn get_strides(&self) -> Strides {
        self.strides.clone()
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, offset: usize) -> Self {
        let shape = shape.into();
        let strides = shape.strides_contiguous();
        Self {
            shape,
            strides,
            offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }
}
