#![allow(dead_code)]
use crate::shape::Shape;

pub type Stride = Vec<usize>;

#[derive(Debug)]
pub struct Layout {
    shape: Shape,
    stride: Stride,
    pub offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, offset: usize) -> Self {
        Self {
            shape,
            stride,
            offset,
        }
    }

    pub fn get_shape(&self) -> Shape {
        self.shape.clone()
    }

    pub fn get_stride(&self) -> Stride {
        self.stride.clone()
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    // Return true if the array is C contiguous (aka Row Major)
    pub fn is_contiguous(&self) -> bool {
        return self.shape.is_contiguous(&self.stride);
    }
}
