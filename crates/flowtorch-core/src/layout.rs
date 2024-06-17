#![allow(dead_code)]

use crate::{
    error::{Error, LayoutError},
    shape::Shape,
};

pub type Stride = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Layout {
    shape: Shape,
    stride: Stride,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    pub fn stride(&self) -> Stride {
        self.stride.clone()
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    // Return true if the array is C contiguous (aka Row Major)
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    pub fn broadcast_as(&self, _shape: Shape) -> Result<Self, Error> {
        todo!()
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, Error> {
        let dims = self.shape.dims();
        if dim >= dims.len() {
            return Err(
                LayoutError::Narrow(String::from("Dim not in current Dimenesions.")).into(),
            );
        }
        let mut dims = dims.to_vec();
        dims[dim] = len;

        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }
}
