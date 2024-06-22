use std::fmt::Display;

use crate::layout::Stride;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape {:?}", self.0)
    }
}

impl Shape {
    // Method to create a new Shape
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    // Method to get the dimensions
    pub fn dims(&self) -> &Vec<usize> {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    pub fn is_1d(&self) -> bool {
        self.0.len() == 1
    }

    // Method to calculate the number of elements (product of dimensions)
    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn stride_contiguous(&self) -> Vec<usize> {
        let shape = &self.0;
        let mut stride: Vec<usize> = vec![0; shape.len()];

        if shape.is_empty() {
            return stride;
        }

        //as this is first time insertion, the last element is 1
        stride[shape.len() - 1] = 1;

        //Early return as 1D array has strides as 1
        if shape.len() == 1 {
            return stride;
        }

        for i in (0..=shape.len() - 2).rev() {
            stride[i] = shape[i + 1] * stride[i + 1];
        }

        stride
    }

    // Return true if the array is C contiguous (aka Row Major)
    pub fn is_contiguous(&self, strides: &Stride) -> bool {
        if self.0.len() != strides.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in strides.iter().zip(self.0.iter()).rev() {
            if dim > 1 && acc != stride {
                return false;
            }
            acc *= dim;
        }
        true
    }

    // Return true if the array is Fotran contiguous (aka Column Major)
    pub fn is_fotran_contiguous(&self, strides: &Stride) -> bool {
        if self.0.len() != strides.len() {
            return false;
        }
        let mut acc = 1;
        for (&stride, &dim) in strides.iter().zip(self.0.iter()) {
            if dim > 1 && acc != stride {
                return false;
            }
            acc *= dim;
        }
        true
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self(shape.0.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

impl From<&Shape> for Vec<usize> {
    fn from(shape: &Shape) -> Self {
        shape.0.clone()
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self(vec![])
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self(vec![d1])
    }
}

impl From<(usize,)> for Shape {
    fn from(d1: (usize,)) -> Self {
        Self(vec![d1.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d12: (usize, usize)) -> Self {
        Self(vec![d12.0, d12.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d123: (usize, usize, usize)) -> Self {
        Self(vec![d123.0, d123.1, d123.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d1234: (usize, usize, usize, usize)) -> Self {
        Self(vec![d1234.0, d1234.1, d1234.2, d1234.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(d12345: (usize, usize, usize, usize, usize)) -> Self {
        Self(vec![d12345.0, d12345.1, d12345.2, d12345.3, d12345.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(d123456: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self(vec![
            d123456.0, d123456.1, d123456.2, d123456.3, d123456.4, d123456.5,
        ])
    }
}
